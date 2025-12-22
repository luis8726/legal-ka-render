from __future__ import annotations

import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

from config import EMBED_MODEL
from retrieve import HybridRetriever
from answer import answer_question

st.set_page_config(page_title="Legal KA (Societario)", layout="wide")

# Carga .env solo en local; en Render se usan Environment Variables
load_dotenv()

# --- Validación de API key (mensaje compatible Render + local) ---
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "Falta OPENAI_API_KEY. En Render cargala en Environment Variables. "
        "En local podés usar un archivo .env."
    )
    st.stop()

st.title("⚖️ Chalk Legal powered by TCA")

@st.cache_resource
def get_retriever() -> HybridRetriever:
    return HybridRetriever()

@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI()

# Instancias cacheadas
try:
    retriever = get_retriever()
except Exception as e:
    st.error(f"Error inicializando el retriever: {e}")
    st.stop()

client = get_openai_client()

# ---------------------------
# Estado del chat (historial)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Podras hacer consultas sobre normativa societaria argentina. "
        }
    ]

# Render de historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Input estilo chat
q = st.chat_input("Escribí tu consulta…")

# Sidebar opcional: acciones del chat
with st.sidebar:
    st.header("Opciones")
    if st.button("🧹 Limpiar conversación"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Listo. Empecemos de nuevo 👋",
            }
        ]
        st.rerun()

# Procesamiento del turno
if q and q.strip():
    # 1) Mostrar el mensaje del usuario + guardarlo
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    # 2) Generar respuesta
    with st.chat_message("assistant"):

        # Embedding
        with st.spinner("Procesando la consulta."):
            emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding

        # Retrieval
        with st.spinner("Recuperando contexto."):
            chunks = retriever.retrieve(query_embedding=emb, query_text=q)

        # Detectar exact match (artículo explícito)
        exact_chunk = None
        if chunks and chunks[0].get("match_type") == "exact_metadata":
            exact_chunk = chunks[0]

        # (Opcional) Encabezado legal cuando hay exact match
        if exact_chunk:
            meta = exact_chunk.get("meta", {}) or {}
            documento = meta.get("documento_nombre") or meta.get("source_file") or "Documento"
            art = meta.get("articulo_nro") or meta.get("article") or ""
            numero = meta.get("numero") or ""
            p1 = meta.get("page_start")
            p2 = meta.get("page_end")
            fuente = meta.get("source_file") or meta.get("doc_id") or ""

            header = f"**{documento}"
            if art:
                header += f" – Artículo {art}"
            if numero:
                header += f" (Ley {numero})"
            header += "**"

            footer = []
            if fuente:
                footer.append(f"Fuente: {fuente}")
            if p1 is not None and p2 is not None:
                footer.append(f"páginas {p1}–{p2}")

            if footer:
                st.markdown(header + "  \n" + f"*{' · '.join(footer)}*")
            else:
                st.markdown(header)

        # LLM Answer
        with st.spinner("Generando respuesta."):
            # Si es exact match: usamos SOLO ese chunk y sin historial/memoria
            if exact_chunk:
                ans = answer_question(
                    q,
                    [exact_chunk],
                    memory_summary=None,
                    recent_messages=None,
                )
            else:
                ans = answer_question(
                    q,
                    chunks,
                    memory_summary=st.session_state.get("memory_summary"),
                    recent_messages=st.session_state.messages[-4:],
                )

        st.write(ans)

    # 3) Guardar respuesta en historial
    st.session_state.messages.append({"role": "assistant", "content": ans})
