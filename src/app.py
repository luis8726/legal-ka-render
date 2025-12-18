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

    # 2) Generar respuesta (sin reenviar historial => no suma tokens por historial)
    with st.chat_message("assistant"):
        with st.spinner("Procesando la consulta..."):
            emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding

        with st.spinner("Recuperando contexto..."):
            chunks = retriever.retrieve(query_embedding=emb, query_text=q)

        with st.spinner("Generando respuesta..."):
            ans = answer_question(
                q,
                chunks,
                memory_summary=st.session_state.get("memory_summary"),
                recent_messages=st.session_state.messages[-4:],
            )


        st.write(ans)

    # 3) Guardar respuesta en historial
    st.session_state.messages.append({"role": "assistant", "content": ans})
