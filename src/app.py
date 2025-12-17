from __future__ import annotations

import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

from config import EMBED_MODEL, CHROMA_DIR, BM25_PATH, META_PATH
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
# st.caption("RAG híbrido (Vector + BM25) con citas por página cuando es posible.")

# --- Panel de estado (debug) ---
""" 
with st.sidebar:
    st.header("Estado")
    st.write("RENDER_DISK_PATH:", os.getenv("RENDER_DISK_PATH", "(no seteado)"))
    st.write("CHROMA_DIR:", str(CHROMA_DIR))
    st.write("BM25_PATH:", str(BM25_PATH), "exists=", BM25_PATH.exists())
    st.write("META_PATH:", str(META_PATH), "exists=", META_PATH.exists())
    st.divider()
    st.caption("Si BM25/META no existen, el retriever debería degradar a solo vector.")
"""

@st.cache_resource
def get_retriever() -> HybridRetriever:
    return HybridRetriever()

@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI()

# Instanciar de forma cacheada
try:
    retriever = get_retriever()
except Exception as e:
    st.error(f"Error inicializando el retriever: {e}")
    st.stop()

client = get_openai_client()

q = st.text_area(
    "Consulta",
    height=110,
    placeholder="Ej: ¿Qué requisitos y órganos exige una SAS para su administración?"
)

run = st.button("Buscar y responder", type="primary")

if run and q.strip():
    # Spinner genérico: no menciona embeddings ni modelo
    with st.spinner("Procesando la consulta..."):
        emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding

    with st.spinner("Recuperando contexto..."):
        chunks = retriever.retrieve(query_embedding=emb, query_text=q)

    with st.spinner("Generando respuesta..."):
        ans = answer_question(q, chunks)

    st.subheader("✅ Respuesta")
    st.write(ans)
