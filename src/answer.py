from __future__ import annotations
from typing import List, Dict, Optional
from openai import OpenAI
from config import LLM_MODEL

SYSTEM = """
Sos un asistente legal especializado en derecho societario argentino.
Respondés únicamente con base en el CONTEXTO recuperado (fragmentos de documentos).

Reglas:
- Si el contexto NO alcanza para responder con certeza, decí: "No surge del material provisto".
- No inventes artículos, números, jurisprudencia, fechas ni definiciones.
- Citá siempre la fuente por afirmación relevante, usando el formato: (doc_id, pág X, chunk_id).
- Si hay conflicto entre fuentes, señalalo y citá ambas.
- Redactá en español jurídico claro y estructurado:
  Respuesta → Fundamento (con citas) → Alcances/Advertencias → Próximos pasos.
- La memoria conversacional NO es fuente legal.
- No es asesoramiento legal definitivo; es información basada en documentos provistos.
""".strip()


def _format_citation(meta: Dict, chunk_id: str) -> str:
    doc_id = meta.get("doc_id", "doc")
    p1 = meta.get("page_start", "?")
    p2 = meta.get("page_end", "?")
    page = f"{p1}" if p1 == p2 else f"{p1}-{p2}"
    return f"({doc_id}, pág {page}, {chunk_id})"


def build_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        meta = c["meta"]
        cid = c["chunk_id"]
        cite = _format_citation(meta, cid)

        header_bits = [
            f"doc_id={meta.get('doc_id')}",
            f"pages={meta.get('page_start')}-{meta.get('page_end')}",
            f"chunk_id={cid}",
        ]

        if meta.get("article"):
            header_bits.append(f"art={meta.get('article')}")

        if meta.get("section"):
            header_bits.append("section=" + meta.get("section")[:80].replace("\n", " "))

        parts.append(
            f"[SOURCE {' | '.join(header_bits)}]\n"
            f"CITA_SUGERIDA: {cite}\n"
            f"{c['text'].strip()}\n"
        )

    return "\n---\n".join(parts)


def answer_question(
    question: str,
    chunks: List[Dict],
    memory_summary: Optional[str] = None,
    recent_messages: Optional[List[Dict]] = None,
) -> str:
    client = OpenAI()
    context = build_context(chunks)

    memory_block = ""
    if memory_summary:
        memory_block += f"\nMEMORIA RESUMIDA DEL CHAT (no es fuente legal):\n{memory_summary}\n"

    if recent_messages:
        recent_txt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent_messages
        )
        memory_block += f"\nÚLTIMOS INTERCAMBIOS (contexto conversacional):\n{recent_txt}\n"

    user_prompt = f"""
{memory_block}

CONTEXTO LEGAL (usar exclusivamente esto como fuente):
{context}

PREGUNTA:
{question}

INSTRUCCIONES:
- Respondé solo con el CONTEXTO LEGAL.
- No uses la memoria como fuente jurídica.
- Cada afirmación relevante debe incluir una cita.
- Si no surge del material provisto, decilo explícitamente.
- Estructura: Respuesta → Fundamento → Alcances/Advertencias → Próximos pasos.
""".strip()

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.output_text
