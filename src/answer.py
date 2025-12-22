from __future__ import annotations
from typing import List, Dict, Optional
from openai import OpenAI
from config import LLM_MODEL


SYSTEM = """
Sos un asistente legal especializado en derecho societario argentino.

Respondés exclusivamente en base al CONTEXTO LEGAL recuperado (fragmentos de documentos).
No usás conocimiento externo ni memoria como fuente jurídica.

Reglas de oro:
- NO inventes artículos, textos, números, fechas ni definiciones.
- Si el texto completo de un artículo no está íntegramente en el contexto:
  * Explicá con precisión QUÉ SÍ surge del material aportado.
  * Aclaralo expresamente en la respuesta.
- Solo usá "No surge del material provisto" cuando:
  * No exista ningún fragmento relevante en el contexto.
- Citá siempre las fuentes por afirmación relevante.
- Si hay conflicto entre fuentes, señalalo y citá ambas.

Formato obligatorio:
Breve resumen clave →
Respuesta →
Fundamento (con citas) →
Alcances / Advertencias →
Próximos pasos

Esto no constituye asesoramiento legal definitivo.
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

        if meta.get("articulo_nro"):
            header_bits.append(f"art={meta.get('articulo_nro')}")

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

    if not chunks:
        # Caso extremo: no hay ningún fragmento relevante
        return (
            "Breve resumen clave\n"
            "No surge información relevante del material provisto.\n\n"
            "Respuesta\n"
            "No surge del material provisto contenido aplicable a la consulta formulada.\n\n"
            "Alcances / Advertencias\n"
            "La respuesta se limita estrictamente a los documentos incorporados al contexto.\n\n"
            "Próximos pasos\n"
            "Podés уточar la norma, artículo o documento específico para ampliar la búsqueda."
        )

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

CONTEXTO LEGAL (única fuente jurídica):
{context}

PREGUNTA:
{question}

INSTRUCCIONES:
- Respondé únicamente con el CONTEXTO LEGAL.
- Si el texto completo no está, explicá claramente el alcance de lo que sí surge.
- Cada afirmación relevante debe tener cita.
- No inventes ni completes con conocimiento externo.
- Estructura obligatoria:
  Breve resumen clave →
  Respuesta →
  Fundamento (con citas) →
  Alcances / Advertencias →
  Próximos pasos
""".strip()

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.output_text
