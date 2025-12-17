from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str
    article: Optional[str] = None
    section: Optional[str] = None

_ARTICLE_RE = re.compile(r"(?im)^\s*(art[íi]?culo|art\.)\s+(\d+)\b.*$", re.MULTILINE)
_SECTION_RE = re.compile(r"(?im)^\s*(t[íi]tulo|cap[íi]tulo|secci[óo]n)\s+([ivxlcdm0-9]+)\b.*$", re.MULTILINE)

def _approx_token_len(s: str) -> int:
    # Aproximación razonable: ~4 chars por token en español
    return max(1, len(s) // 4)

def chunk_pages_legal_aware(
    doc_id: str,
    pages: List[Tuple[int, str]],
    chunk_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    """
    pages: list of (page_number, page_text)
    Estrategia:
      1) Intentar detectar artículos en el texto concatenado por páginas.
      2) Si hay artículos: cortar por artículo (manteniendo pages).
      3) Si no: chunking por tamaño aproximado de tokens con overlap.
    """
    full = "\n\n".join([f"[PAGE {p}]\n{t}" for p, t in pages])

    # Detectar artículos
    matches = list(_ARTICLE_RE.finditer(full))
    if len(matches) >= 2:
        chunks: List[Chunk] = []
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
            art_num = m.group(2)

            block = full[start:end].strip()
            page_nums = [int(x) for x in re.findall(r"\[PAGE (\d+)\]", block)]
            if not page_nums:
                p1 = pages[0][0]
                p2 = pages[-1][0]
            else:
                p1 = min(page_nums)
                p2 = max(page_nums)

            # Limpiar marcadores de página
            block_clean = re.sub(r"\[PAGE \d+\]\n?", "", block).strip()

            chunk_id = f"{doc_id}__art_{art_num}__{idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_start=p1,
                    page_end=p2,
                    text=block_clean,
                    article=art_num,
                    section=None,
                )
            )
        return chunks

    # Fallback: chunking fijo por tamaño
    flat_pages = []
    for p, t in pages:
        flat_pages.append((p, t))

    text_all = "\n\n".join([t for _, t in flat_pages]).strip()
    if not text_all:
        return []

    # Encontrar una sección “cercana” (opcional)
    sec_m = _SECTION_RE.search(text_all)
    section = sec_m.group(0).strip() if sec_m else None

    chunks: List[Chunk] = []
    step = max(1, chunk_tokens - overlap_tokens)

    # Troceo por caracteres aproximando tokens
    approx_chars_per_token = 4
    chunk_chars = chunk_tokens * approx_chars_per_token
    step_chars = step * approx_chars_per_token

    n = len(text_all)
    i = 0
    idx = 0
    while i < n:
        j = min(n, i + chunk_chars)
        block = text_all[i:j].strip()
        if not block:
            i += step_chars
            continue

        # Mapear páginas de forma aproximada: como tenemos texto por página,
        # aquí citamos el rango total del documento si no podemos mapear exacto.
        p1 = flat_pages[0][0]
        p2 = flat_pages[-1][0]

        chunk_id = f"{doc_id}__chunk__{idx:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_start=p1,
                page_end=p2,
                text=block,
                article=None,
                section=section,
            )
        )
        idx += 1
        i += step_chars

    return chunks
