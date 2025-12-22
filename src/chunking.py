from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Chunk:
    """A single chunk of legal text with basic provenance metadata."""

    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str
    article: Optional[str] = None
    section: Optional[str] = None


# -----------------------------
# Normalization helpers
# -----------------------------
_NBSP = "\u00A0"
_ZERO_WIDTH = "\u200B"


# Encabezado REAL típico: ARTICULO 13. — / ARTÍCULO 13.- / ART. 13. —
# (lo usamos solo para “despegar” headers pegados a frases)
_HEADER_GLUE_RE = re.compile(
    r"""
    (?ix)
    (?<!\n)                              # no ya en línea nueva
    (?:\s+)                              # espacios
    (ART(?:[ÍI]CULO)?|ART\.)             # ARTICULO / ART. (en mayúsculas)
    \s*
    (?:N[º°O]\s*)?                       # N° / Nº / NO (OCR)
    (\d{1,4})                            # número
    \s*(?:[º°])?                         # ordinal opcional
    \s*(?:\.\s*[—\-–]|[—\-–]\s*)         # separador típico: ". —" o ".-" o "—" o "-"
    """,
    re.VERBOSE,
)


def normalize_legal_text(s: str) -> str:
    """Normalize common PDF/OCR artifacts that break matching.

    Fixes:
    - NBSP -> space
    - zero-width removed
    - If a REAL header got glued to previous text, force a newline before it:
        "... texto. ARTICULO 13. —" -> "... texto.\nARTICULO 13. —"

    IMPORTANT:
    - Do NOT split references like "en el artículo 248" inside a sentence.
      Only split when we detect a REAL header shape (uppercase + separator).
    """
    if not s:
        return s

    s = s.replace(_NBSP, " ").replace(_ZERO_WIDTH, "")

    # Despegar solo encabezados reales pegados a texto anterior.
    s = _HEADER_GLUE_RE.sub(r"\n\1 \2. —", s)

    return s


# -----------------------------
# Regexes (robust to PDF/OCR)
# -----------------------------
# Encabezado REAL de artículo (NO referencias dentro de la frase).
# Requisitos:
# - inicio de línea (^ con MULTILINE)
# - ARTICULO/ARTÍCULO/ART. (cualquier caso, pero típico encabezado)
# - número
# - separador típico (., -, —) justo después del número
_ARTICLE_RE = re.compile(
    r"""
    (?im)^\s*
    (art(?:[íi]culo)?|art\.)          # Articulo / Art. (permitimos minúsculas por OCR)
    \s*
    (?:n[º°o]\s*)?                    # N° / Nº / No (OCR)
    (?:\n\s*)?                        # salto opcional entre palabra y número
    (\d{1,4})                         # número
    (?:\s*[º°])?                      # ordinal opcional
    \s*
    (?:\.\s*[—\-–]|[—\-–]|\.)         # separador OBLIGATORIO de encabezado
    """,
    re.MULTILINE | re.VERBOSE,
)

# Section-like headings (fallback metadata only)
_SECTION_RE = re.compile(
    r"(?im)^\s*(t[íi]tulo|cap[íi]tulo|secci[óo]n)\s+[ivxlcdm0-9]+\b.*$",
    re.MULTILINE,
)


def chunk_pages_legal_aware(
    doc_id: str,
    pages: List[Tuple[int, str]],
    chunk_tokens: int = 800,
    overlap_tokens: int = 120,
    debug: bool = False,
) -> List[Chunk]:
    """Chunk a document trying to respect legal structure (Articles) when possible."""
    if not pages:
        return []

    # Normalize each page to avoid OCR/PDF artifacts.
    norm_pages = [(p, normalize_legal_text(t or "")) for p, t in pages]

    # Build a single string with explicit page markers for provenance.
    full = "\n\n".join([f"[PAGE {p}]\n{t}" for p, t in norm_pages])

    matches = list(_ARTICLE_RE.finditer(full))

    if debug:
        print(f"[chunking] doc_id={doc_id} article_matches={len(matches)}")
        for m in matches[:100]:
            hdr_line = m.group(0).strip().splitlines()[0]
            print(f"  - {hdr_line}")

    chunks: List[Chunk] = []

    # -----------------------------
    # Mode A: chunk by article headers
    # -----------------------------
    if len(matches) >= 1:
        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)

            block = full[start:end]

            # page provenance from markers inside the block
            page_nums = [int(x) for x in re.findall(r"\[PAGE (\d+)\]", block)]
            if page_nums:
                p1, p2 = min(page_nums), max(page_nums)
            else:
                p1, p2 = norm_pages[0][0], norm_pages[-1][0]

            art_num = m.group(2)

            # clean markers from text
            block_clean = re.sub(r"\[PAGE \d+\]\n?", "", block).strip()

            chunk_id = f"{doc_id}__art_{art_num}__{idx:04d}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_start=p1,
                    page_end=p2,
                    text=block_clean,
                    article=str(art_num),
                    section=None,
                )
            )
        return chunks

    # -----------------------------
    # Mode B: fallback chunking by size
    # -----------------------------
    text_all = "\n\n".join([t for _, t in norm_pages]).strip()
    if not text_all:
        return []

    sec_m = _SECTION_RE.search(text_all)
    section = sec_m.group(0).strip() if sec_m else None

    # Approx tokens -> chars (simple heuristic)
    chunk_chars = max(200, int(chunk_tokens) * 4)
    step_tokens = max(1, int(chunk_tokens) - int(overlap_tokens))
    step_chars = max(50, step_tokens * 4)

    p1 = norm_pages[0][0]
    p2 = norm_pages[-1][0]

    i = 0
    idx = 0
    while i < len(text_all):
        block = text_all[i : i + chunk_chars].strip()
        if not block:
            break

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
