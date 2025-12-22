from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List
import fitz  # PyMuPDF


@dataclass
class PageText:
    page: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str) -> List[PageText]:
    doc = fitz.open(pdf_path)
    try:
        pages: List[PageText] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = page.get_text("text") or ""
            txt = clean_text(txt)
            if txt:
                pages.append(PageText(page=i + 1, text=txt))
        return pages
    finally:
        doc.close()
