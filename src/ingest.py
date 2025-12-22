from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from dotenv import load_dotenv

from openai import OpenAI
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from config import (
    PDF_DIR,
    CHROMA_DIR,
    BM25_PATH,
    META_PATH,
    INDEX_DIR,
    EMBED_MODEL,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
)

from pdf_extract import extract_pdf_pages
from chunking import chunk_pages_legal_aware


# -----------------------------
# Normalize paths
# -----------------------------
NORMALIZED_PDF_DIR = Path("data/pdfs_normalized")
MANIFEST_PATH = NORMALIZED_PDF_DIR / "manifest.json"
COLLECTION_NAME = "legal_chunks"


# -----------------------------
# Utils
# -----------------------------
def simple_tokenize_es(text: str) -> List[str]:
    return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if len(t) > 1]


def infer_doc_id(pdf_path: Path) -> str:
    # doc_id estable desde nombre de archivo
    stem = pdf_path.stem.lower().strip()
    stem = stem.replace(" ", "_")
    return stem


def load_manifest() -> List[Dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def build_manifest_by_normalized_file(rows: List[Dict]) -> Dict[str, Dict]:
    """
    normalized_file -> row
    """
    out: Dict[str, Dict] = {}
    for r in rows:
        nf = r.get("normalized_file")
        if nf:
            out[nf] = r
    return out


def pick_pdf_dir_and_manifest() -> tuple[Path, Dict[str, Dict]]:
    """
    Usa PDFs normalizados si están disponibles; sino fallback a PDF_DIR.
    """
    if NORMALIZED_PDF_DIR.exists() and MANIFEST_PATH.exists():
        rows = load_manifest()
        m = build_manifest_by_normalized_file(rows)
        pdfs = list(NORMALIZED_PDF_DIR.glob("*.pdf"))
        if m and pdfs:
            return NORMALIZED_PDF_DIR, m
    return PDF_DIR, {}


def norma_meta_from_manifest(pdf_path: Path, manifest_map: Dict[str, Dict]) -> Dict[str, str]:
    """
    Devuelve meta “fuente de verdad” desde manifest.
    pdf_path.name debe ser normalized_file.
    """
    row = manifest_map.get(pdf_path.name, {}) or {}

    tipo_norma = str(row.get("tipo_norma", "")).strip()
    siglas = str(row.get("siglas", "")).strip()
    numero = str(row.get("numero", "")).strip()

    # Documento nombre “humano”: detalle si existe, sino stem del normalized
    detalle = str(row.get("detalle", "")).strip()
    documento_nombre = detalle if detalle else pdf_path.stem

    # source_file: nombre original (pre-normalize)
    source_file = str(row.get("source_file", "")).strip() or pdf_path.name

    return {
        "tipo_norma": tipo_norma,
        "siglas": siglas,
        "numero": numero,
        "documento_nombre": documento_nombre,
        "source_file": source_file,
        "normalized_file": pdf_path.name,
    }


def norma_meta_fallback(pdf_path: Path) -> Dict[str, str]:
    """
    Fallback si no hay manifest (comportamiento original minimalista).
    """
    return {
        "tipo_norma": "",
        "siglas": "",
        "numero": "",
        "documento_nombre": pdf_path.stem,
        "source_file": pdf_path.name,
        "normalized_file": pdf_path.name,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY. Copiá .env.example a .env y completalo.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Elegir PDFs normalizados si existen
    pdf_dir, manifest_map = pick_pdf_dir_and_manifest()
    using_manifest = bool(manifest_map)

    pdf_files = sorted([p for p in pdf_dir.glob("*.pdf")])
    if not pdf_files:
        raise RuntimeError(f"No encontré PDFs en {pdf_dir}. Poné tus PDFs ahí.")

    print(f"📄 PDFs dir: {pdf_dir}")
    print(f"🧾 Manifest: {'OK' if using_manifest else 'NO (fallback)'}")

    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metas: List[Dict] = []
    bm25_docs_tokens: List[List[str]] = []
    bm25_ids: List[str] = []

    # Limpiar colección para re-ingesta
    try:
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    for pdf_path in tqdm(pdf_files, desc="Ingestando PDFs"):
        doc_id = infer_doc_id(pdf_path)

        pages = extract_pdf_pages(str(pdf_path))
        pages_tuples = [(p.page, p.text) for p in pages]

        if using_manifest:
            norma_meta = norma_meta_from_manifest(pdf_path, manifest_map)
        else:
            norma_meta = norma_meta_fallback(pdf_path)

        chunks = chunk_pages_legal_aware(
            doc_id=doc_id,
            pages=pages_tuples,
            chunk_tokens=DEFAULT_CHUNK_TOKENS,
            overlap_tokens=DEFAULT_OVERLAP_TOKENS,
        )

        for c in chunks:
            text = (c.text or "").strip()
            if not text:
                continue

            meta = {
                "doc_id": c.doc_id,
                "page_start": int(c.page_start),
                "page_end": int(c.page_end),

                # ✅ meta desde manifest (si existe)
                "tipo_norma": norma_meta.get("tipo_norma", ""),
                "siglas": norma_meta.get("siglas", ""),
                "numero": norma_meta.get("numero", ""),
                "documento_nombre": norma_meta.get("documento_nombre", ""),
                "source_file": norma_meta.get("source_file", pdf_path.name),
                "normalized_file": norma_meta.get("normalized_file", pdf_path.name),
            }

            # Artículo
            if c.article is not None:
                meta["article"] = str(c.article)          # legacy
                meta["articulo_nro"] = str(c.article)     # clave estable (exact-match)

            # Sección
            if c.section is not None:
                meta["section"] = str(c.section)

            all_texts.append(text)
            all_ids.append(c.chunk_id)
            all_metas.append(meta)

            bm25_docs_tokens.append(simple_tokenize_es(text))
            bm25_ids.append(c.chunk_id)

    # Embeddings (batch)
    embeddings: List[List[float]] = []
    batch_size = 64
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Creando embeddings"):
        batch = all_texts[i:i + batch_size]
        emb = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([e.embedding for e in emb.data])

    # Guardar en Chroma
    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metas,
    )

    # BM25
    bm25 = BM25Okapi(bm25_docs_tokens)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": bm25_ids, "docs_tokens": bm25_docs_tokens}, f)

    # Meta map para citas / display
    with open(META_PATH, "wb") as f:
        pickle.dump({"id_to_meta": {cid: m for cid, m in zip(all_ids, all_metas)}}, f)

    print("✅ Ingesta completa.")
    print(f"- PDFs usados: {len(pdf_files)}")
    print(f"- Chroma: {CHROMA_DIR}")
    print(f"- BM25:   {BM25_PATH}")
    print(f"- Meta:   {META_PATH}")
    print(f"- Chunks: {len(all_ids)}")


if __name__ == "__main__":
    main()
