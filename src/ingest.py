from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from dotenv import load_dotenv

from openai import OpenAI
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from config import PDF_DIR, CHROMA_DIR, BM25_PATH, META_PATH, INDEX_DIR, EMBED_MODEL, DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS
from pdf_extract import extract_pdf_pages
from chunking import chunk_pages_legal_aware

def simple_tokenize_es(text: str) -> List[str]:
    # Tokenizador simple para BM25 (suficiente para 9 PDFs)
    return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if len(t) > 1]

def infer_doc_id(pdf_path: Path) -> str:
    # doc_id estable desde nombre de archivo
    stem = pdf_path.stem.lower().strip()
    stem = stem.replace(" ", "_")
    return stem

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
        name="legal_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metas: List[Dict] = []
    bm25_docs_tokens: List[List[str]] = []
    bm25_ids: List[str] = []

    pdf_files = sorted([p for p in PDF_DIR.glob("*.pdf")])
    if not pdf_files:
        raise RuntimeError(f"No encontré PDFs en {PDF_DIR}. Poné tus 9 PDFs ahí.")

    # Limpiar colección para re-ingesta (opcional: comentá si querés incremental)
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

        chunks = chunk_pages_legal_aware(
            doc_id=doc_id,
            pages=pages_tuples,
            chunk_tokens=DEFAULT_CHUNK_TOKENS,
            overlap_tokens=DEFAULT_OVERLAP_TOKENS,
        )

        for c in chunks:
            text = c.text.strip()
            if not text:
                continue

            meta = {
            "doc_id": c.doc_id,
            "page_start": int(c.page_start),
            "page_end": int(c.page_end),
            "source_file": pdf_path.name,
            }

            # Chroma no permite None -> solo agregamos si existe
            if c.article is not None:
                meta["article"] = str(c.article)

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
        batch = all_texts[i:i+batch_size]
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
    print(f"- Chroma: {CHROMA_DIR}")
    print(f"- BM25:   {BM25_PATH}")
    print(f"- Meta:   {META_PATH}")
    print(f"- Chunks: {len(all_ids)}")

if __name__ == "__main__":
    main()
