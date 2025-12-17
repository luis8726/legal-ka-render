from __future__ import annotations
import pickle
import numpy as np
from typing import Dict, List, Tuple
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from config import CHROMA_DIR, BM25_PATH, META_PATH, TOPK_VECTOR, TOPK_BM25, TOPK_FINAL, W_VECTOR, W_BM25

def simple_tokenize_es(text: str) -> List[str]:
    return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if len(t) > 1]

def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

class HybridRetriever:
    def __init__(self):
        self.chroma = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        self.collection = self.chroma.get_or_create_collection(name="legal_chunks")
        with open(BM25_PATH, "rb") as f:
            obj = pickle.load(f)
        self.bm25: BM25Okapi = obj["bm25"]
        self.bm25_ids: List[str] = obj["ids"]

        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.id_to_meta: Dict[str, Dict] = meta["id_to_meta"]

    def retrieve(self, query_embedding: List[float], query_text: str) -> List[Dict]:
        # 1) Vector search
        vres = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOPK_VECTOR,
            include=["documents", "metadatas", "distances"],
        )

        v_ids = vres["ids"][0]
        v_docs = vres["documents"][0]
        v_metas = vres["metadatas"][0]
        v_dist = np.array(vres["distances"][0], dtype=float)  # cosine distance
        v_sim = 1.0 - v_dist
        v_sim_n = _minmax_norm(v_sim)

        vector_map = {
            cid: {"text": doc, "meta": meta, "v_score": float(vs)}
            for cid, doc, meta, vs in zip(v_ids, v_docs, v_metas, v_sim_n)
        }

        # 2) BM25
        q_tok = simple_tokenize_es(query_text)
        bm25_scores = np.array(self.bm25.get_scores(q_tok), dtype=float)
        # tomar topK_BM25 por score
        top_idx = np.argsort(-bm25_scores)[:TOPK_BM25]
        b_ids = [self.bm25_ids[i] for i in top_idx]
        b_scores = bm25_scores[top_idx]
        b_scores_n = _minmax_norm(b_scores)

        bm25_map = {cid: float(sc) for cid, sc in zip(b_ids, b_scores_n)}

        # 3) Merge + score híbrido
        candidates = set(list(vector_map.keys()) + list(bm25_map.keys()))
        results = []
        for cid in candidates:
            v = vector_map.get(cid, {}).get("v_score", 0.0)
            b = bm25_map.get(cid, 0.0)
            score = W_VECTOR * v + W_BM25 * b

            # Recuperar doc/meta desde chroma si viene solo de bm25
            if cid in vector_map:
                text = vector_map[cid]["text"]
                meta = vector_map[cid]["meta"]
            else:
                got = self.collection.get(ids=[cid], include=["documents", "metadatas"])
                text = got["documents"][0]
                meta = got["metadatas"][0]

            results.append({
                "chunk_id": cid,
                "score": float(score),
                "text": text,
                "meta": meta,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:TOPK_FINAL]
