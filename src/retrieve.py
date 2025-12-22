from __future__ import annotations

import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from config import CHROMA_DIR, BM25_PATH, META_PATH, TOPK_VECTOR, TOPK_BM25, TOPK_FINAL, W_VECTOR, W_BM25


# -----------------------------
# Normalize / Manifest support
# -----------------------------
NORMALIZED_PDF_DIR = Path("data/pdfs_normalized")
MANIFEST_PATH = NORMALIZED_PDF_DIR / "manifest.json"


def load_manifest_map() -> Dict[str, Dict]:
    """
    normalized_file -> row
    """
    if not MANIFEST_PATH.exists():
        return {}
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out: Dict[str, Dict] = {}
    for row in data:
        nf = row.get("normalized_file")
        if nf:
            out[nf] = row
    return out


def build_alias_index(manifest_map: Dict[str, Dict]) -> Dict[str, Dict[str, str]]:
    """
    Arma aliases para resolver anclas a partir del texto de la consulta.
    Devuelve: alias_lower -> {"numero":..., "siglas":..., "tipo_norma":...}
    """
    aliases: Dict[str, Dict[str, str]] = {}

    for _, row in manifest_map.items():
        tipo = str(row.get("tipo_norma", "")).strip()
        siglas = str(row.get("siglas", "")).strip()
        numero = str(row.get("numero", "")).strip()
        source_file = str(row.get("source_file", "")).strip()
        normalized_file = str(row.get("normalized_file", "")).strip()
        detalle = str(row.get("detalle", "")).strip()

        payload = {"tipo_norma": tipo, "siglas": siglas, "numero": numero}

        def add_alias(a: str):
            a = (a or "").lower().strip()
            if not a:
                return
            aliases[a] = payload

        add_alias(source_file)
        add_alias(normalized_file)
        add_alias(Path(source_file).stem)
        add_alias(Path(normalized_file).stem)
        add_alias(detalle)

        # alias por numero
        if numero:
            add_alias(numero)
            add_alias(numero.replace("-", "/"))
            add_alias(numero.replace("/", "-"))

        # alias por siglas
        if siglas:
            add_alias(siglas)

    # aliases “universales” (sin hardcodear números)
    aliases["lgs"] = {"tipo_norma": "ley", "siglas": "lgs", "numero": ""}     # numero vacío: no lo forzamos
    aliases["ccycn"] = {"tipo_norma": "codigo", "siglas": "ccycn", "numero": ""}
    aliases["igj"] = {"tipo_norma": "resolucion", "siglas": "igj", "numero": ""}

    # si usaste sigla "sas" para ley 27349 en normalize
    aliases["sas"] = {"tipo_norma": "ley", "siglas": "sas", "numero": ""}

    return aliases


# -----------------------------
# Token / scoring
# -----------------------------
def simple_tokenize_es(text: str) -> List[str]:
    return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if len(t) > 1]


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# -----------------------------
# Chroma where helper
# -----------------------------
def chroma_where(eq_dict: Optional[Dict[str, str]]) -> Optional[Dict]:
    """
    {"a":"1","b":"2"} -> {"$and":[{"a":{"$eq":"1"}},{"b":{"$eq":"2"}}]}
    """
    if not eq_dict:
        return None
    items = []
    for k, v in eq_dict.items():
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue
        items.append({k: {"$eq": v}})
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return {"$and": items}


# -----------------------------
# Parser robusto (no usa \b por underscores)
# -----------------------------
def _extract_numero_generic(q: str) -> Optional[str]:
    ql = (q or "").lower()

    # 27.349 -> 27349
    m = re.search(r"(?<!\d)(\d{2})\.(\d{3})(?!\d)", ql)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    # 15/24, 09-20, 09_20 -> 15/24
    m = re.search(r"(?<!\d)(\d{1,3})[\/_\-](\d{2,4})(?!\d)", ql)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # 27349 / 19550 / 26994 etc (sin \b)
    m = re.search(r"(?<!\d)(\d{4,6})(?!\d)", ql)
    if m:
        return m.group(1)

    return None


def _extract_tipo_norma(q: str) -> Optional[str]:
    ql = (q or "").lower()
    if "ley" in ql:
        return "ley"
    if "decreto" in ql or "dec." in ql or "dnu" in ql:
        return "decreto"
    if "resol" in ql or "rg" in ql or "res." in ql:
        return "resolucion"
    if "codigo" in ql or "código" in ql or "ccycn" in ql:
        return "codigo"
    return None


def _extract_articulo(q: str) -> Optional[str]:
    ql = (q or "").lower()
    m = re.search(r"(?<!\w)art(?:[íi]culo)?\.?\s*(\d{1,4})(?!\d)", ql)
    return m.group(1) if m else None


def resolve_alias_anchors(query_text: str, alias_index: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Si la consulta contiene un alias del manifest (source_file, normalized_file, detalle, sigla),
    devolvemos anchors sugeridos.
    """
    ql = (query_text or "").lower()

    # Estrategia simple: buscar ocurrencia de alias como substring (no regex)
    for alias, payload in alias_index.items():
        if alias and alias in ql:
            return {k: v for k, v in payload.items() if v}

    return {}


def parse_query(query_text: str, alias_index: Dict[str, Dict[str, str]]) -> Dict[str, Optional[str]]:
    """
    Señales:
      - articulo_nro
      - tipo_norma
      - numero
      - siglas
    Con soporte manifest/normalize.
    """
    q = (query_text or "").lower()

    articulo = _extract_articulo(q)
    tipo_norma = _extract_tipo_norma(q)
    numero = _extract_numero_generic(q)

    siglas = None
    if "lgs" in q or "ley general de sociedades" in q:
        siglas = "lgs"
    elif "ccycn" in q:
        siglas = "ccycn"
    elif "igj" in q:
        siglas = "igj"
    elif "sas" in q:
        siglas = "sas"

    # Resolver anclas por alias del manifest
    alias_anchors = resolve_alias_anchors(query_text, alias_index)

    # Merge: el usuario explícito gana; si falta, completamos con alias
    if not tipo_norma and alias_anchors.get("tipo_norma"):
        tipo_norma = alias_anchors["tipo_norma"]
    if not siglas and alias_anchors.get("siglas"):
        siglas = alias_anchors["siglas"]
    if not numero and alias_anchors.get("numero"):
        numero = alias_anchors["numero"]

    return {
        "articulo_nro": articulo,
        "tipo_norma": tipo_norma,
        "numero": numero,
        "siglas": siglas,
    }


# -----------------------------
# Reranking rules
# -----------------------------
def _is_from_ley(meta: Dict) -> bool:
    if not meta:
        return False
    tipo = str(meta.get("tipo_norma", "")).lower()
    if tipo == "ley":
        return True
    # fallback
    v = " ".join([str(meta.get(k, "")) for k in ("source_file", "documento_nombre", "doc_id", "normalized_file")]).lower()
    return "ley" in v


def _matches_article_request(meta: Dict, intent: Dict[str, Optional[str]]) -> bool:
    if not meta:
        return False

    art = intent.get("articulo_nro")
    if not art:
        return False
    if str(meta.get("articulo_nro", "")).strip() != str(art).strip():
        return False

    for k in ("numero", "siglas", "tipo_norma"):
        v = intent.get(k)
        if v:
            if str(meta.get(k, "")).lower().strip() != str(v).lower().strip():
                return False
    return True


def keyword_bonus(text: str, intent: Dict[str, Optional[str]]) -> float:
    """
    Bonus suave si aparecen keywords ancla en el texto.
    NO reemplaza BM25; solo mejora el orden en empates.
    """
    t = (text or "").lower()
    bonus = 0.0

    # anclas
    if intent.get("numero") and intent["numero"].replace("/", "") in t.replace(".", "").replace("/", "").replace("-", ""):
        bonus += 0.04
    if intent.get("siglas") and intent["siglas"] in t:
        bonus += 0.03
    if intent.get("tipo_norma") and intent["tipo_norma"] in t:
        bonus += 0.01

    # si pide artículo, darle un empujón al “ARTICULO X”
    if intent.get("articulo_nro"):
        if re.search(rf"(?im)^\s*art(?:[íi]culo)?\.?\s*{re.escape(intent['articulo_nro'])}\b", text or ""):
            bonus += 0.06

    return bonus


# -----------------------------
# Retriever híbrido
# -----------------------------
class HybridRetriever:
    def __init__(self):
        self.chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(name="legal_chunks")

        with open(BM25_PATH, "rb") as f:
            obj = pickle.load(f)
        self.bm25: BM25Okapi = obj["bm25"]
        self.bm25_ids: List[str] = obj["ids"]

        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.id_to_meta: Dict[str, Dict] = meta["id_to_meta"]

        # Manifest
        self.manifest_map = load_manifest_map()
        self.alias_index = build_alias_index(self.manifest_map)

    def retrieve(self, query_embedding: List[float], query_text: str) -> List[Dict]:
        intent = parse_query(query_text, self.alias_index)

        has_article = bool(intent.get("articulo_nro"))

        # ---------------------------
        # 0) EXACT MATCH determinístico:
        #    articulo + (numero o siglas o tipo) => metadata
        #    Si no hay ancla, intentamos igual con articulo+siglas inferida (por alias)
        # ---------------------------
        where_exact: Dict[str, str] = {}
        if intent.get("articulo_nro"):
            where_exact["articulo_nro"] = intent["articulo_nro"]
        if intent.get("numero"):
            where_exact["numero"] = intent["numero"]
        if intent.get("siglas"):
            where_exact["siglas"] = intent["siglas"]
        if intent.get("tipo_norma"):
            where_exact["tipo_norma"] = intent["tipo_norma"]

        has_anchor = bool(intent.get("numero") or intent.get("siglas") or intent.get("tipo_norma"))

        if has_article and has_anchor:
            w = chroma_where(where_exact)
            if w is not None:
                exact = self.collection.get(where=w, include=["documents", "metadatas"])
                ids = exact.get("ids", [])
                if ids:
                    results = []
                    for cid, doc, meta in zip(exact["ids"], exact["documents"], exact["metadatas"]):
                        results.append(
                            {"chunk_id": cid, "score": 2.0, "text": doc, "meta": meta, "match_type": "exact_metadata"}
                        )
                    return results[:TOPK_FINAL]

        # ---------------------------
        # 1) VECTOR SEARCH (con filtro si hay norma detectada)
        # ---------------------------
        v_filter: Dict[str, str] = {}
        if intent.get("numero"):
            v_filter["numero"] = intent["numero"]
        if intent.get("siglas"):
            v_filter["siglas"] = intent["siglas"]
        if intent.get("tipo_norma"):
            v_filter["tipo_norma"] = intent["tipo_norma"]

        v_where = chroma_where(v_filter) if v_filter else None

        vres = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOPK_VECTOR,
            where=v_where,
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

        # ---------------------------
        # 2) BM25 + filtro opcional por norma
        # ---------------------------
        q_tok = simple_tokenize_es(query_text)
        bm25_scores = np.array(self.bm25.get_scores(q_tok), dtype=float)

        pool = np.argsort(-bm25_scores)[: TOPK_BM25 * 5]
        cand_ids = [self.bm25_ids[i] for i in pool]
        cand_scores = bm25_scores[pool]

        if v_filter:
            filtered_ids = []
            filtered_scores = []
            for cid, sc in zip(cand_ids, cand_scores):
                meta = self.id_to_meta.get(cid)
                if not meta:
                    continue
                ok = True
                for k, v in v_filter.items():
                    if str(meta.get(k, "")).lower() != str(v).lower():
                        ok = False
                        break
                if ok:
                    filtered_ids.append(cid)
                    filtered_scores.append(sc)

            if filtered_ids:
                b_ids = filtered_ids[:TOPK_BM25]
                b_scores = np.array(filtered_scores[:TOPK_BM25], dtype=float)
            else:
                b_ids = cand_ids[:TOPK_BM25]
                b_scores = cand_scores[:TOPK_BM25]
        else:
            b_ids = cand_ids[:TOPK_BM25]
            b_scores = cand_scores[:TOPK_BM25]

        b_scores_n = _minmax_norm(b_scores)
        bm25_map = {cid: float(sc) for cid, sc in zip(b_ids, b_scores_n)}

        # ---------------------------
        # 3) MERGE + híbrido + dedupe
        # ---------------------------
        candidates = set(list(vector_map.keys()) + list(bm25_map.keys()))
        results = []

        for cid in candidates:
            v = vector_map.get(cid, {}).get("v_score", 0.0)
            b = bm25_map.get(cid, 0.0)
            score = W_VECTOR * v + W_BM25 * b

            if cid in vector_map:
                text = vector_map[cid]["text"]
                meta = vector_map[cid]["meta"]
            else:
                got = self.collection.get(ids=[cid], include=["documents", "metadatas"])
                text = got["documents"][0]
                meta = got["metadatas"][0]

            # keyword bonus suave
            score += keyword_bonus(text, intent)

            results.append({"chunk_id": cid, "score": float(score), "text": text, "meta": meta, "match_type": "hybrid"})

        # dedupe por norma+artículo+páginas
        dedup: Dict[tuple, Dict] = {}
        for r in results:
            m = r["meta"] or {}
            key = (
                str(m.get("siglas", "")),
                str(m.get("tipo_norma", "")),
                str(m.get("numero", "")),
                str(m.get("articulo_nro", "")),
                str(m.get("page_start", "")),
                str(m.get("page_end", "")),
            )
            if key not in dedup or r["score"] > dedup[key]["score"]:
                dedup[key] = r

        out = list(dedup.values())

        # ---------------------------
        # 4) Bonus “ley” + hard boosting artículo top5
        # ---------------------------
        LEY_BONUS = 0.04
        for r in out:
            if _is_from_ley(r.get("meta") or {}):
                r["score"] = float(r.get("score", 0.0)) + LEY_BONUS
                r["match_type"] = r.get("match_type", "hybrid") + "+ley_bonus"

        out.sort(key=lambda x: x["score"], reverse=True)

        # Hard boost: si pidió artículo, asegurarlo arriba
        if has_article:
            forced = [r for r in out if _matches_article_request(r.get("meta") or {}, intent)]

            if not forced:
                # fallback: buscar metadata exacta pero solo por articulo y anclas disponibles
                where_force: Dict[str, str] = {"articulo_nro": intent["articulo_nro"]}
                if intent.get("numero"):
                    where_force["numero"] = intent["numero"]
                if intent.get("siglas"):
                    where_force["siglas"] = intent["siglas"]
                if intent.get("tipo_norma"):
                    where_force["tipo_norma"] = intent["tipo_norma"]

                w_force = chroma_where(where_force)
                if w_force is not None:
                    exact2 = self.collection.get(where=w_force, include=["documents", "metadatas"])
                    ids2 = exact2.get("ids", [])
                    if ids2:
                        injected = []
                        for cid, doc, meta in zip(exact2["ids"], exact2["documents"], exact2["metadatas"]):
                            injected.append(
                                {
                                    "chunk_id": cid,
                                    "score": 1.6,
                                    "text": doc,
                                    "meta": meta,
                                    "match_type": "forced_article_metadata",
                                }
                            )
                        existing = {r["chunk_id"] for r in out}
                        for r in injected:
                            if r["chunk_id"] not in existing:
                                out.append(r)

                        forced = [r for r in out if _matches_article_request(r.get("meta") or {}, intent)]

            if forced:
                forced.sort(key=lambda x: x["score"], reverse=True)
                forced_ids = {r["chunk_id"] for r in forced}
                rest = [r for r in out if r["chunk_id"] not in forced_ids]
                out = forced + rest

        return out[:TOPK_FINAL]

