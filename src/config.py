from pathlib import Path
import os

# -----------------------------
# Paths (Local defaults + Render override via ENV)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# Defaults locales (repo)
_DEFAULT_PDF_DIR = BASE_DIR / "data" / "pdfs"
_DEFAULT_INDEX_DIR = BASE_DIR / "data" / "index"

# ✅ Permite override por variables de entorno (Render Disk)
# En Render recomendación: INDEX_DIR=/var/data/index
PDF_DIR = Path(os.getenv("PDF_DIR", str(_DEFAULT_PDF_DIR)))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(_DEFAULT_INDEX_DIR)))

# Si querés granularidad, también se puede setear cada una:
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(INDEX_DIR / "chroma")))
BM25_PATH  = Path(os.getenv("BM25_PATH",  str(INDEX_DIR / "bm25.pkl")))
META_PATH  = Path(os.getenv("META_PATH",  str(INDEX_DIR / "meta.pkl")))

# (Opcional) normalización + manifest (si tus scripts lo usan)
NORMALIZED_PDF_DIR = Path(os.getenv("NORMALIZED_PDF_DIR", str(BASE_DIR / "data" / "pdfs_normalized")))
MANIFEST_PATH      = Path(os.getenv("MANIFEST_PATH", str(NORMALIZED_PDF_DIR / "manifest.json")))

# -----------------------------
# OpenAI
# -----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")  # 3072 dims
LLM_MODEL   = os.getenv("LLM_MODEL", "gpt-5.2")                   # máxima calidad legal

# -----------------------------
# Chunking
# -----------------------------
DEFAULT_CHUNK_TOKENS   = int(os.getenv("DEFAULT_CHUNK_TOKENS", "950"))
DEFAULT_OVERLAP_TOKENS = int(os.getenv("DEFAULT_OVERLAP_TOKENS", "180"))

# -----------------------------
# Retrieval (optimizado para legal)
# -----------------------------
TOPK_VECTOR = int(os.getenv("TOPK_VECTOR", "50"))
TOPK_BM25   = int(os.getenv("TOPK_BM25", "50"))
TOPK_FINAL  = int(os.getenv("TOPK_FINAL", "40"))

# Hybrid scoring weights
W_VECTOR = float(os.getenv("W_VECTOR", "0.80"))
W_BM25   = float(os.getenv("W_BM25", "0.20"))
