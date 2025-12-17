from pathlib import Path
import os

# Paths
BASE_DIR = Path(os.getenv("RENDER_DISK_PATH", Path(__file__).resolve().parents[1]))
PDF_DIR = BASE_DIR / "data" / "pdfs"
INDEX_DIR = BASE_DIR / "data" / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"
META_PATH = INDEX_DIR / "meta.pkl"

# OpenAI
EMBED_MODEL = "text-embedding-3-large"   # 3072 dims
LLM_MODEL = "gpt-5.2"                    # calidad

# Chunking
DEFAULT_CHUNK_TOKENS = 950
DEFAULT_OVERLAP_TOKENS = 180

# Retrieval
TOPK_VECTOR = 20
TOPK_BM25 = 20
TOPK_FINAL = 8

# Hybrid scoring weights
W_VECTOR = 0.60
W_BM25 = 0.40
