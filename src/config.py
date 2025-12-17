from pathlib import Path
import os

# En Render: /var/data
# En local: raíz del proyecto
BASE_DIR = Path(os.getenv("RENDER_DISK_PATH", Path(__file__).resolve().parents[1]))

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"

BM25_PATH = INDEX_DIR / "bm25.pkl"
META_PATH = INDEX_DIR / "meta.pkl"
CHROMA_DIR = INDEX_DIR / "chroma"



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
