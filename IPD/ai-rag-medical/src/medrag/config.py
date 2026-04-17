from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "medrag.sqlite3"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_LIBRARY_DB_PATH = PROJECT_ROOT / "data" / "library" / "library.sqlite3"
LIBRARY_ARTICLES_ROOT = PROJECT_ROOT / "data" / "library" / "articles"
MATERI_GLOB = "IPD/Materi/**/pages/page-*/markdown.md"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
