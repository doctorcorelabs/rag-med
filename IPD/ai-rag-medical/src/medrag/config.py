from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "medrag.sqlite3"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
MATERI_GLOB = "IPD/Materi/**/pages/page-*/markdown.md"
