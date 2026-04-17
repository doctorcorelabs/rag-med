from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "medrag.sqlite3"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_LIBRARY_DB_PATH = PROJECT_ROOT / "data" / "library" / "library.sqlite3"
LIBRARY_ARTICLES_ROOT = PROJECT_ROOT / "data" / "library" / "articles"

# Legacy single-stase glob — kept for backward compatibility with existing code
# New code should use STASE_MATERI_ROOTS instead.
MATERI_GLOB = "IPD/Materi/**/pages/page-*/markdown.md"

# Multi-stase configuration.
# Each entry is (stase_slug, materi_dir_relative_to_workspace_root).
# Add a new row here to register a new stase — no code changes needed elsewhere.
STASE_MATERI_ROOTS: list[tuple[str, str]] = [
    ("ipd",   "IPD/Materi"),
    # ("saraf", "Saraf/Materi"),  # uncomment when Saraf materials are ready
    # ("anak",  "Anak/Materi"),
    # ("obgyn", "ObGyn/Materi"),
]

# Glob pattern appended to each materi root to discover markdown pages
MATERI_PAGE_GLOB = "**/pages/page-*/markdown.md"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
