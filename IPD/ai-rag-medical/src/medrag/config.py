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


def _discover_workspace_stase_roots(workspace_root: Path) -> list[tuple[str, str]]:
    """Discover local stase folders that follow the <Stase>/Materi layout."""
    discovered: list[tuple[str, str]] = []
    if not workspace_root.is_dir():
        return discovered

    for child in sorted(workspace_root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        materi_dir = child / "Materi"
        if not materi_dir.is_dir():
            continue
        slug = child.name.strip().lower()
        if not slug:
            continue
        discovered.append((slug, f"{child.name}/Materi"))
    return discovered

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Retrieval mode constants ──────────────────────────────────────────────────
# "relevant"   → default focused QA mode (bounded top_k)
# "exhaustive" → listing/catalog mode (higher top_k, relaxed filtering)
DEFAULT_TOP_K_RELEVANT: int = 8
DEFAULT_TOP_K_EXHAUSTIVE: int = 80
MAX_TOP_K_EXHAUSTIVE: int = 200
ENABLE_EXHAUSTIVE_AUTO_MODE: bool = True

# Path untuk stase dinamis (dibuat via UI Admin)
STASE_OVERRIDES_PATH = PROJECT_ROOT / "stase_overrides.json"


def load_stase_roots(workspace_root: Path | None = None) -> list[tuple[str, str]]:
    """
    Gabungkan STASE_MATERI_ROOTS (hardcoded) + stase_overrides.json (dinamis dari UI).
    Return list of (slug, materi_dir_relative_to_workspace_root).

    Ditambah dengan discovery lokal berbasis folder <Stase>/Materi
    agar stase baru yang dibuat manual langsung ikut ter-index.
    """
    import json as _json

    base = list(STASE_MATERI_ROOTS)
    if STASE_OVERRIDES_PATH.exists():
        try:
            overrides = _json.loads(STASE_OVERRIDES_PATH.read_text(encoding="utf-8"))
            for entry in overrides.get("stases", []):
                pair = (entry["slug"], entry["materi_dir"])
                if pair not in base:
                    base.append(pair)
        except Exception:
            pass

    local_root = workspace_root or DEFAULT_WORKSPACE_ROOT
    seen_slugs = {slug for slug, _ in base}
    for slug, materi_dir in _discover_workspace_stase_roots(local_root):
        if slug in seen_slugs:
            continue
        base.append((slug, materi_dir))
        seen_slugs.add(slug)

    return base
