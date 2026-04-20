"""
Pytest configuration: pre-patch heavy optional dependencies that are not
available in the test environment (ChromaDB, sentence-transformers, full
FastAPI app initialisation) so that unit tests can import medrag.api and
medrag.retriever without errors.
"""

import sys
import types


def _make_stub(mod_name: str, attrs: list[str]) -> types.ModuleType:
    m = types.ModuleType(mod_name)
    for a in attrs:
        setattr(m, a, lambda *x, **kw: None)
    return m


def _patch_heavy_deps() -> None:
    # ── chromadb / sentence_transformers ─────────────────────────────────────
    for heavy in ("chromadb", "sentence_transformers"):
        if heavy not in sys.modules:
            sys.modules[heavy] = types.ModuleType(heavy)

    # ── medrag.library ───────────────────────────────────────────────────────
    if "medrag.library" not in sys.modules:
        lib_attrs = [
            "clear_library_article", "combine_preview_markdown", "content_hash",
            "ensure_library_initialized", "get_disease_bundle", "get_disease_list",
            "get_stase_by_slug", "get_stases", "load_mindmap", "mindmap_path",
            "run_article_generation_pipeline", "save_mindmap", "sync_all_stases",
            "update_article_meta", "update_library_article_row", "write_article_files",
            "init_library_schema", "_connect", "article_dir", "_utc_now_iso",
        ]
        sys.modules["medrag.library"] = _make_stub("medrag.library", lib_attrs)

    # ── other medrag heavy modules ────────────────────────────────────────────
    stubs = {
        "medrag.copilot_client": [
            "ask_copilot_adaptive", "ask_copilot_for_list",
            "generate_mindmap_from_article", "merge_two_markdown_articles",
            "refine_markdown_with_instruction",
        ],
        "medrag.source_manager": [
            "list_sources", "create_source", "upload_page", "upload_zip",
            "delete_source", "get_source_tree", "get_page_content",
        ],
        "medrag.stase_manager": ["create_stase", "delete_stase", "list_all_stases"],
        "medrag.indexer": ["build_index_for_source", "build_index"],
        "medrag.library_rag_hook": ["maybe_index_article_for_rag"],
    }
    for mod_name, attrs in stubs.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub(mod_name, attrs)


_patch_heavy_deps()
