"""
Optional hook: index Medical Library Markdown into the main RAG index.

Currently a no-op stub. Call maybe_index_article_for_rag() after saving an article
so a future implementation can push chunks to Chroma/SQLite without API changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _file_checksum(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_index_article_for_rag(
    content_md_path: Path,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Placeholder for future RAG indexing of library articles.

    Returns metadata useful for callers to store (e.g. indexed_checksum when implemented).
    """
    checksum = _file_checksum(content_md_path)
    # Future: chunk content_md_path, upsert into medrag chunks + chroma with source_type=library
    return {
        "indexed_into_rag": False,
        "indexed_checksum": None,
        "content_checksum": checksum,
        "note": "RAG indexing hook not enabled; stub only.",
    }
