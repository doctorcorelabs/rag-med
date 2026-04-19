"""
Source Manager — Medical RAG

Mengelola sumber materi (folder struktur) per stase:
- List sumber yang ada
- Buat folder sumber baru
- Upload halaman markdown (per-page)
- Upload ZIP (bulk pages)
- Hapus folder sumber
- Info index status dari DB
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT, DEFAULT_DB_PATH
from .stase_manager import stase_slug_to_dirname


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_materi_root(
    slug: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> Path:
    """Return path ke folder Materi/ untuk stase tertentu."""
    dirname = stase_slug_to_dirname(slug)
    return workspace_root / dirname / "Materi"


def _get_indexed_info(
    source_name: str,
    stase_slug: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Query DB untuk chunk count & index status satu sumber."""
    if not db_path.exists():
        return {"indexed": False, "chunk_count": 0, "image_count": 0}
    try:
        conn = sqlite3.connect(db_path)
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_name = ? AND stase_slug = ?",
            (source_name, stase_slug),
        ).fetchone()[0]
        image_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE source_name = ? AND stase_slug = ?",
            (source_name, stase_slug),
        ).fetchone()[0]
        conn.close()
        return {
            "indexed": chunk_count > 0,
            "chunk_count": chunk_count,
            "image_count": image_count,
        }
    except Exception:
        return {"indexed": False, "chunk_count": 0, "image_count": 0}


# ─── Public API ───────────────────────────────────────────────────────────────

def list_sources(
    slug: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """
    Scan folder Materi/ dan kembalikan daftar sumber yang ada,
    dilengkapi info index status dari DB.
    """
    root = get_materi_root(slug, workspace_root)
    if not root.is_dir():
        return []

    sources: list[dict[str, Any]] = []
    for source_dir in sorted(root.iterdir()):
        if not source_dir.is_dir():
            continue
        pages_dir = source_dir / "pages"
        page_count = (
            len([p for p in pages_dir.iterdir() if p.is_dir() and p.name.startswith("page-")])
            if pages_dir.is_dir()
            else 0
        )
        index_info = _get_indexed_info(source_dir.name, slug, db_path)
        sources.append({
            "source_name": source_dir.name,
            "page_count": page_count,
            "path": str(source_dir),
            **index_info,
        })
    return sources


def create_source(
    slug: str,
    source_name: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> dict[str, Any]:
    """
    Buat folder sumber baru dengan struktur:
      Materi/<source_name>/pages/page-1/markdown.md

    source_name HARUS diawali '(Sumber) '.
    """
    if not source_name.startswith("(Sumber) "):
        raise ValueError("Nama sumber harus diawali '(Sumber) '")

    root = get_materi_root(slug, workspace_root)
    root.mkdir(parents=True, exist_ok=True)

    source_dir = root / source_name
    if source_dir.exists():
        raise ValueError(f"Sumber '{source_name}' sudah ada")

    # Struktur awal
    page1_dir = source_dir / "pages" / "page-1"
    page1_dir.mkdir(parents=True)
    (page1_dir / "markdown.md").write_text(
        f"# {source_name}\n\n_Tambahkan konten di sini._\n",
        encoding="utf-8",
    )

    return {
        "source_name": source_name,
        "path": str(source_dir),
        "page_count": 1,
        "indexed": False,
    }


def upload_page(
    slug: str,
    source_name: str,
    page_no: int,
    markdown_content: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> dict[str, Any]:
    """
    Upload / update konten satu halaman (page-N/markdown.md).
    Folder page-N dibuat otomatis jika belum ada.
    """
    root = get_materi_root(slug, workspace_root)
    page_dir = root / source_name / "pages" / f"page-{page_no}"
    page_dir.mkdir(parents=True, exist_ok=True)
    md_path = page_dir / "markdown.md"
    md_path.write_text(markdown_content, encoding="utf-8")
    return {
        "page_no": page_no,
        "chars": len(markdown_content),
        "path": str(md_path),
    }


def get_page_content(
    slug: str,
    source_name: str,
    page_no: int,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> str | None:
    """Baca isi markdown.md satu halaman. Return None jika tidak ada."""
    root = get_materi_root(slug, workspace_root)
    md_path = root / source_name / "pages" / f"page-{page_no}" / "markdown.md"
    if not md_path.exists():
        return None
    return md_path.read_text(encoding="utf-8")


def upload_zip(
    slug: str,
    source_name: str,
    zip_bytes: bytes,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> dict[str, Any]:
    """
    Upload ZIP berisi batch pages, extract ke Materi/<source_name>/pages/.

    Format ZIP yang diterima:
      page-1/markdown.md
      page-1/img-0.jpeg  (opsional)
      page-2/markdown.md
      ...

    - Folder page-N yang sudah ada akan di-REPLACE.
    - source_name folder dibuat otomatis jika belum ada.
    """
    root = get_materi_root(slug, workspace_root)
    source_dir = root / source_name
    pages_dir = source_dir / "pages"

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "upload.zip"
        zip_path.write_bytes(zip_bytes)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            # Validasi: harus ada minimal satu page-N/markdown.md
            valid_pages = [
                m for m in members
                if m.startswith("page-") and m.endswith("markdown.md")
            ]
            if not valid_pages:
                raise ValueError(
                    "ZIP tidak valid. Harus berisi: page-1/markdown.md, page-2/markdown.md, ..."
                )
            zf.extractall(tmp)

        # Copy pages ke destination
        tmp_root = Path(tmp)
        page_dirs = sorted(
            [p for p in tmp_root.iterdir() if p.is_dir() and p.name.startswith("page-")],
            key=lambda p: int(p.name.replace("page-", "")),
        )

        pages_dir.mkdir(parents=True, exist_ok=True)
        for p_dir in page_dirs:
            dest = pages_dir / p_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(p_dir, dest)

    return {
        "source_name": source_name,
        "pages_uploaded": len(page_dirs),
        "path": str(source_dir),
    }


def delete_source(
    slug: str,
    source_name: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> None:
    """
    Hapus folder sumber beserta seluruh isinya.
    PERINGATAN: operasi ini TIDAK dapat dibatalkan.
    Setelah hapus, perlu re-index untuk update DB.
    """
    root = get_materi_root(slug, workspace_root)
    source_dir = root / source_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Sumber '{source_name}' tidak ditemukan di stase '{slug}'")
    shutil.rmtree(source_dir)


def get_source_tree(
    slug: str,
    source_name: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> dict[str, Any]:
    """
    Kembalikan tree lengkap satu sumber: semua page + file.
    Berguna untuk preview di UI.
    """
    root = get_materi_root(slug, workspace_root)
    source_dir = root / source_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Sumber '{source_name}' tidak ditemukan")

    pages_dir = source_dir / "pages"
    pages_info: list[dict[str, Any]] = []
    if pages_dir.is_dir():
        for p_dir in sorted(
            [p for p in pages_dir.iterdir() if p.is_dir() and p.name.startswith("page-")],
            key=lambda p: int(p.name.replace("page-", "")),
        ):
            page_no = int(p_dir.name.replace("page-", ""))
            md = p_dir / "markdown.md"
            images = [f.name for f in p_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            pages_info.append({
                "page_no": page_no,
                "has_markdown": md.exists(),
                "markdown_chars": len(md.read_text(encoding="utf-8")) if md.exists() else 0,
                "images": images,
            })

    return {
        "source_name": source_name,
        "path": str(source_dir),
        "page_count": len(pages_info),
        "pages": pages_info,
    }
