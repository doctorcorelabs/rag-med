"""
Stase Manager — Medical RAG

Mengelola stase medis:
- Buat stase baru (folder PascalCase/Materi/ + CSV placeholder + register SQLite)
- Hapus stase dari registry (tidak hapus folder materi)
- List semua stase (hardcoded + stase_overrides.json)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT, PROJECT_ROOT
from . import library as library_mod

# Path file JSON untuk stase dinamis (di luar hardcoded STASE_MATERI_ROOTS)
STASE_OVERRIDES_PATH = PROJECT_ROOT / "stase_overrides.json"

# Mapping slug → dirname untuk stase bawaan (tidak perlu di overrides)
_BUILTIN_DIRNAMES: dict[str, str] = {
    "ipd": "IPD",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_stase_overrides() -> dict[str, Any]:
    """Load stase_overrides.json. Returns empty structure if not found."""
    if STASE_OVERRIDES_PATH.exists():
        try:
            return json.loads(STASE_OVERRIDES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"stases": []}


def save_stase_overrides(data: dict[str, Any]) -> None:
    """Persist stase_overrides.json."""
    STASE_OVERRIDES_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stase_slug_to_dirname(slug: str) -> str:
    """
    Convert slug → PascalCase folder name.
    'ipd' → 'IPD' (builtin), 'saraf' → 'Saraf' (from overrides or capitalize).
    """
    if slug in _BUILTIN_DIRNAMES:
        return _BUILTIN_DIRNAMES[slug]
    overrides = load_stase_overrides()
    for entry in overrides.get("stases", []):
        if entry["slug"] == slug:
            return entry.get("dirname", slug.capitalize())
    return slug.capitalize()


def _next_sort_order(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 FROM stase")
    return cur.fetchone()[0]


# ─── Public API ───────────────────────────────────────────────────────────────

def create_stase(
    slug: str,
    display_name: str,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> dict[str, Any]:
    """
    Buat stase baru:
    1. Buat folder <PascalCase>/Materi/ di workspace_root
    2. Buat CSV placeholder katalog penyakit
    3. Register ke library SQLite
    4. Simpan ke stase_overrides.json

    Konvensi folder: slug 'saraf' → folder 'Saraf/Materi/'
    """
    slug = slug.lower().strip()
    if not slug or not slug.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Slug tidak valid: '{slug}'. Gunakan huruf kecil, angka, - atau _")

    dirname = slug.capitalize()
    stase_root = workspace_root / dirname
    materi_dir = stase_root / "Materi"
    materi_dir.mkdir(parents=True, exist_ok=True)

    # CSV placeholder katalog penyakit
    csv_name = f"Daftar Penyakit - {display_name}.csv"
    csv_path = stase_root / csv_name
    if not csv_path.exists():
        csv_path.write_text(
            "No,Daftar Penyakit,Level Kompetensi\n",
            encoding="utf-8-sig",
        )

    # Relative materi_dir path (untuk STASE_MATERI_ROOTS format)
    materi_rel = f"{dirname}/Materi"

    # Register ke library SQLite
    conn = library_mod._connect()
    try:
        library_mod.init_library_schema(conn)
        sort_order = _next_sort_order(conn)
        conn.execute(
            """INSERT OR IGNORE INTO stase (slug, display_name, csv_path, sort_order)
               VALUES (?, ?, ?, ?)""",
            (slug, display_name, str(csv_path), sort_order),
        )
        conn.commit()
    finally:
        conn.close()

    # Simpan ke stase_overrides.json
    overrides = load_stase_overrides()
    existing_slugs = {e["slug"] for e in overrides["stases"]}
    if slug not in existing_slugs and slug not in _BUILTIN_DIRNAMES:
        overrides["stases"].append({
            "slug": slug,
            "display_name": display_name,
            "dirname": dirname,
            "materi_dir": materi_rel,
            "csv_path": str(csv_path),
        })
        save_stase_overrides(overrides)

    return {
        "slug": slug,
        "display_name": display_name,
        "dirname": dirname,
        "materi_dir": str(materi_dir),
        "csv_path": str(csv_path),
    }


def delete_stase(slug: str) -> None:
    """
    Hapus stase dari overrides.json & library DB.
    TIDAK menghapus folder materi (user harus manual).
    Stase builtin (ipd) tidak dapat dihapus.
    """
    if slug in _BUILTIN_DIRNAMES:
        raise ValueError(f"Stase builtin '{slug}' tidak dapat dihapus")

    # Hapus dari overrides.json
    overrides = load_stase_overrides()
    overrides["stases"] = [e for e in overrides["stases"] if e["slug"] != slug]
    save_stase_overrides(overrides)

    # Hapus dari library SQLite
    conn = library_mod._connect()
    try:
        conn.execute("DELETE FROM stase WHERE slug = ?", (slug,))
        conn.commit()
    finally:
        conn.close()


def list_all_stases(
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> list[dict[str, Any]]:
    """
    List semua stase terdaftar dari library DB + info folder dari overrides.
    """
    conn = library_mod._connect()
    try:
        library_mod.init_library_schema(conn)
        rows = library_mod.get_stases(conn)
    finally:
        conn.close()

    # Enrich dengan info folder
    overrides = load_stase_overrides()
    override_map = {e["slug"]: e for e in overrides.get("stases", [])}

    enriched = []
    for row in rows:
        slug = row["slug"]
        dirname = stase_slug_to_dirname(slug)
        materi_dir = workspace_root / dirname / "Materi"
        entry = dict(row)
        entry["dirname"] = dirname
        entry["materi_dir"] = str(materi_dir)
        entry["materi_exists"] = materi_dir.is_dir()
        entry["is_builtin"] = slug in _BUILTIN_DIRNAMES
        enriched.append(entry)

    return enriched
