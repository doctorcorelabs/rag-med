"""
Medical Library: multi-stase disease catalog (CSV), SQLite metadata, on-disk Markdown articles.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from .config import DEFAULT_LIBRARY_DB_PATH, DEFAULT_WORKSPACE_ROOT, LIBRARY_ARTICLES_ROOT
from .library_rag_hook import maybe_index_article_for_rag

# Default CSV for IPD stase (same layout as Materi: under IPD/ next to ai-rag-medical)
# DEFAULT_WORKSPACE_ROOT is repo root (e.g. E:\Coas), so IPD/… not the root alone.
DEFAULT_IPD_CSV = (
    "IPD/Daftar Penyakit dan Level Kompetensi - Daftar Penyakit dan Level Kompetensi.csv"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_LIBRARY_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_library_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stase (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            csv_path TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS disease_catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stase_id INTEGER NOT NULL REFERENCES stase(id) ON DELETE CASCADE,
            catalog_no INTEGER NOT NULL,
            name TEXT NOT NULL,
            competency_level TEXT,
            group_label TEXT,
            stable_key TEXT NOT NULL,
            UNIQUE (stase_id, catalog_no),
            UNIQUE (stase_id, stable_key)
        );

        CREATE INDEX IF NOT EXISTS idx_disease_stase ON disease_catalog(stase_id);

        CREATE TABLE IF NOT EXISTS library_article (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            catalog_id INTEGER NOT NULL UNIQUE REFERENCES disease_catalog(id) ON DELETE CASCADE,
            status TEXT NOT NULL DEFAULT 'missing'
                CHECK (status IN ('missing', 'draft', 'published')),
            content_path TEXT,
            meta_path TEXT,
            content_hash TEXT,
            updated_at TEXT
        );
        """
    )


def ensure_library_initialized(db_path: Path | None = None) -> None:
    conn = _connect(db_path)
    try:
        init_library_schema(conn)
        _ensure_default_stases(conn)
        conn.commit()
    finally:
        conn.close()


def _ensure_default_stases(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT COUNT(*) FROM stase")
    if cur.fetchone()[0] > 0:
        return
    csv_abs = (DEFAULT_WORKSPACE_ROOT / DEFAULT_IPD_CSV).resolve()
    conn.execute(
        """
        INSERT INTO stase (slug, display_name, csv_path, sort_order)
        VALUES ('ipd', 'Stase IPD', ?, 0)
        """,
        (str(csv_abs),),
    )


def resolve_csv_path(stase_row: sqlite3.Row) -> Path:
    """Resolve catalog CSV; tolerate old DB rows that pointed at repo root instead of IPD/."""
    stored = Path(stase_row["csv_path"])
    name = stored.name
    candidates = [
        stored,
        DEFAULT_WORKSPACE_ROOT / "IPD" / name,
        DEFAULT_WORKSPACE_ROOT / name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return stored


def stable_key_for(stase_slug: str, catalog_no: int) -> str:
    return f"{stase_slug}-{catalog_no}"


def article_dir(stase_slug: str, catalog_no: int) -> Path:
    return LIBRARY_ARTICLES_ROOT / stase_slug / stable_key_for(stase_slug, catalog_no)


def parse_disease_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse catalog CSV: section rows (no No) update group; numbered rows are diseases."""
    rows: list[dict[str, Any]] = []
    current_group = ""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return rows
        # Normalize expected columns
        for line in reader:
            no_raw = (line.get("No") or "").strip()
            name = (line.get("Daftar Penyakit") or "").strip()
            level = (line.get("Level Kompetensi") or "").strip()
            if not name:
                continue
            if no_raw.isdigit():
                rows.append(
                    {
                        "type": "disease",
                        "catalog_no": int(no_raw),
                        "name": name,
                        "competency_level": level or None,
                        "group_label": current_group or None,
                    }
                )
            else:
                current_group = name
    return rows


def sync_catalog_from_csv(conn: sqlite3.Connection, stase_id: int, stase_slug: str, csv_path: Path) -> int:
    """Upsert disease rows from CSV. Returns number of disease rows processed."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    parsed = parse_disease_csv(csv_path)
    count = 0
    for item in parsed:
        if item["type"] != "disease":
            continue
        catalog_no = item["catalog_no"]
        sk = stable_key_for(stase_slug, catalog_no)
        conn.execute(
            """
            INSERT INTO disease_catalog (stase_id, catalog_no, name, competency_level, group_label, stable_key)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(stase_id, catalog_no) DO UPDATE SET
                name = excluded.name,
                competency_level = excluded.competency_level,
                group_label = excluded.group_label,
                stable_key = excluded.stable_key
            """,
            (
                stase_id,
                catalog_no,
                item["name"],
                item["competency_level"],
                item["group_label"],
                sk,
            ),
        )
        # Ensure library_article row exists
        cur = conn.execute(
            "SELECT id FROM disease_catalog WHERE stase_id = ? AND catalog_no = ?",
            (stase_id, catalog_no),
        )
        cid = cur.fetchone()[0]
        conn.execute(
            """
            INSERT OR IGNORE INTO library_article (catalog_id, status)
            VALUES (?, 'missing')
            """,
            (cid,),
        )
        count += 1
    return count


def sync_all_stases(db_path: Path | None = None) -> dict[str, int]:
    ensure_library_initialized(db_path)
    conn = _connect(db_path)
    out: dict[str, int] = {}
    try:
        init_library_schema(conn)
        stases = conn.execute("SELECT id, slug, csv_path FROM stase ORDER BY sort_order, id").fetchall()
        for s in stases:
            path = resolve_csv_path(s)
            if path.is_file() and str(path) != str(s["csv_path"]):
                conn.execute(
                    "UPDATE stase SET csv_path = ? WHERE id = ?",
                    (str(path), s["id"]),
                )
            n = sync_catalog_from_csv(conn, s["id"], s["slug"], path)
            out[s["slug"]] = n
        conn.commit()
    finally:
        conn.close()
    return out


def combine_preview_markdown(
    base_md: str | None,
    candidate_md: str,
    mode: Literal["append", "replace"],
) -> str:
    """Merge base article with a new generation. `replace` ignores base (or uses candidate only)."""
    cand = candidate_md.strip()
    if not cand.endswith("\n"):
        cand += "\n"
    if mode == "replace":
        return cand
    base = (base_md or "").strip()
    if not base:
        return cand
    sep = "\n\n---\n\n## Pembaruan (regenerate)\n\n"
    return (base.rstrip() + sep + cand.rstrip()).strip() + "\n"


def run_article_generation_pipeline(
    database: Path,
    disease_name: str,
    extra_prompt: str | None,
    top_k: int,
    image_limit: int,
    to_image_url: Callable[[str], str],
) -> dict[str, Any]:
    """RAG retrieval + synthesis for one disease; returns evidence, images (with URLs), draft_answer, markdown_candidate."""
    import os

    from .copilot_client import ask_copilot_adaptive
    from .retriever import related_images, search_chunks, synthesize_answer

    query = disease_name
    if extra_prompt:
        query = f"{disease_name}. {extra_prompt}"

    evidence = search_chunks(database, query, top_k=top_k, chat_history=None)
    images_raw = related_images(database, query, evidence, limit=image_limit)
    images = [{**img, "image_url": to_image_url(img["image_abs_path"])} for img in images_raw]

    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        answer = ask_copilot_adaptive(
            query,
            evidence,
            github_token,
            chat_history=None,
            images=images,
        )
    else:
        answer = synthesize_answer(disease_name, evidence)

    markdown_candidate = draft_answer_to_markdown(disease_name, answer)
    return {
        "query": query,
        "evidence": evidence,
        "images": images,
        "draft_answer": answer,
        "markdown_candidate": markdown_candidate,
    }


def draft_answer_to_markdown(disease_title: str, draft: dict[str, Any]) -> str:
    parts: list[str] = [f"# {disease_title}\n"]
    for sec in draft.get("sections") or []:
        title = sec.get("title") or "Section"
        md = (sec.get("markdown") or "").strip()
        if not md:
            pts = sec.get("points")
            if pts:
                md = "\n".join(f"- {p}" for p in pts)
        if md:
            parts.append(f"\n## {title}\n\n{md}\n")
    return "".join(parts).strip() + "\n"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mindmap_path(stase_slug: str, catalog_no: int) -> Path:
    return article_dir(stase_slug, catalog_no) / "mindmap.json"


def save_mindmap(stase_slug: str, catalog_no: int, data: dict[str, Any]) -> Path:
    path = mindmap_path(stase_slug, catalog_no)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["saved_at"] = _utc_now_iso()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_mindmap(stase_slug: str, catalog_no: int) -> dict[str, Any] | None:
    path = mindmap_path(stase_slug, catalog_no)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_article_files(
    stase_slug: str,
    catalog_no: int,
    disease_name: str,
    draft_answer: dict[str, Any],
    images: list[dict[str, Any]],
    extra: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    d = article_dir(stase_slug, catalog_no)
    d.mkdir(parents=True, exist_ok=True)
    md_path = d / "content.md"
    meta_path = d / "meta.json"
    body = draft_answer_to_markdown(disease_name, draft_answer)
    md_path.write_text(body, encoding="utf-8")

    meta: dict[str, Any] = {
        "version": int(extra.get("version", 1)),
        "disease_name": disease_name,
        "catalog_no": catalog_no,
        "stase_slug": stase_slug,
        "generated_at": _utc_now_iso(),
        "last_refine_instruction": extra.get("last_refine_instruction"),
        "extra_prompt": extra.get("extra_prompt"),
        "last_operation": extra.get("last_operation", "generate"),
        "images": [
            {
                "image_abs_path": img.get("image_abs_path", ""),
                "heading": img.get("heading", ""),
                "source_name": img.get("source_name", ""),
                "page_no": img.get("page_no", 0),
            }
            for img in images
        ],
        "indexed_into_rag": False,
        "indexed_checksum": None,
    }
    hook = maybe_index_article_for_rag(md_path, meta)
    meta["indexed_into_rag"] = hook.get("indexed_into_rag", False)
    meta["indexed_checksum"] = hook.get("indexed_checksum")
    meta["content_checksum"] = hook.get("content_checksum")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, meta_path, meta


def update_article_meta(meta_path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if meta_path.is_file():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    data.update(patch)
    data["updated_at"] = _utc_now_iso()
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# --- Query helpers ---

def get_stases(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT s.id, s.slug, s.display_name, s.sort_order,
               COUNT(dc.id) AS disease_count,
               SUM(CASE WHEN la.status IN ('draft','published') THEN 1 ELSE 0 END) AS filled_count
        FROM stase s
        LEFT JOIN disease_catalog dc ON dc.stase_id = s.id
        LEFT JOIN library_article la ON la.catalog_id = dc.id
        GROUP BY s.id
        ORDER BY s.sort_order, s.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_stase_by_slug(conn: sqlite3.Connection, slug: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM stase WHERE slug = ?", (slug,)).fetchone()


def get_disease_list(conn: sqlite3.Connection, stase_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT dc.id, dc.catalog_no, dc.name, dc.competency_level, dc.group_label, dc.stable_key,
               la.status, la.content_path, la.updated_at
        FROM disease_catalog dc
        LEFT JOIN library_article la ON la.catalog_id = dc.id
        WHERE dc.stase_id = ?
        ORDER BY dc.catalog_no
        """,
        (stase_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_disease_bundle(conn: sqlite3.Connection, stase_slug: str, catalog_id: int) -> dict[str, Any] | None:
    st = get_stase_by_slug(conn, stase_slug)
    if not st:
        return None
    row = conn.execute(
        """
        SELECT dc.*, la.status, la.content_path, la.meta_path, la.content_hash, la.updated_at
        FROM disease_catalog dc
        LEFT JOIN library_article la ON la.catalog_id = dc.id
        WHERE dc.stase_id = ? AND dc.id = ?
        """,
        (st["id"], catalog_id),
    ).fetchone()
    if not row:
        return None
    return dict(row)


def delete_article_files(stase_slug: str, catalog_no: int) -> None:
    import shutil

    d = article_dir(stase_slug, catalog_no)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


def update_library_article_row(
    conn: sqlite3.Connection,
    catalog_id: int,
    status: str,
    content_path: str | None,
    meta_path: str | None,
    content_hash: str | None,
) -> None:
    conn.execute(
        """
        UPDATE library_article
        SET status = ?, content_path = ?, meta_path = ?, content_hash = ?, updated_at = ?
        WHERE catalog_id = ?
        """,
        (status, content_path, meta_path, content_hash, _utc_now_iso(), catalog_id),
    )


def clear_library_article(
    conn: sqlite3.Connection,
    catalog_id: int,
    stase_slug: str,
    catalog_no: int,
) -> None:
    delete_article_files(stase_slug, catalog_no)
    update_library_article_row(conn, catalog_id, "missing", None, None, None)
