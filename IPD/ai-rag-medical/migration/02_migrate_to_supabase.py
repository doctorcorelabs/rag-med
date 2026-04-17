"""
Medical RAG — Migration Script: SQLite + ChromaDB → Supabase
=============================================================
Run ONCE locally before deploying to Cloudflare Workers.

Prerequisites:
    pip install supabase sentence-transformers chromadb python-dotenv tqdm

Usage:
    python migration/02_migrate_to_supabase.py

Set these env vars (or .env.migration):
    SUPABASE_URL=https://YOUR_PROJECT.supabase.co
    SUPABASE_SERVICE_KEY=eyJh... (service_role key, NOT anon key!)
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

MEDRAG_DB = PROJECT_ROOT / "data" / "medrag.sqlite3"
LIBRARY_DB = PROJECT_ROOT / "data" / "library" / "library.sqlite3"
ARTICLES_ROOT = PROJECT_ROOT / "data" / "library" / "articles"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

# ── Load env ─────────────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.migration", override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def get_supabase_client():
    """Initialize Supabase client with service role key."""
    try:
        from supabase import create_client
    except ImportError:
        print("❌ supabase-py not installed. Run: pip install supabase")
        sys.exit(1)

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("❌ SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env.migration")
        print("   Create .env.migration with:")
        print("   SUPABASE_URL=https://xxxxx.supabase.co")
        print("   SUPABASE_SERVICE_KEY=eyJh...")
        sys.exit(1)

    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_embedding_model():
    """Load embedding model for generating vectors."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    print(f"📦 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model loaded.")
    return model


# =============================================================================
# Phase 1: Migrate medrag.sqlite3 → chunks, images, graph_edges tables
# =============================================================================

def migrate_chunks(supabase, model):
    """Read all chunks from SQLite, generate embeddings, upsert to Supabase."""
    if not MEDRAG_DB.exists():
        print(f"⚠️  medrag.sqlite3 not found at {MEDRAG_DB}. Skipping chunk migration.")
        return

    conn = sqlite3.connect(str(MEDRAG_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT id, source_name, page_no, heading, content, disease_tags,
                   markdown_path, checksum, section_category, parent_heading,
                   chunk_index, total_chunks, heading_level, content_type
            FROM chunks
        """).fetchall()
    finally:
        conn.close()

    if not rows:
        print("⚠️  No chunks found in medrag.sqlite3.")
        return

    print(f"\n🔄 Migrating {len(rows)} chunks to Supabase...")
    batch_size = 32

    # Get existing checksums to avoid duplicates
    try:
        existing_res = supabase.table("chunks").select("checksum").execute()
        existing_checksums = {r["checksum"] for r in existing_res.data}
    except Exception:
        existing_checksums = set()

    for i in tqdm(range(0, len(rows), batch_size), desc="Chunks"):
        batch = [dict(r) for r in rows[i:i + batch_size]]
        
        # Filter out existing chunks
        batch = [c for c in batch if c["checksum"] not in existing_checksums]
        if not batch:
            continue

        # Generate embeddings for this batch
        texts = [f"passage: {c['heading']}: {c['content']}" for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        # Build upsert payload
        records = []
        for chunk, emb in zip(batch, embeddings):
            records.append({
                "source_name": chunk["source_name"],
                "page_no": chunk["page_no"],
                "heading": chunk["heading"],
                "content": chunk["content"],
                "disease_tags": chunk["disease_tags"],
                "markdown_path": chunk.get("markdown_path", ""),
                "checksum": chunk["checksum"],
                "section_category": chunk.get("section_category", "Ringkasan_Klinis"),
                "parent_heading": chunk.get("parent_heading", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "heading_level": chunk.get("heading_level", 2),
                "content_type": chunk.get("content_type", "prose"),
                "embedding": emb,
            })

        try:
            supabase.table("chunks").insert(records).execute()
        except Exception as e:
            print(f"\n❌ Error inserting chunk batch {i}: {e}")


def migrate_images(supabase):
    """Read all images from SQLite, upsert to Supabase images table."""
    if not MEDRAG_DB.exists():
        return

    conn = sqlite3.connect(str(MEDRAG_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT source_name, page_no, alt_text, image_ref,
                   image_abs_path, heading, nearby_text, markdown_path, checksum
            FROM images
        """).fetchall()
    finally:
        conn.close()

    if not rows:
        print("⚠️  No images found in medrag.sqlite3.")
        return

    print(f"\n🔄 Migrating {len(rows)} image references to Supabase...")

    try:
        existing_res = supabase.table("images").select("checksum").execute()
        existing_img_checksums = {r["checksum"] for r in existing_res.data}
    except Exception:
        existing_img_checksums = set()

    records = []
    for raw_r in rows:
        r = dict(raw_r)
        if r["checksum"] in existing_img_checksums:
            continue
        records.append({
            "source_name": r["source_name"],
            "page_no": r["page_no"],
            "alt_text": r["alt_text"] or "",
            "image_ref": r["image_ref"],
            "image_abs_path": r["image_abs_path"],
            "storage_url": "",  # Will be populated after Supabase Storage upload
            "heading": r["heading"],
            "nearby_text": r["nearby_text"],
            "markdown_path": r.get("markdown_path", ""),
            "checksum": r["checksum"],
        })

    # Insert in batches of 100
    for i in tqdm(range(0, len(records), 100), desc="Images"):
        if not records: break
        batch = records[i:i+100]
        try:
            supabase.table("images").insert(batch).execute()
        except Exception as e:
            print(f"\n❌ Error inserting image batch {i}: {e}")


def migrate_graph_edges(supabase):
    """Migrate knowledge graph edges."""
    if not MEDRAG_DB.exists():
        return

    conn = sqlite3.connect(str(MEDRAG_DB))
    conn.row_factory = sqlite3.Row
    try:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'"
        ).fetchone()
        if not table_exists:
            print("⚠️  No graph_edges table in medrag.sqlite3. Skipping.")
            conn.close()
            return
        rows = conn.execute(
            "SELECT source_disease, relation, target_node, target_type, source_name, page_no FROM graph_edges"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return

    print(f"\n🔄 Migrating {len(rows)} graph edges...")
    records = [dict(r) for r in rows]

    for i in tqdm(range(0, len(records), 500), desc="Graph edges"):
        batch = records[i:i+500]
        try:
            supabase.table("graph_edges").insert(batch).execute()
        except Exception as e:
            print(f"\n⚠️  Graph edge batch {i}: {e}")


# =============================================================================
# Phase 2: Migrate library.sqlite3 → stase, disease_catalog, library_article
# =============================================================================

def migrate_library(supabase):
    """Migrate library catalog and articles from SQLite to Supabase."""
    if not LIBRARY_DB.exists():
        print(f"⚠️  library.sqlite3 not found at {LIBRARY_DB}. Skipping library migration.")
        return

    conn = sqlite3.connect(str(LIBRARY_DB))
    conn.row_factory = sqlite3.Row
    try:
        stases = conn.execute("SELECT * FROM stase ORDER BY sort_order, id").fetchall()
    except Exception as e:
        print(f"⚠️  Failed to read stase table: {e}")
        conn.close()
        return

    for stase_row in stases:
        stase_data = {
            "slug": stase_row["slug"],
            "display_name": stase_row["display_name"],
            "sort_order": stase_row["sort_order"],
        }
        try:
            stase_result = supabase.table("stase").upsert(stase_data, on_conflict="slug").execute()
            new_stase_id = stase_result.data[0]["id"]
            print(f"\n✅ Stase '{stase_row['slug']}' → id={new_stase_id}")
        except Exception as e:
            print(f"\n❌ Failed to upsert stase {stase_row['slug']}: {e}")
            continue

        # Migrate disease_catalog for this stase
        diseases = conn.execute(
            """SELECT dc.*, la.status, la.content_path, la.meta_path, la.content_hash, la.updated_at
               FROM disease_catalog dc
               LEFT JOIN library_article la ON la.catalog_id = dc.id
               WHERE dc.stase_id = ?
               ORDER BY dc.catalog_no""",
            (stase_row["id"],)
        ).fetchall()

        print(f"   📋 Migrating {len(diseases)} diseases for {stase_row['slug']}...")

        for raw_d in tqdm(diseases, desc=f"  {stase_row['slug']}"):
            d = dict(raw_d)
            cat_data = {
                "stase_id": new_stase_id,
                "catalog_no": d["catalog_no"],
                "name": d["name"],
                "competency_level": d["competency_level"],
                "group_label": d["group_label"],
                "stable_key": d["stable_key"],
            }
            try:
                cat_result = supabase.table("disease_catalog").upsert(
                    cat_data, on_conflict="stase_id,catalog_no"
                ).execute()
                catalog_id = cat_result.data[0]["id"]
            except Exception as e:
                print(f"\n❌ Failed to upsert disease {d['name']}: {e}")
                continue

            # Read article content from disk
            content_markdown = None
            meta_dict = {}
            cp = d.get("content_path")
            mp = d.get("meta_path")

            if cp and Path(cp).is_file():
                try:
                    content_markdown = Path(cp).read_text(encoding="utf-8")
                except Exception:
                    pass

            if mp and Path(mp).is_file():
                try:
                    meta_dict = json.loads(Path(mp).read_text(encoding="utf-8"))
                except Exception:
                    pass

            # Read mindmap if exists
            slug = stase_row["slug"]
            catalog_no = d["catalog_no"]
            mindmap_file = ARTICLES_ROOT / slug / f"{slug}-{catalog_no}" / "mindmap.json"
            mindmap_data = None
            if mindmap_file.is_file():
                try:
                    mindmap_data = json.loads(mindmap_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            article_data = {
                "catalog_id": catalog_id,
                "status": d.get("status") or "missing",
                "content_markdown": content_markdown,
                "meta": meta_dict,
                "mindmap": mindmap_data,
                "content_hash": d.get("content_hash"),
            }
            try:
                supabase.table("library_article").upsert(
                    article_data, on_conflict="catalog_id"
                ).execute()
            except Exception as e:
                print(f"\n❌ Failed to upsert article {d['name']}: {e}")

    conn.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  Medical RAG — Migration to Supabase")
    print("=" * 60)

    supabase = get_supabase_client()
    print(f"✅ Connected to Supabase: {SUPABASE_URL}")

    # Phase 1: Core RAG data
    print("\n📚 Phase 1: Migrating RAG data (chunks + images + graph)...")
    model = get_embedding_model()
    migrate_chunks(supabase, model)
    migrate_images(supabase)
    migrate_graph_edges(supabase)

    # Phase 2: Library data
    print("\n📚 Phase 2: Migrating Medical Library (stase + diseases + articles)...")
    migrate_library(supabase)

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("   Next steps:")
    print("   1. Verify data in Supabase Table Editor")
    print("   2. Deploy Cloudflare Worker (cd worker && npx wrangler deploy)")
    print("   3. Deploy Frontend to Netlify")
    print("=" * 60)


if __name__ == "__main__":
    main()
