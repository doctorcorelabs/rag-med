"""
Medical RAG — Upload local images to Cloudflare R2
====================================================
Reads image records from local medrag.sqlite3, uploads each image file to
Cloudflare R2 bucket 'coas-medical-images' via the S3-compatible API, then
updates the storage_url AND stase_slug columns in the Supabase images table.

Object key structure:
    {stase_slug}/{source_name}/{image_ref}
    e.g.  ipd/Cardiomegali/img-12.jpeg
          saraf/Stroke_Iskemik/img-3.png

Prerequisites:
    pip install boto3 supabase python-dotenv tqdm

Environment (.env.migration):
    SUPABASE_URL=https://...supabase.co
    SUPABASE_SERVICE_KEY=eyJh...
    R2_ACCESS_KEY_ID=...         (from Cloudflare Dashboard > R2 > Manage R2 API Tokens)
    R2_SECRET_ACCESS_KEY=...
    R2_ACCOUNT_ID=...            (Cloudflare account ID, found in dashboard URL)
    R2_BUCKET_NAME=coas-medical-images
    R2_PUBLIC_BASE_URL=https://pub-56e3e682628340078755345d5a0d6c05.r2.dev

Usage:
    python migration/05_upload_images_to_r2.py
    python migration/05_upload_images_to_r2.py --stase ipd
    python migration/05_upload_images_to_r2.py --stase saraf
    python migration/05_upload_images_to_r2.py --dry-run
"""

import argparse
import mimetypes
import os
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDRAG_DB = PROJECT_ROOT / "data" / "medrag.sqlite3"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.migration", override=True)
load_dotenv(Path(__file__).parent / ".env.migration", override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "coas-medical-images")
R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "https://pub-56e3e682628340078755345d5a0d6c05.r2.dev").rstrip("/")

# Default stase for data that doesn't have a stase_slug column yet
DEFAULT_STASE_SLUG = "ipd"


def _clean_path_segment(name: str) -> str:
    """Sanitize a name for use as part of an S3 key."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _build_r2_key(stase_slug: str, source_name: str, image_ref: str) -> str:
    """Build the R2 object key: {stase}/{source}/{filename}."""
    clean_source = _clean_path_segment(source_name)
    # image_ref may contain path separators from the markdown (e.g. img-12.jpeg or ./img-12.jpeg)
    filename = Path(image_ref).name
    return f"{stase_slug}/{clean_source}/{filename}"


def get_r2_client():
    """Create a boto3 S3 client pointed at Cloudflare R2."""
    try:
        import boto3
    except ImportError:
        print("❌ boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY or not R2_ACCOUNT_ID:
        print("❌ R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_ACCOUNT_ID must be set in .env.migration")
        print("   Get them from: Cloudflare Dashboard → R2 → Manage R2 API Tokens")
        sys.exit(1)

    endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )
    return client


def get_supabase_client():
    from supabase import create_client
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("❌ SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env.migration")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def load_images_from_sqlite(stase_filter: str | None = None) -> list[dict]:
    """Load image records from the local medrag.sqlite3."""
    if not MEDRAG_DB.exists():
        print(f"❌ medrag.sqlite3 not found at {MEDRAG_DB}")
        sys.exit(1)

    conn = sqlite3.connect(str(MEDRAG_DB))
    conn.row_factory = sqlite3.Row
    try:
        # Check if stase_slug column exists
        columns = [row[1] for row in conn.execute("PRAGMA table_info(images)").fetchall()]
        has_stase = "stase_slug" in columns

        if has_stase and stase_filter:
            rows = conn.execute(
                "SELECT * FROM images WHERE stase_slug = ?", (stase_filter,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM images").fetchall()

        records = []
        for row in rows:
            r = dict(row)
            if not has_stase:
                r["stase_slug"] = DEFAULT_STASE_SLUG
            records.append(r)
        return records
    finally:
        conn.close()


def upload_images(
    r2,
    supabase,
    records: list[dict],
    dry_run: bool = False,
) -> None:
    """Upload each image to R2 and update its storage_url in Supabase."""
    skipped = 0
    uploaded = 0
    errors = 0

    # Pre-fetch existing storage_urls from Supabase to skip already-uploaded images
    print("📋 Fetching existing storage_url entries from Supabase...")
    try:
        existing_res = supabase.table("images").select("checksum,storage_url").execute()
        existing: dict[str, str] = {
            r["checksum"]: r.get("storage_url", "")
            for r in (existing_res.data or [])
        }
    except Exception as e:
        print(f"⚠️  Could not fetch existing URLs: {e}")
        existing = {}

    for record in tqdm(records, desc="Uploading to R2"):
        checksum = record.get("checksum", "")
        image_abs_path = record.get("image_abs_path", "")
        source_name = record.get("source_name", "unknown")
        image_ref = record.get("image_ref", "")
        stase_slug = record.get("stase_slug", DEFAULT_STASE_SLUG)

        # Skip if already has an R2 URL
        existing_url = existing.get(checksum, "")
        if existing_url and existing_url.startswith(R2_PUBLIC_BASE_URL):
            skipped += 1
            continue

        # Find the local file
        local_path = Path(image_abs_path)
        if not local_path.is_file():
            # Try relative resolution from project root
            alt = PROJECT_ROOT.parents[1] / image_abs_path.lstrip("/\\")
            if alt.is_file():
                local_path = alt
            else:
                tqdm.write(f"⚠️  File not found: {image_abs_path}")
                errors += 1
                continue

        r2_key = _build_r2_key(stase_slug, source_name, image_ref)
        public_url = f"{R2_PUBLIC_BASE_URL}/{r2_key}"
        content_type, _ = mimetypes.guess_type(str(local_path))
        content_type = content_type or "image/jpeg"

        if dry_run:
            tqdm.write(f"[DRY RUN] {local_path.name} → s3://{R2_BUCKET_NAME}/{r2_key}")
            uploaded += 1
            continue

        try:
            with open(local_path, "rb") as f:
                r2.put_object(
                    Bucket=R2_BUCKET_NAME,
                    Key=r2_key,
                    Body=f.read(),
                    ContentType=content_type,
                )

            # Update storage_url and stase_slug in Supabase
            supabase.table("images").update({
                "storage_url": public_url,
                "stase_slug": stase_slug,
            }).eq("checksum", checksum).execute()

            uploaded += 1
        except Exception as e:
            tqdm.write(f"❌ Error uploading {local_path.name}: {e}")
            errors += 1

    print(f"\n✅ Done. uploaded={uploaded}, skipped={skipped}, errors={errors}")
    if not dry_run:
        print(f"   R2 public URL prefix: {R2_PUBLIC_BASE_URL}/")


def main():
    parser = argparse.ArgumentParser(description="Upload medical images to Cloudflare R2")
    parser.add_argument("--stase", default=None, help="Filter by stase slug (e.g. ipd, saraf)")
    parser.add_argument("--dry-run", action="store_true", help="List uploads without actually uploading")
    args = parser.parse_args()

    print(f"🪣 R2 bucket: {R2_BUCKET_NAME}")
    print(f"🔗 Public URL: {R2_PUBLIC_BASE_URL}")
    if args.stase:
        print(f"🏷️  Stase filter: {args.stase}")
    if args.dry_run:
        print("🔍 DRY RUN — no files will be uploaded or Supabase rows updated\n")

    records = load_images_from_sqlite(stase_filter=args.stase)
    print(f"📦 {len(records)} image records loaded from SQLite")

    if not records:
        print("Nothing to do.")
        return

    if args.dry_run:
        upload_images(None, None, records, dry_run=True)
    else:
        r2 = get_r2_client()
        supabase = get_supabase_client()
        upload_images(r2, supabase, records, dry_run=False)


if __name__ == "__main__":
    main()
