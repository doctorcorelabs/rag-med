"""
Medical RAG — Upload local images to Supabase Storage
=====================================================
Reads image_abs_path from local SQLite, uploads each file to
Supabase Storage bucket "medical-images", then updates the
storage_url column in the Supabase images table.

Usage:
    python migration/03_upload_images_to_supabase.py
"""

import mimetypes
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDRAG_DB = PROJECT_ROOT / "data" / "medrag.sqlite3"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.migration", override=True)
load_dotenv(PROJECT_ROOT / "migration" / ".env.migration", override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
BUCKET_NAME = "medical-images"


def main():
    try:
        from supabase import create_client
    except ImportError:
        print("pip install supabase")
        sys.exit(1)

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env.migration")
        sys.exit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print(f"Connected to {SUPABASE_URL}")

    # Ensure bucket exists
    try:
        supabase.storage.get_bucket(BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' exists.")
    except Exception:
        print(f"Creating public bucket '{BUCKET_NAME}'...")
        supabase.storage.create_bucket(BUCKET_NAME, options={"public": True})
        print(f"Bucket '{BUCKET_NAME}' created.")

    conn = sqlite3.connect(str(MEDRAG_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, image_abs_path, image_ref, source_name, page_no, checksum FROM images"
    ).fetchall()
    conn.close()

    print(f"\nUploading {len(rows)} images...")
    uploaded = 0
    skipped = 0
    errors = 0

    for row in tqdm(rows, desc="Uploading"):
        local_path = Path(row["image_abs_path"])
        if not local_path.is_file():
            skipped += 1
            continue

        source_clean = row["source_name"].replace(" ", "_").replace("(", "").replace(")", "")
        storage_path = f"{source_clean}/p{row['page_no']}/{row['image_ref']}"

        content_type = mimetypes.guess_type(str(local_path))[0] or "image/jpeg"

        try:
            file_bytes = local_path.read_bytes()
            supabase.storage.from_(BUCKET_NAME).upload(
                storage_path,
                file_bytes,
                file_options={"content-type": content_type, "upsert": "true"},
            )

            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{storage_path}"

            supabase.table("images").update(
                {"storage_url": public_url}
            ).eq("checksum", row["checksum"]).execute()

            uploaded += 1
        except Exception as e:
            err_msg = str(e)
            if "Duplicate" in err_msg or "already exists" in err_msg:
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{storage_path}"
                supabase.table("images").update(
                    {"storage_url": public_url}
                ).eq("checksum", row["checksum"]).execute()
                uploaded += 1
            else:
                errors += 1
                if errors <= 5:
                    print(f"\n  Error: {storage_path}: {e}")

    print(f"\nDone! Uploaded: {uploaded}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
