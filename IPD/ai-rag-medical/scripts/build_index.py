import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medrag.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown+image+vector index for Medical RAG")
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--chroma-path", type=Path, default=None)
    parser.add_argument("--skip-vector", action="store_true", help="Skip building ChromaDB vector index (faster, no semantic search)")
    args = parser.parse_args()

    print("[INFO] Starting Medical RAG index build...")
    result = build_index(
        workspace_root=args.workspace_root,
        db_path=args.db_path,
        chroma_path=args.chroma_path,
        skip_vector=args.skip_vector,
    )
    print("\n[OK] Index build completed!")
    print(f"  Source pages : {result['source_pages']}")
    print(f"  Chunks       : {result['chunks']}")
    print(f"  Images       : {result['images']}")
    if not args.skip_vector:
        print("  Vector index : ChromaDB (data/chroma_db/)")


if __name__ == "__main__":
    main()
