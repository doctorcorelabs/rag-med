import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medrag.api import create_app
from medrag.config import DEFAULT_DB_PATH, DEFAULT_WORKSPACE_ROOT
from medrag.retriever import related_images, search_chunks, synthesize_answer

mcp = FastMCP("medical-rag")


def _materials_root() -> Path:
    return DEFAULT_WORKSPACE_ROOT / "IPD" / "Materi"


def _to_image_url(image_abs_path: str) -> str:
    image_path = Path(image_abs_path)
    return image_path.resolve().as_uri()


@mcp.tool()
def search_disease_context(disease_name: str, detail_level: str = "detail", top_k: int = 8, include_images: bool = True) -> dict:
    """Search disease explanation evidence from indexed markdown materials."""
    database = DEFAULT_DB_PATH
    evidence = search_chunks(database, disease_name, top_k=top_k)
    answer = synthesize_answer(disease_name, evidence)
    images = []
    if include_images:
        images = related_images(database, disease_name, evidence, limit=3)
        images = [
            {**image, "image_url": _to_image_url(image["image_abs_path"])}
            for image in images
        ]

    return {
        "query": disease_name,
        "detail_level": detail_level,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "draft_answer": answer,
        "images": images,
        "note": "Gunakan draft_answer sebagai basis penjelasan grounded.",
    }


@mcp.tool()
def get_related_images(disease_name: str, limit: int = 3) -> dict:
    """Return related images for a disease query."""
    database = DEFAULT_DB_PATH
    evidence = search_chunks(database, disease_name, top_k=8)
    images = related_images(database, disease_name, evidence, limit=limit)
    images = [
        {**image, "image_url": _to_image_url(image["image_abs_path"])}
        for image in images
    ]
    return {
        "query": disease_name,
        "images": images,
        "count": len(images),
    }


if __name__ == "__main__":
    mcp.run()
