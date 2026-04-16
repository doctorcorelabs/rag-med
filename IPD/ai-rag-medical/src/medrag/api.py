from pathlib import Path
from typing import Literal, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv

load_dotenv()

from .config import DEFAULT_DB_PATH, DEFAULT_WORKSPACE_ROOT
from .retriever import (
    related_images, search_chunks, synthesize_answer, get_knowledge_graph,
    _extract_disease_name, _extract_topic_intent, _is_detail_request,
)
from .copilot_client import ask_copilot_adaptive


class ChatHistoryItem(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class SearchDiseaseRequest(BaseModel):
    disease_name: str = Field(min_length=2)
    detail_level: Literal["ringkas", "detail"] = "detail"
    top_k: int = Field(default=8, ge=3, le=20)
    include_images: bool = True
    chat_history: list[ChatHistoryItem] = Field(default_factory=list)


class ImageRequest(BaseModel):
    disease_name: str = Field(min_length=2)
    limit: int = Field(default=3, ge=1, le=10)


def create_app(db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Medical RAG API", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    database = db_path or DEFAULT_DB_PATH
    materials_root = DEFAULT_WORKSPACE_ROOT / "IPD" / "Materi"

    if materials_root.exists():
        app.mount("/materials", StaticFiles(directory=str(materials_root)), name="materials")

    def to_image_url(image_abs_path: str) -> str:
        image_path = Path(image_abs_path)
        try:
            relative_path = image_path.relative_to(materials_root)
            return "/materials/" + relative_path.as_posix()
        except ValueError:
            return image_path.as_uri()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "db_path": str(database), "version": "2.0.0"}

    @app.post("/search_disease_context")
    def search_disease_context(payload: SearchDiseaseRequest) -> dict:
        # Convert chat_history pydantic models to plain dicts
        history_dicts: list[dict[str, Any]] = [
            {"role": h.role, "content": h.content} for h in payload.chat_history
        ]

        # Query analysis
        detected_disease = _extract_disease_name(payload.disease_name)
        detected_intent = _extract_topic_intent(payload.disease_name)
        is_detail = _is_detail_request(payload.disease_name)

        # Hybrid + multi-turn + hierarchical + disease-filtered search
        evidence = search_chunks(
            database,
            payload.disease_name,
            top_k=payload.top_k,
            chat_history=history_dicts if history_dicts else None,
        )

        # Fetch images FIRST so we can pass them to the Vision AI model
        images = []
        if payload.include_images:
            images = related_images(database, payload.disease_name, evidence, limit=3)
            images = [
                {
                    **image,
                    "image_url": to_image_url(image["image_abs_path"]),
                }
                for image in images
            ]

        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            answer = ask_copilot_adaptive(
                payload.disease_name,
                evidence,
                github_token,
                chat_history=history_dicts if history_dicts else None,
                images=images,
            )
        else:
            answer = synthesize_answer(payload.disease_name, evidence)

        return {
            "query": payload.disease_name,
            "query_analysis": {
                "detected_disease": detected_disease,
                "detected_intent": detected_intent,
                "is_detail_request": is_detail,
            },
            "detail_level": payload.detail_level,
            "evidence_count": len(evidence),
            "evidence": evidence,
            "draft_answer": answer,
            "images": images,
            "note": "Gunakan draft_answer sebagai basis penjelasan grounded.",
        }

    @app.post("/get_related_images")
    def get_related_images_endpoint(payload: ImageRequest) -> dict:
        images = related_images(database, payload.disease_name, evidence=[], limit=payload.limit)
        images = [
            {
                **image,
                "image_url": to_image_url(image["image_abs_path"]),
            }
            for image in images
        ]
        return {
            "query": payload.disease_name,
            "images": images,
            "count": len(images),
        }

    @app.get("/knowledge_graph/{disease_name}")
    def knowledge_graph(disease_name: str, max_nodes: int = 40) -> dict:
        """Return knowledge graph nodes and edges for a disease (Ide 18)."""
        return get_knowledge_graph(database, disease_name, max_nodes=max_nodes)

    return app


app = create_app()
