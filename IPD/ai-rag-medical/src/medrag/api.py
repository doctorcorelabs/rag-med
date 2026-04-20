from __future__ import annotations

from pathlib import Path
from typing import Literal, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import json
import os
from dotenv import load_dotenv

load_dotenv()

from .config import DEFAULT_DB_PATH, DEFAULT_WORKSPACE_ROOT
from .retriever import (
    related_images, search_chunks, synthesize_answer, get_knowledge_graph,
    _extract_disease_name, _extract_topic_intent, _is_detail_request,
    detect_list_intent, get_topics_from_db,
    _resolve_retrieval_mode, extract_disease_list_from_chunks,
)
from .copilot_client import (
    ask_copilot_adaptive,
    generate_mindmap_from_article,
    merge_two_markdown_articles,
    refine_markdown_with_instruction,
)
from .source_manager import (
    list_sources, create_source, upload_page as sm_upload_page,
    upload_zip, delete_source, get_source_tree, get_page_content,
)
from .stase_manager import create_stase, delete_stase, list_all_stases
from .indexer import build_index_for_source, build_index
from .library_rag_hook import maybe_index_article_for_rag
from . import library as library_mod
from .library import (
    clear_library_article,
    combine_preview_markdown,
    content_hash,
    ensure_library_initialized,
    get_disease_bundle,
    get_disease_list,
    get_stase_by_slug,
    get_stases,
    load_mindmap,
    mindmap_path,
    run_article_generation_pipeline,
    save_mindmap,
    sync_all_stases,
    update_article_meta,
    update_library_article_row,
    write_article_files,
)


class ChatHistoryItem(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class SearchDiseaseRequest(BaseModel):
    disease_name: str = Field(min_length=2)
    detail_level: Literal["ringkas", "detail"] = "detail"
    top_k: int = Field(default=8, ge=3, le=20)
    include_images: bool = True
    chat_history: list[ChatHistoryItem] = Field(default_factory=list)
    # Dynamic retrieval mode
    # "relevant" → focused QA (default), "exhaustive" → listing/catalog mode.
    # When omitted, the mode is auto-detected from the query text.
    retrieval_mode: Literal["relevant", "exhaustive"] | None = None
    # Optional paging controls (used when retrieval_mode="exhaustive")
    max_items: int | None = Field(default=None, ge=1, le=500)
    page: int | None = Field(default=None, ge=1)
    page_size: int | None = Field(default=None, ge=1, le=200)


class ImageRequest(BaseModel):
    disease_name: str = Field(min_length=2)
    limit: int = Field(default=3, ge=1, le=10)


# ── Admin Models ─────────────────────────────────────────────────────────────

class CreateSourceRequest(BaseModel):
    source_name: str = Field(min_length=3, description="Harus diawali '(Sumber) '")


class UploadPageRequest(BaseModel):
    page_no: int = Field(ge=1)
    markdown: str = Field(min_length=1)


class CreateStaseRequest(BaseModel):
    slug: str = Field(min_length=2, pattern=r"^[a-z][a-z0-9_-]*$")
    display_name: str = Field(min_length=2)


class LibraryGenerateRequest(BaseModel):
    extra_prompt: str | None = None
    top_k: int = Field(default=10, ge=3, le=20)
    image_limit: int = Field(default=5, ge=1, le=10)


class LibraryPreviewRequest(LibraryGenerateRequest):
    combine_with_existing: bool = False
    combine_mode: Literal["append", "replace"] = "replace"
    persist: bool = False  # save markdown_combined to disk + SQLite (same draft/published rules as generate)


class LibraryRefineRequest(BaseModel):
    instruction: str = Field(min_length=3)


class LibraryPatchContentRequest(BaseModel):
    markdown: str = Field(min_length=1)
    preview_commit: bool = False


class LibraryImageRef(BaseModel):
    image_abs_path: str = Field(min_length=3)
    heading: str = ""
    source_name: str = ""
    page_no: int = 0


class LibraryUpdateVisualRefsRequest(BaseModel):
    images: list[LibraryImageRef] = Field(default_factory=list)


class MindmapNodeModel(BaseModel):
    id: str
    label: str
    type: str = "concept"
    level: int = 2
    summary: str = ""
    val: int = 7
    group: int = 2


class MindmapEdgeModel(BaseModel):
    source: str
    target: str


class VisualRefModel(BaseModel):
    image_url: str = ""
    heading: str = ""
    description: str = ""


class MindmapSaveRequest(BaseModel):
    nodes: list[MindmapNodeModel]
    edges: list[MindmapEdgeModel]
    visual_refs: list[VisualRefModel] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    summary_root: str = ""


class LibraryMergeMarkdownRequest(BaseModel):
    """Menggabung artikel utama + kandidat via Copilot (stateless; untuk modal pratinjau)."""

    markdown_base: str = ""
    markdown_candidate: str = Field(min_length=1)


def create_app(db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Medical RAG API", version="3.1.0")

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

    @app.on_event("startup")
    def _library_startup() -> None:
        try:
            ensure_library_initialized()
            sync_all_stases()
        except Exception as exc:
            print(f"[Library] startup sync: {exc}")

    def _library_images_from_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for img in meta.get("images") or []:
            p = img.get("image_abs_path") or ""
            if not p:
                continue
            out.append({**img, "image_url": to_image_url(p)})
        return out

    def _enrich_evidence_with_urls(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add source_url to each evidence item for frontend traceability."""
        enriched = []
        for item in evidence:
            item_copy = dict(item)
            md_path = item.get("markdown_path", "")
            if md_path:
                try:
                    rel = Path(md_path).relative_to(materials_root)
                    item_copy["source_url"] = "/materials/" + rel.as_posix()
                except (ValueError, TypeError):
                    item_copy["source_url"] = ""
            else:
                item_copy["source_url"] = ""
            enriched.append(item_copy)
        return enriched

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "db_path": str(database), "version": "3.1.0"}

    # ── List Topics (Enumerative) ─────────────────────────────────────────────

    @app.get("/list_topics")
    def list_topics_endpoint(slug: str | None = None) -> dict:
        """
        Enumerasi semua topik/heading dari knowledge base, dikelompokkan per sumber.
        Tidak menggunakan search — langsung baca metadata DB.
        """
        return get_topics_from_db(database, stase_slug=slug)

    @app.get("/list_topics/catalog")
    def list_topics_catalog(slug: str = "ipd") -> dict:
        """
        Daftar lengkap penyakit dari katalog CSV (termasuk yang belum punya artikel).
        Berguna untuk menampilkan progress coverage RAG.
        """
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            st = library_mod.get_stase_by_slug(conn, slug)
            if not st:
                raise HTTPException(status_code=404, detail=f"Stase '{slug}' tidak ditemukan")
            diseases = library_mod.get_disease_list(conn, st["id"])
            # Enrich dengan info apakah sudah ada chunk di DB
            indexed_names = set()
            try:
                import sqlite3 as _sq
                c = _sq.connect(database)
                for r in c.execute("SELECT DISTINCT source_name FROM chunks WHERE stase_slug = ?", (slug,)):
                    indexed_names.add(r[0].lower())
                c.close()
            except Exception:
                pass
            for d in diseases:
                d["has_rag_content"] = any(
                    d["name"].lower() in src for src in indexed_names
                )
            return {
                "stase": {"slug": st["slug"], "display_name": st["display_name"]},
                "diseases": diseases,
                "total": len(diseases),
                "with_article": sum(1 for d in diseases if d.get("status") not in ("missing", None)),
            }
        finally:
            conn.close()

    @app.post("/search_disease_context")
    def search_disease_context(payload: SearchDiseaseRequest) -> dict:
        history_dicts: list[dict[str, Any]] = [
            {"role": h.role, "content": h.content} for h in payload.chat_history
        ]

        # Resolve retrieval mode (explicit override or auto-detected from query)
        mode = _resolve_retrieval_mode(payload.disease_name, payload.retrieval_mode)
        is_exhaustive = mode == "exhaustive"

        # ── Cek list intent sebelum semantic search ────────────────────────
        if is_exhaustive:
            # Exhaustive mode: use semantic search with higher top_k then build
            # a deduplicated disease list.  Topics DB is always added for
            # completeness.
            topics = get_topics_from_db(database)

            evidence = search_chunks(
                database,
                payload.disease_name,
                top_k=payload.top_k,
                chat_history=history_dicts if history_dicts else None,
                retrieval_mode="exhaustive",
            )

            # Build disease list from retrieved chunks
            raw_disease_list = extract_disease_list_from_chunks(evidence)

            # Apply optional pagination / max_items
            page_size = payload.page_size or 50
            page = payload.page or 1
            max_items = payload.max_items
            total_found = len(raw_disease_list)
            if max_items:
                raw_disease_list = raw_disease_list[:max_items]

            start = (page - 1) * page_size
            paged_list = raw_disease_list[start:start + page_size]
            returned_count = len(paged_list)
            is_truncated = (start + returned_count) < len(raw_disease_list)

            # Build a structured answer for list queries
            sections = []
            for src in topics.get("sources", []):
                topics_md = "\n".join(
                    f"{i+1}. **{t['heading']}**"
                    for i, t in enumerate(src.get("topics", []))
                )
                sections.append({"title": src["source_name"], "markdown": topics_md})
            list_answer: dict[str, Any] = {
                "disease": "Daftar Topik Knowledge Base",
                "sections": sections,
                "citations": [],
                "grounded": True,
            }
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                from .copilot_client import ask_copilot_for_list
                try:
                    list_answer = ask_copilot_for_list(topics, github_token)
                except Exception:
                    pass  # fallback ke list_answer manual di atas

            return {
                "query": payload.disease_name,
                "query_analysis": {
                    "is_list_intent": True,
                    "retrieval_mode": mode,
                },
                "retrieval_mode": mode,
                "detail_level": payload.detail_level,
                "evidence_count": len(evidence),
                "evidence": [],
                "draft_answer": list_answer,
                "images": [],
                "topics": topics,
                "disease_list": paged_list,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_found": total_found,
                    "returned_count": returned_count,
                    "is_truncated": is_truncated,
                },
                "note": "Query terdeteksi sebagai permintaan daftar; mengembalikan enumerasi dari knowledge base.",
            }

        detected_disease = _extract_disease_name(payload.disease_name)
        detected_intent = _extract_topic_intent(payload.disease_name)
        is_detail = _is_detail_request(payload.disease_name)

        evidence = search_chunks(
            database,
            payload.disease_name,
            top_k=payload.top_k,
            chat_history=history_dicts if history_dicts else None,
            retrieval_mode="relevant",
        )

        evidence_with_urls = _enrich_evidence_with_urls(evidence)

        images: list[dict[str, Any]] = []
        if payload.include_images:
            images = related_images(database, payload.disease_name, evidence, limit=3)
            images = [
                {**image, "image_url": to_image_url(image["image_abs_path"])}
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
                "retrieval_mode": mode,
            },
            "retrieval_mode": mode,
            "detail_level": payload.detail_level,
            "evidence_count": len(evidence),
            "evidence": evidence_with_urls,
            "draft_answer": answer,
            "images": images,
            "pagination": {
                "page": 1,
                "page_size": len(evidence),
                "total_found": len(evidence),
                "returned_count": len(evidence),
                "is_truncated": False,
            },
            "note": "Gunakan draft_answer sebagai basis penjelasan grounded.",
        }

    @app.post("/get_related_images")
    def get_related_images_endpoint(payload: ImageRequest) -> dict:
        images = related_images(database, payload.disease_name, evidence=[], limit=payload.limit)
        images = [
            {**image, "image_url": to_image_url(image["image_abs_path"])}
            for image in images
        ]
        return {
            "query": payload.disease_name,
            "images": images,
            "count": len(images),
        }

    @app.get("/knowledge_graph/{disease_name}")
    def knowledge_graph(disease_name: str, max_nodes: int = 40) -> dict:
        return get_knowledge_graph(database, disease_name, max_nodes=max_nodes)

    def _build_mindmap_for_disease(bundle: dict[str, Any]) -> dict[str, Any]:
        """Internal helper: call Copilot, build mindmap, enrich URLs. Does NOT save."""
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise HTTPException(status_code=503, detail="GITHUB_TOKEN required for mindmap generation")

        disease_name: str = bundle["name"]
        competency_level: str | None = bundle.get("competency_level")
        cp = bundle.get("content_path")
        mp = bundle.get("meta_path")

        if not cp or not Path(cp).is_file():
            raise HTTPException(status_code=400, detail="Article not found. Generate article in Medical Library first.")

        markdown_content = Path(cp).read_text(encoding="utf-8")

        images_for_vision: list[dict[str, Any]] = []
        if mp and Path(mp).is_file():
            meta_data = json.loads(Path(mp).read_text(encoding="utf-8"))
            for img in meta_data.get("images") or []:
                p = img.get("image_abs_path", "")
                if p and Path(p).is_file():
                    images_for_vision.append(img)

        result = generate_mindmap_from_article(
            disease_name=disease_name,
            markdown_content=markdown_content,
            github_token=github_token,
            competency_level=competency_level,
            images=images_for_vision,
        )

        for vref in result.get("visual_refs") or []:
            raw_path = vref.get("image_url", "")
            if raw_path and not raw_path.startswith("/"):
                vref["image_url"] = to_image_url(raw_path)

        return result

    @app.get("/library/stases/{slug}/diseases/{catalog_id}/mindmap")
    def library_mindmap_get(slug: str, catalog_id: int) -> dict:
        """Return saved mindmap.json if it exists; otherwise return not_generated flag."""
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")

            saved = load_mindmap(slug, bundle["catalog_no"])
            if saved:
                return saved

            return {
                "disease": bundle["name"],
                "competency_level": bundle.get("competency_level"),
                "nodes": [],
                "edges": [],
                "visual_refs": [],
                "key_takeaways": [],
                "not_generated": True,
            }
        finally:
            conn.close()

    @app.post("/library/stases/{slug}/diseases/{catalog_id}/mindmap/generate")
    def library_mindmap_generate(slug: str, catalog_id: int) -> dict:
        """(Re)generate mindmap via Copilot and persist to mindmap.json."""
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")

            result = _build_mindmap_for_disease(bundle)
            save_mindmap(slug, bundle["catalog_no"], result)
            return result
        finally:
            conn.close()

    @app.patch("/library/stases/{slug}/diseases/{catalog_id}/mindmap")
    def library_mindmap_save(slug: str, catalog_id: int, payload: MindmapSaveRequest) -> dict:
        """Persist hand-edited mindmap data (nodes/edges/takeaways) to mindmap.json."""
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")

            existing = load_mindmap(slug, bundle["catalog_no"]) or {}
            data: dict[str, Any] = {
                **existing,
                "disease": bundle["name"],
                "competency_level": bundle.get("competency_level"),
                "nodes": [n.model_dump() for n in payload.nodes],
                "edges": [e.model_dump() for e in payload.edges],
                "visual_refs": [v.model_dump() for v in payload.visual_refs],
                "key_takeaways": payload.key_takeaways,
                "summary_root": payload.summary_root,
            }
            save_mindmap(slug, bundle["catalog_no"], data)
            return {"ok": True, "disease": bundle["name"], "node_count": len(payload.nodes)}
        finally:
            conn.close()

    # ── Medical Library (multi-stase catalog + on-disk articles) ─────────────

    @app.get("/library/stases")
    def library_list_stases() -> dict[str, Any]:
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            rows = get_stases(conn)
            return {"stases": rows}
        finally:
            conn.close()

    @app.post("/library/sync")
    def library_sync_catalogs() -> dict[str, Any]:
        """Re-import CSV catalogs (idempotent)."""
        try:
            counts = sync_all_stases()
            return {"ok": True, "synced": counts}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/library/merge_markdown_copilot")
    def library_merge_markdown_copilot(payload: LibraryMergeMarkdownRequest) -> dict[str, Any]:
        """Gabungkan artikel utama dan kandidat baru dengan Copilot (modal pratinjau)."""
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise HTTPException(status_code=503, detail="GITHUB_TOKEN required for AI merge")
        try:
            merged = merge_two_markdown_articles(
                payload.markdown_base,
                payload.markdown_candidate,
                github_token,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"ok": True, "markdown_merged": merged}

    @app.get("/library/stases/{slug}/diseases")
    def library_list_diseases(slug: str) -> dict[str, Any]:
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            st = get_stase_by_slug(conn, slug)
            if not st:
                raise HTTPException(status_code=404, detail="Stase not found")
            diseases = get_disease_list(conn, st["id"])
            total = len(diseases)
            filled = sum(1 for d in diseases if d.get("status") in ("draft", "published"))
            return {
                "stase": {"slug": st["slug"], "display_name": st["display_name"]},
                "diseases": diseases,
                "progress": {"filled": filled, "total": total, "percent": round(100 * filled / total, 1) if total else 0},
            }
        finally:
            conn.close()

    @app.get("/library/stases/{slug}/diseases/{catalog_id}")
    def library_get_disease(slug: str, catalog_id: int) -> dict[str, Any]:
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")
            meta: dict[str, Any] | None = None
            markdown: str | None = None
            mp = bundle.get("meta_path")
            cp = bundle.get("content_path")
            if mp and Path(mp).is_file():
                meta = json.loads(Path(mp).read_text(encoding="utf-8"))
            if cp and Path(cp).is_file():
                markdown = Path(cp).read_text(encoding="utf-8")
            images_out: list[dict[str, Any]] = []
            if meta:
                images_out = _library_images_from_meta(meta)
            return {
                "disease": {k: bundle[k] for k in bundle if k not in ("meta_path", "content_path")},
                "markdown": markdown,
                "meta": meta,
                "images": images_out,
            }
        finally:
            conn.close()

    @app.post("/library/stases/{slug}/diseases/{catalog_id}/preview")
    def library_preview(slug: str, catalog_id: int, payload: LibraryPreviewRequest) -> dict[str, Any]:
        """Regenerate synthesis without writing disk; optional combine with existing article."""
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")
            disease_name = bundle["name"]

            gen = run_article_generation_pipeline(
                database,
                disease_name,
                payload.extra_prompt,
                payload.top_k,
                payload.image_limit,
                to_image_url,
            )
            evidence = gen["evidence"]
            evidence_with_urls = _enrich_evidence_with_urls(evidence)
            answer = gen["draft_answer"]
            markdown_candidate = gen["markdown_candidate"]
            images = gen["images"]

            markdown_base = ""
            cp = bundle.get("content_path")
            if cp and Path(cp).is_file():
                markdown_base = Path(cp).read_text(encoding="utf-8")

            if not payload.combine_with_existing or payload.combine_mode == "replace":
                markdown_combined = combine_preview_markdown(
                    markdown_base if markdown_base.strip() else None,
                    markdown_candidate,
                    "replace",
                )
                preview_note = "Kandidat baru saja (ganti penuh). Belum disimpan."
            else:
                markdown_combined = combine_preview_markdown(
                    markdown_base if markdown_base.strip() else None,
                    markdown_candidate,
                    "append",
                )
                preview_note = (
                    "Gabungan artikel lama + pembaruan. Belum disimpan."
                    if markdown_base.strip()
                    else "Belum ada artikel lama; kandidat sama dengan generate baru. Belum disimpan."
                )

            if payload.persist:
                status = "published"
                if not evidence:
                    status = "draft"
                if isinstance(answer, dict) and answer.get("grounded") is False:
                    status = "draft"
                ad = library_mod.article_dir(slug, bundle["catalog_no"])
                ad.mkdir(parents=True, exist_ok=True)
                md_path = ad / "content.md"
                meta_path = ad / "meta.json"
                md_path.write_text(markdown_combined, encoding="utf-8")
                meta: dict[str, Any] = {}
                ver = 1
                if meta_path.is_file():
                    try:
                        old = json.loads(meta_path.read_text(encoding="utf-8"))
                        ver = int(old.get("version", 0)) + 1
                        meta = old
                    except Exception:
                        ver = 1
                meta.update({
                    "version": ver,
                    "disease_name": disease_name,
                    "catalog_no": bundle["catalog_no"],
                    "stase_slug": slug,
                    "generated_at": library_mod._utc_now_iso(),
                    "extra_prompt": payload.extra_prompt,
                    "last_operation": "preview_persist",
                    "images": [
                        {
                            "image_abs_path": img.get("image_abs_path", ""),
                            "heading": img.get("heading", ""),
                            "source_name": img.get("source_name", ""),
                            "page_no": img.get("page_no", 0),
                        }
                        for img in images
                    ],
                })
                hook = maybe_index_article_for_rag(md_path, meta)
                meta.update({k: hook[k] for k in ("indexed_into_rag", "indexed_checksum", "content_checksum") if k in hook})
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                ch = content_hash(markdown_combined)
                update_library_article_row(
                    conn,
                    catalog_id,
                    status,
                    str(md_path),
                    str(meta_path),
                    ch,
                )
                conn.commit()
                preview_note = preview_note.replace("Belum disimpan.", "Tersimpan ke artikel utama.").replace(
                    "Belum disimpan", "Tersimpan ke artikel utama"
                )

            return {
                "ok": True,
                "markdown_base": markdown_base,
                "markdown_candidate": markdown_candidate,
                "markdown_combined": markdown_combined,
                "draft_answer": answer,
                "evidence_count": len(evidence),
                "evidence": evidence_with_urls,
                "images": images,
                "preview_note": preview_note,
                "persisted": payload.persist,
            }
        finally:
            conn.close()

    @app.post("/library/stases/{slug}/diseases/{catalog_id}/generate")
    def library_generate(slug: str, catalog_id: int, payload: LibraryGenerateRequest) -> dict[str, Any]:
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")
            disease_name = bundle["name"]

            gen = run_article_generation_pipeline(
                database,
                disease_name,
                payload.extra_prompt,
                payload.top_k,
                payload.image_limit,
                to_image_url,
            )
            evidence = gen["evidence"]
            evidence_with_urls = _enrich_evidence_with_urls(evidence)
            answer = gen["draft_answer"]
            images = gen["images"]

            status = "published"
            if not evidence:
                status = "draft"
            if isinstance(answer, dict) and answer.get("grounded") is False:
                status = "draft"

            ad = library_mod.article_dir(slug, bundle["catalog_no"])
            prev_meta = ad / "meta.json"
            ver = 1
            if prev_meta.is_file():
                try:
                    old = json.loads(prev_meta.read_text(encoding="utf-8"))
                    ver = int(old.get("version", 0)) + 1
                except Exception:
                    ver = 1
            md_path, meta_path, meta = write_article_files(
                slug,
                bundle["catalog_no"],
                disease_name,
                answer,
                images,
                {
                    "extra_prompt": payload.extra_prompt,
                    "last_operation": "generate",
                    "version": ver,
                },
            )

            ch = content_hash(Path(md_path).read_text(encoding="utf-8"))
            update_library_article_row(
                conn,
                catalog_id,
                status,
                str(md_path),
                str(meta_path),
                ch,
            )
            conn.commit()
            return {
                "ok": True,
                "status": status,
                "draft_answer": answer,
                "evidence_count": len(evidence),
                "evidence": evidence_with_urls,
                "content_path": str(md_path),
                "meta": meta,
                "images": images,
            }
        finally:
            conn.close()

    @app.post("/library/stases/{slug}/diseases/{catalog_id}/refine")
    def library_refine(slug: str, catalog_id: int, payload: LibraryRefineRequest) -> dict[str, Any]:
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise HTTPException(status_code=503, detail="GITHUB_TOKEN required for refine")
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")
            cp = bundle.get("content_path")
            if not cp or not Path(cp).is_file():
                raise HTTPException(status_code=400, detail="No article to refine; generate first")
            md = Path(cp).read_text(encoding="utf-8")
            new_md = refine_markdown_with_instruction(md, payload.instruction, github_token)
            Path(cp).write_text(new_md, encoding="utf-8")
            ad = library_mod.article_dir(slug, bundle["catalog_no"])
            meta_path = ad / "meta.json"
            meta = update_article_meta(
                meta_path,
                {
                    "last_refine_instruction": payload.instruction,
                    "last_operation": "refine",
                },
            )
            hook = maybe_index_article_for_rag(Path(cp), meta)
            meta.update({k: hook[k] for k in ("indexed_into_rag", "indexed_checksum", "content_checksum") if k in hook})
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            ch = content_hash(new_md)
            update_library_article_row(
                conn,
                catalog_id,
                "published",
                str(cp),
                str(meta_path),
                ch,
            )
            conn.commit()
            return {"ok": True, "markdown": new_md, "meta": meta}
        finally:
            conn.close()

    @app.patch("/library/stases/{slug}/diseases/{catalog_id}/content")
    def library_patch_content(slug: str, catalog_id: int, payload: LibraryPatchContentRequest) -> dict[str, Any]:
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")
            ad = library_mod.article_dir(slug, bundle["catalog_no"])
            ad.mkdir(parents=True, exist_ok=True)
            md_path = ad / "content.md"
            meta_path = ad / "meta.json"
            md_path.write_text(payload.markdown, encoding="utf-8")
            meta: dict[str, Any] = {}
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["last_operation"] = "preview_commit" if payload.preview_commit else "manual_edit"
            meta["updated_at"] = library_mod._utc_now_iso()
            hook = maybe_index_article_for_rag(md_path, meta)
            meta.update({k: hook[k] for k in ("indexed_into_rag", "indexed_checksum", "content_checksum") if k in hook})
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            ch = content_hash(payload.markdown)
            update_library_article_row(
                conn,
                catalog_id,
                "published",
                str(md_path),
                str(meta_path),
                ch,
            )
            conn.commit()
            return {"ok": True, "content_path": str(md_path), "meta": meta}
        finally:
            conn.close()

    @app.patch("/library/stases/{slug}/diseases/{catalog_id}/visual_refs")
    def library_update_visual_refs(slug: str, catalog_id: int, payload: LibraryUpdateVisualRefsRequest) -> dict[str, Any]:
        """Replace visual references stored in meta.json for this article."""
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")
            ad = library_mod.article_dir(slug, bundle["catalog_no"])
            ad.mkdir(parents=True, exist_ok=True)
            meta_path = ad / "meta.json"
            meta: dict[str, Any] = {}
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))

            meta["images"] = [img.model_dump() for img in payload.images]
            meta["last_operation"] = "update_visual_refs"
            meta["updated_at"] = library_mod._utc_now_iso()
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            images_out = _library_images_from_meta(meta)
            conn.commit()
            return {"ok": True, "images": images_out, "count": len(images_out), "meta": meta}
        finally:
            conn.close()

    @app.delete("/library/stases/{slug}/diseases/{catalog_id}/article")
    def library_delete_article(slug: str, catalog_id: int) -> dict[str, Any]:
        conn = library_mod._connect()
        try:
            library_mod.init_library_schema(conn)
            bundle = get_disease_bundle(conn, slug, catalog_id)
            if not bundle:
                raise HTTPException(status_code=404, detail="Disease not found")
            clear_library_article(conn, catalog_id, slug, bundle["catalog_no"])
            conn.commit()
            return {"ok": True}
        finally:
            conn.close()

    # ══ Admin: Source Manager ══════════════════════════════════════════════════════════════════════

    @app.get("/admin/stases/{slug}/sources")
    def admin_list_sources(slug: str) -> dict:
        """List semua folder sumber di Materi/ untuk stase ini + index status."""
        try:
            sources = list_sources(slug)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"slug": slug, "sources": sources, "count": len(sources)}

    @app.get("/admin/stases/{slug}/sources/{source_name}")
    def admin_get_source_tree(slug: str, source_name: str) -> dict:
        """Detail tree satu sumber: semua page + file yang ada."""
        try:
            return get_source_tree(slug, source_name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/admin/stases/{slug}/sources/{source_name}/pages/{page_no}")
    def admin_get_page(slug: str, source_name: str, page_no: int) -> dict:
        """Baca konten markdown satu halaman."""
        content = get_page_content(slug, source_name, page_no)
        if content is None:
            raise HTTPException(status_code=404, detail=f"Halaman {page_no} tidak ditemukan")
        return {"page_no": page_no, "markdown": content, "chars": len(content)}

    @app.post("/admin/stases/{slug}/sources")
    def admin_create_source(slug: str, payload: CreateSourceRequest) -> dict:
        """Buat folder sumber baru dengan template page-1/markdown.md."""
        try:
            result = create_source(slug, payload.source_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, **result}

    @app.post("/admin/stases/{slug}/sources/{source_name}/pages/{page_no}")
    def admin_upload_page(
        slug: str, source_name: str, page_no: int, payload: UploadPageRequest,
    ) -> dict:
        """Upload / update konten markdown satu halaman tertentu."""
        try:
            result = sm_upload_page(slug, source_name, page_no, payload.markdown)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"ok": True, **result}

    @app.post("/admin/stases/{slug}/sources/{source_name}/upload_zip")
    async def admin_upload_zip(
        slug: str,
        source_name: str,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> dict:
        """Upload ZIP berisi batch pages. Auto-trigger partial re-index."""
        content = await file.read()
        try:
            result = upload_zip(slug, source_name, content)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        background_tasks.add_task(
            _run_partial_reindex_bg, slug, source_name
        )
        return {"ok": True, **result, "reindex_queued": True}

    @app.delete("/admin/stases/{slug}/sources/{source_name}")
    def admin_delete_source(slug: str, source_name: str) -> dict:
        """Hapus folder sumber beserta isinya (tidak bisa dibatalkan)."""
        try:
            delete_source(slug, source_name)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"ok": True, "deleted": source_name}

    @app.post("/admin/reindex")
    def admin_trigger_reindex(
        background_tasks: BackgroundTasks,
        slug: str | None = None,
        source_name: str | None = None,
        skip_vector: bool = True,
    ) -> dict:
        """
        Trigger rebuild index:
        - Tanpa params → full rebuild semua sumber
        - ?slug=X → rebuild semua sumber di stase X
        - ?slug=X&source_name=Y → partial rebuild satu sumber
        """
        from fastapi import BackgroundTasks
        if source_name and slug:
            background_tasks.add_task(
                _run_partial_reindex_bg, slug, source_name
            )
            return {"ok": True, "mode": "partial", "target": source_name, "reindex_queued": True}
        else:
            background_tasks.add_task(_run_full_reindex_bg, skip_vector)
            return {"ok": True, "mode": "full", "reindex_queued": True}

    def _run_partial_reindex_bg(slug: str, source_name: str) -> None:
        try:
            result = build_index_for_source(
                source_name, slug, DEFAULT_WORKSPACE_ROOT, database
            )
            print(f"[Admin ReIndex] {source_name}: {result}")
        except Exception as e:
            print(f"[Admin ReIndex ERROR] {source_name}: {e}")

    def _run_full_reindex_bg(skip_vector: bool) -> None:
        try:
            result = build_index(db_path=database, skip_vector=skip_vector)
            print(f"[Admin ReIndex Full]: {result}")
        except Exception as e:
            print(f"[Admin ReIndex Full ERROR]: {e}")

    # ══ Admin: Stase Manager ══════════════════════════════════════════════════════════════════════

    @app.get("/admin/stases")
    def admin_list_stases() -> dict:
        """List semua stase (hardcoded + stase_overrides.json)."""
        return {"stases": list_all_stases()}

    @app.post("/admin/stases")
    def admin_create_stase(payload: CreateStaseRequest) -> dict:
        """Buat stase baru: folder, CSV placeholder, register DB + overrides.json."""
        try:
            result = create_stase(payload.slug, payload.display_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, **result}

    @app.delete("/admin/stases/{slug}")
    def admin_delete_stase(slug: str) -> dict:
        """Hapus stase dari registry. TIDAK menghapus folder materi."""
        try:
            delete_stase(slug)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, "deleted": slug}

    return app


app = create_app()
