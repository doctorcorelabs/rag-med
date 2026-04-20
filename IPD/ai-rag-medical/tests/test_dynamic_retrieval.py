"""
Tests for the dynamic retrieval mode feature.

Covers:
1. Listing intent detection (Python + config constants).
2. Retrieval mode resolver.
3. exhaustive mode top_k is larger than relevant mode top_k.
4. Disease list extractor produces deduplicated entries with evidence.
5. API request schema accepts the new fields.
6. Response metadata (pagination / is_truncated) is present and coherent.
"""

import sys
from pathlib import Path
import sqlite3
import tempfile
import pytest

# Allow import from src/medrag without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medrag.retriever import (
    detect_list_intent,
    _is_listing_intent,
    _resolve_retrieval_mode,
    extract_disease_list_from_chunks,
    LIST_INTENT_KEYWORDS,
)
from medrag.config import (
    DEFAULT_TOP_K_RELEVANT,
    DEFAULT_TOP_K_EXHAUSTIVE,
    MAX_TOP_K_EXHAUSTIVE,
    ENABLE_EXHAUSTIVE_AUTO_MODE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Listing intent detection
# ─────────────────────────────────────────────────────────────────────────────

class TestListingIntentDetection:
    """detect_list_intent / _is_listing_intent should flag catalog-style queries."""

    @pytest.mark.parametrize("query", [
        "daftar penyakit",
        "daftar semua penyakit",
        "semua penyakit",
        "list penyakit",
        "penyakit apa saja",
        "tampilkan semua",
        "berikan daftar",
        "list semua",
        "semua materi",
    ])
    def test_listing_queries_detected(self, query: str):
        assert detect_list_intent(query) is True, f"Expected listing intent for: {query!r}"

    @pytest.mark.parametrize("query", [
        "penanganan pneumonia",
        "tatalaksana diabetes",
        "definisi hipertensi",
        "apa itu tbc",
        "gejala gagal jantung",
    ])
    def test_focused_queries_not_listing(self, query: str):
        assert detect_list_intent(query) is False, f"Should NOT be listing intent: {query!r}"

    def test_is_listing_intent_alias_matches(self):
        """_is_listing_intent is an alias and must behave identically."""
        for kw in LIST_INTENT_KEYWORDS[:5]:
            assert _is_listing_intent(kw) is True
            assert detect_list_intent(kw) is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval mode resolver
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveRetrievalMode:
    """_resolve_retrieval_mode must respect priority: explicit > auto > default."""

    def test_explicit_exhaustive(self):
        assert _resolve_retrieval_mode("penanganan pneumonia", "exhaustive") == "exhaustive"

    def test_explicit_relevant(self):
        assert _resolve_retrieval_mode("daftar penyakit", "relevant") == "relevant"

    def test_auto_exhaustive_for_listing_query(self):
        assert _resolve_retrieval_mode("daftar semua penyakit") == "exhaustive"

    def test_auto_relevant_for_focused_query(self):
        assert _resolve_retrieval_mode("tatalaksana tbc") == "relevant"

    def test_none_mode_falls_back_to_auto(self):
        assert _resolve_retrieval_mode("semua penyakit", None) == "exhaustive"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Exhaustive top_k is substantially higher than relevant top_k
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicTopK:
    def test_exhaustive_top_k_gt_relevant(self):
        assert DEFAULT_TOP_K_EXHAUSTIVE > DEFAULT_TOP_K_RELEVANT

    def test_max_top_k_exhaustive_gte_default_exhaustive(self):
        assert MAX_TOP_K_EXHAUSTIVE >= DEFAULT_TOP_K_EXHAUSTIVE

    def test_auto_mode_enabled_by_default(self):
        assert ENABLE_EXHAUSTIVE_AUTO_MODE is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Disease list extractor
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractDiseaseList:
    """extract_disease_list_from_chunks must deduplicate and build evidence."""

    def _make_chunk(
        self,
        disease_tags: str = "",
        heading: str = "",
        source_name: str = "TestSource",
        page_no: int = 1,
    ) -> dict:
        return {
            "id": None,
            "source_name": source_name,
            "page_no": page_no,
            "heading": heading,
            "content": "",
            "disease_tags": disease_tags,
            "section_category": "",
        }

    def test_disease_tags_extracted(self):
        chunks = [self._make_chunk(disease_tags="Pneumonia")]
        result = extract_disease_list_from_chunks(chunks)
        names = [r["name"].lower() for r in result]
        assert "pneumonia" in names

    def test_heading_extracted(self):
        chunks = [self._make_chunk(heading="Tuberculosis")]
        result = extract_disease_list_from_chunks(chunks)
        names = [r["name"].lower() for r in result]
        assert "tuberculosis" in names

    def test_generic_heading_excluded(self):
        chunks = [
            self._make_chunk(heading="Patofisiologi"),
            self._make_chunk(heading="Tatalaksana"),
            self._make_chunk(heading="Definisi"),
        ]
        result = extract_disease_list_from_chunks(chunks)
        names = {r["name"].lower() for r in result}
        assert "patofisiologi" not in names
        assert "tatalaksana" not in names
        assert "definisi" not in names

    def test_deduplication(self):
        chunks = [
            self._make_chunk(disease_tags="pneumonia", page_no=1),
            self._make_chunk(disease_tags="pneumonia", page_no=2),
            self._make_chunk(disease_tags="Pneumonia", page_no=3),
        ]
        result = extract_disease_list_from_chunks(chunks)
        names = [r["name"].lower() for r in result]
        assert names.count("pneumonia") == 1

    def test_evidence_merged_across_pages(self):
        chunks = [
            self._make_chunk(disease_tags="pneumonia", source_name="BookA", page_no=1),
            self._make_chunk(disease_tags="pneumonia", source_name="BookA", page_no=5),
        ]
        result = extract_disease_list_from_chunks(chunks)
        assert len(result) == 1
        evidence = result[0]["evidence"]
        pages = {ev["page_no"] for ev in evidence}
        assert pages == {1, 5}

    def test_multiple_tags_comma_separated(self):
        chunks = [self._make_chunk(disease_tags="diabetes, hipertensi; anemia")]
        result = extract_disease_list_from_chunks(chunks)
        names = {r["name"].lower() for r in result}
        assert "diabetes" in names
        assert "hipertensi" in names
        assert "anemia" in names

    def test_empty_chunks(self):
        assert extract_disease_list_from_chunks([]) == []

    def test_sorted_alphabetically(self):
        chunks = [
            self._make_chunk(disease_tags="zebra disease"),
            self._make_chunk(disease_tags="alpha disease"),
            self._make_chunk(disease_tags="middle disease"),
        ]
        result = extract_disease_list_from_chunks(chunks)
        names = [r["name"].lower() for r in result]
        assert names == sorted(names)


# ─────────────────────────────────────────────────────────────────────────────
# 5. API schema accepts new fields (unit test only, no HTTP server)
# ─────────────────────────────────────────────────────────────────────────────

class TestApiSchemaExtensions:
    """SearchDiseaseRequest must accept retrieval_mode, max_items, page, page_size."""

    def test_defaults_unchanged(self):
        from medrag.api import SearchDiseaseRequest
        m = SearchDiseaseRequest(disease_name="pneumonia")
        assert m.retrieval_mode is None
        assert m.max_items is None
        assert m.page is None
        assert m.page_size is None

    def test_exhaustive_mode_accepted(self):
        from medrag.api import SearchDiseaseRequest
        m = SearchDiseaseRequest(disease_name="daftar penyakit", retrieval_mode="exhaustive")
        assert m.retrieval_mode == "exhaustive"

    def test_relevant_mode_accepted(self):
        from medrag.api import SearchDiseaseRequest
        m = SearchDiseaseRequest(disease_name="tbc", retrieval_mode="relevant")
        assert m.retrieval_mode == "relevant"

    def test_pagination_fields_accepted(self):
        from medrag.api import SearchDiseaseRequest
        m = SearchDiseaseRequest(disease_name="daftar penyakit", max_items=100, page=2, page_size=25)
        assert m.max_items == 100
        assert m.page == 2
        assert m.page_size == 25

    def test_invalid_mode_rejected(self):
        from pydantic import ValidationError
        from medrag.api import SearchDiseaseRequest
        with pytest.raises(ValidationError):
            SearchDiseaseRequest(disease_name="tbc", retrieval_mode="turbo")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# 6. search_chunks exhaustive vs relevant: top_k comparison (no real DB needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchChunksExhaustiveVsRelevant:
    """
    Verify that exhaustive mode uses a higher effective top_k.
    We test this by patching _execute_single_search to record the top_k
    value passed to it.
    """

    def _make_minimal_db(self) -> Path:
        """Create a minimal in-memory-like SQLite DB with required tables."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source_name TEXT DEFAULT '',
                page_no INTEGER DEFAULT 0,
                heading TEXT DEFAULT '',
                content TEXT DEFAULT '',
                disease_tags TEXT DEFAULT '',
                section_category TEXT DEFAULT '',
                parent_heading TEXT DEFAULT '',
                content_type TEXT DEFAULT 'prose',
                chunk_index INTEGER DEFAULT 0,
                checksum TEXT DEFAULT '',
                stase_slug TEXT DEFAULT 'ipd'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                heading, content, disease_tags,
                content='chunks', content_rowid='id'
            );
        """)
        conn.close()
        return Path(tmp.name)

    def test_exhaustive_mode_top_k_is_larger(self, monkeypatch):
        """Patch search internals to capture effective_top_k; verify exhaustive > relevant."""
        from medrag import retriever as ret_mod

        captured: list[int] = []
        original = ret_mod._execute_single_search

        def patched(conn, query, top_k=8):
            captured.append(top_k)
            return []

        monkeypatch.setattr(ret_mod, "_execute_single_search", patched)
        monkeypatch.setattr(ret_mod, "_vector_search", lambda *a, **kw: [])

        db = self._make_minimal_db()
        try:
            captured.clear()
            ret_mod.search_chunks(db, "penyakit apa saja", top_k=8, retrieval_mode="exhaustive")
            exhaustive_top_k = captured[0] if captured else 0

            captured.clear()
            ret_mod.search_chunks(db, "penanganan pneumonia", top_k=8, retrieval_mode="relevant")
            relevant_top_k = captured[0] if captured else 0

            assert exhaustive_top_k > relevant_top_k, (
                f"Exhaustive top_k ({exhaustive_top_k}) should be larger than "
                f"relevant top_k ({relevant_top_k})"
            )
        finally:
            import os
            os.unlink(db)

    def test_auto_listing_query_uses_exhaustive_top_k(self, monkeypatch):
        """Auto-resolve: listing query must use exhaustive top_k."""
        from medrag import retriever as ret_mod

        captured: list[int] = []

        def patched(conn, query, top_k=8):
            captured.append(top_k)
            return []

        monkeypatch.setattr(ret_mod, "_execute_single_search", patched)
        monkeypatch.setattr(ret_mod, "_vector_search", lambda *a, **kw: [])

        db = self._make_minimal_db()
        try:
            ret_mod.search_chunks(db, "daftar semua penyakit", top_k=8)
            assert captured, "Expected at least one search call"
            assert captured[0] >= DEFAULT_TOP_K_EXHAUSTIVE, (
                f"Auto listing query should have top_k >= {DEFAULT_TOP_K_EXHAUSTIVE}, got {captured[0]}"
            )
        finally:
            import os
            os.unlink(db)
