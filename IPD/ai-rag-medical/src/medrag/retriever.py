"""
Medical RAG Retriever — v2.0 (Smart Retrieval)

Key improvements over v1:
- Medical synonym & abbreviation expansion (TBC↔Tuberkulosis, DM↔Diabetes, etc.)
- Typo tolerance for common medical misspellings
- Intent-aware hierarchical re-ranking
- Precision image filtering (no blind fallback)
- Dynamic section categorization (supports Patogenesis/Patofisiologi)
"""

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_DB_PATH, DEFAULT_CHROMA_PATH, EMBEDDING_MODEL, RERANKER_MODEL,
    DEFAULT_TOP_K_RELEVANT, DEFAULT_TOP_K_EXHAUSTIVE, MAX_TOP_K_EXHAUSTIVE,
    ENABLE_EXHAUSTIVE_AUTO_MODE,
)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9/-]{1,}")

# ─────────────────────────────────────────────
# Medical Vocabulary: Synonyms & Abbreviations
# ─────────────────────────────────────────────
MEDICAL_SYNONYMS: dict[str, list[str]] = {
    # Pulmo
    "tbc": ["tuberkulosis", "tb", "tuberculosis", "kp", "koch pulmonum"],
    "tuberkulosis": ["tbc", "tb", "tuberculosis"],
    "ppok": ["copd", "penyakit paru obstruktif kronis", "bronkitis kronis", "emfisema"],
    "copd": ["ppok", "penyakit paru obstruktif kronis"],
    "asma": ["asthma", "bengek", "mengi"],
    "pneumonia": ["paru-paru basah", "bronkopneumonia", "cap", "hap", "vap"],
    "ards": ["acute respiratory distress syndrome", "gagal napas akut"],
    # Cardio
    "acs": ["sindrom koroner akut", "ska", "serangan jantung", "stemi", "nstemi", "infark miokard"],
    "stemi": ["infark miokard", "acs", "ska", "serangan jantung"],
    "nstemi": ["infark miokard", "acs", "ska", "serangan jantung"],
    "hfpef": ["gagal jantung", "heart failure", "chf"],
    "hfrgef": ["gagal jantung", "heart failure", "chf"],
    "chf": ["gagal jantung", "congestive heart failure", "hfpef", "hfref"],
    "hipertensi": ["ht", "darah tinggi", "hypertension", "htn"],
    "cad": ["penyakit jantung koroner", "pjk", "coronary artery disease"],
    # Gastro/Hepa
    "gerd": ["asam lambung", "refluks", "gastroesophageal reflux disease"],
    "sirosis": ["cirrhosis", "pengerasan hati"],
    # Endo
    "dm": ["diabetes", "diabetes melitus", "kencing manis", "hiperglikemia"],
    "diabetes": ["dm", "kencing manis", "diabetes melitus"],
    # Infeksi
    "dbd": ["dengue", "dengue hemorrhagic fever", "dhf", "demam berdarah"],
    "hiv": ["aids", "odhav", "odha", "human immunodeficiency virus"],
    "aids": ["hiv", "odha"],
    "odha": ["hiv", "aids"],
    # Pediatri
    "hmd": ["hyaline membrane disease", "rds", "respiratory distress syndrome"],
    # General
    "ca": ["kanker", "karsinoma", "neoplasma"],
    "anemia": ["anemis"],
    "isk": ["infeksi saluran kemih"],
    "ispa": ["infeksi saluran pernapasan atas"],
    "patofisiologi": ["patogenesis", "fisiologi", "mekanisme"],
    "patogenesis": ["patofisiologi", "fisiologi", "mekanisme"],
    "etiologi": ["penyebab", "kausa"],
    "manifestasi": ["gejala", "keluhan", "simptom"],
    "gejala": ["manifestasi", "keluhan", "simptom"],
    "tatalaksana": ["terapi", "pengobatan", "penanganan", "manajemen"],
    "terapi": ["tatalaksana", "pengobatan", "penanganan"],
    "pengobatan": ["tatalaksana", "terapi", "penanganan"],
    "diagnosis": ["diagnosa", "diagnostik"],
    "diagnosa": ["diagnosis", "diagnostik"],
    "komplikasi": ["penyulit"],
    "prognosis": ["luaran", "outcome"],
    "definisi": ["pengertian", "arti"],
    "anamnesis": ["anamnesa", "keluhan", "riwayat"],
    "pemeriksaan fisik": ["physical examination"],
    "penunjang": ["laboratorium", "lab", "radiologi"],
    "farmakologi": ["obat", "medikasi"],
    "obat": ["farmakologi", "medikasi", "dosis"],
}

# Common typos → correct form
TYPO_CORRECTIONS: dict[str, str] = {
    "patofisilogi": "patofisiologi",
    "patofisiolgi": "patofisiologi",
    "patofsiologi": "patofisiologi",
    "patofisiogi": "patofisiologi",
    "tuberkulosisi": "tuberkulosis",
    "tuberkulosi": "tuberkulosis",
    "tuberkolusis": "tuberkulosis",
    "tuberkuosis": "tuberkulosis",
    "manitestasi": "manifestasi",
    "manfestasi": "manifestasi",
    "tatalaksna": "tatalaksana",
    "tatlaksana": "tatalaksana",
    "etiologis": "etiologi",
    "diagnoss": "diagnosis",
    "diagnossis": "diagnosis",
    "komplikassi": "komplikasi",
    "progosis": "prognosis",
    "prognsis": "prognosis",
    "anamesis": "anamnesis",
    "ananmesis": "anamnesis",
    "anamnesa": "anamnesis",
    "penunujang": "penunjang",
    "diabtes": "diabetes",
    "diabetis": "diabetes",
    "hipertensu": "hipertensi",
    "hipertensei": "hipertensi",
    "pnemonia": "pneumonia",
    "pnuemonia": "pneumonia",
    "pneumoni": "pneumonia",
}

# ─────────────────────────────────────────────
# Section Classification Rules
# ─────────────────────────────────────────────
SECTION_RULES: dict[str, list[str]] = {
    "Definisi": ["definisi", "pengertian", "arti"],
    "Etiologi dan Faktor Risiko": [
        "etiologi", "faktor risiko", "risk", "penyebab", "kausa",
    ],
    "Patogenesis dan Patofisiologi": [
        "patogenesis", "patofisiologi", "fisiologi", "mekanisme",
    ],
    "Anamnesis": ["anamnesis", "anamnesa", "riwayat", "keluhan utama"],
    "Manifestasi Klinis": [
        "manifestasi", "gejala", "klinis", "keluhan", "simptom",
    ],
    "Pemeriksaan Fisik": ["pemeriksaan fisik", "physical exam"],
    "Pemeriksaan Penunjang": [
        "penunjang", "laboratorium", "radiologi", "rontgen",
    ],
    "Diagnosis": ["diagnosis", "diagnosa", "diagnostik", "kriteria"],
    "Tatalaksana": [
        "tata laksana", "tatalaksana", "terapi", "pengobatan",
        "penanganan", "obat", "farmakologi", "dosis",
    ],
    "Komplikasi dan Prognosis": [
        "komplikasi", "prognosis", "penyulit", "luaran",
    ],
}

# Intent keywords → section_category for hierarchical boost
INTENT_MAP: dict[str, str] = {
    # Tatalaksana
    "obat": "Tatalaksana",
    "terapi": "Tatalaksana",
    "pengobatan": "Tatalaksana",
    "dosis": "Tatalaksana",
    "tatalaksana": "Tatalaksana",
    "penanganan": "Tatalaksana",
    "regimen": "Tatalaksana",
    "farmakologi": "Tatalaksana",
    # Manifestasi Klinis
    "gejala": "Manifestasi_Klinis",
    "keluhan": "Manifestasi_Klinis",
    "manifestasi": "Manifestasi_Klinis",
    "simptom": "Manifestasi_Klinis",
    # Diagnosis
    "diagnosis": "Diagnosis",
    "diagnosa": "Diagnosis",
    "pemeriksaan": "Diagnosis",
    "penunjang": "Diagnosis",
    "diagnostik": "Diagnosis",
    "kriteria": "Diagnosis",
    # Etiologi
    "etiologi": "Etiologi",
    "penyebab": "Etiologi",
    "kausa": "Etiologi",
    "faktor risiko": "Etiologi",
    # Patogenesis / Patofisiologi
    "patofisiologi": "Patogenesis",
    "patogenesis": "Patogenesis",
    "fisiologi": "Patogenesis",
    "mekanisme": "Patogenesis",
    "patofisilogi": "Patogenesis",  # common typo
    # Komplikasi & Prognosis
    "komplikasi": "Komplikasi",
    "prognosis": "Prognosis",
    # Definisi
    "definisi": "Definisi",
    "pengertian": "Definisi",
    # Anamnesis
    "anamnesis": "Anamnesis",
    "anamnesa": "Anamnesis",
    # Pemeriksaan fisik
    "pemeriksaan fisik": "Pemeriksaan_Fisik",
    "pemeriksaan fisik": "Pemeriksaan_Fisik",
}

# Known disease keywords for image relevance gating
DISEASE_KEYWORDS: list[str] = [
    # Pulmonologi
    "tuberkulosis", "tbc", "tb", "pneumonia", "asma", "copd", "ppok",
    "bronkitis", "bronkiolitis", "bronkiektasis", "emboli", "abses paru",
    "efusi pleura", "pneumotoraks", "atelektasis", "fibrosis",
    "kanker paru", "mesotelioma", "sarkoidosis", "hemoptisis",
    "gagal napas", "ards", "edema paru", "cor pulmonale",
    "laringitis", "trakeitis", "epiglotitis", "croup",
    "pneumonitis hipersensitif", "pneumonitis",
    "aspirasi benda asing", "pneumonia aspirasi",
    "abses paru", "pleuritis", "hidropneumotoraks",
    # Neonatologi/Pediatri
    "hyaline membrane disease", "hmd", "rds",
    "transient tachypnea", "respiratory distress",
    "meconium aspiration", "sepsis neonatorum",
    # Penyakit Dalam (Cardio, Gastro, Endo, dll)
    "diabetes", "hipertensi", "gagal jantung", "chf", "acs", "stemi", "nstemi",
    "anemia", "hiv", "aids", "meningitis", "hepatitis",
    "sirosis", "gerd", "dispepsia", "gagal ginjal", "ckd", "aki",
    "lupus", "rheumatoid", "stroke", "infark miokard", "aritmia",
    "demam tifoid", "malaria", "dengue", "dbd", "leptospirosis",
    "sindrom koroner akut", "ska", "penyakit jantung koroner", "pjk",
]


# ─────────────────────────────────────────────
# Enumerative / List Intent Detection
# ─────────────────────────────────────────────
LIST_INTENT_KEYWORDS: list[str] = [
    "daftar penyakit", "list penyakit", "semua penyakit",
    "penyakit apa saja", "apa saja penyakit", "sebutkan penyakit",
    "penyakit yang ada", "semua topik", "topik apa saja",
    "daftar topik", "apa yang tersedia", "berikan daftar",
    "tampilkan semua", "list semua", "semua materi",
    "materi apa saja", "semua sumber", "list sumber",
    # Additional keywords (parity with TypeScript worker)
    "daftar semua", "semua diagnosis", "katalog penyakit",
    "all diseases", "list all", "list diseases",
    "seluruh penyakit",
]

# Headings to filter out in listing mode
_HEADING_NOISE: frozenset[str] = frozenset([
    "definisi", "etiologi", "patofisiologi", "patogenesis",
    "manifestasi klinis", "diagnosis", "tatalaksana", "tata laksana",
    "komplikasi", "prognosis", "pemeriksaan fisik", "pemeriksaan penunjang",
    "anamnesis", "faktor risiko", "diagnosis banding", "ringkasan",
    "daftar isi", "pendahuluan", "referensi", "daftar pustaka",
    "image from page folder", "general",
])


def detect_list_intent(query: str) -> bool:
    """Deteksi apakah query ingin enumerasi daftar penyakit/topik."""
    q = query.lower()
    return any(kw in q for kw in LIST_INTENT_KEYWORDS)


<<<<<<< HEAD
# Alias used internally and exported for tests
_is_listing_intent = detect_list_intent


def _resolve_retrieval_mode(
    query: str,
    requested_mode: str | None = None,
) -> str:
    """
    Resolve retrieval mode to either 'relevant' or 'exhaustive'.

    Priority:
      1. Explicit ``requested_mode`` from the caller (API field).
      2. Auto-detection via listing intent (when ENABLE_EXHAUSTIVE_AUTO_MODE is True).
      3. Default: 'relevant'.
    """
    if requested_mode in ("relevant", "exhaustive"):
        return requested_mode
    if ENABLE_EXHAUSTIVE_AUTO_MODE and _is_listing_intent(query):
        return "exhaustive"
    return "relevant"
=======
# Heading noise words that are not disease names
_HEADING_NOISE: set[str] = {
    "general", "image from page folder", "unknown", "introduction",
    "pendahuluan", "daftar isi", "referensi", "bibliography", "lampiran",
}
>>>>>>> 8f35e3e (daftar)


def get_topics_from_db(
    db_path: "Path | None",
    stase_slug: str | None = None,
    source_filter: str | None = None,
) -> dict:
    """
    Enumerasi semua heading/topik dari DB, dikelompokkan per sumber.
    Tidak menggunakan BM25/vector — langsung baca metadata chunks.
    Mengembalikan SELURUH daftar tanpa pemotongan; tidak memanggil LLM.
    """
    database = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        params: list = []
        noise_placeholders = ",".join("?" for _ in _HEADING_NOISE)
        where = f"WHERE lower(heading) NOT IN ({noise_placeholders})"
        params.extend(_HEADING_NOISE)
<<<<<<< HEAD
        
=======
>>>>>>> 8f35e3e (daftar)
        if stase_slug:
            where += " AND stase_slug = ?"
            params.append(stase_slug)
        if source_filter:
            where += " AND source_name LIKE ?"
            params.append(f"%{source_filter}%")

        rows = conn.execute(
            f"""
            SELECT DISTINCT source_name, heading, section_category, stase_slug
            FROM chunks
            {where}
            ORDER BY source_name, heading
            """,
            params,
        ).fetchall()

        grouped: dict[str, dict] = {}
        for r in rows:
            h = r["heading"].strip()
<<<<<<< HEAD
            # Filter noise: ignore very short headings, single digits, or pure symbols
=======
            # Filer noise: ignore very short headings, single digits, or pure symbols
>>>>>>> 8f35e3e (daftar)
            if len(h) <= 2 or h.isdigit() or all(not c.isalnum() for c in h):
                continue
                
            src = r["source_name"]
            if src not in grouped:
                grouped[src] = {
                    "source_name": src,
                    "stase_slug": r["stase_slug"],
                    "topics": [],
                }
            # Dedup heading per source
            existing_headings = {t["heading"] for t in grouped[src]["topics"]}
            if h not in existing_headings:
                grouped[src]["topics"].append({
                    "heading": h,
                    "section_category": r["section_category"],
                })

        sources = list(grouped.values())
        for s in sources:
            s["topic_count"] = len(s["topics"])

        return {
            "sources": sources,
            "total_topics": sum(s["topic_count"] for s in sources),
            "source_count": len(sources),
        }
    finally:
        conn.close()


def get_disease_names_from_db(
    db_path: "Path | None",
    stase_slug: str | None = None,
) -> dict:
    """
    Enumerasi ringkas: hanya nama sumber unik (= nama penyakit/buku),
    tanpa memuat semua heading. Lebih ringan untuk tampilan daftar singkat.
    Mengembalikan SELURUH daftar tanpa pemotongan; tidak memanggil LLM.
    """
    database = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        params: list = []
        where = "WHERE 1=1"
        if stase_slug:
            where += " AND stase_slug = ?"
            params.append(stase_slug)

        rows = conn.execute(
            f"""
            SELECT source_name, stase_slug, COUNT(DISTINCT heading) as heading_count
            FROM chunks
            {where}
            GROUP BY source_name, stase_slug
            ORDER BY source_name
            """,
            params,
        ).fetchall()

        diseases = [
            {
                "source_name": r["source_name"],
                "stase_slug": r["stase_slug"],
                "heading_count": r["heading_count"],
            }
            for r in rows
        ]

        return {
            "diseases": diseases,
            "total": len(diseases),
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Query Processing Pipeline
# ─────────────────────────────────────────────
def _correct_typo(token: str) -> str:
    """Fix common medical typos."""
    return TYPO_CORRECTIONS.get(token.lower(), token)


def _expand_synonyms(tokens: list[str]) -> list[str]:
    """Expand tokens with medical synonyms/abbreviations."""
    expanded = list(tokens)
    for token in tokens:
        lower = token.lower()
        if lower in MEDICAL_SYNONYMS:
            for syn in MEDICAL_SYNONYMS[lower]:
                if syn.lower() not in [t.lower() for t in expanded]:
                    expanded.append(syn)
    return expanded


def _extract_disease_name(query: str) -> str | None:
    """Extract the primary disease name from a query for relevance gating."""
    q_lower = query.lower()
    for disease in DISEASE_KEYWORDS:
        if disease in q_lower:
            return disease
    # Also check medical synonyms to find canonical names
    tokens = TOKEN_RE.findall(q_lower)
    for token in tokens:
        corrected = _correct_typo(token)
        if corrected in MEDICAL_SYNONYMS:
            for syn in MEDICAL_SYNONYMS[corrected]:
                if syn in DISEASE_KEYWORDS:
                    return syn
    return None


def _extract_topic_intent(query: str) -> str | None:
    """Extract the specific clinical topic the user is asking about."""
    q_lower = query.lower()
    # Check multi-word intents first
    for keyword in sorted(INTENT_MAP.keys(), key=len, reverse=True):
        if keyword in q_lower:
            return INTENT_MAP[keyword]
    # Check after typo correction
    tokens = TOKEN_RE.findall(q_lower)
    for token in tokens:
        corrected = _correct_typo(token)
        if corrected in INTENT_MAP:
            return INTENT_MAP[corrected]
    return None


def _is_detail_request(query: str) -> bool:
    """Detect if user wants a comprehensive/detail response."""
    detail_keywords = [
        "detail", "lengkap", "jelaskan", "jelasin", "menjelaskan",
        "komprehensif", "selengkap", "keseluruhan", "semua aspek",
        "secara detail", "secara lengkap", "apa itu", "jelaskan tentang",
    ]
    q_lower = query.lower()
    return any(kw in q_lower for kw in detail_keywords)


def _filter_by_disease_relevance(
    results: list[dict[str, Any]],
    disease_name: str | None,
) -> list[dict[str, Any]]:
    """
    Balanced disease relevance filter (v3).
    - KEEPS chunks that explicitly mention the target disease in heading OR content
    - KEEPS chunks with generic clinical headings (Patofisiologi, Tatalaksana, dll)
    - REMOVES chunks whose HEADING explicitly names a DIFFERENT specific disease
      AND whose content does NOT mention the target disease at all
    """
    if not disease_name:
        return results

    acceptable = {disease_name.lower()}
    if disease_name.lower() in MEDICAL_SYNONYMS:
        for syn in MEDICAL_SYNONYMS[disease_name.lower()]:
            acceptable.add(syn.lower())

    # Hanya filter penyakit lain yang panjang nama-nya (>4 char) untuk menghindari false positive
    other_diseases = {
        kw.lower() for kw in DISEASE_KEYWORDS
        if kw.lower() not in acceptable and len(kw) > 4
    }

    # Generic clinical headings that are safe to keep regardless
    generic_headings = {
        "patofisiologi", "patogenesis", "definisi", "etiologi",
        "tatalaksana", "tata laksana", "manifestasi klinis",
        "diagnosis", "komplikasi", "prognosis", "general",
        "pemeriksaan fisik", "pemeriksaan penunjang",
        "anamnesis", "faktor risiko", "alur diagnosis",
        "klasifikasi", "epidemiologi", "ringkasan klinis",
    }

    filtered = []
    for item in results:
        heading_lower = item.get("heading", "").lower().rstrip(":")
        content_lower = item.get("content", "").lower()[:600]

        mentions_target = any(term in heading_lower or term in content_lower for term in acceptable)
        is_generic_heading = any(g in heading_lower for g in generic_headings)
        heading_mentions_other = any(disease in heading_lower for disease in other_diseases)

        if mentions_target:
            # Target disease is explicitly present → always include
            filtered.append(item)
        elif is_generic_heading and not heading_mentions_other:
            # Generic clinical heading with no explicit foreign disease → include
            filtered.append(item)
        elif heading_mentions_other and not mentions_target:
            # Heading is specifically about another disease → exclude
            continue
        else:
            # Neutral chunk → include
            filtered.append(item)

    # Safety fallback: jika filter terlalu agresif, kembalikan semua
    return filtered if len(filtered) >= 2 else results


def search_chunks(
    db_path: Path | None,
    query: str,
    top_k: int = 8,
    chat_history: list[dict[str, Any]] | None = None,
    chroma_path: Path | None = None,
    retrieval_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid FTS + Semantic Search with Ph.D.-Level Multi-Query Decomposition.

    Args:
        db_path: Path to the SQLite database.
        query: User query string.
        top_k: Base number of results for relevant mode.
        chat_history: Optional prior conversation turns for query enrichment.
        chroma_path: Optional path to ChromaDB vector store.
        retrieval_mode: One of ``'relevant'`` (default focused QA) or
            ``'exhaustive'`` (listing/catalog mode).  When ``None``, the mode
            is resolved automatically via :func:`_resolve_retrieval_mode`.
    """
    database = db_path or DEFAULT_DB_PATH

    mode = _resolve_retrieval_mode(query, retrieval_mode)
    is_exhaustive = mode == "exhaustive"

    enriched_query = _enrich_query_from_history(query, chat_history)

    detected_disease = _extract_disease_name(enriched_query)
    is_detail = _is_detail_request(enriched_query)
    intent = _extract_topic_intent(enriched_query)

    # Adaptive top_k — exhaustive mode overrides all other scaling
    if is_exhaustive:
        effective_top_k = min(
            max(top_k, DEFAULT_TOP_K_EXHAUSTIVE),
            MAX_TOP_K_EXHAUSTIVE,
        )
    elif is_detail:
        effective_top_k = max(top_k, DEFAULT_TOP_K_RELEVANT * 2)
    elif not detected_disease and not intent:
        # Broad/generic query: pakai minimum 12 untuk coverage lebih baik
        effective_top_k = max(top_k, 12)
    else:
        effective_top_k = top_k

    # In exhaustive mode, run a single broad query to maximise coverage.
    # In focused/detail modes, decompose into clinical sub-queries.
    if is_exhaustive:
        queries_to_run = [enriched_query]
    elif is_detail or (detected_disease and not intent):
        base = detected_disease if detected_disease else enriched_query
        queries_to_run = [
            f"{base} definisi etiologi patogenesis",
            f"{base} anamnesis gejala klinis pemeriksaan fisik",
            f"{base} diagnosis tatalaksana dosis obat algoritma",
            f"{base} komplikasi prognosis",
        ]
    else:
        queries_to_run = [enriched_query]

    all_results: list[dict[str, Any]] = []
    seen_ids: set = set()

    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        bm25_all: list[dict[str, Any]] = []
        vector_all: list[dict[str, Any]] = []

        for sub_query in queries_to_run:
            sub_results = _execute_single_search(conn, sub_query, top_k=effective_top_k)
            bm25_all.extend(sub_results)

            vec_res = _vector_search(sub_query, top_k=effective_top_k, chroma_path=chroma_path)
            if vec_res:
                vector_all.extend(vec_res)

        # Sparse-Dense Hybrid Fusion (RRF)
        if vector_all:
            merged_candidates = _reciprocal_rank_fusion(bm25_all, vector_all)
        else:
            merged_candidates = bm25_all

        for row in merged_candidates:
            cid = row.get("id") or row.get("checksum") or row.get("content", "")[:50]
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_results.append(row)

        # Cross-encoder reranking for precision
        reranked_results = _rerank_cross_encoder(
            enriched_query, all_results, top_k=effective_top_k * len(queries_to_run)
        )

        # Disease relevance gating — skip in exhaustive mode to preserve coverage
        if is_exhaustive:
            filtered_results = reranked_results
        else:
            filtered_results = _filter_by_disease_relevance(reranked_results, detected_disease)

        # Jaccard dynamic pruning — use a looser threshold in exhaustive mode
        prune_threshold = 0.85 if is_exhaustive else 0.6
        pruned_results = _prune_redundant_chunks(filtered_results, similarity_threshold=prune_threshold)

        # Hierarchical sort by intent
        sorted_results = _hierarchical_sort(
            pruned_results, intent_category=intent, top_k=effective_top_k * len(queries_to_run)
        )

        # Context window expansion: fetch sibling chunks
        if not is_exhaustive:
            expanded_results = _expand_context(conn, sorted_results, expand_siblings=1)
            # Second disease filter pass: after expand_context, some siblings from OTHER diseases may have been pulled in
            expanded_results = _filter_by_disease_relevance(expanded_results, detected_disease)
            # Final dedup after expansion
            final = _prune_redundant_chunks(expanded_results, similarity_threshold=0.7)
        else:
            final = sorted_results

        return final[:max(effective_top_k, 20)]
    finally:
        conn.close()

def _execute_single_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Executes FTS5 and fuzzy text search for a single query string."""
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        return []

    corrected = [_correct_typo(t) for t in tokens if len(t) > 2]
    expanded = _expand_synonyms(corrected)

    search_terms = " OR ".join(f'"{term}"*' for term in expanded)

    # 1. Exact/Prefix matches using FTS5
    fts_rows = conn.execute(
        f"""
        SELECT c.*, bm25(chunks_fts, 5.0, 3.0, 1.0, 2.0) as score
        FROM chunks_fts f
        JOIN chunks c ON f.rowid = c.id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (search_terms, top_k * 2),
    ).fetchall()

    # 2. Fuzzy fallback (LIKE) if FTS is sparse
    fuzzy_rows = []
    if len(fts_rows) < top_k:
        like_clauses = " OR ".join("(lower(content) LIKE ? OR lower(heading) LIKE ? OR lower(disease_tags) LIKE ?)" for _ in expanded)
        params: list[Any] = []
        for term in expanded:
            params.extend([f"%{term.lower()}%", f"%{term.lower()}%", f"%{term.lower()}%"])
        fuzzy_rows = conn.execute(
            f"""
            SELECT *, 100.0 as score
            FROM chunks
            WHERE {like_clauses}
            LIMIT ?
            """,
            params,
        ).fetchall()

    seen = set()
    combined: list[dict[str, Any]] = []

    for row in fts_rows + fuzzy_rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            combined.append(dict(row))

    return combined

def _hierarchical_sort(results: list[dict[str, Any]], intent_category: str | None, top_k: int) -> list[dict[str, Any]]:
    """Sorts results based on intent category."""
    if not intent_category:
        return results[:top_k]
    
    priority = []
    others = []
    for r in results:
        if r.get("section_category") == intent_category:
            priority.append(r)
        else:
            others.append(r)
    return (priority + others)[:top_k]

# ─────────────────────────────────────────────
# Apex RAG Advanced Methods
# ─────────────────────────────────────────────

def _vector_search(query: str, top_k: int = 10, chroma_path: Path | None = None) -> list[dict[str, Any]]:
    """Semantic vector search using ChromaDB."""
    chroma = chroma_path or DEFAULT_CHROMA_PATH
    if not chroma.exists():
        return []

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return []

    try:
        client = chromadb.PersistentClient(path=str(chroma))
        collection = client.get_collection("medrag_chunks")
        model = SentenceTransformer(EMBEDDING_MODEL)

        query_embedding = model.encode([f"query: {query}"]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results and results["metadatas"]:
            for meta, doc, dist in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
                items.append({
                    "source_name": meta.get("source_name", ""),
                    "page_no": meta.get("page_no", 0),
                    "heading": meta.get("heading", ""),
                    "content": meta.get("content", "") or doc,
                    "section_category": meta.get("section_category", "Ringkasan_Klinis"),
                    "markdown_path": "",
                    "parent_heading": meta.get("parent_heading", ""),
                    "content_type": meta.get("content_type", "prose"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "vector_score": 1.0 - float(dist),
                })
        return items
    except Exception as e:
        return []

def _reciprocal_rank_fusion(bm25_results: list[dict[str, Any]], vector_results: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    """Combine BM25 and vector results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    data: dict[str, dict[str, Any]] = {}

    def get_key(item: dict[str, Any]) -> str:
        content_hash = hashlib.md5(item.get("content", "")[:200].encode()).hexdigest()[:8]
        return f"{item.get('source_name','')}|{item.get('page_no','')}|{item.get('heading','')}|{content_hash}"

    for rank, item in enumerate(bm25_results):
        key = get_key(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        data[key] = item

    for rank, item in enumerate(vector_results):
        key = get_key(item)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in data:
            data[key] = item

    sorted_keys = sorted(scores, key=lambda key_: scores[key_], reverse=True)
    return [data[k_] for k_ in sorted_keys]

def _jaccard_similarity(s1: str, s2: str) -> float:
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def _prune_redundant_chunks(results: list[dict[str, Any]], similarity_threshold: float = 0.6) -> list[dict[str, Any]]:
    """Prune chunks that are highly overlapping via Jaccard text similarity to save LLM context window."""
    pruned = []
    for item in results:
        content = item.get("content", "").lower()
        is_redundant = False
        for saved in pruned:
            if _jaccard_similarity(content, saved.get("content", "").lower()) > similarity_threshold:
                is_redundant = True
                break
        if not is_redundant:
            pruned.append(item)
    return pruned


_reranker = None


def _rerank_cross_encoder(query: str, candidates: list[dict[str, Any]], top_k: int = 12) -> list[dict[str, Any]]:
    """Rerank candidates using a cross-encoder model for higher precision."""
    if not candidates:
        return candidates

    global _reranker
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return candidates[:top_k]

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)

    pairs = [(query, c.get("content", "")) for c in candidates]
    scores = _reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _s in ranked[:top_k]]


def _expand_context(conn: sqlite3.Connection, retrieved_chunks: list[dict[str, Any]], expand_siblings: int = 1) -> list[dict[str, Any]]:
    """Expand each retrieved chunk with its sibling chunks from the same section."""
    if not retrieved_chunks:
        return retrieved_chunks

    expanded = []
    seen_ids: set[int] = set()

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("id")
        if not chunk_id:
            expanded.append(chunk)
            continue

        if chunk_id in seen_ids:
            continue

        rows = conn.execute(
            """
            SELECT * FROM chunks
            WHERE source_name = ? AND heading = ?
            AND id BETWEEN ? AND ?
            ORDER BY id
            """,
            (
                chunk["source_name"],
                chunk["heading"],
                chunk_id - expand_siblings,
                chunk_id + expand_siblings,
            ),
        ).fetchall()

        if len(rows) > 1:
            merged_content = "\n\n".join(dict(r)["content"] for r in rows)
            merged_chunk = dict(chunk)
            merged_chunk["content"] = merged_content
            merged_chunk["expanded"] = True
            for r in rows:
                seen_ids.add(r["id"])
            expanded.append(merged_chunk)
        else:
            seen_ids.add(chunk_id)
            expanded.append(chunk)

    return expanded


def _enrich_query_from_history(query: str, chat_history: list[dict[str, Any]] | None) -> str:
    """Enrich query with disease context from chat history for follow-up questions."""
    if not chat_history:
        return query

    recent_context = " ".join(
        turn["content"] for turn in chat_history[-4:]
        if turn.get("role") == "user"
    )

    current_disease = _extract_disease_name(query)
    if not current_disease:
        history_disease = _extract_disease_name(recent_context)
        if history_disease:
            return f"{history_disease} {query}"

    return query


# ─────────────────────────────────────────────
# Precision Image Retrieval
# ─────────────────────────────────────────────
def related_images(
    db_path: Path | None,
    query: str,
    evidence: list[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Find relevant images with strict disease-relevance gating and intent alignment."""
    database = db_path or DEFAULT_DB_PATH
    disease_name = _extract_disease_name(query)

    search_terms: list[str] = []
    if disease_name:
        search_terms.append(disease_name)
        if disease_name in MEDICAL_SYNONYMS:
            search_terms.extend(MEDICAL_SYNONYMS[disease_name])
    else:
        tokens = TOKEN_RE.findall(query.lower())
        search_terms = [_correct_typo(t) for t in tokens if len(t) > 2]

    if not search_terms:
        return []
        
    intent = _extract_topic_intent(query)
    intent_keywords = []
    if intent in ("Diagnosis", "Tatalaksana") or any(k in query.lower() for k in ["penanganan", "alur", "skema"]):
        intent_keywords = ["alur", "diagnosis", "tatalaksana", "algoritma", "skema", "bagan"]

    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        where_pairs = set()
        for item in evidence:
            src = item["source_name"]
            pn = item["page_no"]
            where_pairs.add((src, pn))
            where_pairs.add((src, pn - 1))
            where_pairs.add((src, pn + 1))
            
        phase1_results = []
        if where_pairs:
            page_clauses = " OR ".join("(source_name = ? AND page_no = ?)" for _ in where_pairs)
            disease_clauses = " OR ".join(
                "(lower(heading) LIKE ? OR lower(nearby_text) LIKE ? OR lower(alt_text) LIKE ?)"
                for _ in search_terms
            )

            params: list[Any] = []
            for pair in where_pairs:
                params.extend(pair)
            for term in search_terms:
                params.extend([f"%{term.lower()}%", f"%{term.lower()}%", f"%{term.lower()}%"])

            rows = conn.execute(
                f"""
                SELECT source_name, page_no, heading, alt_text, nearby_text,
                       image_ref, image_abs_path, markdown_path
                FROM images
                WHERE ({page_clauses})
                  AND ({disease_clauses})
                """,
                params,
            ).fetchall()
            phase1_results = [dict(row) for row in rows]

        # Phase 2: Global search
        phase2_results = []
        if not phase1_results:
            disease_clauses = " OR ".join(
                "(lower(heading) LIKE ? OR lower(nearby_text) LIKE ? OR lower(alt_text) LIKE ?)"
                for _ in search_terms
            )
            params2: list[Any] = []
            for term in search_terms:
                params2.extend([f"%{term.lower()}%", f"%{term.lower()}%", f"%{term.lower()}%"])

            rows = conn.execute(
                f"""
                SELECT source_name, page_no, heading, alt_text, nearby_text,
                       image_ref, image_abs_path, markdown_path
                FROM images
                WHERE ({disease_clauses})
                """,
                params2,
            ).fetchall()
            phase2_results = [dict(row) for row in rows]

        all_results = phase1_results + phase2_results
        
        if not all_results:
            return []
            
        # Scoring based on intent keywords
        scored_results = []
        unique_paths = set()
        for item in all_results:
            if item["image_abs_path"] in unique_paths:
                continue
            unique_paths.add(item["image_abs_path"])
            
            score = 1.0 # Base score
            combined_text = f"{item['heading']} {item['alt_text']} {item['nearby_text']}".lower()
            for kw in intent_keywords:
                if kw in combined_text:
                    score += 2.0  # +200% boost for intent matching!
            scored_results.append((score, item))
            
        scored_results.sort(key=lambda x: x[0], reverse=True)
        final_list = [item for score, item in scored_results[:limit]]
        return final_list

    finally:
        conn.close()


# ─────────────────────────────────────────────
# Knowledge Graph
# ─────────────────────────────────────────────
def get_knowledge_graph(db_path: Path | None, disease_name: str, max_nodes: int = 40) -> dict[str, Any]:
    """Build knowledge graph data for a disease from the graph_edges table."""
    database = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'"
        ).fetchone()

        if not table_exists:
            return {"nodes": [], "edges": [], "disease": disease_name}

        # Expand search with synonyms
        search_terms = [disease_name.lower()]
        if disease_name.lower() in MEDICAL_SYNONYMS:
            search_terms.extend(MEDICAL_SYNONYMS[disease_name.lower()])

        all_rows = []
        for term in search_terms:
            query_lower = f"%{term}%"
            rows = conn.execute(
                """
                SELECT DISTINCT source_disease, relation, target_node, target_type, source_name
                FROM graph_edges
                WHERE lower(source_disease) LIKE ?
                   OR lower(target_node) LIKE ?
                LIMIT ?
                """,
                (query_lower, query_lower, max_nodes * 2),
            ).fetchall()
            all_rows.extend(rows)
    finally:
        conn.close()

    if not all_rows:
        return {"nodes": [], "edges": [], "disease": disease_name}

    nodes_map: dict[str, dict[str, Any]] = {}
    edges_list: list[dict[str, Any]] = []

    group_map = {
        "Definisi": 1,
        "Etiologi": 2,
        "Patogenesis": 3,
        "Manifestasi_Klinis": 4,
        "Diagnosis": 5,
        "Tatalaksana": 6,
        "Komplikasi": 7,
        "Prognosis": 8,
        "Ringkasan_Klinis": 9,
        "disease": 0,
        "concept": 10,
    }

    for row in all_rows:
        src_id = row["source_disease"]
        tgt_id = row["target_node"]

        if src_id not in nodes_map:
            nodes_map[src_id] = {
                "id": src_id,
                "label": src_id.replace("_", " "),
                "type": "disease",
                "group": group_map.get("disease", 0),
                "val": 12,
            }

        if tgt_id not in nodes_map:
            nodes_map[tgt_id] = {
                "id": tgt_id,
                "label": tgt_id.replace("_", " "),
                "type": row["target_type"],
                "group": group_map.get(row["relation"], 10),
                "val": 6,
            }

        edges_list.append({
            "source": src_id,
            "target": tgt_id,
            "relation": row["relation"].replace("_", " "),
        })

    nodes_list = list(nodes_map.values())[:max_nodes]

    return {
        "nodes": nodes_list,
        "edges": edges_list[:max_nodes * 2],
        "disease": disease_name,
    }


# ─────────────────────────────────────────────
# Disease List Extractor (for exhaustive mode)
# ─────────────────────────────────────────────

def extract_disease_list_from_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a deduplicated list of disease/topic names from retrieved chunks.

    Each entry contains the canonical name plus evidence metadata.
    This is used in exhaustive/listing mode to produce a structured catalog
    rather than a narrative QA answer.
    """
    # Generic clinical section headings that are NOT disease names
    _GENERIC_HEADINGS: frozenset[str] = frozenset([
        "patofisiologi", "patogenesis", "definisi", "etiologi", "tatalaksana",
        "tata laksana", "manifestasi klinis", "manifestasi", "diagnosis",
        "komplikasi", "prognosis", "general", "pemeriksaan fisik",
        "pemeriksaan penunjang", "anamnesis", "faktor risiko", "etiologi dan faktor risiko",
        "komplikasi dan prognosis", "ringkasan klinis", "penunjang", "farmakologi",
        "image from page folder",
    ])

    seen: dict[str, dict[str, Any]] = {}  # normalized_name → entry

    for chunk in chunks:
        evidence_item = {
            "source_name": chunk.get("source_name", ""),
            "page_no": chunk.get("page_no", 0),
            "heading": chunk.get("heading", ""),
        }

        candidates: list[str] = []

        # 1. disease_tags (comma/semicolon-separated)
        tags_raw: str = chunk.get("disease_tags", "") or ""
        if tags_raw.strip():
            for tag in re.split(r"[,;|]", tags_raw):
                t = tag.strip()
                if t:
                    candidates.append(t)

        # 2. heading — skip generic clinical section names
        heading: str = (chunk.get("heading", "") or "").strip()
        if heading and heading.lower().rstrip(":") not in _GENERIC_HEADINGS:
            candidates.append(heading)

        # 3. source_name as last-resort label
        src_name: str = (chunk.get("source_name", "") or "").strip()
        if src_name and not candidates:
            candidates.append(src_name)

        for cand in candidates:
            norm = cand.lower().strip()
            if not norm or len(norm) < 3:
                continue
            if norm in seen:
                # Merge evidence, keep unique source+page combos
                existing_ev = seen[norm]["evidence"]
                ev_key = f"{evidence_item['source_name']}:{evidence_item['page_no']}"
                existing_keys = {
                    f"{e['source_name']}:{e['page_no']}" for e in existing_ev
                }
                if ev_key not in existing_keys:
                    existing_ev.append(evidence_item)
            else:
                seen[norm] = {
                    "name": cand,
                    "evidence": [evidence_item],
                }

    # Sort alphabetically by normalized name
    return [entry for _norm, entry in sorted(seen.items())]


# ─────────────────────────────────────────────
# Smart Section Picker for Rule-based Synthesis
# ─────────────────────────────────────────────
def _pick_section(heading: str, content: str) -> str:
    merged = f"{heading} {content[:250]}".lower()
    for section, keywords in SECTION_RULES.items():
        if any(keyword in merged for keyword in keywords):
            return section
    return "Ringkasan Klinis"


def synthesize_answer(disease_name: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule-based synthesis fallback (when AI is unavailable)."""
    output_sections: dict[str, list[str]] = {}
    citations: list[str] = []

    for item in evidence:
        section = _pick_section(item["heading"], item["content"])
        output_sections.setdefault(section, [])
        excerpt = item["content"][:450].strip()
        if excerpt and excerpt not in output_sections[section]:
            output_sections[section].append(excerpt)
        citation = f"{item['source_name']} p.{item['page_no']}"
        if citation not in citations:
            citations.append(citation)

    ordered_sections = []
    for key in [
        "Definisi",
        "Etiologi dan Faktor Risiko",
        "Patogenesis dan Patofisiologi",
        "Anamnesis",
        "Manifestasi Klinis",
        "Pemeriksaan Fisik",
        "Pemeriksaan Penunjang",
        "Diagnosis",
        "Tatalaksana",
        "Komplikasi dan Prognosis",
        "Ringkasan Klinis",
    ]:
        if key in output_sections:
            ordered_sections.append(
                {
                    "title": key,
                    "points": output_sections[key][:3],
                }
            )

    return {
        "disease": disease_name,
        "sections": ordered_sections,
        "citations": citations,
        "grounded": True,
    }
