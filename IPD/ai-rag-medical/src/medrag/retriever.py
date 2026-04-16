"""
Medical RAG Retriever — v2.0 (Smart Retrieval)

Key improvements over v1:
- Medical synonym & abbreviation expansion (TBC↔Tuberkulosis, DM↔Diabetes, etc.)
- Typo tolerance for common medical misspellings
- Intent-aware hierarchical re-ranking
- Precision image filtering (no blind fallback)
- Dynamic section categorization (supports Patogenesis/Patofisiologi)
"""

import re
import sqlite3
from pathlib import Path
from typing import Any

from .config import DEFAULT_DB_PATH, DEFAULT_CHROMA_PATH

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9/-]{1,}")

# ─────────────────────────────────────────────
# Medical Vocabulary: Synonyms & Abbreviations
# ─────────────────────────────────────────────
MEDICAL_SYNONYMS: dict[str, list[str]] = {
    "tbc": ["tuberkulosis", "tuberculosis", "tb"],
    "tuberkulosis": ["tbc", "tuberculosis", "tb"],
    "tuberculosis": ["tbc", "tuberkulosis", "tb"],
    "tb": ["tbc", "tuberkulosis", "tuberculosis"],
    "dm": ["diabetes", "mellitus"],
    "diabetes": ["dm"],
    "ht": ["hipertensi"],
    "hipertensi": ["ht", "hipertensif"],
    "chf": ["gagal jantung", "heart failure"],
    "ckd": ["gagal ginjal", "chronic kidney"],
    "copd": ["ppok"],
    "ppok": ["copd"],
    "hiv": ["aids", "odha"],
    "aids": ["hiv", "odha"],
    "odha": ["hiv", "aids"],
    "pneumonia": ["pneumonitis", "radang paru"],
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
}

# Known disease keywords for image relevance gating
DISEASE_KEYWORDS: list[str] = [
    # Pulmonologi
    "tuberkulosis", "tbc", "tb", "pneumonia", "asma", "copd", "ppok",
    "bronkitis", "bronkiolitis", "bronkiektasis", "emboli", "abses",
    "efusi pleura", "pneumotoraks", "atelektasis", "fibrosis",
    "kanker paru", "mesotelioma", "sarkoidosis", "hemoptisis",
    "gagal napas", "ards", "edema paru", "cor pulmonale",
    "laringitis", "trakeitis", "epiglotitis", "croup",
    "pneumonitis hipersensitif", "pneumonitis",
    # Neonatologi/Pediatri
    "hyaline membrane disease", "hmd",
    "transient tachypnea", "respiratory distress",
    "meconium aspiration", "sepsis neonatorum",
    # Penyakit Dalam
    "diabetes", "hipertensi", "gagal jantung", "anemia",
    "hiv", "aids", "meningitis", "hepatitis",
    "sirosis", "gagal ginjal", "lupus", "rheumatoid",
    "stroke", "infark miokard", "aritmia",
    "demam tifoid", "malaria", "dengue", "leptospirosis",
]


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


def _normalize_query(query: str) -> list[str]:
    """Normalize, correct typos, and expand query tokens."""
    raw_tokens: list[str] = []
    for token in TOKEN_RE.findall(query.lower()):
        corrected = _correct_typo(token)
        if corrected not in raw_tokens:
            raw_tokens.append(corrected)

    # Expand with synonyms
    expanded = _expand_synonyms(raw_tokens[:12])
    return expanded[:20]


def _fts_query(query: str) -> str:
    tokens = _normalize_query(query)
    if not tokens:
        return ""
    return " OR ".join(f"{token}*" for token in tokens)


def _detect_intent(query: str) -> str | None:
    """Detect hierarchical intent from query to boost relevant sections."""
    return _extract_topic_intent(query)


# ─────────────────────────────────────────────
# Search Functions
# ─────────────────────────────────────────────
def _fts_search(db_path: Path, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """BM25-based FTS5 search with synonym expansion."""
    fts_query = _fts_query(query)
    if not fts_query:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.source_name,
                c.page_no,
                c.heading,
                c.content,
                c.markdown_path,
                c.section_category,
                bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


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
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Also expand the semantic query with corrected terms
        tokens = TOKEN_RE.findall(query.lower())
        corrected_query = " ".join(_correct_typo(t) for t in tokens)
        query_embedding = model.encode([corrected_query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results and results["metadatas"]:
            for meta, doc, dist in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            ):
                items.append({
                    "source_name": meta.get("source_name", ""),
                    "page_no": meta.get("page_no", 0),
                    "heading": meta.get("heading", ""),
                    "content": meta.get("content", doc[:500]),
                    "section_category": meta.get("section_category", "Ringkasan_Klinis"),
                    "markdown_path": "",
                    "vector_score": 1.0 - float(dist),
                })
        return items
    except Exception as e:
        print(f"[WARN] Vector search failed: {e}")
        return []


def _reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Combine BM25 and vector results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    data: dict[str, dict[str, Any]] = {}

    def key(item: dict[str, Any]) -> str:
        return f"{item['source_name']}|{item['page_no']}|{item['heading'][:50]}"

    for rank, item in enumerate(bm25_results):
        k_ = key(item)
        scores[k_] = scores.get(k_, 0.0) + 1.0 / (k + rank + 1)
        data[k_] = item

    for rank, item in enumerate(vector_results):
        k_ = key(item)
        scores[k_] = scores.get(k_, 0.0) + 1.0 / (k + rank + 1)
        if k_ not in data:
            data[k_] = item

    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    return [data[k_] for k_ in sorted_keys]


def _filter_by_disease_relevance(
    results: list[dict[str, Any]],
    disease_name: str | None,
    query: str,
) -> list[dict[str, Any]]:
    """Filter out chunks that don't belong to the target disease.

    This prevents cross-contamination where chunks about Abses Paru
    appear when searching for Tuberkulosis, etc.

    Strategy:
    - A chunk PASSES if it explicitly mentions the target disease
    - A chunk FAILS if its heading mentions a DIFFERENT disease
    - A chunk with a GENERIC heading (e.g. "Patofisiologi:") FAILS
      unless its content mentions the target disease
    """
    if not disease_name:
        return results

    # Build list of acceptable disease terms (including synonyms)
    acceptable = {disease_name.lower()}
    if disease_name.lower() in MEDICAL_SYNONYMS:
        for syn in MEDICAL_SYNONYMS[disease_name.lower()]:
            acceptable.add(syn.lower())

    # Also gather all disease keywords that are NOT acceptable
    other_diseases = {
        kw.lower() for kw in DISEASE_KEYWORDS
        if kw.lower() not in acceptable
    }

    filtered = []
    for item in results:
        heading_lower = item.get("heading", "").lower()
        content_lower = item.get("content", "").lower()[:400]
        source_lower = item.get("source_name", "").lower()
        combined = f"{heading_lower} {content_lower}"

        # Does this chunk mention the target disease?
        mentions_target = any(term in combined for term in acceptable)

        # Does the heading mention a DIFFERENT specific disease?
        heading_mentions_other = any(
            disease in heading_lower
            for disease in other_diseases
            if len(disease) > 2  # skip very short abbreviations
        )

        # Is the heading generic (like "Patofisiologi:", "Definisi", etc.)?
        heading_is_generic = heading_lower.rstrip(":").strip() in {
            "patofisiologi", "patogenesis", "definisi", "etiologi",
            "tatalaksana", "tata laksana", "manifestasi klinis",
            "diagnosis", "komplikasi", "prognosis", "general",
            "pemeriksaan fisik", "pemeriksaan penunjang",
            "anamnesis", "faktor risiko",
        }

        if mentions_target:
            # Explicitly mentions target → always include
            filtered.append(item)
        elif heading_mentions_other:
            # Heading explicitly about another disease → exclude
            continue
        elif heading_is_generic and not mentions_target:
            # Generic heading without target disease mention → exclude
            # (e.g. "Patofisiologi:" for Emboli Paru when searching TBC)
            continue
        else:
            # Neutral chunk → include
            filtered.append(item)

    # Safety net: if filtering removed too many, relax
    if len(filtered) < 3 and len(results) >= 3:
        # Re-add items that at least mention target in content
        for item in results:
            if item not in filtered:
                content_lower = item.get("content", "").lower()[:400]
                if any(term in content_lower for term in acceptable):
                    filtered.append(item)
            if len(filtered) >= 3:
                break
        # If still too few, return original (degraded but non-empty)
        if len(filtered) < 2:
            return results[:top_k] if 'top_k' in dir() else results[:8]

    return filtered


def search_chunks(
    db_path: Path | None,
    query: str,
    top_k: int = 8,
    chat_history: list[dict[str, Any]] | None = None,
    chroma_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Hybrid (BM25 + Vector) + Hierarchical + Disease-filtered search."""
    database = db_path or DEFAULT_DB_PATH
    chroma = chroma_path or DEFAULT_CHROMA_PATH

    # Step 1: Enrich query with multi-turn context
    enriched_query = query
    if chat_history:
        recent_context = " ".join(
            msg["content"] for msg in chat_history[-4:]
            if msg.get("role") == "user"
        )
        if recent_context:
            enriched_query = f"{recent_context} {query}"

    # Step 2: Extract disease name and topic intent
    disease_name = _extract_disease_name(enriched_query)
    intent_category = _detect_intent(query)  # Use original query for intent
    is_detail = _is_detail_request(query)

    # Step 3: BM25 search (with synonym expansion)
    fetch_k = top_k + 8 if is_detail else top_k + 4
    bm25_results = _fts_search(database, enriched_query, top_k=fetch_k)

    # Step 4: Vector search
    vector_results = _vector_search(enriched_query, top_k=fetch_k, chroma_path=chroma)

    # Step 5: Merge with RRF
    if vector_results:
        merged = _reciprocal_rank_fusion(bm25_results, vector_results)
    else:
        merged = bm25_results

    # Step 6: Filter by disease relevance (prevent cross-contamination)
    if not is_detail:
        merged = _filter_by_disease_relevance(merged, disease_name, query)

    # Step 7: Hierarchical re-ranking by intent
    if intent_category and not is_detail:
        # Boost chunks matching detected intent to top
        priority = []
        secondary = []
        others = []
        for c in merged:
            cat = c.get("section_category", "")
            heading_lower = c.get("heading", "").lower()
            content_lower = c.get("content", "").lower()[:200]

            # Check if chunk matches intent via category OR heading/content keywords
            matches_intent = (
                cat == intent_category
                or any(
                    kw in heading_lower or kw in content_lower
                    for kw in _get_intent_keywords(intent_category)
                )
            )

            if matches_intent:
                priority.append(c)
            elif disease_name and disease_name.lower() in heading_lower:
                secondary.append(c)
            else:
                others.append(c)

        merged = priority + secondary + others

    return merged[:top_k]


def _get_intent_keywords(intent_category: str) -> list[str]:
    """Get all keywords associated with an intent category."""
    keywords = []
    for kw, cat in INTENT_MAP.items():
        if cat == intent_category:
            keywords.append(kw)
    return keywords


# ─────────────────────────────────────────────
# Precision Image Retrieval
# ─────────────────────────────────────────────
def related_images(
    db_path: Path | None,
    query: str,
    evidence: list[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Find relevant images with strict disease-relevance gating and intent alignment.
    
    1. Fuzzy-Page Search: Expands evidence pages (+/- 1 page) to catch large diagrams.
    2. Intent-Driven Retrieval: Boosts images containing keywords like "alur" or "diagnosis" 
       if the user's querying about clinical pathways.
    """
    database = db_path or DEFAULT_DB_PATH
    disease_name = _extract_disease_name(query)

    # Build disease search terms (including synonyms)
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
        # Phase 1: Fuzzy Page Search (+/- 1 Page)
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
            # Find ANY image in these pages mentioning the disease
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
