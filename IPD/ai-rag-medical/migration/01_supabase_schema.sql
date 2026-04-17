-- =============================================================
-- Medical RAG — Supabase PostgreSQL Schema
-- Run this in Supabase SQL Editor ONCE before migrating data
-- =============================================================

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================
-- CORE RAG TABLES
-- =============================================================

-- Main chunks table (replaces SQLite chunks + chunks_fts)
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    page_no INTEGER NOT NULL DEFAULT 0,
    heading TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    disease_tags TEXT NOT NULL DEFAULT '',
    markdown_path TEXT NOT NULL DEFAULT '',
    checksum TEXT NOT NULL,
    section_category TEXT NOT NULL DEFAULT 'Ringkasan_Klinis',
    parent_heading TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 1,
    heading_level INTEGER NOT NULL DEFAULT 2,
    content_type TEXT NOT NULL DEFAULT 'prose',
    -- pgvector embedding column (768 dims = multilingual-e5-base)
    embedding VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Full-text search index (PostgreSQL built-in, replaces FTS5)
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN (to_tsvector('indonesian', content || ' ' || disease_tags || ' ' || heading));

-- Vector similarity index (ivfflat for large datasets, hnsw for smaller)
CREATE INDEX IF NOT EXISTS idx_chunks_vector
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source_name, page_no);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks (section_category);
CREATE INDEX IF NOT EXISTS idx_chunks_checksum ON chunks (checksum);

-- Images table
CREATE TABLE IF NOT EXISTS images (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    page_no INTEGER NOT NULL DEFAULT 0,
    alt_text TEXT DEFAULT '',
    image_ref TEXT NOT NULL DEFAULT '',
    image_abs_path TEXT NOT NULL DEFAULT '',
    -- Supabase Storage public URL (replaces local file path)
    storage_url TEXT DEFAULT '',
    heading TEXT NOT NULL DEFAULT '',
    nearby_text TEXT NOT NULL DEFAULT '',
    markdown_path TEXT NOT NULL DEFAULT '',
    checksum TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_images_source ON images (source_name, page_no);
CREATE INDEX IF NOT EXISTS idx_images_fts
    ON images USING GIN (to_tsvector('simple', heading || ' ' || alt_text || ' ' || nearby_text));

-- Knowledge graph edges
CREATE TABLE IF NOT EXISTS graph_edges (
    id BIGSERIAL PRIMARY KEY,
    source_disease TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_node TEXT NOT NULL,
    target_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    page_no INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges (lower(source_disease));

-- =============================================================
-- LIBRARY TABLES (replaces library.sqlite3)
-- =============================================================

CREATE TABLE IF NOT EXISTS stase (
    id BIGSERIAL PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS disease_catalog (
    id BIGSERIAL PRIMARY KEY,
    stase_id BIGINT NOT NULL REFERENCES stase(id) ON DELETE CASCADE,
    catalog_no INTEGER NOT NULL,
    name TEXT NOT NULL,
    competency_level TEXT,
    group_label TEXT,
    stable_key TEXT NOT NULL,
    UNIQUE (stase_id, catalog_no),
    UNIQUE (stase_id, stable_key)
);

CREATE INDEX IF NOT EXISTS idx_disease_stase ON disease_catalog (stase_id);

CREATE TABLE IF NOT EXISTS library_article (
    id BIGSERIAL PRIMARY KEY,
    catalog_id BIGINT NOT NULL UNIQUE REFERENCES disease_catalog(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'missing'
        CHECK (status IN ('missing', 'draft', 'published')),
    -- Markdown content stored directly in DB (replaces file system)
    content_markdown TEXT,
    -- JSON metadata stored as JSONB
    meta JSONB DEFAULT '{}'::jsonb,
    -- Mindmap JSON
    mindmap JSONB,
    content_hash TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================
-- HELPER FUNCTIONS
-- =============================================================

-- Function for vector similarity search (called from Cloudflare Worker)
CREATE OR REPLACE FUNCTION search_chunks_vector(
    query_embedding VECTOR(768),
    match_count INTEGER DEFAULT 10,
    filter_section TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    source_name TEXT,
    page_no INTEGER,
    heading TEXT,
    content TEXT,
    disease_tags TEXT,
    section_category TEXT,
    parent_heading TEXT,
    content_type TEXT,
    chunk_index INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF filter_section IS NOT NULL THEN
        RETURN QUERY
        SELECT
            c.id, c.source_name, c.page_no, c.heading, c.content,
            c.disease_tags, c.section_category, c.parent_heading,
            c.content_type, c.chunk_index,
            1 - (c.embedding <=> query_embedding) AS similarity
        FROM chunks c
        WHERE c.section_category = filter_section
          AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count;
    ELSE
        RETURN QUERY
        SELECT
            c.id, c.source_name, c.page_no, c.heading, c.content,
            c.disease_tags, c.section_category, c.parent_heading,
            c.content_type, c.chunk_index,
            1 - (c.embedding <=> query_embedding) AS similarity
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count;
    END IF;
END;
$$;

-- Function for full-text BM25-like search
CREATE OR REPLACE FUNCTION search_chunks_fts(
    query_text TEXT,
    match_count INTEGER DEFAULT 10,
    expanded_terms TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    source_name TEXT,
    page_no INTEGER,
    heading TEXT,
    content TEXT,
    disease_tags TEXT,
    section_category TEXT,
    parent_heading TEXT,
    content_type TEXT,
    chunk_index INTEGER,
    fts_rank FLOAT
)
LANGUAGE plpgsql
AS $$
DECLARE
    tsquery_val TSQUERY;
BEGIN
    -- Build tsquery from main query
    tsquery_val := plainto_tsquery('simple', query_text);

    IF expanded_terms IS NOT NULL AND array_length(expanded_terms, 1) > 0 THEN
        -- Add synonym expansion
        DECLARE
            term TEXT;
        BEGIN
            FOREACH term IN ARRAY expanded_terms LOOP
                tsquery_val := tsquery_val || plainto_tsquery('simple', term);
            END LOOP;
        END;
    END IF;

    RETURN QUERY
    SELECT
        c.id, c.source_name, c.page_no, c.heading, c.content,
        c.disease_tags, c.section_category, c.parent_heading,
        c.content_type, c.chunk_index,
        ts_rank(
            to_tsvector('simple', c.content || ' ' || c.disease_tags || ' ' || c.heading),
            tsquery_val
        )::FLOAT AS fts_rank
    FROM chunks c
    WHERE to_tsvector('simple', c.content || ' ' || c.disease_tags || ' ' || c.heading) @@ tsquery_val
    ORDER BY fts_rank DESC
    LIMIT match_count;
END;
$$;

-- =============================================================
-- ROW LEVEL SECURITY (RLS) — Enable for production
-- =============================================================

-- For now, allow public reads (you can tighten this later)
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE images ENABLE ROW LEVEL SECURITY;
ALTER TABLE stase ENABLE ROW LEVEL SECURITY;
ALTER TABLE disease_catalog ENABLE ROW LEVEL SECURITY;
ALTER TABLE library_article ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;

-- Public read access
CREATE POLICY "Allow public reads on chunks" ON chunks FOR SELECT USING (true);
CREATE POLICY "Allow public reads on images" ON images FOR SELECT USING (true);
CREATE POLICY "Allow public reads on stase" ON stase FOR SELECT USING (true);
CREATE POLICY "Allow public reads on disease_catalog" ON disease_catalog FOR SELECT USING (true);
CREATE POLICY "Allow public reads on library_article" ON library_article FOR SELECT USING (true);
CREATE POLICY "Allow public reads on graph_edges" ON graph_edges FOR SELECT USING (true);

-- Service role can write everything (used by migration script and CF Worker via service key)
CREATE POLICY "Service role write chunks" ON chunks FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role write images" ON images FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role write stase" ON stase FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role write disease_catalog" ON disease_catalog FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role write library_article" ON library_article FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role write graph_edges" ON graph_edges FOR ALL USING (auth.role() = 'service_role');
