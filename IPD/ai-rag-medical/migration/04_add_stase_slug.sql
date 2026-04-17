-- =============================================================
-- Medical RAG — Migration 04: Add stase_slug to RAG tables
-- Run this in Supabase SQL Editor ONCE
-- =============================================================

-- Step 1: Add stase_slug column to RAG tables (default 'ipd' for existing data)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS stase_slug TEXT NOT NULL DEFAULT 'ipd';
ALTER TABLE images ADD COLUMN IF NOT EXISTS stase_slug TEXT NOT NULL DEFAULT 'ipd';
ALTER TABLE graph_edges ADD COLUMN IF NOT EXISTS stase_slug TEXT NOT NULL DEFAULT 'ipd';

-- Step 2: Indexes for fast stase-filtered queries
CREATE INDEX IF NOT EXISTS idx_chunks_stase ON chunks (stase_slug);
CREATE INDEX IF NOT EXISTS idx_images_stase ON images (stase_slug);
CREATE INDEX IF NOT EXISTS idx_graph_stase ON graph_edges (stase_slug);

-- Step 3: Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_chunks_stase_section ON chunks (stase_slug, section_category);
CREATE INDEX IF NOT EXISTS idx_images_stase_source ON images (stase_slug, source_name);

-- Step 4: Update search_chunks_vector RPC to support optional stase_slug filter
CREATE OR REPLACE FUNCTION search_chunks_vector(
    query_embedding VECTOR(768),
    match_count INTEGER DEFAULT 10,
    filter_section TEXT DEFAULT NULL,
    stase_slug TEXT DEFAULT NULL
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
    RETURN QUERY
    SELECT
        c.id, c.source_name, c.page_no, c.heading, c.content,
        c.disease_tags, c.section_category, c.parent_heading,
        c.content_type, c.chunk_index,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.embedding IS NOT NULL
      AND (filter_section IS NULL OR c.section_category = filter_section)
      AND (stase_slug IS NULL OR c.stase_slug = stase_slug)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Step 5: Update search_chunks_fts RPC to support optional stase_slug filter
CREATE OR REPLACE FUNCTION search_chunks_fts(
    query_text TEXT,
    match_count INTEGER DEFAULT 10,
    expanded_terms TEXT[] DEFAULT NULL,
    stase_slug TEXT DEFAULT NULL
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
    tsquery_val := plainto_tsquery('simple', query_text);

    IF expanded_terms IS NOT NULL AND array_length(expanded_terms, 1) > 0 THEN
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
      AND (stase_slug IS NULL OR c.stase_slug = stase_slug)
    ORDER BY fts_rank DESC
    LIMIT match_count;
END;
$$;
