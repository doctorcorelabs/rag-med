-- =============================================================
-- Medical RAG — Migration 06: Cloud source page storage
-- Stores raw markdown pages so Worker batch uploads can retry and reindex
-- =============================================================

CREATE TABLE IF NOT EXISTS source_pages (
    id BIGSERIAL PRIMARY KEY,
    stase_slug TEXT NOT NULL DEFAULT 'ipd',
    source_name TEXT NOT NULL,
    page_no INTEGER NOT NULL DEFAULT 0,
    markdown TEXT NOT NULL DEFAULT '',
    markdown_path TEXT NOT NULL DEFAULT '',
    checksum TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (stase_slug, source_name, page_no)
);

CREATE INDEX IF NOT EXISTS idx_source_pages_stase ON source_pages (stase_slug);
CREATE INDEX IF NOT EXISTS idx_source_pages_source ON source_pages (stase_slug, source_name, page_no);
