// Medical RAG — Cloudflare Worker Type Definitions
// Translated from Python models.py + config.py

export interface Env {
  // Supabase
  SUPABASE_URL: string;
  SUPABASE_SERVICE_KEY: string;
  // GitHub Copilot
  GITHUB_TOKEN: string;
  // Cloudflare Workers AI binding
  AI: Ai;
  // Cloudflare R2 — medical images bucket
  MEDICAL_IMAGES: R2Bucket;
  // CORS
  CORS_ORIGIN: string;
  // R2 public CDN base URL (from wrangler vars)
  R2_PUBLIC_BASE_URL: string;
}

// ── RAG Data Types ────────────────────────────────────────────────────────────

export interface ChunkRecord {
  id?: number;
  source_name: string;
  page_no: number;
  heading: string;
  content: string;
  disease_tags: string;
  section_category: string;
  parent_heading: string;
  content_type: string;
  chunk_index: number;
  similarity?: number;
  fts_rank?: number;
  rrf_score?: number;
  vector_score?: number;
  markdown_path?: string;
  source_url?: string;
  expanded?: boolean;
}

export interface ImageRecord {
  id?: number;
  source_name: string;
  page_no: number;
  heading: string;
  alt_text: string;
  image_ref: string;
  image_abs_path?: string;
  storage_url?: string;
  image_url: string; // computed URL for frontend
  nearby_text?: string;
}

export interface GraphEdge {
  source_disease: string;
  relation: string;
  target_node: string;
  target_type: string;
  source_name: string;
  page_no: number;
}

// ── Chat & Request Types ──────────────────────────────────────────────────────

export interface ChatHistoryItem {
  role: "user" | "assistant";
  content: string;
}

export interface SearchDiseaseRequest {
  disease_name: string;
  detail_level?: "ringkas" | "detail";
  top_k?: number;
  include_images?: boolean;
  chat_history?: ChatHistoryItem[];
  /** Retrieval mode override.  When omitted, auto-detected from the query. */
  retrieval_mode?: "relevant" | "exhaustive" | null;
  /** Maximum items to include in the disease list (exhaustive mode). */
  max_items?: number | null;
  /** 1-based page index for disease list pagination (exhaustive mode). */
  page?: number | null;
  /** Page size for disease list pagination (exhaustive mode). */
  page_size?: number | null;
}

// ── Retrieval Diagnostics ─────────────────────────────────────────────────────

/** Metadata returned alongside search results to indicate coverage/truncation. */
export interface RetrievalDiagnostics {
  /** Total candidate chunks evaluated before paging. */
  total_candidates: number;
  /** Number of results actually returned. */
  returned_count: number;
  /** True if results were truncated due to max_items or page_size. */
  is_truncated: boolean;
  /** Resolved retrieval mode used for the request. */
  retrieval_mode: "relevant" | "exhaustive";
  /** Total retrieval passes executed. */
  retrieval_passes?: number;
  /** Retry strategy used on second pass when present. */
  retry_mode?: "none" | "recall" | "precision";
  /** Human-readable reason why retry was triggered. */
  retry_reason?: string;
  /** Disease/entity resolver method used by retriever. */
  resolution_method?: "exact" | "synonym" | "embedding" | "history" | "fallback";
  /** Resolver confidence (0-1). */
  resolution_confidence?: number;
}

export interface DiseaseResolutionInfo {
  disease: string | null;
  method: "exact" | "synonym" | "embedding" | "history" | "fallback";
  confidence: number;
  candidates: string[];
}

export interface LibraryGenerateRequest {
  extra_prompt?: string;
  top_k?: number;
  image_limit?: number;
}

export interface LibraryPreviewRequest extends LibraryGenerateRequest {
  combine_with_existing?: boolean;
  combine_mode?: "append" | "replace";
  /** Persist combined preview to DB (markdown_combined). */
  persist?: boolean;
}

export interface LibraryRefineRequest {
  instruction: string;
}

export interface LibraryPatchContentRequest {
  markdown: string;
  preview_commit?: boolean;
}

export interface MindmapNode {
  id: string;
  label: string;
  type: "root" | "section" | "concept" | "fact";
  level: number;
  summary: string;
  val: number;
  group: number;
}

export interface MindmapEdge {
  source: string;
  target: string;
}

export interface VisualRef {
  image_url: string;
  heading: string;
  description?: string;
}

export interface MindmapSaveRequest {
  nodes: MindmapNode[];
  edges: MindmapEdge[];
  visual_refs?: VisualRef[];
  key_takeaways?: string[];
  summary_root?: string;
}

export interface LibraryMergeMarkdownRequest {
  markdown_base?: string;
  markdown_candidate: string;
}

export interface SourcePageRecord {
  id?: number;
  stase_slug: string;
  source_name: string;
  page_no: number;
  markdown: string;
  markdown_path?: string;
  checksum?: string;
  created_at?: string;
  updated_at?: string;
}

export interface BatchUploadPageInput {
  page_no: number;
  markdown: string;
  markdown_path?: string;
}

export interface BatchUploadRequest {
  pages: BatchUploadPageInput[];
  reset_source?: boolean;
  batch_index?: number;
  total_batches?: number;
  source_name?: string;
}

// ── Library Entities ──────────────────────────────────────────────────────────

export interface StaseRow {
  id: number;
  slug: string;
  display_name: string;
  sort_order: number;
  disease_count?: number;
  filled_count?: number;
}

export interface DiseaseRow {
  id: number;
  catalog_no: number;
  name: string;
  competency_level: string | null;
  group_label: string | null;
  stable_key: string;
  status: string | null;
  updated_at: string | null;
}

export interface LibraryArticle {
  id: number;
  catalog_id: number;
  status: string;
  content_markdown: string | null;
  meta: Record<string, unknown>;
  mindmap: Record<string, unknown> | null;
  content_hash: string | null;
  updated_at: string | null;
}

// ── Copilot Response Types ────────────────────────────────────────────────────

export interface DraftSection {
  title: string;
  markdown?: string;
  points?: string[];
}

export interface DraftAnswer {
  disease: string;
  verification_log?: string;
  sections: DraftSection[];
  citations: string[];
  grounded: boolean;
  question_style?: string;
  style_confidence?: number;
  intent_confidence?: number;
  ambiguity?: boolean;
  ambiguity_candidates?: string[];
  detection_method?: string;
  detection_confidence?: number;
  answer_confidence?: number;
  section_confidence_map?: Record<string, number>;
  citation_quality?: {
    overall_precision: number;
    section_precision_map: Record<string, number>;
    total_citations: number;
    relevant_citations: number;
    filtered_citations: number;
    unsupported_claims_removed: number;
    section_unsupported_removed_map: Record<string, number>;
  };
  retrieval_passes?: number;
  retry_mode?: "none" | "recall" | "precision";
  retry_reason?: string;
  regenerated_sections?: string[];
  section_regeneration_passes?: number;
  evidence_coverage?: {
    total_evidence: number;
    used_evidence: number[];
    unused_evidence: number[];
    coverage_percent: number;
  };
}
