// Medical RAG — Cloudflare Workers Main Entrypoint (Hono.js)
// Full API parity with api.py (FastAPI), translated to TypeScript

import { Hono } from "hono";
import { cors } from "hono/cors";
import { z, ZodError } from "zod";

import type { Env, ChunkRecord } from "./types";
import {
  searchChunks,
  relatedImages,
  getKnowledgeGraph,
  synthesizeFallback,
  getSupabase,
  getTopicsFromDb,
  extractDiseaseListFromChunks,
} from "./retriever";
import {
  askCopilotAdaptive,
  askCopilotForPureList,
  refineMarkdownWithInstruction,
  mergeTwoMarkdownArticles,
  generateMindmapFromArticle,
} from "./copilot-client";
import {
  extractDiseaseName,
  extractTopicIntent,
  isDetailRequest,
  isListingIntent,
  resolveRetrievalMode,
} from "./medical-vocab";
import { embedTexts, parsePageForIndexing } from "./indexer";

// ─────────────────────────────────────────────────────────────────────────────
// App setup
// ─────────────────────────────────────────────────────────────────────────────

const app = new Hono<{ Bindings: Env }>();

app.use("*", async (c, next) => {
  const corsMiddleware = cors({
    origin: c.env.CORS_ORIGIN || "*",
    allowMethods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
  });
  return corsMiddleware(c, next);
});

// ─────────────────────────────────────────────────────────────────────────────
// Health check
// ─────────────────────────────────────────────────────────────────────────────

app.get("/health", (c) => {
  return c.json({ status: "ok", version: "3.0.0", runtime: "cloudflare-workers" });
});

// ─────────────────────────────────────────────────────────────────────────────
// Zod schemas (equivalent to Pydantic models in api.py)
// ─────────────────────────────────────────────────────────────────────────────

const ChatHistoryItemSchema = z.object({
  role: z.string(),
  content: z.string(),
});

const SearchDiseaseSchema = z.object({
  disease_name: z.string().min(2),
  detail_level: z.enum(["ringkas", "detail"]).default("detail"),
  top_k: z.number().int().min(3).max(20).default(8),
  include_images: z.boolean().default(true),
  chat_history: z.array(ChatHistoryItemSchema).default([]),
  stase_slug: z.string().optional().default("ipd"),
  // Dynamic retrieval mode; when omitted, auto-detected from the query text.
  retrieval_mode: z.enum(["relevant", "exhaustive"]).nullable().optional(),
  // Listing/pagination controls (used in exhaustive mode)
  max_items: z.number().int().min(1).max(500).nullable().optional(),
  page: z.number().int().min(1).nullable().optional(),
  page_size: z.number().int().min(1).max(200).nullable().optional(),
});

const ImageRequestSchema = z.object({
  disease_name: z.string().min(2),
  limit: z.number().int().min(1).max(10).default(3),
  stase_slug: z.string().optional().default("ipd"),
});

// extra_prompt: frontend sends `null` when empty (parity with FastAPI Optional[str] = None).
// Plain z.string().optional() rejects JSON null.
const LibraryGenerateSchema = z.object({
  extra_prompt: z.preprocess(
    (v) => (v === null ? undefined : v),
    z.string().optional(),
  ),
  top_k: z.number().int().min(3).max(20).default(10),
  image_limit: z.number().int().min(1).max(10).default(5),
});

const LibraryPreviewSchema = LibraryGenerateSchema.extend({
  combine_with_existing: z.boolean().default(false),
  combine_mode: z.enum(["append", "replace"]).default("replace"),
  /** When true, persist markdown_combined to library_article (same rules as generate for draft/published). */
  persist: z.boolean().default(false),
});

const LibraryRefineSchema = z.object({
  instruction: z.string().min(3),
});

const LibraryPatchContentSchema = z.object({
  markdown: z.string().min(1),
  preview_commit: z.boolean().default(false),
});

const SourceCreateSchema = z.object({
  source_name: z.string().min(3),
});

const SourcePageUploadSchema = z.object({
  page_no: z.number().int().min(1),
  markdown: z.string().min(1),
  markdown_path: z.string().optional(),
});

const BatchUploadPageSchema = z.object({
  page_no: z.number().int().min(1),
  markdown: z.string().min(1),
  markdown_path: z.string().optional(),
});

const BatchUploadSchema = z.object({
  pages: z.array(BatchUploadPageSchema).min(1).max(10),
  reset_source: z.boolean().default(false),
  batch_index: z.number().int().min(0).default(0),
  total_batches: z.number().int().min(1).optional(),
  source_name: z.string().optional(),
});

type BatchUploadPage = z.infer<typeof BatchUploadPageSchema>;

type StoredSourcePage = {
  id: number;
  stase_slug: string;
  source_name: string;
  page_no: number;
  markdown: string;
  markdown_path: string | null;
};

type AdminSourceSummary = {
  source_name: string;
  page_count: number;
  chunk_count: number;
  indexed: boolean;
};

const MindmapNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string().default("concept"),
  level: z.number().int().default(2),
  summary: z.string().default(""),
  val: z.number().default(7),
  group: z.number().default(2),
});

const MindmapSaveSchema = z.object({
  nodes: z.array(MindmapNodeSchema),
  edges: z.array(z.object({ source: z.string(), target: z.string() })),
  visual_refs: z
    .array(
      z.object({
        image_url: z.string().default(""),
        heading: z.string().default(""),
        description: z.string().optional(),
      }),
    )
    .default([]),
  key_takeaways: z.array(z.string()).default([]),
  summary_root: z.string().default(""),
});

const MergeMarkdownSchema = z.object({
  markdown_base: z.string().default(""),
  markdown_candidate: z.string().min(1),
});

const LibraryVisualRefItemSchema = z.object({
  image_abs_path: z.string().optional(),
  image_ref: z.string().optional(),
  storage_url: z.string().optional(),
  image_url: z.string().optional(),
  heading: z.string().default(""),
  source_name: z.string().default(""),
  page_no: z.number().default(0),
});

const LibraryUpdateVisualRefsSchema = z.object({
  images: z.array(LibraryVisualRefItemSchema),
});

type LibraryVisualRefItemIn = z.infer<typeof LibraryVisualRefItemSchema>;
const MIN_EVIDENCE_FOR_AI = 3;

/** At least one of path/ref/URL so non-local R2 images can be persisted (parity with FastAPI meta.images). */
function normalizeLibraryVisualRefItem(img: LibraryVisualRefItemIn): Record<string, unknown> | null {
  const a = (img.image_abs_path ?? "").trim();
  const r = (img.image_ref ?? "").trim();
  const s = (img.storage_url ?? "").trim();
  const u = (img.image_url ?? "").trim();
  if (!a && !r && !s && !u) return null;
  const out: Record<string, unknown> = {
    heading: img.heading ?? "",
    source_name: img.source_name ?? "",
    page_no: img.page_no ?? 0,
  };
  if (a) out.image_abs_path = a;
  if (r) out.image_ref = r;
  if (s) out.storage_url = s;
  if (u) out.image_url = u;
  return out;
}

function isLikelyMedicalTopic(text: string): boolean {
  const clean = text.trim();
  if (!clean || clean.length < 3) return false;
  if (/^\d+$/.test(clean)) return false;
  if (/^[^a-zA-Z]+$/.test(clean)) return false;
  const lowered = clean.toLowerCase();
  const blocked = ["pendahuluan", "daftar isi", "lampiran", "kata pengantar", "bab ", "chapter "];
  return !blocked.some((kw) => lowered.includes(kw));
}

function buildPureListFallback(topicsData: Record<string, any>): Record<string, unknown> {
  const sections: Array<{ title: string; markdown: string }> = [];

  for (const src of (topicsData.sources as any[]) ?? []) {
    const cleaned = ((src.topics as any[]) ?? [])
      .map((t: any) => (t.heading ?? "").toString().trim())
      .filter((heading: string) => isLikelyMedicalTopic(heading));

    const unique = [...new Set(cleaned.map((h: string) => h.replace(/\s+/g, " ")))].sort((a, b) =>
      a.localeCompare(b),
    );

    if (unique.length === 0) continue;
    sections.push({
      title: src.source_name,
      markdown: unique.map((topic: string, i: number) => `${i + 1}. **${topic}**`).join("\n"),
    });
  }

  return {
    disease: "Daftar Murni Penyakit & Kondisi Medis",
    sections,
    citations: [],
    grounded: true,
  };
}

function injectLowEvidenceWarning(
  answer: Record<string, unknown>,
  evidenceCount: number,
): Record<string, unknown> {
  const sections = Array.isArray(answer.sections) ? [...(answer.sections as Array<Record<string, unknown>>)] : [];
  const hasWarning = sections.some((sec) =>
    ((sec.title ?? "").toString().toLowerCase().includes("catatan kualitas")),
  );
  if (!hasWarning) {
    sections.unshift({
      title: "Catatan Kualitas Evidence",
      markdown:
        `Evidence yang ditemukan terbatas (${evidenceCount} dokumen). Jawaban disusun secara konservatif dari bukti yang tersedia; gunakan verifikasi klinis lanjutan bila diperlukan.`,
    });
  }
  return {
    ...answer,
    sections,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: parse + validate JSON body via Zod (replaces zValidator middleware)
// Returns parsed data or sends 400 and returns null.
// ─────────────────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function parseBody<T extends z.ZodTypeAny>(c: any, schema: T): Promise<z.infer<T> | null> {
  try {
    const raw = await c.req.json();
    return schema.parse(raw) as z.infer<T>;
  } catch (err) {
    if (err instanceof ZodError) {
      c.status(422);
      await c.res; // flush
      return null;
    }
    c.status(400);
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Content hash (SHA-256 via Web Crypto API)
// ─────────────────────────────────────────────────────────────────────────────

async function contentHash(text: string): Promise<string> {
  const enc = new TextEncoder();
  const buf = await crypto.subtle.digest("SHA-256", enc.encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** PostgREST/Supabase embeds one-to-one FK relations as a single object; one-to-many as an array. */
function embeddedLibraryArticle(raw: unknown): Record<string, unknown> | undefined {
  if (raw == null) return undefined;
  if (Array.isArray(raw)) return raw[0] as Record<string, unknown> | undefined;
  if (typeof raw === "object") return raw as Record<string, unknown>;
  return undefined;
}

function sourceSummaryKey(sourceName: string): string {
  return sourceName.trim().toLowerCase();
}

async function deleteSourceArtifacts(
  supabase: ReturnType<typeof getSupabase>,
  staseSlug: string,
  sourceName: string,
  pageNos?: number[],
): Promise<void> {
  let chunksQuery = supabase.from("chunks").delete().eq("stase_slug", staseSlug).eq("source_name", sourceName);
  let imagesQuery = supabase.from("images").delete().eq("stase_slug", staseSlug).eq("source_name", sourceName);
  let edgesQuery = supabase.from("graph_edges").delete().eq("stase_slug", staseSlug).eq("source_name", sourceName);

  if (pageNos && pageNos.length > 0) {
    chunksQuery = chunksQuery.in("page_no", pageNos);
    imagesQuery = imagesQuery.in("page_no", pageNos);
    edgesQuery = edgesQuery.in("page_no", pageNos);
  }

  await Promise.all([chunksQuery, imagesQuery, edgesQuery]);
}

async function clearSourcePages(
  supabase: ReturnType<typeof getSupabase>,
  staseSlug: string,
  sourceName: string,
  pageNos?: number[],
): Promise<void> {
  let query = supabase.from("source_pages").delete().eq("stase_slug", staseSlug).eq("source_name", sourceName);
  if (pageNos && pageNos.length > 0) {
    query = query.in("page_no", pageNos);
  }
  await query;
}

async function persistSourcePages(
  supabase: ReturnType<typeof getSupabase>,
  staseSlug: string,
  sourceName: string,
  pages: BatchUploadPage[],
): Promise<void> {
  const rows = await Promise.all(
    pages.map(async (page) => ({
      stase_slug: staseSlug,
      source_name: sourceName,
      page_no: page.page_no,
      markdown: page.markdown,
      markdown_path: page.markdown_path ?? `page-${page.page_no}/markdown.md`,
      checksum: await contentHash(page.markdown),
      updated_at: new Date().toISOString(),
    })),
  );

  await supabase.from("source_pages").upsert(rows, { onConflict: "stase_slug,source_name,page_no" });
}

async function loadSourcePages(
  supabase: ReturnType<typeof getSupabase>,
  staseSlug: string,
  sourceName?: string,
): Promise<StoredSourcePage[]> {
  let query = supabase
    .from("source_pages")
    .select("id, stase_slug, source_name, page_no, markdown, markdown_path")
    .eq("stase_slug", staseSlug)
    .order("source_name")
    .order("page_no");

  if (sourceName) {
    query = query.eq("source_name", sourceName);
  }

  const { data, error } = await query;
  if (error || !data) return [];
  return (data as StoredSourcePage[]).filter((row) => row.page_no > 0);
}

async function insertParsedPage(
  env: Env,
  supabase: ReturnType<typeof getSupabase>,
  staseSlug: string,
  sourceName: string,
  page: BatchUploadPage,
  skipVector = false,
): Promise<{ chunk_count: number; image_count: number; edge_count: number }> {
  const parsed = parsePageForIndexing({
    source_name: sourceName,
    page_no: page.page_no,
    markdown: page.markdown,
    stase_slug: staseSlug,
    markdown_path: page.markdown_path,
  });

  await deleteSourceArtifacts(supabase, staseSlug, sourceName, [page.page_no]);

  if (parsed.chunks.length > 0) {
    const embeddings = skipVector ? [] : await embedTexts(env, parsed.chunks.map((chunk) => `passage: ${chunk.heading}: ${chunk.content}`));

    const chunkRows = parsed.chunks.map((chunk, index) => ({
      source_name: chunk.source_name,
      page_no: chunk.page_no,
      heading: chunk.heading,
      content: chunk.content,
      disease_tags: chunk.disease_tags,
      markdown_path: chunk.markdown_path,
      checksum: chunk.checksum,
      section_category: chunk.section_category,
      parent_heading: chunk.parent_heading,
      chunk_index: chunk.chunk_index,
      total_chunks: chunk.total_chunks,
      heading_level: chunk.heading_level,
      content_type: chunk.content_type,
      stase_slug: chunk.stase_slug,
      embedding: skipVector ? null : embeddings[index] ?? null,
    }));

    await supabase.from("chunks").insert(chunkRows);
  }

  if (parsed.images.length > 0) {
    await supabase.from("images").insert(
      parsed.images.map((img) => ({
        source_name: img.source_name,
        page_no: img.page_no,
        alt_text: img.alt_text,
        image_ref: img.image_ref,
        image_abs_path: img.image_abs_path,
        heading: img.heading,
        nearby_text: img.nearby_text,
        markdown_path: img.markdown_path,
        checksum: img.checksum,
        stase_slug: img.stase_slug,
      })),
    );
  }

  if (parsed.graph_edges.length > 0) {
    await supabase.from("graph_edges").insert(
      parsed.graph_edges.map((edge) => ({
        ...edge,
      })),
    );
  }

  await persistSourcePages(supabase, staseSlug, sourceName, [page]);

  return {
    chunk_count: parsed.chunks.length,
    image_count: parsed.images.length,
    edge_count: parsed.graph_edges.length,
  };
}

async function rebuildSourceFromStoredPages(
  env: Env,
  supabase: ReturnType<typeof getSupabase>,
  staseSlug: string,
  sourceName?: string,
  skipVector = false,
): Promise<{ source_name: string; pages: number; chunks: number; images: number; graph_edges: number }[]> {
  const pages = await loadSourcePages(supabase, staseSlug, sourceName);
  if (pages.length === 0) return [];

  const bySource = new Map<string, StoredSourcePage[]>();
  for (const page of pages) {
    const key = page.source_name;
    if (!bySource.has(key)) bySource.set(key, []);
    bySource.get(key)!.push(page);
  }

  const results: { source_name: string; pages: number; chunks: number; images: number; graph_edges: number }[] = [];

  for (const [currentSource, sourcePages] of bySource.entries()) {
    const pageNos = sourcePages.map((page) => page.page_no);
    await deleteSourceArtifacts(supabase, staseSlug, currentSource);

    let totalChunks = 0;
    let totalImages = 0;
    let totalEdges = 0;

    for (const page of sourcePages) {
      const parsed = parsePageForIndexing({
        source_name: currentSource,
        page_no: page.page_no,
        markdown: page.markdown,
        stase_slug: staseSlug,
        markdown_path: page.markdown_path ?? undefined,
      });

      if (parsed.chunks.length > 0) {
        const embeddings = skipVector ? [] : await embedTexts(env, parsed.chunks.map((chunk) => `passage: ${chunk.heading}: ${chunk.content}`));
        await supabase.from("chunks").insert(
          parsed.chunks.map((chunk, index) => ({
            source_name: chunk.source_name,
            page_no: chunk.page_no,
            heading: chunk.heading,
            content: chunk.content,
            disease_tags: chunk.disease_tags,
            markdown_path: chunk.markdown_path,
            checksum: chunk.checksum,
            section_category: chunk.section_category,
            parent_heading: chunk.parent_heading,
            chunk_index: chunk.chunk_index,
            total_chunks: chunk.total_chunks,
            heading_level: chunk.heading_level,
            content_type: chunk.content_type,
            stase_slug: chunk.stase_slug,
            embedding: skipVector ? null : embeddings[index] ?? null,
          })),
        );
      }

      if (parsed.images.length > 0) {
        await supabase.from("images").insert(
          parsed.images.map((img) => ({
            source_name: img.source_name,
            page_no: img.page_no,
            alt_text: img.alt_text,
            image_ref: img.image_ref,
            image_abs_path: img.image_abs_path,
            heading: img.heading,
            nearby_text: img.nearby_text,
            markdown_path: img.markdown_path,
            checksum: img.checksum,
            stase_slug: img.stase_slug,
          })),
        );
      }

      if (parsed.graph_edges.length > 0) {
        await supabase.from("graph_edges").insert(
          parsed.graph_edges.map((edge) => ({ ...edge })),
        );
      }

      totalChunks += parsed.chunks.length;
      totalImages += parsed.images.length;
      totalEdges += parsed.graph_edges.length;
    }

    results.push({
      source_name: currentSource,
      pages: pageNos.length,
      chunks: totalChunks,
      images: totalImages,
      graph_edges: totalEdges,
    });
  }

  return results;
}

async function getAdminSourceSummaries(env: Env, staseSlug: string): Promise<AdminSourceSummary[]> {
  const supabase = getSupabase(env);
  const [pageRows, chunkRows] = await Promise.all([
    supabase.from("source_pages").select("source_name, page_no").eq("stase_slug", staseSlug),
    supabase.from("chunks").select("source_name, page_no").eq("stase_slug", staseSlug),
  ]);

  const pageMap = new Map<string, Set<number>>();
  const chunkCountMap = new Map<string, number>();

  for (const row of (pageRows.data ?? []) as Array<{ source_name: string; page_no: number }>) {
    const key = sourceSummaryKey(row.source_name);
    if (!pageMap.has(key)) pageMap.set(key, new Set());
    pageMap.get(key)!.add(row.page_no);
  }

  for (const row of (chunkRows.data ?? []) as Array<{ source_name: string; page_no: number }>) {
    const key = sourceSummaryKey(row.source_name);
    chunkCountMap.set(key, (chunkCountMap.get(key) ?? 0) + 1);
  }

  const names = new Map<string, string>();
  for (const row of (pageRows.data ?? []) as Array<{ source_name: string }>) {
    names.set(sourceSummaryKey(row.source_name), row.source_name);
  }
  for (const row of (chunkRows.data ?? []) as Array<{ source_name: string }>) {
    if (!names.has(sourceSummaryKey(row.source_name))) {
      names.set(sourceSummaryKey(row.source_name), row.source_name);
    }
  }

  return [...names.entries()]
    .map(([key, sourceName]) => {
      const pageCount = [...(pageMap.get(key) ?? new Set())].filter((pageNo) => pageNo > 0).length;
      const chunkCount = chunkCountMap.get(key) ?? 0;
      return {
        source_name: sourceName,
        page_count: pageCount,
        chunk_count: chunkCount,
        indexed: chunkCount > 0,
      };
    })
    .sort((a, b) => a.source_name.localeCompare(b.source_name));
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: persist library article (shared by generate + preview persist)
// ─────────────────────────────────────────────────────────────────────────────

type ArticleGenResult = Awaited<ReturnType<typeof runArticleGenerationPipeline>>;

async function upsertLibraryArticleMarkdown(
  supabase: ReturnType<typeof getSupabase>,
  catalogId: number,
  slug: string,
  bundle: Record<string, unknown>,
  markdownBody: string,
  gen: ArticleGenResult,
  extraPrompt: string | undefined,
  lastOperation: "generate" | "preview_persist",
): Promise<{ status: string; newMeta: Record<string, unknown> }> {
  const isGrounded = (gen.draft_answer as Record<string, unknown>)["grounded"] !== false;
  const status = gen.evidence.length === 0 || !isGrounded ? "draft" : "published";
  const hash = await contentHash(markdownBody);
  const la = embeddedLibraryArticle(bundle["library_article"]);
  const prevMeta = (la?.["meta"] as Record<string, unknown>) ?? {};
  const ver = ((prevMeta["version"] as number) ?? 0) + 1;

  const newMeta: Record<string, unknown> = {
    version: ver,
    disease_name: bundle["name"],
    catalog_no: bundle["catalog_no"],
    stase_slug: slug,
    generated_at: new Date().toISOString(),
    extra_prompt: extraPrompt ?? null,
    last_operation: lastOperation,
    images: gen.images,
  };

  await supabase.from("library_article").upsert(
    {
      catalog_id: catalogId,
      status,
      content_markdown: markdownBody,
      meta: newMeta,
      content_hash: hash,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "catalog_id" },
  );

  return { status, newMeta };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: draft answer → markdown
// ─────────────────────────────────────────────────────────────────────────────

function draftAnswerToMarkdown(
  diseaseTitle: string,
  draft: Record<string, unknown>,
): string {
  const parts = [`# ${diseaseTitle}\n`];
  const sections = draft["sections"] as
    | Array<{ title?: string; markdown?: string; points?: string[] }>
    | undefined;
  for (const sec of sections ?? []) {
    const title = sec.title ?? "Section";
    let md = (sec.markdown ?? "").trim();
    if (!md && sec.points) md = sec.points.map((p) => `- ${p}`).join("\n");
    if (md) parts.push(`\n## ${title}\n\n${md}\n`);
  }
  return parts.join("").trim() + "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: combine preview markdown
// ─────────────────────────────────────────────────────────────────────────────

function combinePreviewMarkdown(
  baseMd: string | null,
  candidateMd: string,
  mode: "append" | "replace",
): string {
  let cand = candidateMd.trim();
  if (!cand.endsWith("\n")) cand += "\n";
  if (mode === "replace") return cand;
  const base = (baseMd ?? "").trim();
  if (!base) return cand;
  const sep = "\n\n---\n\n## Pembaruan (regenerate)\n\n";
  return (base + sep + cand.trim()).trim() + "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run article generation pipeline
// ─────────────────────────────────────────────────────────────────────────────

async function runArticleGenerationPipeline(
  env: Env,
  diseaseName: string,
  extraPrompt?: string,
  topK = 10,
  imageLimit = 5,
  staseSlug = "ipd",
): Promise<{
  query: string;
  evidence: ChunkRecord[];
  images: Record<string, unknown>[];
  draft_answer: Record<string, unknown>;
  markdown_candidate: string;
}> {
  const query = extraPrompt ? `${diseaseName}. ${extraPrompt}` : diseaseName;
  const evidence = await searchChunks(env, query, topK, undefined, staseSlug);
  const rawImages = await relatedImages(env, query, evidence, imageLimit, staseSlug);
  const images: Record<string, unknown>[] = rawImages.map((img) => ({
    source_name: img.source_name,
    page_no: img.page_no,
    heading: img.heading,
    alt_text: img.alt_text,
    image_ref: img.image_ref,
    storage_url: img.storage_url ?? "",
    image_url: img.storage_url ?? img.image_url ?? "",
    nearby_text: img.nearby_text ?? "",
  }));

  let draftAnswer: Record<string, unknown>;
  if (env.GITHUB_TOKEN) {
    draftAnswer = (await askCopilotAdaptive(
      query,
      evidence,
      env.GITHUB_TOKEN,
      undefined,
      rawImages,
    )) as unknown as Record<string, unknown>;
  } else {
    draftAnswer = synthesizeFallback(query, evidence) as unknown as Record<string, unknown>;
  }

  const markdownCandidate = draftAnswerToMarkdown(diseaseName, draftAnswer);
  return { query, evidence, images, draft_answer: draftAnswer, markdown_candidate: markdownCandidate };
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helper: get disease bundle with library_article joined
// ─────────────────────────────────────────────────────────────────────────────

async function getBundleOrFail(
  env: Env,
  slug: string,
  catalogId: number,
): Promise<Record<string, unknown> | null> {
  const supabase = getSupabase(env);
  const { data: stase } = await supabase.from("stase").select("id").eq("slug", slug).single();
  if (!stase) return null;

  const staseId = (stase as Record<string, unknown>)["id"];
  const { data: bundle } = await supabase
    .from("disease_catalog")
    .select(
      "*, library_article ( status, content_markdown, meta, mindmap, content_hash, updated_at )",
    )
    .eq("stase_id", staseId as number)
    .eq("id", catalogId)
    .single();

  if (!bundle) return null;
  return bundle as Record<string, unknown>;
}

// ─────────────────────────────────────────────────────────────────────────────
// ROUTE: POST /search_disease_context
// ─────────────────────────────────────────────────────────────────────────────

app.post("/search_disease_context", async (c) => {
  const payload = await parseBody(c, SearchDiseaseSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const history = payload.chat_history as Array<{ role: string; content: string }>;
  const staseSlug = payload.stase_slug ?? "ipd";

  // Resolve retrieval mode (explicit override or auto-detected)
  const mode = resolveRetrievalMode(
    payload.disease_name,
    (payload.retrieval_mode as "relevant" | "exhaustive" | null | undefined) ?? null,
  );
  const isExhaustive = mode === "exhaustive";

  const detectedDisease = extractDiseaseName(payload.disease_name);
  const detectedIntent = extractTopicIntent(payload.disease_name);
  const det = isDetailRequest(payload.disease_name);
  const detectedListingIntent = isListingIntent(payload.disease_name);

  // Compute pagination metadata
  const pageSize = payload.page_size ?? 50;
  const page = payload.page ?? 1;

  // ── Cek list intent sebelum semantic search ────────────────────────
  if (detectedListingIntent) {
    // Extract source filter if any (simple keyword matching)
    const sourceKeywords = ["atria", "mediko", "pppk", "p3k", "kaplan"];
    let foundFilter: string | undefined;
    const qLower = payload.disease_name.toLowerCase();
    for (const kw of sourceKeywords) {
      if (qLower.includes(kw)) {
        foundFilter = kw;
        break;
      }
    }

    const topics = await getTopicsFromDb(c.env, staseSlug, foundFilter);

    let listAnswer: Record<string, any> = buildPureListFallback(topics);
    let listRefinedByAi = false;

    // --- AI Refinement (Optional) ---
    if (c.env.GITHUB_TOKEN) {
      try {
        // AI will filter noise (like symbols and page numbers) while maintaining completeness
        listAnswer = (await askCopilotForPureList(
          topics,
          c.env.GITHUB_TOKEN,
        )) as unknown as Record<string, any>;
        if (!Array.isArray(listAnswer.sections) || listAnswer.sections.length === 0) {
          listAnswer = buildPureListFallback(topics);
        } else {
          listRefinedByAi = true;
        }
      } catch (e) {
        console.warn("[WARN] AI list refinement failed, using raw DB list:", e);
        listAnswer = buildPureListFallback(topics);
      }
    }

    return c.json({
      query: payload.disease_name,
      query_analysis: {
        is_list_intent: true,
        retrieval_mode: mode,
      },
      retrieval_mode: mode,
      detail_level: payload.detail_level,
      evidence_count: 0,
      evidence: [],
      draft_answer: listAnswer,
      images: [],
      topics: topics,
      disease_list: extractDiseaseListFromChunks(
        topics.sources.flatMap((s: any) =>
          s.topics.map((t: any) => ({
            ...t,
            source_name: s.source_name,
            page_no: Number.isFinite(t.page_no) ? Number(t.page_no) : 0,
          }))
        )
      ),
      pagination: {
        page: page,
        page_size: pageSize,
        total_found: 0,
        returned_count: 0,
        is_truncated: false,
      },
      note: listRefinedByAi
        ? "Output disaring menggunakan AI untuk menampilkan entitas penyakit murni."
        : "Output menggunakan fallback deterministic untuk menampilkan entitas penyakit murni.",
    });
  }

  const evidence = await searchChunks(
    c.env,
    payload.disease_name,
    payload.top_k,
    history.length > 0 ? history : undefined,
    staseSlug,
    mode,
  );

  const maxItems = payload.max_items ?? null;
  const totalCandidates = evidence.length;

  let returnedEvidence = evidence;
  if (isExhaustive) {
    let pool = evidence;
    if (maxItems) pool = pool.slice(0, maxItems);
    const start = (page - 1) * pageSize;
    returnedEvidence = pool.slice(start, start + pageSize);
  }

  const returnedCount = returnedEvidence.length;
  const isTruncated = isExhaustive
    ? returnedCount < (maxItems != null ? Math.min(totalCandidates, maxItems) : totalCandidates)
    : false;

  const diagnostics = {
    total_candidates: totalCandidates,
    returned_count: returnedCount,
    is_truncated: isTruncated,
    retrieval_mode: mode,
  };

  let images: Record<string, unknown>[] = [];
  if (payload.include_images && !isExhaustive) {
    const raw = await relatedImages(c.env, payload.disease_name, evidence, 3, staseSlug);
    images = raw.map((img) => ({
      source_name: img.source_name,
      page_no: img.page_no,
      heading: img.heading,
      alt_text: img.alt_text,
      image_url: img.storage_url ?? img.image_url ?? "",
    }));
  }

  const evidenceQuality = returnedCount >= MIN_EVIDENCE_FOR_AI ? "ok" : "low";

  let answer: Record<string, unknown>;
  if (c.env.GITHUB_TOKEN && returnedCount >= MIN_EVIDENCE_FOR_AI) {
    answer = (await askCopilotAdaptive(
      payload.disease_name,
      returnedEvidence,
      c.env.GITHUB_TOKEN,
      history.length > 0 ? history : undefined,
    )) as unknown as Record<string, unknown>;
  } else {
    answer = synthesizeFallback(payload.disease_name, returnedEvidence) as unknown as Record<string, unknown>;
  }

  if (evidenceQuality === "low") {
    answer = injectLowEvidenceWarning(answer, returnedCount);
  }

  return c.json({
    query: payload.disease_name,
    query_analysis: {
      detected_disease: detectedDisease,
      detected_intent: detectedIntent,
      is_detail_request: det,
      is_list_intent: detectedListingIntent,
      retrieval_mode: mode,
    },
    retrieval_mode: mode,
    detail_level: payload.detail_level,
    evidence_count: returnedCount,
    evidence: returnedEvidence,
    draft_answer: answer,
    evidence_quality: evidenceQuality,
    images,
    retrieval_diagnostics: diagnostics,
    note: "Gunakan draft_answer sebagai basis penjelasan grounded.",
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// ROUTE: POST /get_related_images
// ─────────────────────────────────────────────────────────────────────────────

app.post("/get_related_images", async (c) => {
  const payload = await parseBody(c, ImageRequestSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const staseSlug = payload.stase_slug ?? "ipd";
  const images = await relatedImages(c.env, payload.disease_name, [], payload.limit, staseSlug);
  return c.json({
    query: payload.disease_name,
    images: images.map((img) => ({
      ...img,
      image_url: img.storage_url ?? img.image_url ?? "",
    })),
    count: images.length,
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// ADMIN ROUTES: Supabase-backed source management for cloud uploads
// ─────────────────────────────────────────────────────────────────────────────

app.get("/admin/stases", async (c) => {
  const supabase = getSupabase(c.env);
  const { data: stases } = await supabase
    .from("stase")
    .select(
      "id, slug, display_name, sort_order, disease_catalog ( id, library_article ( status ) )",
    )
    .order("sort_order");

  const enriched = (stases ?? []).map((s: unknown) => {
    const sr = s as Record<string, unknown>;
    const diseases = (sr["disease_catalog"] as Array<Record<string, unknown>>) ?? [];
    const diseaseCount = diseases.length;
    const filledCount = diseases.filter((d) => {
      const la = embeddedLibraryArticle(d["library_article"]);
      const status = la?.["status"] as string;
      return status === "draft" || status === "published";
    }).length;
    return {
      id: sr["id"],
      slug: sr["slug"],
      display_name: sr["display_name"],
      sort_order: sr["sort_order"],
      disease_count: diseaseCount,
      filled_count: filledCount,
    };
  });

  return c.json({ stases: enriched });
});

app.post("/admin/stases", async (c) => {
  const payload = await parseBody(c, z.object({ slug: z.string().min(2), display_name: z.string().min(2) }));
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const { error } = await supabase.from("stase").upsert(
    { slug: payload.slug, display_name: payload.display_name, sort_order: 0 },
    { onConflict: "slug" },
  );
  if (error) return c.json({ error: error.message }, 400);
  return c.json({ ok: true, slug: payload.slug, display_name: payload.display_name });
});

app.delete("/admin/stases/:slug", async (c) => {
  const slug = c.req.param("slug");
  const supabase = getSupabase(c.env);
  const { error } = await supabase.from("stase").delete().eq("slug", slug);
  if (error) return c.json({ error: error.message }, 400);
  return c.json({ ok: true, slug });
});

app.get("/admin/stases/:slug/sources", async (c) => {
  const slug = c.req.param("slug");
  const sources = await getAdminSourceSummaries(c.env, slug);
  return c.json({ sources });
});

app.post("/admin/stases/:slug/sources", async (c) => {
  const slug = c.req.param("slug");
  const payload = await parseBody(c, SourceCreateSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const sourceName = payload.source_name.trim();
  const supabase = getSupabase(c.env);
  await supabase.from("source_pages").upsert(
    {
      stase_slug: slug,
      source_name: sourceName,
      page_no: 0,
      markdown: "",
      markdown_path: "",
      checksum: await contentHash(`placeholder:${slug}:${sourceName}`),
      updated_at: new Date().toISOString(),
    },
    { onConflict: "stase_slug,source_name,page_no" },
  );

  return c.json({ ok: true, source_name: sourceName });
});

app.post("/admin/stases/:slug/sources/:sourceName/pages/:pageNo", async (c) => {
  const slug = c.req.param("slug");
  const sourceName = c.req.param("sourceName");
  const pageNo = parseInt(c.req.param("pageNo"), 10);
  const payload = await parseBody(c, SourcePageUploadSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const page: BatchUploadPage = {
    page_no: Number.isFinite(pageNo) && pageNo > 0 ? pageNo : payload.page_no,
    markdown: payload.markdown,
    markdown_path: payload.markdown_path,
  };

  const result = await insertParsedPage(c.env, supabase, slug, sourceName, page);
  return c.json({ ok: true, source_name: sourceName, ...result });
});

app.delete("/admin/stases/:slug/sources/:sourceName", async (c) => {
  const slug = c.req.param("slug");
  const sourceName = c.req.param("sourceName");
  const supabase = getSupabase(c.env);

  await Promise.all([
    clearSourcePages(supabase, slug, sourceName),
    deleteSourceArtifacts(supabase, slug, sourceName),
  ]);

  return c.json({ ok: true, deleted: sourceName });
});

app.post("/admin/stases/:slug/sources/:sourceName/batch_upload", async (c) => {
  const slug = c.req.param("slug");
  const sourceName = c.req.param("sourceName");
  const payload = await parseBody(c, BatchUploadSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const pages = [...payload.pages].sort((a, b) => a.page_no - b.page_no);

  if (payload.reset_source || payload.batch_index === 0) {
    await Promise.all([
      clearSourcePages(supabase, slug, sourceName),
      deleteSourceArtifacts(supabase, slug, sourceName),
    ]);
  }

  let chunkCount = 0;
  let imageCount = 0;
  let edgeCount = 0;
  for (const page of pages) {
    const result = await insertParsedPage(c.env, supabase, slug, sourceName, page);
    chunkCount += result.chunk_count;
    imageCount += result.image_count;
    edgeCount += result.edge_count;
  }

  return c.json({
    ok: true,
    source_name: sourceName,
    pages_uploaded: pages.length,
    chunk_count: chunkCount,
    image_count: imageCount,
    edge_count: edgeCount,
    batch_index: payload.batch_index,
    next_batch_index: payload.batch_index + 1,
    reset_source: payload.reset_source,
  });
});

app.post("/admin/reindex", async (c) => {
  const slug = c.req.query("slug") ?? undefined;
  const sourceName = c.req.query("source_name") ?? undefined;
  const skipVector = (c.req.query("skip_vector") ?? "false").toLowerCase() === "true";

  if (!slug) return c.json({ error: "slug query parameter is required" }, 422);

  const supabase = getSupabase(c.env);
  const rebuilt = await rebuildSourceFromStoredPages(c.env, supabase, slug, sourceName, skipVector);
  return c.json({ ok: true, mode: sourceName ? "partial" : "full", rebuilt, skip_vector: skipVector });
});

// ─────────────────────────────────────────────────────────────────────────────
// ROUTE: GET /knowledge_graph/:disease_name
// ─────────────────────────────────────────────────────────────────────────────

app.get("/knowledge_graph/:disease_name", async (c) => {
  const diseaseName = c.req.param("disease_name");
  const maxNodes = parseInt(c.req.query("max_nodes") ?? "40", 10);
  const graph = await getKnowledgeGraph(c.env, diseaseName, maxNodes);
  return c.json(graph);
});

// ─────────────────────────────────────────────────────────────────────────────
// LIBRARY ROUTES
// ─────────────────────────────────────────────────────────────────────────────

app.get("/library/stases", async (c) => {
  const supabase = getSupabase(c.env);
  const { data: stases } = await supabase
    .from("stase")
    .select(
      "id, slug, display_name, sort_order, disease_catalog ( id, library_article ( status ) )",
    )
    .order("sort_order");

  const enriched = (stases ?? []).map((s: unknown) => {
    const sr = s as Record<string, unknown>;
    const diseases = (sr["disease_catalog"] as Array<Record<string, unknown>>) ?? [];
    const diseaseCount = diseases.length;
    const filledCount = diseases.filter((d) => {
      const la = embeddedLibraryArticle(d["library_article"]);
      const status = la?.["status"] as string;
      return status === "draft" || status === "published";
    }).length;
    return {
      id: sr["id"],
      slug: sr["slug"],
      display_name: sr["display_name"],
      sort_order: sr["sort_order"],
      disease_count: diseaseCount,
      filled_count: filledCount,
    };
  });

  return c.json({ stases: enriched });
});

app.post("/library/sync", (c) =>
  c.json({ ok: true, message: "Sync is managed via migration script on Supabase." }),
);

app.post("/library/merge_markdown_copilot", async (c) => {
  if (!c.env.GITHUB_TOKEN) return c.json({ error: "GITHUB_TOKEN required" }, 503);
  const payload = await parseBody(c, MergeMarkdownSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const merged = await mergeTwoMarkdownArticles(
    payload.markdown_base ?? "",
    payload.markdown_candidate,
    c.env.GITHUB_TOKEN,
  );
  return c.json({ ok: true, markdown_merged: merged });
});

app.get("/library/stases/:slug/diseases", async (c) => {
  const slug = c.req.param("slug");
  const supabase = getSupabase(c.env);

  const { data: stase } = await supabase
    .from("stase")
    .select("id, slug, display_name")
    .eq("slug", slug)
    .single();
  if (!stase) return c.json({ error: "Stase not found" }, 404);

  const staseData = stase as Record<string, unknown>;
  const { data: diseases } = await supabase
    .from("disease_catalog")
    .select(
      "id, catalog_no, name, competency_level, group_label, stable_key, library_article ( status, updated_at )",
    )
    .eq("stase_id", staseData["id"] as number)
    .order("catalog_no");

  const rows = (diseases ?? []).map((d: unknown) => {
    const dr = d as Record<string, unknown>;
    const la = embeddedLibraryArticle(dr["library_article"]);
    return {
      id: dr["id"],
      catalog_no: dr["catalog_no"],
      name: dr["name"],
      competency_level: dr["competency_level"],
      group_label: dr["group_label"],
      stable_key: dr["stable_key"],
      status: la?.["status"] ?? "missing",
      updated_at: la?.["updated_at"] ?? null,
    };
  });

  const total = rows.length;
  const filled = rows.filter((d) => d.status === "draft" || d.status === "published").length;

  return c.json({
    stase: { slug: staseData["slug"], display_name: staseData["display_name"] },
    diseases: rows,
    progress: {
      filled,
      total,
      percent: total > 0 ? Math.round((filled / total) * 1000) / 10 : 0,
    },
  });
});

app.get("/library/stases/:slug/diseases/:catalog_id", async (c) => {
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const la = embeddedLibraryArticle(bundle["library_article"]);
  const meta = (la?.["meta"] as Record<string, unknown>) ?? {};
  const images = ((meta["images"] as Array<Record<string, unknown>>) ?? []).map((img) => ({
    ...img,
    image_url: (img["storage_url"] as string) ?? (img["image_url"] as string) ?? "",
  }));

  const { library_article: _ignored, ...bundleWithoutArticle } = bundle;

  return c.json({
    disease: bundleWithoutArticle,
    markdown: la?.["content_markdown"] ?? null,
    meta,
    images,
  });
});

// ─── Article CRUD ─────────────────────────────────────────────────────────────

app.post("/library/stases/:slug/diseases/:catalog_id/preview", async (c) => {
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const payload = await parseBody(c, LibraryPreviewSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const gen = await runArticleGenerationPipeline(
    c.env,
    bundle["name"] as string,
    payload.extra_prompt,
    payload.top_k,
    payload.image_limit,
    slug,
  );

  const la = embeddedLibraryArticle(bundle["library_article"]);
  const markdownBase = (la?.["content_markdown"] as string) ?? "";
  const mode =
    payload.combine_with_existing && payload.combine_mode === "append" ? "append" : "replace";
  const markdownCombined = combinePreviewMarkdown(markdownBase || null, gen.markdown_candidate, mode);

  let previewNote: string;
  if (mode === "replace") {
    previewNote = "Kandidat baru saja (ganti penuh). Belum disimpan.";
  } else if (markdownBase.trim()) {
    previewNote = "Gabungan artikel lama + pembaruan. Belum disimpan.";
  } else {
    previewNote = "Belum ada artikel lama; kandidat sama dengan generate baru. Belum disimpan.";
  }

  if (payload.persist) {
    const supabase = getSupabase(c.env);
    await upsertLibraryArticleMarkdown(
      supabase,
      catalogId,
      slug,
      bundle,
      markdownCombined,
      gen,
      payload.extra_prompt,
      "preview_persist",
    );
    previewNote = previewNote.replace("Belum disimpan.", "Tersimpan ke artikel utama.");
  }

  return c.json({
    ok: true,
    markdown_base: markdownBase,
    markdown_candidate: gen.markdown_candidate,
    markdown_combined: markdownCombined,
    draft_answer: gen.draft_answer,
    evidence_count: gen.evidence.length,
    evidence: gen.evidence,
    images: gen.images,
    preview_note: previewNote,
    persisted: payload.persist,
  });
});

app.post("/library/stases/:slug/diseases/:catalog_id/generate", async (c) => {
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const payload = await parseBody(c, LibraryGenerateSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const gen = await runArticleGenerationPipeline(
    c.env,
    bundle["name"] as string,
    payload.extra_prompt,
    payload.top_k,
    payload.image_limit,
    slug,
  );

  const { status, newMeta } = await upsertLibraryArticleMarkdown(
    supabase,
    catalogId,
    slug,
    bundle,
    gen.markdown_candidate,
    gen,
    payload.extra_prompt,
    "generate",
  );

  return c.json({
    ok: true,
    status,
    draft_answer: gen.draft_answer,
    evidence_count: gen.evidence.length,
    evidence: gen.evidence,
    meta: newMeta,
    images: gen.images,
  });
});

app.post("/library/stases/:slug/diseases/:catalog_id/refine", async (c) => {
  if (!c.env.GITHUB_TOKEN) return c.json({ error: "GITHUB_TOKEN required" }, 503);
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const payload = await parseBody(c, LibraryRefineSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const la = embeddedLibraryArticle(bundle["library_article"]);
  const currentMd = (la?.["content_markdown"] as string) ?? "";
  if (!currentMd) return c.json({ error: "No article to refine; generate first." }, 400);

  const refined = await refineMarkdownWithInstruction(
    currentMd,
    payload.instruction,
    c.env.GITHUB_TOKEN,
  );
  const hash = await contentHash(refined);
  const prevMeta = (la?.["meta"] as Record<string, unknown>) ?? {};
  const newMeta = {
    ...prevMeta,
    last_refine_instruction: payload.instruction,
    last_operation: "refine",
    updated_at: new Date().toISOString(),
  };

  await supabase.from("library_article").upsert(
    {
      catalog_id: catalogId,
      status: "published",
      content_markdown: refined,
      meta: newMeta,
      content_hash: hash,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "catalog_id" },
  );

  return c.json({ ok: true, markdown: refined, meta: newMeta });
});

app.patch("/library/stases/:slug/diseases/:catalog_id/content", async (c) => {
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const payload = await parseBody(c, LibraryPatchContentSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const hash = await contentHash(payload.markdown);
  const la = embeddedLibraryArticle(bundle["library_article"]);
  const prevMeta = (la?.["meta"] as Record<string, unknown>) ?? {};
  const newMeta = {
    ...prevMeta,
    last_operation: payload.preview_commit ? "preview_commit" : "manual_edit",
    updated_at: new Date().toISOString(),
  };

  await supabase.from("library_article").upsert(
    {
      catalog_id: catalogId,
      status: "published",
      content_markdown: payload.markdown,
      meta: newMeta,
      content_hash: hash,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "catalog_id" },
  );

  return c.json({ ok: true, meta: newMeta });
});

app.patch("/library/stases/:slug/diseases/:catalog_id/visual_refs", async (c) => {
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const payload = await parseBody(c, LibraryUpdateVisualRefsSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const normalizedImages: Record<string, unknown>[] = [];
  for (const it of payload.images) {
    const n = normalizeLibraryVisualRefItem(it);
    if (n === null) {
      return c.json(
        {
          error:
            "Each image entry must include at least one of: image_abs_path, image_ref, storage_url, image_url",
        },
        422,
      );
    }
    normalizedImages.push(n);
  }

  const la = embeddedLibraryArticle(bundle["library_article"]);
  const prevMeta = (la?.["meta"] as Record<string, unknown>) ?? {};
  const newMeta = {
    ...prevMeta,
    images: normalizedImages,
    last_operation: "update_visual_refs",
    updated_at: new Date().toISOString(),
  };

  await supabase.from("library_article").upsert(
    {
      catalog_id: catalogId,
      status: (la?.["status"] as string) ?? "draft",
      content_markdown: (la?.["content_markdown"] as string | null) ?? null,
      meta: newMeta,
      content_hash: (la?.["content_hash"] as string | null) ?? null,
      mindmap: la?.["mindmap"] ?? null,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "catalog_id" },
  );

  const imagesOut = normalizedImages.map((img) => ({
    ...img,
    image_url:
      (img["storage_url"] as string | undefined) ??
      (img["image_url"] as string | undefined) ??
      "",
  }));

  return c.json({
    ok: true,
    images: imagesOut,
    count: imagesOut.length,
    meta: newMeta,
  });
});

app.delete("/library/stases/:slug/diseases/:catalog_id/article", async (c) => {
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const supabase = getSupabase(c.env);

  await supabase.from("library_article").upsert(
    {
      catalog_id: catalogId,
      status: "missing",
      content_markdown: null,
      meta: {},
      mindmap: null,
      content_hash: null,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "catalog_id" },
  );

  return c.json({ ok: true });
});

// ─── Mindmap routes ───────────────────────────────────────────────────────────

app.get("/library/stases/:slug/diseases/:catalog_id/mindmap", async (c) => {
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);

  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const la = embeddedLibraryArticle(bundle["library_article"]);
  if (la?.["mindmap"]) return c.json(la["mindmap"]);

  return c.json({
    disease: bundle["name"],
    competency_level: bundle["competency_level"] ?? null,
    nodes: [],
    edges: [],
    visual_refs: [],
    key_takeaways: [],
    not_generated: true,
  });
});

app.post("/library/stases/:slug/diseases/:catalog_id/mindmap/generate", async (c) => {
  if (!c.env.GITHUB_TOKEN) return c.json({ error: "GITHUB_TOKEN required" }, 503);
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const supabase = getSupabase(c.env);

  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const la = embeddedLibraryArticle(bundle["library_article"]);
  const markdownContent = (la?.["content_markdown"] as string) ?? "";
  if (!markdownContent)
    return c.json(
      { error: "Article not found. Generate article in Medical Library first." },
      400,
    );

  const result = await generateMindmapFromArticle(
    bundle["name"] as string,
    markdownContent,
    c.env.GITHUB_TOKEN,
    bundle["competency_level"] as string | null,
  );

  await supabase.from("library_article").upsert(
    { catalog_id: catalogId, mindmap: result, updated_at: new Date().toISOString() },
    { onConflict: "catalog_id" },
  );

  return c.json(result);
});

app.patch("/library/stases/:slug/diseases/:catalog_id/mindmap", async (c) => {
  const slug = c.req.param("slug");
  const catalogId = parseInt(c.req.param("catalog_id"), 10);
  const payload = await parseBody(c, MindmapSaveSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const supabase = getSupabase(c.env);
  const bundle = await getBundleOrFail(c.env, slug, catalogId);
  if (!bundle) return c.json({ error: "Disease not found" }, 404);

  const la = embeddedLibraryArticle(bundle["library_article"]);
  const existing = (la?.["mindmap"] as Record<string, unknown>) ?? {};
  const newMindmap = {
    ...existing,
    disease: bundle["name"],
    competency_level: bundle["competency_level"] ?? null,
    nodes: payload.nodes,
    edges: payload.edges,
    visual_refs: payload.visual_refs ?? [],
    key_takeaways: payload.key_takeaways ?? [],
    summary_root: payload.summary_root ?? "",
    saved_at: new Date().toISOString(),
  };

  await supabase.from("library_article").upsert(
    { catalog_id: catalogId, mindmap: newMindmap, updated_at: new Date().toISOString() },
    { onConflict: "catalog_id" },
  );

  return c.json({ ok: true, disease: bundle["name"], node_count: payload.nodes.length });
});

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

export default app;
