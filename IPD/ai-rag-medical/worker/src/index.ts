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
} from "./retriever";
import {
  askCopilotAdaptive,
  refineMarkdownWithInstruction,
  mergeTwoMarkdownArticles,
  generateMindmapFromArticle,
} from "./copilot-client";
import {
  extractDiseaseName,
  extractTopicIntent,
  isDetailRequest,
} from "./medical-vocab";

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
});

const ImageRequestSchema = z.object({
  disease_name: z.string().min(2),
  limit: z.number().int().min(1).max(10).default(3),
});

const LibraryGenerateSchema = z.object({
  extra_prompt: z.string().optional(),
  top_k: z.number().int().min(3).max(20).default(10),
  image_limit: z.number().int().min(1).max(10).default(5),
});

const LibraryPreviewSchema = LibraryGenerateSchema.extend({
  combine_with_existing: z.boolean().default(false),
  combine_mode: z.enum(["append", "replace"]).default("replace"),
});

const LibraryRefineSchema = z.object({
  instruction: z.string().min(3),
});

const LibraryPatchContentSchema = z.object({
  markdown: z.string().min(1),
  preview_commit: z.boolean().default(false),
});

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
): Promise<{
  query: string;
  evidence: ChunkRecord[];
  images: Record<string, unknown>[];
  draft_answer: Record<string, unknown>;
  markdown_candidate: string;
}> {
  const query = extraPrompt ? `${diseaseName}. ${extraPrompt}` : diseaseName;
  const evidence = await searchChunks(env, query, topK);
  const rawImages = await relatedImages(env, query, evidence, imageLimit);
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

  const detectedDisease = extractDiseaseName(payload.disease_name);
  const detectedIntent = extractTopicIntent(payload.disease_name);
  const det = isDetailRequest(payload.disease_name);

  const evidence = await searchChunks(
    c.env,
    payload.disease_name,
    payload.top_k,
    history.length > 0 ? history : undefined,
  );

  let images: Record<string, unknown>[] = [];
  if (payload.include_images) {
    const raw = await relatedImages(c.env, payload.disease_name, evidence, 3);
    images = raw.map((img) => ({
      source_name: img.source_name,
      page_no: img.page_no,
      heading: img.heading,
      alt_text: img.alt_text,
      image_url: img.storage_url ?? img.image_url ?? "",
    }));
  }

  let answer: Record<string, unknown>;
  if (c.env.GITHUB_TOKEN) {
    answer = (await askCopilotAdaptive(
      payload.disease_name,
      evidence,
      c.env.GITHUB_TOKEN,
      history.length > 0 ? history : undefined,
    )) as unknown as Record<string, unknown>;
  } else {
    answer = synthesizeFallback(payload.disease_name, evidence) as unknown as Record<string, unknown>;
  }

  return c.json({
    query: payload.disease_name,
    query_analysis: {
      detected_disease: detectedDisease,
      detected_intent: detectedIntent,
      is_detail_request: det,
    },
    detail_level: payload.detail_level,
    evidence_count: evidence.length,
    evidence,
    draft_answer: answer,
    images,
    note: "Gunakan draft_answer sebagai basis penjelasan grounded.",
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// ROUTE: POST /get_related_images
// ─────────────────────────────────────────────────────────────────────────────

app.post("/get_related_images", async (c) => {
  const payload = await parseBody(c, ImageRequestSchema);
  if (!payload) return c.json({ error: "Invalid request body" }, 422);

  const images = await relatedImages(c.env, payload.disease_name, [], payload.limit);
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
      const la = d["library_article"] as Array<Record<string, unknown>>;
      const status = la?.[0]?.["status"] as string;
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
    const la = (dr["library_article"] as Array<Record<string, unknown>>)?.[0];
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

  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
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
  );

  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
  const markdownBase = (la?.["content_markdown"] as string) ?? "";
  const mode =
    payload.combine_with_existing && payload.combine_mode === "append" ? "append" : "replace";
  const markdownCombined = combinePreviewMarkdown(markdownBase || null, gen.markdown_candidate, mode);

  return c.json({
    ok: true,
    markdown_base: markdownBase,
    markdown_candidate: gen.markdown_candidate,
    markdown_combined: markdownCombined,
    draft_answer: gen.draft_answer,
    evidence_count: gen.evidence.length,
    evidence: gen.evidence,
    images: gen.images,
    preview_note:
      mode === "append"
        ? "Gabungan artikel lama + pembaruan. Belum disimpan."
        : "Kandidat baru saja (ganti penuh). Belum disimpan.",
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
  );

  const isGrounded = (gen.draft_answer as Record<string, unknown>)["grounded"] !== false;
  const status = gen.evidence.length === 0 || !isGrounded ? "draft" : "published";
  const hash = await contentHash(gen.markdown_candidate);

  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
  const prevMeta = (la?.["meta"] as Record<string, unknown>) ?? {};
  const ver = ((prevMeta["version"] as number) ?? 0) + 1;

  const newMeta = {
    version: ver,
    disease_name: bundle["name"],
    catalog_no: bundle["catalog_no"],
    stase_slug: slug,
    generated_at: new Date().toISOString(),
    extra_prompt: payload.extra_prompt ?? null,
    last_operation: "generate",
    images: gen.images,
  };

  await supabase.from("library_article").upsert(
    {
      catalog_id: catalogId,
      status,
      content_markdown: gen.markdown_candidate,
      meta: newMeta,
      content_hash: hash,
      updated_at: new Date().toISOString(),
    },
    { onConflict: "catalog_id" },
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

  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
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
  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
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

  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
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

  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
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

  const la = (bundle["library_article"] as Array<Record<string, unknown>>)?.[0];
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
