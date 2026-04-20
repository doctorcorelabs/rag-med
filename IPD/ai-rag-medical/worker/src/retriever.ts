// Medical RAG — Supabase retrieval client
// Replaces: retriever.py (search_chunks, related_images, get_knowledge_graph)

import { createClient, SupabaseClient } from "@supabase/supabase-js";
import type { Env, ChunkRecord, ImageRecord } from "./types";
import { WORKERS_EMBEDDING_MODEL } from "./indexer";
import {
  extractDiseaseName,
  extractTopicIntent,
  isDetailRequest,
  enrichQueryFromHistory,
  getExpandedTerms,
  reciprocalRankFusion,
  pruneRedundantChunks,
  MEDICAL_SYNONYMS,
  DISEASE_KEYWORDS,
  CLINICAL_ORDER,
  resolveRetrievalMode,
} from "./medical-vocab";

export function getSupabase(env: Env): SupabaseClient {
  return createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY, {
    auth: { persistSession: false },
  });
}

// ── Retrieval mode constants ──────────────────────────────────────────────────
const DEFAULT_TOP_K_EXHAUSTIVE = 80;
const MAX_TOP_K_EXHAUSTIVE = 200;

// ─────────────────────────────────────────────────────────────────────────────
// Core search: Hybrid BM25 (PostgreSQL FTS) + Vector (pgvector via Workers AI)
// ─────────────────────────────────────────────────────────────────────────────

async function vectorSearch(
  supabase: SupabaseClient,
  env: Env,
  query: string,
  topK: number,
  filterSection?: string,
  staseSlug?: string,
): Promise<ChunkRecord[]> {
  try {
    // Use Cloudflare Workers AI to generate embedding (replaces sentence-transformers)
    const embeddingResult = await env.AI.run(WORKERS_EMBEDDING_MODEL, {
      text: [query],
    }) as unknown as { data: number[][] };
    const embedding = embeddingResult.data[0];

    if (!embedding) return [];

    // Call pgvector RPC function
    const params: Record<string, unknown> = {
      query_embedding: embedding,
      match_count: topK,
    };
    if (filterSection) params.filter_section = filterSection;
    if (staseSlug) params.stase_slug = staseSlug;

    const { data, error } = await supabase.rpc("search_chunks_vector", params);

    if (error || !data) return [];

    return (data as ChunkRecord[]).map((row) => ({
      ...row,
      vector_score: row.similarity,
    }));
  } catch (e) {
    console.error("[vectorSearch] error:", e);
    return [];
  }
}

async function ftsSearch(
  supabase: SupabaseClient,
  query: string,
  topK: number,
  staseSlug?: string,
): Promise<ChunkRecord[]> {
  const expandedTerms = getExpandedTerms(query);

  const params: Record<string, unknown> = {
    query_text: query,
    match_count: topK * 2,
    expanded_terms: expandedTerms.length > 0 ? expandedTerms : null,
  };
  if (staseSlug) params.stase_slug = staseSlug;

  const { data, error } = await supabase.rpc("search_chunks_fts", params);

  if (error || !data) {
    // Fallback: direct ILIKE (with optional stase filter)
    let q = supabase
      .from("chunks")
      .select("id,source_name,page_no,heading,content,disease_tags,section_category,parent_heading,content_type,chunk_index")
      .or(expandedTerms.slice(0, 5).map((t) => `content.ilike.%${t}%`).join(","));
    if (staseSlug) q = q.eq("stase_slug", staseSlug);
    const { data: fallback } = await q.limit(topK);
    return (fallback ?? []) as ChunkRecord[];
  }

  return (data as ChunkRecord[]).map((row) => ({
    ...row,
    fts_rank: row.fts_rank,
  }));
}

function hierarchicalSort(
  results: ChunkRecord[],
  intentCategory: string | null,
  topK: number,
): ChunkRecord[] {
  if (!intentCategory) return results.slice(0, topK);
  const priority = results.filter((r) => r.section_category === intentCategory);
  const others = results.filter((r) => r.section_category !== intentCategory);
  return [...priority, ...others].slice(0, topK);
}

function filterByDiseaseRelevance(
  results: ChunkRecord[],
  diseaseName: string | null,
): ChunkRecord[] {
  if (!diseaseName) return results;

  const acceptable = new Set([diseaseName.toLowerCase()]);
  const synonyms = MEDICAL_SYNONYMS[diseaseName.toLowerCase()] ?? [];
  synonyms.forEach((s) => acceptable.add(s.toLowerCase()));

  const otherDiseases = new Set(
    DISEASE_KEYWORDS.map((k) => k.toLowerCase()).filter((k) => !acceptable.has(k)),
  );

  const GENERIC_HEADINGS = new Set([
    "patofisiologi", "patogenesis", "definisi", "etiologi",
    "tatalaksana", "tata laksana", "manifestasi klinis",
    "diagnosis", "komplikasi", "prognosis", "general",
    "pemeriksaan fisik", "pemeriksaan penunjang",
    "anamnesis", "faktor risiko",
  ]);

  return results.filter((item) => {
    const headingLower = item.heading.toLowerCase();
    const contentLower = item.content.toLowerCase().slice(0, 400);
    const combined = `${headingLower} ${contentLower}`;

    const mentionsTarget = [...acceptable].some((term) => combined.includes(term));
    const headingMentionsOther = [...otherDiseases].some(
      (d) => d.length > 2 && headingLower.includes(d),
    );
    const headingIsGeneric = GENERIC_HEADINGS.has(headingLower.replace(/:$/, "").trim());

    if (mentionsTarget) return true;
    if (headingMentionsOther) return false;
    if (headingIsGeneric && !mentionsTarget) return false;
    return true;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Public: searchChunks — full hybrid RAG pipeline
// ─────────────────────────────────────────────────────────────────────────────

export async function searchChunks(
  env: Env,
  query: string,
  topK = 8,
  chatHistory?: Array<{ role: string; content: string }>,
  staseSlug?: string,
  retrievalMode?: "relevant" | "exhaustive" | null,
): Promise<ChunkRecord[]> {
  const supabase = getSupabase(env);
  const mode = resolveRetrievalMode(query, retrievalMode);
  const isExhaustive = mode === "exhaustive";

  const enrichedQuery = enrichQueryFromHistory(query, chatHistory);
  const detectedDisease = extractDiseaseName(enrichedQuery);
  const isDetail = isDetailRequest(enrichedQuery);
  const intent = extractTopicIntent(enrichedQuery);

  // Adaptive top_k — exhaustive mode takes precedence
  let effectiveTopK: number;
  if (isExhaustive) {
    effectiveTopK = Math.min(Math.max(topK, DEFAULT_TOP_K_EXHAUSTIVE), MAX_TOP_K_EXHAUSTIVE);
  } else if (isDetail) {
    effectiveTopK = Math.max(topK * 2, 16);
  } else {
    effectiveTopK = topK;
  }

  // Build sub-queries (multi-query decomposition)
  // In exhaustive mode, run a single broad query to maximise coverage.
  let queriesToRun: string[];
  if (isExhaustive) {
    queriesToRun = [enrichedQuery];
  } else if (isDetail || (detectedDisease && !intent)) {
    const base = detectedDisease ?? enrichedQuery;
    queriesToRun = [
      `${base} definisi etiologi patogenesis`,
      `${base} anamnesis gejala klinis pemeriksaan fisik`,
      `${base} diagnosis tatalaksana dosis obat algoritma`,
      `${base} komplikasi prognosis`,
    ];
  } else {
    queriesToRun = [enrichedQuery];
  }

  // Execute all sub-queries in parallel (with stase filter)
  const [bm25Results, vectorResults] = await Promise.all([
    Promise.all(queriesToRun.map((q) => ftsSearch(supabase, q, effectiveTopK, staseSlug))).then((r) =>
      r.flat(),
    ),
    Promise.all(
      queriesToRun.map((q) => vectorSearch(supabase, env, q, effectiveTopK, undefined, staseSlug)),
    ).then((r) => r.flat()),
  ]);

  // Hybrid fusion
  const merged =
    vectorResults.length > 0
      ? reciprocalRankFusion(bm25Results as unknown as Record<string, unknown>[], vectorResults as unknown as Record<string, unknown>[])
      : bm25Results;

  // Deduplicate → disease filter → prune redundant → hierarchical sort
  const seen = new Set<number>();
  const deduped = (merged as ChunkRecord[]).filter((r) => {
    if (!r.id || seen.has(r.id)) return false;
    seen.add(r.id);
    return true;
  });

  // Disease relevance gating — skip in exhaustive mode to preserve full coverage
  const filtered = isExhaustive ? deduped : filterByDiseaseRelevance(deduped, detectedDisease);

  // Use a looser Jaccard threshold in exhaustive mode to avoid discarding
  // legitimately distinct chunks from a broad catalog search.
  const pruneThreshold = isExhaustive ? 0.85 : 0.6;
  const pruned = pruneRedundantChunks(filtered, pruneThreshold);
  const sorted = hierarchicalSort(pruned, intent, effectiveTopK * queriesToRun.length);

  return sorted.slice(0, effectiveTopK * queriesToRun.length);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public: relatedImages
// ─────────────────────────────────────────────────────────────────────────────

export async function relatedImages(
  env: Env,
  query: string,
  evidence: ChunkRecord[],
  limit = 3,
  staseSlug?: string,
): Promise<ImageRecord[]> {
  const supabase = getSupabase(env);
  const diseaseName = extractDiseaseName(query);
  let searchTerms: string[];

  if (diseaseName) {
    searchTerms = [diseaseName, ...(MEDICAL_SYNONYMS[diseaseName.toLowerCase()] ?? [])];
  } else {
    searchTerms = getExpandedTerms(query)
      .filter((t) => t.length > 2)
      .slice(0, 5);
  }

  if (searchTerms.length === 0) return [];

  const intent = extractTopicIntent(query);
  const intentKeywords =
    intent === "Diagnosis" || intent === "Tatalaksana"
      ? ["alur", "diagnosis", "tatalaksana", "algoritma", "skema", "bagan"]
      : [];

  const filters = searchTerms
    .map((t) => `heading.ilike.%${t}%,alt_text.ilike.%${t}%,nearby_text.ilike.%${t}%`)
    .join(",");

  // Phase 1: Evidence-co-located images
  let results: ImageRecord[] = [];

  if (evidence.length > 0) {
    const sourcePairs = new Set<string>();
    for (const item of evidence) {
      for (const pn of [item.page_no - 1, item.page_no, item.page_no + 1]) {
        sourcePairs.add(`${item.source_name}::${pn}`);
      }
    }

    let q = supabase
      .from("images")
      .select("source_name,page_no,heading,alt_text,image_ref,image_abs_path,storage_url,nearby_text")
      .or(filters);
    if (staseSlug) q = q.eq("stase_slug", staseSlug);

    const { data } = await q.limit(limit * 5);

    if (data) {
      results = (data as ImageRecord[]).filter((img) =>
        sourcePairs.has(`${img.source_name}::${img.page_no}`),
      );
    }
  }

  // Phase 2: Global fallback (same stase filter)
  if (results.length === 0) {
    let q = supabase
      .from("images")
      .select("source_name,page_no,heading,alt_text,image_ref,image_abs_path,storage_url,nearby_text")
      .or(filters);
    if (staseSlug) q = q.eq("stase_slug", staseSlug);
    const { data } = await q.limit(limit * 5);
    results = (data as ImageRecord[]) ?? [];
  }

  if (results.length === 0) return [];

  // Score by intent keywords + deduplicate by image_abs_path
  const scoredMap = new Map<string, { score: number; img: ImageRecord }>();
  for (const img of results) {
    const key = img.image_abs_path ?? img.image_ref;
    if (scoredMap.has(key)) continue;
    const combined =
      `${img.heading} ${img.alt_text ?? ""} ${img.nearby_text ?? ""}`.toLowerCase();
    const score = intentKeywords.reduce((acc, kw) => acc + (combined.includes(kw) ? 2 : 0), 1);
    scoredMap.set(key, { score, img });
  }

  // Resolve image URL: prefer R2 storage_url, fall back to abs path
  const r2Base = env.R2_PUBLIC_BASE_URL ?? "";

  return [...scoredMap.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(({ img }) => {
      let imageUrl = img.storage_url ?? "";
      if (!imageUrl && r2Base && img.image_ref) {
        imageUrl = `${r2Base}/${img.image_ref}`;
      }
      return { ...img, image_url: imageUrl };
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Public: getKnowledgeGraph
// ─────────────────────────────────────────────────────────────────────────────

export async function getKnowledgeGraph(
  env: Env,
  diseaseName: string,
  maxNodes = 40,
): Promise<{ nodes: unknown[]; edges: unknown[]; disease: string }> {
  const supabase = getSupabase(env);
  const searchTerms = [
    diseaseName.toLowerCase(),
    ...(MEDICAL_SYNONYMS[diseaseName.toLowerCase()] ?? []),
  ];

  const orFilter = searchTerms
    .map((t) => `source_disease.ilike.%${t}%`)
    .join(",");

  const { data } = await supabase
    .from("graph_edges")
    .select("source_disease,relation,target_node,target_type")
    .or(orFilter)
    .limit(maxNodes * 3);

  if (!data || data.length === 0) {
    return { nodes: [], edges: [], disease: diseaseName };
  }

  const nodesMap = new Map<string, { id: string; label: string; type: string; group: number }>();
  const edges: Array<{ source: string; target: string; relation: string }> = [];

  const GROUP_MAP: Record<string, number> = {
    Definisi: 1, Etiologi: 2, Patogenesis: 3,
    Manifestasi_Klinis: 4, Diagnosis: 5, Tatalaksana: 6,
    Komplikasi: 7, Prognosis: 8,
  };

  for (const row of data) {
    const srcId = row.source_disease.toLowerCase();
    if (!nodesMap.has(srcId)) {
      nodesMap.set(srcId, { id: srcId, label: row.source_disease, type: "disease", group: 0 });
    }
    const tgtId = row.target_node.toLowerCase();
    if (!nodesMap.has(tgtId)) {
      nodesMap.set(tgtId, {
        id: tgtId,
        label: row.target_node,
        type: row.target_type,
        group: GROUP_MAP[row.relation] ?? 9,
      });
    }
    edges.push({ source: srcId, target: tgtId, relation: row.relation });

    if (nodesMap.size >= maxNodes) break;
  }

  return {
    nodes: [...nodesMap.values()],
    edges,
    disease: diseaseName,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: synthesize fallback (no AI token)
// ─────────────────────────────────────────────────────────────────────────────

export async function getTopicsFromDb(
  env: Env,
  staseSlug?: string,
  sourceFilter?: string,
): Promise<any> {
  const supabase = getSupabase(env);

  const HEADING_NOISE = [
    "definisi", "etiologi", "patofisiologi", "patogenesis",
    "manifestasi klinis", "diagnosis", "tatalaksana", "tata laksana",
    "komplikasi", "prognosis", "pemeriksaan fisik", "pemeriksaan penunjang",
    "anamnesis", "faktor risiko", "diagnosis banding", "ringkasan",
    "daftar isi", "pendahuluan", "referensi", "daftar pustaka",
    "image from page folder", "general",
  ];

  let query = supabase
    .from("chunks")
    .select("source_name, heading, section_category, stase_slug")
    .not("heading", "in", `(${HEADING_NOISE.map((h) => `"${h}"`).join(",")})`);

  if (staseSlug) {
    query = query.eq("stase_slug", staseSlug);
  }
  if (sourceFilter) {
    query = query.ilike("source_name", `%${sourceFilter}%`);
  }

  // Supabase doesn't support SELECT DISTINCT directly via JS client in a clean way for multiple cols
  // without RPC. If RPC is not available, we can deduplicate in JS.
  const { data, error } = await query.order("source_name").order("heading");

  if (error || !data) return { sources: [], total_topics: 0, source_count: 0 };

  const grouped: Record<string, any> = {};
  for (const r of data as any[]) {
    const h = (r.heading || "").trim();
    // Filter noise: ignore very short headings, single digits, or pure symbols
    if (h.length <= 2 || /^\d+$/.test(h) || /^[^a-zA-Z0-9]+$/.test(h)) {
      continue;
    }

    const src = r.source_name;
    if (!grouped[src]) {
      grouped[src] = {
        source_name: src,
        stase_slug: r.stase_slug,
        topics: [],
      };
    }
    // Dedup heading per source
    const existing = new Set(grouped[src].topics.map((t: any) => t.heading));
    if (!existing.has(h)) {
      grouped[src].topics.push({
        heading: h,
        section_category: r.section_category,
      });
    }
  }

  const sources = Object.values(grouped);
  let totalTopics = 0;
  for (const s of sources as any[]) {
    s.topic_count = s.topics.length;
    totalTopics += s.topic_count;
  }

  return {
    sources,
    total_topics: totalTopics,
    source_count: sources.length,
  };
}

export function extractDiseaseListFromChunks(chunks: ChunkRecord[]): any[] {
  const GENERIC_HEADINGS = new Set([
    "patofisiologi", "patogenesis", "definisi", "etiologi", "tatalaksana",
    "tata laksana", "manifestasi klinis", "manifestasi", "diagnosis",
    "komplikasi", "prognosis", "general", "pemeriksaan fisik",
    "pemeriksaan penunjang", "anamnesis", "faktor risiko", "etiologi dan faktor risiko",
    "komplikasi dan prognosis", "ringkasan klinis", "penunjang", "farmakologi",
    "image from page folder",
  ]);

  const seen = new Map<string, any>();

  for (const chunk of chunks) {
    const evidenceItem = {
      source_name: chunk.source_name,
      page_no: chunk.page_no,
      heading: chunk.heading,
    };

    const candidates: string[] = [];

    // 1. disease_tags
    if (chunk.disease_tags) {
      const tags = chunk.disease_tags.split(/[,;|]/);
      for (const tag of tags) {
        const t = tag.trim();
        if (t) candidates.push(t);
      }
    }

    // 2. heading (exclude generic clinical names)
    const heading = (chunk.heading || "").trim();
    if (heading && !GENERIC_HEADINGS.has(heading.toLowerCase().replace(/:$/, ""))) {
      candidates.push(heading);
    }

    // 3. source_name as fallback
    if (candidates.length === 0 && chunk.source_name) {
      candidates.push(chunk.source_name);
    }

    for (const cand of candidates) {
      const norm = cand.toLowerCase().trim();
      if (!norm || norm.length < 3) continue;

      if (seen.has(norm)) {
        const entry = seen.get(norm);
        const evKey = `${evidenceItem.source_name}:${evidenceItem.page_no}`;
        const existingKeys = new Set(entry.evidence.map((e: any) => `${e.source_name}:${e.page_no}`));
        if (!existingKeys.has(evKey)) {
          entry.evidence.push(evidenceItem);
        }
      } else {
        seen.set(norm, {
          name: cand,
          evidence: [evidenceItem],
        });
      }
    }
  }

  return [...seen.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export function synthesizeFallback(
  query: string,
  evidence: ChunkRecord[],
): Record<string, unknown> {
  const sorted = [...evidence].sort(
    (a, b) =>
      (CLINICAL_ORDER[a.section_category] ?? 99) -
      (CLINICAL_ORDER[b.section_category] ?? 99),
  );

  const sectionMap = new Map<string, string[]>();
  for (const chunk of sorted) {
    const key = chunk.section_category || "Ringkasan_Klinis";
    if (!sectionMap.has(key)) sectionMap.set(key, []);
    sectionMap.get(key)!.push(chunk.content.slice(0, 600));
  }

  const sections = [...sectionMap.entries()].map(([title, contents]) => ({
    title: title.replaceAll("_", " "),
    markdown: contents.join("\n\n"),
  }));

  const citations = [...new Set(sorted.map((c) => `${c.source_name} p.${c.page_no}`))];

  return {
    disease: query,
    sections,
    citations,
    grounded: true,
  };
}
