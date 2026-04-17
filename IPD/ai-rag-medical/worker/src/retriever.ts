// Medical RAG — Supabase retrieval client
// Replaces: retriever.py (search_chunks, related_images, get_knowledge_graph)

import { createClient, SupabaseClient } from "@supabase/supabase-js";
import type { Env, ChunkRecord, ImageRecord } from "./types";
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
} from "./medical-vocab";

export function getSupabase(env: Env): SupabaseClient {
  return createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY, {
    auth: { persistSession: false },
  });
}

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
    const embeddingResult = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
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
): Promise<ChunkRecord[]> {
  const supabase = getSupabase(env);
  const enrichedQuery = enrichQueryFromHistory(query, chatHistory);
  const detectedDisease = extractDiseaseName(enrichedQuery);
  const isDetail = isDetailRequest(enrichedQuery);
  const intent = extractTopicIntent(enrichedQuery);

  const effectiveTopK = isDetail ? Math.max(topK * 2, 16) : topK;

  // Build sub-queries (multi-query decomposition)
  let queriesToRun: string[];
  if (isDetail || (detectedDisease && !intent)) {
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

  const filtered = filterByDiseaseRelevance(deduped, detectedDisease);
  const pruned = pruneRedundantChunks(filtered, 0.6);
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
