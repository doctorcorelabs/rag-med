import type { Env } from "./types";

export const WORKERS_EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5";

const HEADING_RE = /^(#{1,6})\s+(.+?)\s*$/gm;
const IMAGE_RE = /!\[(.*?)\]\((.*?)\)/g;
const WHITESPACE_RE = /\s+/g;
const TABLE_ROW_RE = /^\s*\|.+\|/m;
const NUMBERED_LIST_RE = /^\s*\d+[.)]\s+/m;
const DOSE_BLOCK_RE = /\d+\s*(?:mg|mcg|µg|unit|ml|g\/dl|g\/dL|mmol|mEq|IU|gram|tablet|kapsul|ampul|vial)/i;
const TOKEN_RE = /[a-zA-Z][a-zA-Z0-9/-]{2,}/g;

const SECTION_CATEGORY_MAP: Array<[string, string]> = [
  ["definisi", "Definisi"],
  ["pengertian", "Definisi"],
  ["batasan", "Definisi"],
  ["deskripsi", "Definisi"],
  ["etiologi", "Etiologi"],
  ["faktor risiko", "Etiologi"],
  ["penyebab", "Etiologi"],
  ["kausa", "Etiologi"],
  ["predisposisi", "Etiologi"],
  ["epidemiologi", "Etiologi"],
  ["insidensi", "Etiologi"],
  ["prevalensi", "Etiologi"],
  ["patogenesis", "Patogenesis"],
  ["patofisiologi", "Patogenesis"],
  ["mekanisme", "Patogenesis"],
  ["fisiologi", "Patogenesis"],
  ["imunopatologi", "Patogenesis"],
  ["anamnesis", "Anamnesis"],
  ["anamnesa", "Anamnesis"],
  ["riwayat", "Anamnesis"],
  ["keluhan utama", "Anamnesis"],
  ["manifestasi", "Manifestasi_Klinis"],
  ["gejala", "Manifestasi_Klinis"],
  ["klinis", "Manifestasi_Klinis"],
  ["keluhan", "Manifestasi_Klinis"],
  ["gambaran klinis", "Manifestasi_Klinis"],
  ["tanda klinis", "Manifestasi_Klinis"],
  ["simptom", "Manifestasi_Klinis"],
  ["sindrom", "Manifestasi_Klinis"],
  ["presentasi", "Manifestasi_Klinis"],
  ["gejala klinis", "Manifestasi_Klinis"],
  ["pemeriksaan fisik", "Pemeriksaan_Fisik"],
  ["physical exam", "Pemeriksaan_Fisik"],
  ["diagnosis", "Diagnosis"],
  ["pemeriksaan", "Diagnosis"],
  ["penunjang", "Diagnosis"],
  ["laboratorium", "Diagnosis"],
  ["radiologi", "Diagnosis"],
  ["kriteria", "Diagnosis"],
  ["gold standard", "Diagnosis"],
  ["baku emas", "Diagnosis"],
  ["staging", "Diagnosis"],
  ["klasifikasi", "Diagnosis"],
  ["grading", "Diagnosis"],
  ["tata laksana", "Tatalaksana"],
  ["tatalaksana", "Tatalaksana"],
  ["terapi", "Tatalaksana"],
  ["pengobatan", "Tatalaksana"],
  ["penatalaksanaan", "Tatalaksana"],
  ["farmakologi", "Tatalaksana"],
  ["obat", "Tatalaksana"],
  ["dosis", "Tatalaksana"],
  ["regimen", "Tatalaksana"],
  ["medikamentosa", "Tatalaksana"],
  ["non-farmakologi", "Tatalaksana"],
  ["nonfarmakologi", "Tatalaksana"],
  ["intervensi", "Tatalaksana"],
  ["protokol", "Tatalaksana"],
  ["penanganan", "Tatalaksana"],
  ["manajemen", "Tatalaksana"],
  ["komplikasi", "Komplikasi"],
  ["penyulit", "Komplikasi"],
  ["prognosis", "Prognosis"],
  ["luaran", "Prognosis"],
  ["mortalitas", "Prognosis"],
  ["survival", "Prognosis"],
  ["angka kematian", "Prognosis"],
  ["pencegahan", "Prognosis"],
];

export type SourcePageInput = {
  source_name: string;
  page_no: number;
  markdown: string;
  stase_slug?: string;
  markdown_path?: string;
};

export type ParsedChunk = {
  source_name: string;
  page_no: number;
  heading: string;
  content: string;
  disease_tags: string;
  markdown_path: string;
  checksum: string;
  section_category: string;
  parent_heading: string;
  chunk_index: number;
  total_chunks: number;
  heading_level: number;
  content_type: string;
  stase_slug: string;
};

export type ParsedImage = {
  source_name: string;
  page_no: number;
  alt_text: string;
  image_ref: string;
  image_abs_path: string;
  heading: string;
  nearby_text: string;
  markdown_path: string;
  checksum: string;
  stase_slug: string;
};

export type ParsedGraphEdge = {
  source_disease: string;
  relation: string;
  target_node: string;
  target_type: string;
  source_name: string;
  page_no: number;
  stase_slug: string;
};

export type ParsedPage = {
  chunks: ParsedChunk[];
  images: ParsedImage[];
  graph_edges: ParsedGraphEdge[];
};

function checksum(value: string): string {
  return value;
}

function normalizeText(value: string): string {
  return value.replace(/&nbsp;/g, " ").replace(WHITESPACE_RE, " ").trim();
}

function detectSectionCategory(heading: string): string {
  const lower = heading.toLowerCase();
  for (const [keyword, category] of [...SECTION_CATEGORY_MAP].sort((a, b) => b[0].length - a[0].length)) {
    if (lower.includes(keyword)) return category;
  }
  return "Ringkasan_Klinis";
}

function deriveDiseaseTags(heading: string, text: string): string {
  const seeds = `${heading} ${text.slice(0, 300)}`.toLowerCase();
  const tokens = seeds.match(TOKEN_RE) ?? [];
  const unique: string[] = [];
  for (const token of tokens) {
    if (!unique.includes(token)) unique.push(token);
    if (unique.length === 12) break;
  }
  return unique.join(" ");
}

function splitSections(markdown: string): Array<{ heading: string; content: string; level: number; parent_heading: string }> {
  const matches = [...markdown.matchAll(HEADING_RE)];
  if (matches.length === 0) {
    return [{ heading: "General", content: markdown, level: 1, parent_heading: "" }];
  }

  const sections: Array<{ heading: string; content: string; level: number; parent_heading: string }> = [];
  const parentStack: Array<{ level: number; heading: string }> = [];

  matches.forEach((match, index) => {
    const level = match[1].length;
    const heading = normalizeText(match[2] ?? "") || "General";
    const start = (match.index ?? 0) + match[0].length;
    const end = index + 1 < matches.length ? (matches[index + 1].index ?? markdown.length) : markdown.length;
    const content = markdown.slice(start, end);

    while (parentStack.length > 0 && parentStack[parentStack.length - 1].level >= level) {
      parentStack.pop();
    }
    const parentHeading = parentStack[parentStack.length - 1]?.heading ?? "";
    parentStack.push({ level, heading });
    sections.push({ heading, content, level, parent_heading: parentHeading });
  });

  return sections;
}

function splitIntoSemanticBlocks(text: string): Array<{ text: string; block_type: "prose" | "table" | "list" | "dose_block" }> {
  const lines = text.split("\n");
  const blocks: Array<{ text: string; block_type: "prose" | "table" | "list" | "dose_block" }> = [];
  let currentLines: string[] = [];
  let currentType: "prose" | "table" | "list" = "prose";

  const flush = () => {
    const body = currentLines.join("\n").trim();
    if (body) blocks.push({ text: body, block_type: currentType });
    currentLines = [];
    currentType = "prose";
  };

  for (const line of lines) {
    const isTable = TABLE_ROW_RE.test(line);
    const isNumbered = NUMBERED_LIST_RE.test(line);

    if (isTable) {
      if (currentType !== "table") {
        flush();
        currentType = "table";
      }
      currentLines.push(line);
      continue;
    }

    if (isNumbered) {
      if (currentType !== "list") {
        flush();
        currentType = "list";
      }
      currentLines.push(line);
      continue;
    }

    if (currentType === "table" || currentType === "list") {
      if (!line.trim()) {
        currentLines.push(line);
        flush();
      } else {
        flush();
        currentLines.push(line);
        currentType = "prose";
      }
      continue;
    }

    currentLines.push(line);
  }

  flush();

  return blocks.map((block) => {
    if (block.block_type === "prose" && DOSE_BLOCK_RE.test(block.text)) {
      return { ...block, block_type: "dose_block" as const };
    }
    return block;
  });
}

function chunkText(
  value: string,
  maxChars = 1800,
  tableMax = 2500,
  heading = "",
  sourceName = "",
  pageNo = 0,
): Array<{ chunk_content: string; content_type: string }> {
  const text = value.trim();
  if (!text) return [];

  const metadataPrefix = `[Konteks: ${sourceName} - Hal ${pageNo} | Topik Utama: ${heading}]\n`;
  const blocks = splitIntoSemanticBlocks(text);
  const chunks: Array<{ chunk_content: string; content_type: string }> = [];

  let currentChunk = "";
  let currentType = "prose";
  let lastParagraph = "";

  const pushChunk = (content: string, contentType: string) => {
    const normalized = contentType === "prose" ? normalizeText(content) : content.trim();
    if (normalized) {
      chunks.push({ chunk_content: metadataPrefix + normalized, content_type: contentType });
    }
  };

  for (const block of blocks) {
    const budget = block.block_type === "table" ? tableMax : maxChars;
    const maxPayload = budget - metadataPrefix.length;

    if (block.block_type === "table" || block.block_type === "list" || block.block_type === "dose_block") {
      if (currentChunk.trim()) {
        pushChunk(currentChunk, currentType);
        currentChunk = "";
        lastParagraph = "";
      }

      if (block.text.length <= maxPayload) {
        pushChunk(block.text, block.block_type);
      } else {
        const lines = block.text.split("\n");
        let sub = "";
        for (const line of lines) {
          if (sub.length + line.length + 1 > maxPayload && sub) {
            pushChunk(sub, block.block_type);
            sub = "";
          }
          sub += `${line}\n`;
        }
        if (sub.trim()) pushChunk(sub, block.block_type);
      }
      continue;
    }

    const proseMaxPayload = maxChars - metadataPrefix.length;
    const paragraphs = block.text.split(/\n\s*\n/);
    currentType = "prose";

    for (const paragraph of paragraphs) {
      if (currentChunk.length + paragraph.length > proseMaxPayload && currentChunk) {
        pushChunk(currentChunk, currentType);
        currentChunk = lastParagraph.length < proseMaxPayload * 0.25 ? `${lastParagraph}\n\n` : "";
      }

      if (paragraph.length >= proseMaxPayload) {
        if (currentChunk) {
          pushChunk(currentChunk, currentType);
          currentChunk = "";
        }
        const lines = paragraph.split("\n");
        let sub = "";
        for (const line of lines) {
          if (sub.length + line.length < proseMaxPayload) {
            sub += `${line}\n`;
          } else {
            if (sub) pushChunk(sub, currentType);
            sub = `${line}\n`;
          }
        }
        currentChunk = `${sub}\n`;
        lastParagraph = "";
        continue;
      }

      currentChunk += `${paragraph}\n\n`;
      lastParagraph = paragraph;
    }
  }

  if (currentChunk.trim()) {
    pushChunk(currentChunk, currentType);
  }

  return chunks;
}

function extractGraphEdgesFromChunks(chunks: ParsedChunk[]): ParsedGraphEdge[] {
  const edges: ParsedGraphEdge[] = [];
  for (const chunk of chunks) {
    const headingWords = chunk.heading.match(/[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9/ ]{2,}/g) ?? [];
    for (const term of headingWords.slice(0, 3)) {
      const clean = term.trim();
      if (clean && clean.toLowerCase() !== chunk.source_name.toLowerCase()) {
        edges.push({
          source_disease: chunk.source_name,
          relation: chunk.section_category,
          target_node: clean,
          target_type: "concept",
          source_name: chunk.source_name,
          page_no: chunk.page_no,
          stase_slug: chunk.stase_slug,
        });
      }
    }
  }
  return edges;
}

export function parsePageForIndexing(page: SourcePageInput): ParsedPage {
  const markdown = page.markdown.replace(/\r\n/g, "\n");
  const sections = splitSections(markdown);
  const markdownPath = page.markdown_path ?? `page-${page.page_no}/markdown.md`;
  const staseSlug = page.stase_slug ?? "ipd";

  const chunks: ParsedChunk[] = [];
  const images: ParsedImage[] = [];
  const foundImageRefs = new Set<string>();

  for (const section of sections) {
    const rawMatches = [...section.content.matchAll(IMAGE_RE)];
    for (const match of rawMatches) {
      const altText = normalizeText(match[1] ?? "");
      const imageRef = (match[2] ?? "").trim();
      if (!imageRef) continue;
      foundImageRefs.add(imageRef);
      const nearby = normalizeText(section.content.replace(IMAGE_RE, " ")).slice(0, 300);
      images.push({
        source_name: page.source_name,
        page_no: page.page_no,
        alt_text: altText,
        image_ref: imageRef,
        image_abs_path: imageRef,
        heading: section.heading,
        nearby_text: nearby,
        markdown_path: markdownPath,
        checksum: checksum(`${markdownPath}|${section.heading}|${imageRef}|${nearby}`),
        stase_slug: staseSlug,
      });
    }

    const cleaned = section.content.replace(IMAGE_RE, " ");
    const sectionCategory = detectSectionCategory(section.heading);
    const sectionChunks = chunkText(cleaned, 1800, 2500, section.heading, page.source_name, page.page_no);

    sectionChunks.forEach((chunk, chunkIndex) => {
      chunks.push({
        source_name: page.source_name,
        page_no: page.page_no,
        heading: section.heading,
        content: chunk.chunk_content,
        disease_tags: deriveDiseaseTags(section.heading, chunk.chunk_content),
        markdown_path: markdownPath,
        checksum: checksum(`${markdownPath}|${section.heading}|${chunk.chunk_content}`),
        section_category: sectionCategory,
        parent_heading: section.parent_heading,
        chunk_index: chunkIndex,
        total_chunks: sectionChunks.length,
        heading_level: section.level,
        content_type: chunk.content_type,
        stase_slug: staseSlug,
      });
    });
  }

  if (!foundImageRefs.size) {
    // Cloud uploads do not have filesystem access, so there is no safe fallback image discovery.
  }

  return {
    chunks,
    images,
    graph_edges: extractGraphEdgesFromChunks(chunks),
  };
}

export async function embedTexts(env: Env, texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const result = (await env.AI.run(WORKERS_EMBEDDING_MODEL, {
    text: texts,
  })) as unknown as { data?: number[][]; embedding?: number[][] };
  const vectors = result.data ?? result.embedding ?? [];
  return vectors.filter((row): row is number[] => Array.isArray(row));
}
