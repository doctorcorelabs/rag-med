// Medical RAG — Copilot Client (TypeScript)
// Exact translation of copilot_client.py — all prompts preserved verbatim

import type { ChunkRecord, DraftAnswer, ImageRecord } from "./types";
import {
  extractDiseaseName,
  extractTopicIntent,
  isDetailRequest,
  CLINICAL_ORDER,
} from "./medical-vocab";

const COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token";
const COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions";
const EVIDENCE_REF_RE = /\[E(\d+)\]/g;

// ─────────────────────────────────────────────────────────────────────────────
// Token acquisition
// ─────────────────────────────────────────────────────────────────────────────

export async function getCopilotToken(githubToken: string): Promise<string> {
  const res = await fetch(COPILOT_TOKEN_URL, {
    headers: {
      Authorization: `token ${githubToken}`,
      "User-Agent": "GitHubCopilotChat/0.1.0",
      Accept: "application/json",
    },
  });
  if (!res.ok) throw new Error(`Failed to get Copilot token: ${res.status}`);
  const data = (await res.json()) as { token: string };
  return data.token;
}

// ─────────────────────────────────────────────────────────────────────────────
// Evidence formatting (XML-like for LLM referencing)
// ─────────────────────────────────────────────────────────────────────────────

function formatEvidenceStructured(
  evidence: ChunkRecord[],
): [string, ChunkRecord[]] {
  const sorted = [...evidence].sort(
    (a, b) =>
      (CLINICAL_ORDER[a.section_category] ?? 99) -
      (CLINICAL_ORDER[b.section_category] ?? 99),
  );

  const context = sorted
    .map(
      (item, idx) => `
<evidence id="${idx + 1}">
  <source>${item.source_name}</source>
  <page>${item.page_no}</page>
  <heading>${item.heading}</heading>
  <parent_heading>${item.parent_heading ?? ""}</parent_heading>
  <section_type>${item.section_category ?? "General"}</section_type>
  <content_type>${item.content_type ?? "prose"}</content_type>
  <content>
${item.content}
  </content>
</evidence>`,
    )
    .join("\n");

  return [context, sorted];
}

// ─────────────────────────────────────────────────────────────────────────────
// Citation resolution: [E1] → (Source, Hal N)
// ─────────────────────────────────────────────────────────────────────────────

function resolveEvidenceCitations(
  text: string,
  evidence: ChunkRecord[],
): string {
  return text.replace(EVIDENCE_REF_RE, (_match, numStr) => {
    const idx = parseInt(numStr, 10);
    if (idx >= 1 && idx <= evidence.length) {
      const item = evidence[idx - 1];
      return `(${item.source_name}, Hal ${item.page_no})`;
    }
    return _match;
  });
}

function extractUsedEvidenceIds(parsed: DraftAnswer): number[] {
  const used = new Set<number>();
  for (const sec of parsed.sections ?? []) {
    const md = sec.markdown ?? "";
    let m: RegExpExecArray | null;
    const re = /\[E(\d+)\]/g;
    while ((m = re.exec(md)) !== null) used.add(parseInt(m[1], 10));
  }
  const vlog = parsed.verification_log ?? "";
  let vm: RegExpExecArray | null;
  const vre = /\[E(\d+)\]/g;
  while ((vm = vre.exec(vlog)) !== null) used.add(parseInt(vm[1], 10));
  return [...used].sort((a, b) => a - b);
}

// ─────────────────────────────────────────────────────────────────────────────
// System prompt builder (verbatim from Python)
// ─────────────────────────────────────────────────────────────────────────────

function buildSystemPrompt(
  _query: string,
  isDetail: boolean,
  intentCategory: string | null,
  _diseaseName: string | null,
  evidenceCount: number,
): string {
  const topicNames: Record<string, string> = {
    Definisi: "Definisi",
    Etiologi: "Etiologi dan Faktor Risiko",
    Patogenesis: "Patogenesis dan Patofisiologi",
    Anamnesis: "Anamnesis",
    Manifestasi_Klinis: "Manifestasi Klinis",
    Pemeriksaan_Fisik: "Pemeriksaan Fisik",
    Diagnosis: "Diagnosis dan Pemeriksaan Penunjang",
    Tatalaksana: "Tatalaksana",
    Komplikasi: "Komplikasi dan Prognosis",
    Prognosis: "Prognosis",
  };

  let sectionInstruction: string;
  if (isDetail) {
    sectionInstruction = `Buatlah pembahasan KOMPREHENSIF yang mencakup section-section berikut (jika informasi tersedia di referensi):
1. Definisi
2. Etiologi dan Faktor Risiko
3. Patogenesis dan Patofisiologi
4. Anamnesis / Keluhan
5. Pemeriksaan Fisik
6. Pemeriksaan Penunjang
7. Diagnosis
8. Tatalaksana
9. Komplikasi dan Prognosis

Untuk setiap section, gali informasi SEDALAM mungkin dari referensi. Jika suatu section tidak memiliki data di referensi, JANGAN masukkan section tersebut sama sekali (jangan tulis "tidak tersedia").`;
  } else if (intentCategory) {
    const topic = topicNames[intentCategory] ?? intentCategory;
    sectionInstruction = `User bertanya SPESIFIK tentang: **${topic}**.

PENTING: Fokuskan jawaban HANYA pada topik "${topic}". Buatlah pembahasan yang MENDALAM dan DETAIL untuk topik tersebut saja.
- JANGAN menambahkan section lain yang tidak ditanyakan.
- Jika ada informasi diagram/bagan/alur di referensi, DESKRIPSIKAN secara naratif dan terstruktur.
- Buat jumlah section sesuai kebutuhan (bisa 1-3 section yang semuanya relevan dengan topik).
- Gunakan sub-bullet, penomoran, atau tabel untuk memperjelas.`;
  } else {
    sectionInstruction = `Analisis pertanyaan user dan buat section yang paling relevan.
- Jika pertanyaan umum tentang suatu penyakit, buat ringkasan klinis yang mencakup aspek-aspek utama yang tersedia di referensi.
- Jika pertanyaan spesifik, fokuskan pada topik yang diminta.
- JANGAN masukkan section yang tidak memiliki data di referensi.
- Jumlah section fleksibel: bisa 1 hingga 8 tergantung ketersediaan informasi.`;
  }

  return `Anda adalah Asisten Klinis (Medical RAG) profesional berbasis referensi terverifikasi.

PRINSIP UTAMA:
1. Jawab HANYA berdasarkan DOKUMEN REFERENSI yang diberikan dalam tag <evidence>.
2. Informasi dari referensi adalah SUMBER UTAMA. AI hanya bertugas menyusun dan memperjelas informasi tersebut agar lebih mudah dipahami.
3. KEMAMPUAN VISION (EKSTRAKSI PROTOKOL): Jika Anda menerima input gambar berupa flowchart/bagan tatalaksana, Anda WAJIB mengubah alur visual tersebut menjadi pedoman prosedural langkah-demi-langkah (IF-THEN-ELSE) secara berurutan.
4. RESOLUSI KONFLIK PEDOMAN: Jika menemukan perbedaan data antar referensi, buat baris "Perbandingan Pedoman".
5. CLINICAL REASONING ENGINE (LEVEL KONSULEN):
   - Anda HARUS menyertakan "Diagnosis Banding (DDx)" berdasarkan kemiripan gejala klinis, JIKA ada dalam referensi.
   - Anda HARUS memunculkan "Red Flags" atau kondisi gawat darurat yang wajib diwaspadai, JIKA disinggung dalam referensi.
6. AGENTIC SELF-REFLECTION: LLM WAJIB melakukan validasi internal sebelum menjawab! Buat "verification_log" di awal hasil JSON-mu, dan pastikan setiap angka dosis dan prosedur benar-benar tertera di evidence. Jangan berhalusinasi dosis!
7. Gunakan bahasa Indonesia formal medis. Tebalkan (**bold**) istilah medis, gunakan _italic_ untuk nama latin.

EVIDENCE COVERAGE CHECK:
- Anda menerima ${evidenceCount} evidence documents.
- Pastikan SETIAP evidence digunakan minimal 1x jika relevan.
- Di verification_log, sebutkan evidence mana saja yang Anda gunakan dan yang tidak relevan.

SISTEM CITATION GRANULAR:
- Setiap klaim/fakta medis WAJIB diakhiri dengan citation format [E1], [E2], dst. yang merujuk ke <evidence id="N">.
- Setiap kalimat yang menyebut angka, dosis, atau prosedur HARUS memiliki citation.
- Contoh: **Aspirin** diberikan dosis loading **160-320 mg** per oral [E3].
- Anda boleh menggabungkan citation: [E1][E3] atau [E2, E5].

${sectionInstruction}

FORMAT OUTPUT (RAW JSON VALID, TANPA markdown code block):
{
  "verification_log": "Catatan verifikasi: Evidence yang digunakan: [E1], [E2], ... Evidence tidak relevan: [EN] karena ...",
  "disease": "Nama Penyakit/Kondisi",
  "sections": [
    {
      "title": "Judul Section",
      "markdown": "Konten dalam **Markdown** dengan citation [E1] di setiap klaim.\\nGunakan:\\n- **Bold** untuk istilah penting\\n- _Italic_ untuk nama latin\\n- Bullet points untuk daftar\\n- Tabel | untuk data komparatif\\n- Penomoran untuk alur/tahapan"
    }
  ],
  "citations": ["Sumber p.Halaman"]
}

ATURAN KETAT:
- Field WAJIB "markdown" (bukan "points")
- Output HARUS berupa raw JSON valid
- Gunakan \\n untuk baris baru di dalam JSON string
- JANGAN masukkan section dengan konten "Tidak tersedia di referensi"
- Gunakan [E1], [E2] dst. untuk SETIAP klaim medis penting
`;
}

function buildExtractionPrompt(evidenceCount: number): string {
  return `Anda adalah mesin ekstraksi fakta medis. Tugas Anda HANYA mengekstrak fakta dari ${evidenceCount} evidence documents yang diberikan.

INSTRUKSI:
1. Baca SEMUA evidence yang diberikan dalam tag <evidence>.
2. Ekstrak SEMUA fakta penting per section klinis.
3. Setiap fakta HARUS disertai referensi [E1], [E2], dst.
4. JANGAN menambahkan informasi yang tidak ada di evidence.
5. JANGAN membuat narasi. Hanya buat bullet list fakta.

FORMAT OUTPUT (RAW JSON VALID):
{
  "extracted_facts": [
    {
      "section": "Nama Section (Definisi/Etiologi/Patogenesis/dst)",
      "facts": [
        "Fakta 1 dari evidence [E1]",
        "Fakta 2 dari evidence [E3]"
      ]
    }
  ],
  "evidence_used": [1, 2, 3],
  "evidence_unused": [4, 5]
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw Copilot API call
// ─────────────────────────────────────────────────────────────────────────────

async function callCopilot(
  copilotToken: string,
  messages: Array<{ role: string; content: unknown }>,
): Promise<string> {
  const body = {
    messages,
    model: "gpt-4.1",
    temperature: 0.1,
    stream: false,
  };

  const res = await fetch(COPILOT_CHAT_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${copilotToken}`,
      "Content-Type": "application/json",
      "User-Agent": "GitHubCopilotChat/0.1.0",
      "Editor-Version": "vscode/1.92.0",
      "Editor-Plugin-Version": "copilot-chat/0.18.0",
      "Openai-Organization": "github-copilot",
      "Openai-Intent": "conversation-panel",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) throw new Error(`Copilot API error: ${res.status}`);
  const data = (await res.json()) as { choices: Array<{ message: { content: string } }> };
  return data.choices[0].message.content;
}

function parseJsonResponse(raw: string): DraftAnswer {
  let content = raw.trim();
  if (content.startsWith("```json")) content = content.slice(7);
  if (content.startsWith("```")) content = content.slice(3);
  if (content.endsWith("```")) content = content.slice(0, -3);
  return JSON.parse(content.trim()) as DraftAnswer;
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-processing
// ─────────────────────────────────────────────────────────────────────────────

function postprocessResponse(parsed: DraftAnswer, sortedEvidence: ChunkRecord[]): DraftAnswer {
  // Migrate legacy "points" → "markdown"
  if (parsed.sections) {
    for (const sec of parsed.sections) {
      if (sec.points && !sec.markdown) {
        sec.markdown = sec.points.map((p) => `- ${p}`).join("\n");
        delete sec.points;
      }
    }
    parsed.sections = parsed.sections.filter(
      (sec) =>
        (sec.markdown ?? "").trim() &&
        !(sec.markdown ?? "").toLowerCase().includes("tidak tersedia") &&
        !(sec.markdown ?? "").toLowerCase().includes("belum tersedia") &&
        (sec.markdown ?? "").trim() !== "-",
    );
  }

  // Resolve citations
  const usedIds = extractUsedEvidenceIds(parsed);
  for (const sec of parsed.sections ?? []) {
    if (sec.markdown) {
      sec.markdown = resolveEvidenceCitations(sec.markdown, sortedEvidence);
    }
  }
  if (parsed.verification_log) {
    parsed.verification_log = resolveEvidenceCitations(
      parsed.verification_log,
      sortedEvidence,
    );
  }

  // Build real citations
  const seen = new Set<string>();
  const citations: string[] = [];
  for (const item of sortedEvidence) {
    const c = `${item.source_name} p.${item.page_no}`;
    if (!seen.has(c)) {
      seen.add(c);
      citations.push(c);
    }
  }
  parsed.citations = citations;

  // Evidence coverage
  const total = sortedEvidence.length;
  const unusedIds = Array.from({ length: total }, (_, i) => i + 1).filter(
    (i) => !usedIds.includes(i),
  );
  parsed.evidence_coverage = {
    total_evidence: total,
    used_evidence: usedIds,
    unused_evidence: unusedIds,
    coverage_percent: total > 0 ? Math.round((usedIds.length / total) * 1000) / 10 : 0,
  };
  parsed.grounded = true;

  return parsed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-pass synthesis
// ─────────────────────────────────────────────────────────────────────────────

async function singlePassSynthesis(
  copilotToken: string,
  query: string,
  userPrompt: string,
  sortedEvidence: ChunkRecord[],
  isDetail: boolean,
  intentCategory: string | null,
  diseaseName: string | null,
  chatHistory?: Array<{ role: string; content: string }>,
  images?: ImageRecord[],
): Promise<DraftAnswer> {
  const systemPrompt = buildSystemPrompt(
    query,
    isDetail,
    intentCategory,
    diseaseName,
    sortedEvidence.length,
  );
  const messages: Array<{ role: string; content: unknown }> = [
    { role: "system", content: systemPrompt },
  ];

  if (chatHistory) {
    for (const turn of chatHistory.slice(-6)) {
      if (turn.role === "user" || turn.role === "assistant") {
        messages.push({ role: turn.role, content: turn.content });
      }
    }
  }

  if (images && images.length > 0) {
    const userContent: Array<{ type: string; text?: string; image_url?: { url: string } }> = [
      { type: "text", text: userPrompt },
    ];
    for (const img of images) {
      const url = img.storage_url ?? img.image_abs_path ?? "";
      if (url.startsWith("http")) {
        userContent.push({ type: "image_url", image_url: { url } });
      }
    }
    messages.push({ role: "user", content: userContent });
  } else {
    messages.push({ role: "user", content: userPrompt });
  }

  const raw = await callCopilot(copilotToken, messages);
  return parseJsonResponse(raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Two-pass synthesis (for detail + many evidence chunks)
// ─────────────────────────────────────────────────────────────────────────────

async function twoPassSynthesis(
  copilotToken: string,
  query: string,
  userPrompt: string,
  contextText: string,
  sortedEvidence: ChunkRecord[],
  intentCategory: string | null,
  diseaseName: string | null,
  chatHistory?: Array<{ role: string; content: string }>,
  images?: ImageRecord[],
): Promise<DraftAnswer> {
  // Pass 1: Fact extraction
  const extractionPrompt = buildExtractionPrompt(sortedEvidence.length);
  const pass1Raw = await callCopilot(copilotToken, [
    { role: "system", content: extractionPrompt },
    { role: "user", content: userPrompt },
  ]);

  let factsText = "";
  try {
    const extracted = parseJsonResponse(pass1Raw) as unknown as {
      extracted_facts: Array<{ section: string; facts: string[] }>;
    };
    for (const section of extracted.extracted_facts ?? []) {
      factsText += `\n### ${section.section ?? "General"}\n`;
      for (const fact of section.facts ?? []) {
        factsText += `- ${fact}\n`;
      }
    }
  } catch {
    factsText = "(extraction failed — using raw evidence)";
  }

  // Pass 2: Narrative synthesis
  const systemPrompt = buildSystemPrompt(
    query,
    true,
    intentCategory,
    diseaseName,
    sortedEvidence.length,
  );
  const synthesisUser = `Query Klinis: ${query}\n\nFAKTA TEREKSTRAK DARI EVIDENCE (gunakan ini sebagai panduan, tapi tetap rujuk evidence asli):\n${factsText}\n\nEVIDENCE ASLI:\n${contextText}`;

  const pass2Messages: Array<{ role: string; content: unknown }> = [
    { role: "system", content: systemPrompt },
  ];
  if (chatHistory) {
    for (const turn of chatHistory.slice(-6)) {
      if (turn.role === "user" || turn.role === "assistant") {
        pass2Messages.push({ role: turn.role, content: turn.content });
      }
    }
  }
  if (images && images.length > 0) {
    const userContent: Array<{ type: string; text?: string; image_url?: { url: string } }> = [
      { type: "text", text: synthesisUser },
    ];
    for (const img of images) {
      const url = img.storage_url ?? img.image_abs_path ?? "";
      if (url.startsWith("http")) {
        userContent.push({ type: "image_url", image_url: { url } });
      }
    }
    pass2Messages.push({ role: "user", content: userContent });
  } else {
    pass2Messages.push({ role: "user", content: synthesisUser });
  }

  const raw = await callCopilot(copilotToken, pass2Messages);
  return parseJsonResponse(raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public: askCopilotAdaptive
// ─────────────────────────────────────────────────────────────────────────────

export async function askCopilotAdaptive(
  diseaseName: string,
  evidence: ChunkRecord[],
  githubToken: string,
  chatHistory?: Array<{ role: string; content: string }>,
  images?: ImageRecord[],
): Promise<DraftAnswer> {
  if (!githubToken) throw new Error("GITHUB_TOKEN is missing");

  const copilotToken = await getCopilotToken(githubToken);
  const [contextText, sortedEvidence] = formatEvidenceStructured(evidence);
  const intentCategory = extractTopicIntent(diseaseName);
  const isDetail = isDetailRequest(diseaseName);
  const detectedDisease = extractDiseaseName(diseaseName);

  const userPrompt = `Query Klinis: ${diseaseName}\n\nDokumen Referensi Tersedia:\n${contextText}`;

  try {
    let result: DraftAnswer;
    if (isDetail && evidence.length > 6) {
      result = await twoPassSynthesis(
        copilotToken, diseaseName, userPrompt, contextText,
        sortedEvidence, intentCategory, detectedDisease, chatHistory, images,
      );
    } else {
      result = await singlePassSynthesis(
        copilotToken, diseaseName, userPrompt,
        sortedEvidence, isDetail, intentCategory, detectedDisease, chatHistory, images,
      );
    }
    return postprocessResponse(result, sortedEvidence);
  } catch (e) {
    console.error("Copilot API Error:", e);
    return {
      disease: diseaseName,
      sections: [{ title: "AI Processing Error", markdown: `Terjadi kesalahan: \`${e}\`` }],
      citations: evidence.slice(0, 5).map((i) => `${i.source_name} p.${i.page_no}`),
      grounded: false,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Markdown article operations (refine, merge, mindmap)
// ─────────────────────────────────────────────────────────────────────────────

export async function refineMarkdownWithInstruction(
  markdown: string,
  instruction: string,
  githubToken: string,
): Promise<string> {
  const copilotToken = await getCopilotToken(githubToken);
  const system = `Anda adalah editor medis. Revisi artikel Markdown yang diberikan sesuai instruksi pengguna.
Keluaran HANYA berupa Markdown valid. Gunakan bahasa Indonesia medis formal. Jangan mengarang fakta baru.`;
  const user = `## Artikel saat ini (Markdown)\n\n${markdown}\n\n## Instruksi revisi\n\n${instruction}\n\nTuliskan ulang artikel lengkap dalam Markdown. Tanpa fence \`\`\`.`;

  const raw = await callCopilot(copilotToken, [
    { role: "system", content: system },
    { role: "user", content: user },
  ]);
  return stripMarkdownFence(raw);
}

export async function mergeTwoMarkdownArticles(
  markdownBase: string,
  markdownCandidate: string,
  githubToken: string,
): Promise<string> {
  const copilotToken = await getCopilotToken(githubToken);
  const system = `Anda adalah editor medis senior. Tugas Anda MENYATUKAN dua versi artikel Markdown tentang topik yang sama menjadi SATU artikel utuh.

ATURAN:
1. Keluaran: satu dokumen Markdown dalam bahasa Indonesia medis formal.
2. Gabungkan isi kedua sumber secara detail; hindari pengulangan paragraf atau bullet yang redundan.
3. Susun dengan struktur jelas (# judul penyakit/kondisi, ## subbagian sesuai materi klinis).
4. Pertahankan referensi/sitasi dari sumber ((Sumber) ..., Hal ...), [En], atau sejenisnya.
5. JANGAN menambahkan fakta, dosis, atau langkah yang tidak tertera di salah satu dokumen.
6. Jika ada perbedaan antar sumber, nyatakan singkat dalam satu kalimat perbandingan bila perlu.
7. Keluaran HANYA Markdown valid, tanpa fence \`\`\`.`;

  const user = `## Artikel utama (saat ini)\n\n${markdownBase.trim() || "(belum ada teks; gunakan hanya kandidat)"}\n\n---\n\n## Kandidat baru (regenerate)\n\n${markdownCandidate.trim()}\n\n---\n\nTulis SATU artikel Markdown lengkap hasil penggabungan kedua bagian di atas.`;

  const raw = await callCopilot(copilotToken, [
    { role: "system", content: system },
    { role: "user", content: user },
  ]);
  return stripMarkdownFence(raw);
}

export async function generateMindmapFromArticle(
  diseaseName: string,
  markdownContent: string,
  githubToken: string,
  competencyLevel?: string | null,
): Promise<Record<string, unknown>> {
  const copilotToken = await getCopilotToken(githubToken);
  const levelNote = competencyLevel ? ` (Level Kompetensi SKDI: ${competencyLevel})` : "";
  const system = `Anda adalah pakar pendidikan kedokteran yang ahli dalam membuat peta konsep (mindmap) visual untuk membantu mahasiswa kedokteran belajar.

TUGAS ANDA:
Baca artikel medis tentang **${diseaseName}${levelNote}** yang diberikan dan bangun struktur MINDMAP KOMPREHENSIF yang kaya informasi.

PRINSIP MINDMAP:
1. **ROOT NODE** (level 0): Nama penyakit sebagai pusat. Summary berisi definisi singkat 1 kalimat.
2. **SECTION NODES** (level 1): Setiap topik klinis utama = 1 node.
3. **CONCEPT NODES** (level 2): Setiap konsep medis penting dalam tiap section = 1 node.
4. **FACT NODES** (level 3): Fakta spesifik, angka, kriteria, dosis = 1 node.

ATURAN PENTING:
- SUMMARY setiap node WAJIB berisi informasi substantif. Minimum 1 kalimat lengkap.
- JUMLAH NODE: Minimal 25 node, idealnya 40-80 node.
- EDGES: Setiap node non-root HARUS memiliki tepat 1 parent edge.
- ID node: unik, lowercase, gunakan underscore.

FORMAT OUTPUT (RAW JSON VALID, TANPA markdown code block):
{
  "disease": "${diseaseName}",
  "competency_level": "${competencyLevel ?? ""}",
  "summary_root": "Ringkasan 2-3 kalimat.",
  "nodes": [
    {"id": "root", "label": "${diseaseName}", "type": "root", "level": 0, "summary": "Definisi singkat."}
  ],
  "edges": [{"source": "root", "target": "definisi"}],
  "visual_refs": [{"image_url": "", "heading": "", "description": ""}],
  "key_takeaways": ["Poin 1", "Poin 2"]
}`;

  const userText = `Artikel Medis — ${diseaseName}:\n\n${markdownContent}\n\nBangun mindmap komprehensif berdasarkan seluruh isi artikel di atas.`;

  try {
    const raw = await callCopilot(copilotToken, [
      { role: "system", content: system },
      { role: "user", content: userText },
    ]);
    const parsed = parseJsonResponse(raw) as unknown as Record<string, unknown>;

    const typeGroup: Record<string, number> = { root: 0, section: 1, concept: 2, fact: 3 };
    const typeVal: Record<string, number> = { root: 20, section: 12, concept: 7, fact: 4 };
    for (const node of (parsed["nodes"] as Array<Record<string, unknown>>) ?? []) {
      node["group"] = typeGroup[node["type"] as string] ?? 2;
      node["val"] = typeVal[node["type"] as string] ?? 5;
    }

    for (const key of ["disease", "competency_level", "nodes", "edges", "visual_refs", "key_takeaways"]) {
      if (!(key in parsed)) parsed[key] = key === "nodes" || key === "edges" || key === "visual_refs" || key === "key_takeaways" ? [] : "";
    }

    return parsed;
  } catch (e) {
    return { disease: diseaseName, competency_level: competencyLevel ?? "", nodes: [], edges: [], visual_refs: [], key_takeaways: [], error: String(e) };
  }
}

function stripMarkdownFence(text: string): string {
  let t = text.trim();
  if (t.startsWith("```")) {
    const lines = t.split("\n");
    if (lines[0].startsWith("```")) lines.shift();
    if (lines.length && lines[lines.length - 1].trim().startsWith("```")) lines.pop();
    t = lines.join("\n");
  }
  return t.trim();
}
