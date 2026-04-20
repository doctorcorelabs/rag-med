// Medical Vocabulary — TypeScript translation of retriever.py constants
// These are used purely in-memory in the Cloudflare Worker (no Python runtime needed)

export const MEDICAL_SYNONYMS: Record<string, string[]> = {
  // Pulmo
  tbc: ["tuberkulosis", "tb", "tuberculosis", "kp", "koch pulmonum"],
  tuberkulosis: ["tbc", "tb", "tuberculosis"],
  ppok: ["copd", "penyakit paru obstruktif kronis", "bronkitis kronis", "emfisema"],
  copd: ["ppok", "penyakit paru obstruktif kronis"],
  asma: ["asthma", "bengek", "mengi"],
  pneumonia: ["paru-paru basah", "bronkopneumonia", "cap", "hap", "vap"],
  "sindrom mendelson": ["mendelson syndrome", "mendelson", "aspiration pneumonitis", "pneumonitis aspirasi"],
  mendelson: ["sindrom mendelson", "mendelson syndrome", "aspiration pneumonitis", "pneumonitis aspirasi"],
  "aspiration pneumonitis": ["sindrom mendelson", "mendelson syndrome", "mendelson", "pneumonitis aspirasi"],
  "pneumonitis aspirasi": ["sindrom mendelson", "mendelson syndrome", "mendelson", "aspiration pneumonitis"],
  ards: ["acute respiratory distress syndrome", "gagal napas akut"],
  // Cardio
  acs: ["sindrom koroner akut", "ska", "serangan jantung", "stemi", "nstemi", "infark miokard"],
  stemi: ["infark miokard", "acs", "ska", "serangan jantung"],
  nstemi: ["infark miokard", "acs", "ska", "serangan jantung"],
  hfpef: ["gagal jantung", "heart failure", "chf"],
  hfrgef: ["gagal jantung", "heart failure", "chf"],
  chf: ["gagal jantung", "congestive heart failure", "hfpef", "hfref"],
  hipertensi: ["ht", "darah tinggi", "hypertension", "htn"],
  cad: ["penyakit jantung koroner", "pjk", "coronary artery disease"],
  // Gastro/Hepa
  gerd: ["asam lambung", "refluks", "gastroesophageal reflux disease"],
  sirosis: ["cirrhosis", "pengerasan hati"],
  // Endo
  dm: ["diabetes", "diabetes melitus", "kencing manis", "hiperglikemia"],
  diabetes: ["dm", "kencing manis", "diabetes melitus"],
  // Infeksi
  dbd: ["dengue", "dengue hemorrhagic fever", "dhf", "demam berdarah"],
  hiv: ["aids", "odhav", "odha", "human immunodeficiency virus"],
  aids: ["hiv", "odha"],
  // Klinis umum
  patofisiologi: ["patogenesis", "fisiologi", "mekanisme"],
  patogenesis: ["patofisiologi", "fisiologi", "mekanisme"],
  etiologi: ["penyebab", "kausa"],
  manifestasi: ["gejala", "keluhan", "simptom"],
  gejala: ["manifestasi", "keluhan", "simptom"],
  tatalaksana: ["terapi", "pengobatan", "penanganan", "manajemen"],
  terapi: ["tatalaksana", "pengobatan", "penanganan"],
  pengobatan: ["tatalaksana", "terapi", "penanganan"],
  diagnosis: ["diagnosa", "diagnostik"],
  diagnosa: ["diagnosis", "diagnostik"],
  komplikasi: ["penyulit"],
  prognosis: ["luaran", "outcome"],
  definisi: ["pengertian", "arti"],
  anamnesis: ["anamnesa", "keluhan", "riwayat"],
  penunjang: ["laboratorium", "lab", "radiologi"],
  farmakologi: ["obat", "medikasi"],
  obat: ["farmakologi", "medikasi", "dosis"],
};

export const TYPO_CORRECTIONS: Record<string, string> = {
  patofisilogi: "patofisiologi",
  patofisiolgi: "patofisiologi",
  patofsiologi: "patofisiologi",
  patofisiogi: "patofisiologi",
  tuberkulosisi: "tuberkulosis",
  tuberkulosi: "tuberkulosis",
  tuberkolusis: "tuberkulosis",
  tuberkuosis: "tuberkulosis",
  manitestasi: "manifestasi",
  manfestasi: "manifestasi",
  tatalaksna: "tatalaksana",
  tatlaksana: "tatalaksana",
  diagnoss: "diagnosis",
  diagnossis: "diagnosis",
  komplikassi: "komplikasi",
  progosis: "prognosis",
  prognsis: "prognosis",
  anamesis: "anamnesis",
  ananmesis: "anamnesis",
  anamnesa: "anamnesis",
  penunujang: "penunjang",
  diabtes: "diabetes",
  diabetis: "diabetes",
  hipertensu: "hipertensi",
  hipertensei: "hipertensi",
  pnemonia: "pneumonia",
  pnuemonia: "pneumonia",
  pneumoni: "pneumonia",
};

export const INTENT_MAP: Record<string, string> = {
  obat: "Tatalaksana",
  terapi: "Tatalaksana",
  pengobatan: "Tatalaksana",
  dosis: "Tatalaksana",
  tatalaksana: "Tatalaksana",
  penanganan: "Tatalaksana",
  regimen: "Tatalaksana",
  farmakologi: "Tatalaksana",
  gejala: "Manifestasi_Klinis",
  keluhan: "Manifestasi_Klinis",
  manifestasi: "Manifestasi_Klinis",
  simptom: "Manifestasi_Klinis",
  diagnosis: "Diagnosis",
  diagnosa: "Diagnosis",
  pemeriksaan: "Diagnosis",
  penunjang: "Diagnosis",
  diagnostik: "Diagnosis",
  kriteria: "Diagnosis",
  etiologi: "Etiologi",
  penyebab: "Etiologi",
  kausa: "Etiologi",
  "faktor risiko": "Etiologi",
  patofisiologi: "Patogenesis",
  patogenesis: "Patogenesis",
  fisiologi: "Patogenesis",
  mekanisme: "Patogenesis",
  patofisilogi: "Patogenesis",
  komplikasi: "Komplikasi",
  prognosis: "Prognosis",
  definisi: "Definisi",
  pengertian: "Definisi",
  anamnesis: "Anamnesis",
  anamnesa: "Anamnesis",
  "pemeriksaan fisik": "Pemeriksaan_Fisik",
};

export const DISEASE_KEYWORDS: string[] = [
  "tuberkulosis", "tbc", "tb", "pneumonia", "asma", "copd", "ppok",
  "sindrom mendelson", "mendelson syndrome", "mendelson", "aspiration pneumonitis", "pneumonitis aspirasi",
  "bronkitis", "bronkiolitis", "bronkiektasis", "emboli", "abses",
  "efusi pleura", "pneumotoraks", "atelektasis", "fibrosis",
  "kanker paru", "mesotelioma", "sarkoidosis", "hemoptisis",
  "gagal napas", "ards", "edema paru", "cor pulmonale",
  "laringitis", "trakeitis", "epiglotitis", "croup",
  "hyaline membrane disease", "hmd", "rds",
  "diabetes", "hipertensi", "gagal jantung", "chf", "acs", "stemi", "nstemi",
  "anemia", "hiv", "aids", "meningitis", "hepatitis",
  "sirosis", "gerd", "dispepsia", "gagal ginjal", "ckd", "aki",
  "lupus", "rheumatoid", "stroke", "infark miokard", "aritmia",
  "demam tifoid", "malaria", "dengue", "dbd", "leptospirosis",
  "sindrom koroner akut", "ska", "penyakit jantung koroner", "pjk",
];

export const CLINICAL_ORDER: Record<string, number> = {
  Definisi: 0,
  Etiologi: 1,
  Patogenesis: 2,
  Anamnesis: 3,
  Manifestasi_Klinis: 4,
  Pemeriksaan_Fisik: 5,
  Diagnosis: 6,
  Tatalaksana: 7,
  Komplikasi: 8,
  Prognosis: 9,
  Ringkasan_Klinis: 10,
};

// ── Listing / Enumeration Intent ─────────────────────────────────────────────

/** Keywords that signal the user wants a full catalog/list of diseases. */
export const LIST_INTENT_KEYWORDS: string[] = [
  "daftar penyakit", "list penyakit", "semua penyakit",
  "penyakit apa saja", "apa saja penyakit", "sebutkan penyakit",
  "penyakit yang ada", "semua topik", "topik apa saja",
  "daftar topik", "apa yang tersedia", "berikan daftar",
  "tampilkan semua", "list semua", "semua materi",
  "materi apa saja", "semua sumber", "list sumber",
  "daftar semua", "semua diagnosis", "katalog penyakit",
  "all diseases", "list all", "list diseases",
  "semua seluruh", "seluruh penyakit",
];

/**
 * Returns true when the query is requesting a full enumeration / catalog of
 * diseases rather than focused information about a single condition.
 */
export function isListingIntent(query: string): boolean {
  const qLower = query.toLowerCase();
  return LIST_INTENT_KEYWORDS.some((kw) => qLower.includes(kw));
}

/**
 * Resolve retrieval mode to either 'relevant' or 'exhaustive'.
 *
 * Priority:
 *  1. Explicit requestedMode from the caller.
 *  2. Auto-detection via isListingIntent().
 *  3. Default: 'relevant'.
 */
export function resolveRetrievalMode(
  query: string,
  requestedMode?: "relevant" | "exhaustive" | null,
): "relevant" | "exhaustive" {
  if (requestedMode === "relevant" || requestedMode === "exhaustive") {
    return requestedMode;
  }
  return isListingIntent(query) ? "exhaustive" : "relevant";
}

// ── Query Processing Functions ────────────────────────────────────────────────

const TOKEN_RE = /[a-zA-Z][a-zA-Z0-9/-]{1,}/g;

export function correctTypo(token: string): string {
  return TYPO_CORRECTIONS[token.toLowerCase()] ?? token;
}

export function expandSynonyms(tokens: string[]): string[] {
  const expanded = [...tokens];
  for (const token of tokens) {
    const lower = token.toLowerCase();
    const synonyms = MEDICAL_SYNONYMS[lower];
    if (synonyms) {
      for (const syn of synonyms) {
        if (!expanded.some((t) => t.toLowerCase() === syn.toLowerCase())) {
          expanded.push(syn);
        }
      }
    }
  }
  return expanded;
}

export function extractDiseaseName(query: string): string | null {
  const qLower = query.toLowerCase();
  for (const disease of DISEASE_KEYWORDS) {
    if (qLower.includes(disease)) return disease;
  }
  const tokens = query.match(TOKEN_RE) ?? [];
  for (const token of tokens) {
    const corrected = correctTypo(token);
    if (MEDICAL_SYNONYMS[corrected]) {
      for (const syn of MEDICAL_SYNONYMS[corrected]) {
        if (DISEASE_KEYWORDS.includes(syn)) return syn;
      }
    }
  }
  return null;
}

export function extractTopicIntent(query: string): string | null {
  const qLower = query.toLowerCase();
  // Check multi-word intents first (longer first)
  const sortedKeys = Object.keys(INTENT_MAP).sort((a, b) => b.length - a.length);
  for (const keyword of sortedKeys) {
    if (qLower.includes(keyword)) return INTENT_MAP[keyword];
  }
  // Single token check with typo correction
  const tokens = query.match(TOKEN_RE) ?? [];
  for (const token of tokens) {
    const corrected = correctTypo(token);
    if (INTENT_MAP[corrected]) return INTENT_MAP[corrected];
  }
  return null;
}

export function isDetailRequest(query: string): boolean {
  const detail_keywords = [
    "detail", "lengkap", "jelaskan", "jelasin", "menjelaskan",
    "komprehensif", "selengkap", "keseluruhan", "semua aspek",
    "secara detail", "secara lengkap", "apa itu", "jelaskan tentang",
  ];
  const qLower = query.toLowerCase();
  return detail_keywords.some((kw) => qLower.includes(kw));
}

export function enrichQueryFromHistory(
  query: string,
  chatHistory: Array<{ role: string; content: string }> | undefined,
): string {
  if (!chatHistory || chatHistory.length === 0) return query;
  const recentContext = chatHistory
    .slice(-4)
    .filter((t) => t.role === "user")
    .map((t) => t.content)
    .join(" ");
  const currentDisease = extractDiseaseName(query);
  if (!currentDisease) {
    const historyDisease = extractDiseaseName(recentContext);
    if (historyDisease) return `${historyDisease} ${query}`;
  }
  return query;
}

export function getExpandedTerms(query: string): string[] {
  const tokens = (query.match(TOKEN_RE) ?? []).filter((t) => t.length > 2);
  const corrected = tokens.map(correctTypo);
  return expandSynonyms(corrected);
}

export function normalizeMedicalTerm(term: string): string {
  return term
    .toLowerCase()
    .replace(/[_:.,()\[\]{}]/g, " ")
    .replace(/[^a-z0-9/\-\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function scoreMedicalTermSimilarity(query: string, candidate: string): number {
  const queryNorm = normalizeMedicalTerm(query);
  const candidateNorm = normalizeMedicalTerm(candidate);
  if (!queryNorm || !candidateNorm) return 0;

  if (queryNorm === candidateNorm) return 1;
  if (queryNorm.includes(candidateNorm) || candidateNorm.includes(queryNorm)) {
    return 0.92;
  }

  const queryTokens = new Set(queryNorm.split(/\s+/).filter((t) => t.length > 2));
  const candidateTokens = new Set(candidateNorm.split(/\s+/).filter((t) => t.length > 2));
  if (queryTokens.size === 0 || candidateTokens.size === 0) return 0;

  let overlap = 0;
  for (const token of queryTokens) {
    if (candidateTokens.has(token)) overlap++;
  }

  const union = queryTokens.size + candidateTokens.size - overlap;
  const jaccard = union > 0 ? overlap / union : 0;
  const prefixBoost = queryNorm.startsWith(candidateNorm.slice(0, 8)) || candidateNorm.startsWith(queryNorm.slice(0, 8)) ? 0.08 : 0;
  return Math.min(1, jaccard * 0.85 + prefixBoost);
}

export function collectMedicalAliasCandidates(query: string): string[] {
  const expanded = new Set<string>();
  expanded.add(normalizeMedicalTerm(query));

  for (const term of getExpandedTerms(query)) {
    expanded.add(normalizeMedicalTerm(term));
  }

  for (const token of (query.match(TOKEN_RE) ?? []).filter((t) => t.length > 2)) {
    const corrected = correctTypo(token);
    expanded.add(normalizeMedicalTerm(corrected));
    const synonyms = MEDICAL_SYNONYMS[corrected.toLowerCase()] ?? MEDICAL_SYNONYMS[token.toLowerCase()] ?? [];
    for (const synonym of synonyms) {
      expanded.add(normalizeMedicalTerm(synonym));
    }
  }

  return [...expanded].filter((term) => term.length > 2);
}

// Jaccard similarity for dedup
export function jaccardSimilarity(a: string, b: string): number {
  const setA = new Set(a.split(/\s+/));
  const setB = new Set(b.split(/\s+/));
  if (setA.size === 0 || setB.size === 0) return 0;
  let intersection = 0;
  for (const w of setA) {
    if (setB.has(w)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  return intersection / union;
}

export function pruneRedundantChunks<T extends { content: string }>(
  results: T[],
  threshold = 0.6,
): T[] {
  const pruned: T[] = [];
  for (const item of results) {
    const content = item.content.toLowerCase();
    const isRedundant = pruned.some(
      (s) => jaccardSimilarity(content, s.content.toLowerCase()) > threshold,
    );
    if (!isRedundant) pruned.push(item);
  }
  return pruned;
}

// RRF (Reciprocal Rank Fusion)
function getChunkKey(item: Record<string, unknown>): string {
  const content = String(item["content"] ?? "").slice(0, 200);
  // Simple hash via string length + first chars (no crypto needed for key)
  const hash = Array.from(content).reduce((acc, c) => (acc * 31 + c.charCodeAt(0)) & 0xffffffff, 0);
  return `${item["source_name"]}|${item["page_no"]}|${item["heading"]}|${hash}`;
}

export function reciprocalRankFusion<T extends Record<string, unknown>>(
  bm25: T[],
  vector: T[],
  k = 60,
): T[] {
  const scores = new Map<string, number>();
  const data = new Map<string, T>();

  for (let i = 0; i < bm25.length; i++) {
    const key = getChunkKey(bm25[i]);
    scores.set(key, (scores.get(key) ?? 0) + 1 / (k + i + 1));
    data.set(key, bm25[i]);
  }
  for (let i = 0; i < vector.length; i++) {
    const key = getChunkKey(vector[i]);
    scores.set(key, (scores.get(key) ?? 0) + 1 / (k + i + 1));
    if (!data.has(key)) data.set(key, vector[i]);
  }

  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([key]) => data.get(key)!);
}
