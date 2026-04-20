import { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo } from 'react';
import axios from 'axios';
import JSZip from 'jszip';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
// @ts-ignore: missing type declarations
import ForceGraph2D from 'react-force-graph-2d';
import { hierarchy, tree as d3tree } from 'd3-hierarchy';

// ─── TYPES ──────────────────────────────────────────────────────────────────
type EvidenceItem = {
  source_name: string;
  page_no: number;
  heading: string;
  content: string;
  markdown_path?: string;
  section_category?: string;
};

type ImageItem = {
  source_name: string;
  page_no: number;
  heading: string;
  image_url: string;
  image_abs_path?: string;
  /** Worker / RAG — stable key when no local path */
  image_ref?: string;
  storage_url?: string;
};

// Ide 2: sections now use "markdown" string instead of "points" array
type DraftSection = {
  title: string;
  markdown?: string;
  points?: string[]; // legacy fallback
};

type ApiResponse = {
  query: string;
  detail_level: string;
  evidence_count: number;
  evidence_quality?: 'low' | 'ok';
  evidence: EvidenceItem[];
  draft_answer: {
    disease: string;
    sections: DraftSection[];
    citations: string[];
    grounded: boolean;
    answer_confidence?: number;
    section_confidence_map?: Record<string, number>;
    detection_method?: string;
    detection_confidence?: number;
    retrieval_passes?: number;
  };
  images: ImageItem[];
  retrieval_diagnostics?: {
    total_candidates: number;
    returned_count: number;
    is_truncated: boolean;
    retrieval_mode: 'relevant' | 'exhaustive';
  };
};

type ChatMessage =
  | { role: 'user'; content: string }
  | { role: 'bot'; data?: ApiResponse; error?: boolean; content?: string };


type MindmapNode = { id: string; label: string; type: string; level: number; summary: string; val: number; group: number };
type MindmapEdge = { source: string; target: string };
type VisualRef = { image_url: string; heading: string; description?: string };
type MindmapData = {
  disease: string;
  competency_level?: string;
  summary_root?: string;
  nodes: MindmapNode[];
  edges: MindmapEdge[];
  visual_refs: VisualRef[];
  key_takeaways?: string[];
  not_generated?: boolean;
  error?: string;
};

type LibraryStase = {
  id: number;
  slug: string;
  display_name: string;
  sort_order: number;
  disease_count: number;
  filled_count: number;
};

type LibraryDiseaseRow = {
  id: number;
  catalog_no: number;
  name: string;
  competency_level: string | null;
  group_label: string | null;
  stable_key: string;
  status: string | null;
  content_path: string | null;
  updated_at: string | null;
};

type LibraryDiseaseDetail = {
  disease: Record<string, unknown>;
  markdown: string | null;
  meta: Record<string, unknown> | null;
  images: ImageItem[];
};

type LibraryPreviewResponse = {
  ok: boolean;
  markdown_base: string;
  markdown_candidate: string;
  markdown_combined: string;
  preview_note: string;
  evidence_count: number;
  persisted?: boolean;
};

const API_URL = import.meta.env.VITE_API_URL || 'https://medrag-worker.daivanfebrijuansetiya.workers.dev';
const DEFAULT_UPLOAD_MODE: 'legacy' | 'cloud' =
  import.meta.env.VITE_UPLOAD_MODE === 'legacy'
    ? 'legacy'
    : /workers\.dev|pages\.dev|cloudflare/i.test(API_URL)
      ? 'cloud'
      : 'legacy';
const CLOUD_BATCH_SIZE = 5;

const PLACEHOLDER_IMG = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='150' height='150' fill='%23e2e8f0'%3E%3Crect width='150' height='150'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='12' fill='%2394a3b8'%3ENo Image%3C/text%3E%3C/svg%3E";

function resolveImageUrl(url: string | undefined): string {
  if (!url) return PLACEHOLDER_IMG;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  if (url.startsWith('/')) return `${API_URL}${url}`;
  return `${API_URL}/${url}`;
}

/** Stable identity for deduping / PATCH (local path, R2 ref, or display URL). */
function imageItemStableKey(img: ImageItem): string {
  const a = (img.image_abs_path || '').trim();
  if (a) return `abs:${a}`;
  const r = (img.image_ref || '').trim();
  if (r) return `ref:${r}`;
  const s = (img.storage_url || '').trim();
  if (s) return `stor:${s}`;
  return `url:${(img.image_url || '').trim()}`;
}

function imageItemHasStableRef(img: ImageItem): boolean {
  return (
    (img.image_abs_path || '').trim().length > 0 ||
    (img.image_ref || '').trim().length > 0 ||
    (img.storage_url || '').trim().length > 0 ||
    (img.image_url || '').trim().length > 0
  );
}

type CloudUploadPage = {
  page_no: number;
  markdown: string;
  markdown_path: string;
};

type CloudUploadBatch = {
  batch_index: number;
  pages: CloudUploadPage[];
};

type CloudUploadJob = {
  source_name: string;
  stase_slug: string;
  batches: CloudUploadBatch[];
  total_pages: number;
  total_batches: number;
  next_batch_index: number;
  started_at: number;
};

function splitIntoBatches<T>(items: T[], size: number): T[][] {
  const batches: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    batches.push(items.slice(index, index + size));
  }
  return batches;
}

async function readZipMarkdownPages(file: File): Promise<CloudUploadPage[]> {
  const zip = await JSZip.loadAsync(file);
  const pages: CloudUploadPage[] = [];

  const entries = Object.values(zip.files).filter((entry) => {
    const normalized = entry.name.replace(/\\/g, '/');
    return /(?:^|\/)page-(\d+)\/markdown\.md$/i.test(normalized);
  });

  for (const entry of entries) {
    const normalized = entry.name.replace(/\\/g, '/');
    const match = normalized.match(/(?:^|\/)page-(\d+)\/markdown\.md$/i);
    if (!match) continue;
    const pageNo = parseInt(match[1], 10);
    const markdown = await entry.async('string');
    pages.push({ page_no: pageNo, markdown, markdown_path: normalized });
  }

  pages.sort((a, b) => a.page_no - b.page_no);
  return pages;
}

type ActiveView = 'chat' | 'library' | 'kg' | 'analytics' | 'admin';

// ─── RESPONSIVE HOOKS ───────────────────────────────────────────────────────
function useScreenSize() {
  const [size, setSize] = useState({ w: window.innerWidth, h: window.innerHeight });
  useEffect(() => {
    const handler = () => setSize({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, []);
  return { isMobile: size.w < 768, isTablet: size.w >= 768 && size.w < 1024, isDesktop: size.w >= 1024, ...size };
}

// ─── MOBILE BOTTOM NAV ──────────────────────────────────────────────────────
const NAV_ITEMS: { icon: string; label: string; id: ActiveView }[] = [
  { icon: 'chat_bubble', label: 'Chat', id: 'chat' },
  { icon: 'book', label: 'Library', id: 'library' },
  { icon: 'hub', label: 'KG', id: 'kg' },
  { icon: 'query_stats', label: 'Analytics', id: 'analytics' },
  { icon: 'admin_panel_settings', label: 'Admin', id: 'admin' },
];

function MobileBottomNav({ activeView, onChangeView }: { activeView: ActiveView; onChangeView: (v: ActiveView) => void }) {
  return (
    <nav className="mobile-bottom-nav fixed bottom-0 left-0 right-0 z-40 flex items-stretch md:hidden">
      {NAV_ITEMS.map((item) => {
        const active = activeView === item.id;
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => onChangeView(item.id)}
            className={`flex-1 flex flex-col items-center justify-center gap-0.5 py-2 transition-colors ${
              active ? 'text-indigo-600' : 'text-slate-400'
            }`}
          >
            <span
              className={`material-symbols-outlined text-[22px] transition-all ${active ? 'scale-110' : ''}`}
              style={active ? { fontVariationSettings: "'FILL' 1" } : undefined}
            >{item.icon}</span>
            <span className={`text-[10px] font-medium ${active ? 'font-bold' : ''}`}>{item.label}</span>
            {active && <span className="absolute bottom-1 w-5 h-0.5 rounded-full bg-indigo-600" />}
          </button>
        );
      })}
    </nav>
  );
}

// Section icon mapping
const SECTION_ICONS: Record<string, string> = {
  'Definisi': 'info',
  'Etiologi dan Faktor Risiko': 'biotech',
  'Manifestasi Klinis': 'symptoms',
  'Diagnosis': 'labs',
  'Tatalaksana': 'medication',
  'Komplikasi dan Prognosis': 'warning',
  'Ringkasan Klinis': 'summarize',
};

function formatConfidenceLabel(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null;
  return `${Math.round(value * 100)}%`;
}

function confidenceTone(value?: number): 'high' | 'medium' | 'low' {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'low';
  if (value >= 0.75) return 'high';
  if (value >= 0.45) return 'medium';
  return 'low';
}


// ─── MARKDOWN COMPONENTS ────────────────────────────────────────────────────
const mdComponents = {
  h1: ({ children }: any) => <h1 className="text-2xl font-headline font-black text-slate-800 dark:text-slate-100 mt-4 mb-2">{children}</h1>,
  h2: ({ children }: any) => <h2 className="text-xl font-headline font-bold text-indigo-700 dark:text-indigo-300 mt-3 mb-2">{children}</h2>,
  h3: ({ children }: any) => <h3 className="text-base font-headline font-semibold text-slate-700 dark:text-slate-200 mt-2 mb-1">{children}</h3>,
  p: ({ children }: any) => <p className="text-base leading-relaxed text-on-surface mb-3">{children}</p>,
  ul: ({ children }: any) => <ul className="list-disc pl-6 space-y-1.5 mb-3">{children}</ul>,
  ol: ({ children }: any) => <ol className="list-decimal pl-6 space-y-1.5 mb-3">{children}</ol>,
  li: ({ children }: any) => <li className="text-on-surface leading-relaxed marker:text-indigo-400">{children}</li>,
  strong: ({ children }: any) => <strong className="font-bold text-indigo-700 dark:text-indigo-300">{children}</strong>,
  em: ({ children }: any) => <em className="italic text-slate-600 dark:text-slate-400">{children}</em>,
  code: ({ children }: any) => {
    const isBlock = String(children).includes('\n');
    return isBlock
      ? <code className="block bg-slate-100 dark:bg-slate-800 rounded-xl p-3 text-sm font-mono text-slate-700 dark:text-slate-300 mb-3 overflow-x-auto whitespace-pre">{children}</code>
      : <code className="bg-indigo-50 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 px-1.5 py-0.5 rounded text-sm font-mono">{children}</code>;
  },
  table: ({ children }: any) => (
    <div className="overflow-x-auto mb-4 rounded-2xl border border-slate-200 dark:border-slate-700">
      <table className="w-full text-sm text-left">{children}</table>
    </div>
  ),
  thead: ({ children }: any) => <thead className="bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300">{children}</thead>,
  tbody: ({ children }: any) => <tbody className="divide-y divide-slate-100 dark:divide-slate-800">{children}</tbody>,
  tr: ({ children }: any) => <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">{children}</tr>,
  th: ({ children }: any) => <th className="px-4 py-2 font-semibold">{children}</th>,
  td: ({ children }: any) => <td className="px-4 py-2.5 text-on-surface">{children}</td>,
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-4 border-indigo-400 pl-4 my-3 text-slate-500 dark:text-slate-400 italic">{children}</blockquote>
  ),
  a: ({ children, href }: any) => (
    <a href={href} className="text-indigo-600 hover:underline" target="_blank" rel="noopener noreferrer">{children}</a>
  ),
};


// ─── KNOWLEDGE GRAPH PANEL ───────────────────────────────────────────────────

/** Reingold-Tilford tree layout — each node gets an exact (x, y) without overlap. */
const TREE_NODE_W = 230; // horizontal space per node slot
const TREE_NODE_H = 130; // vertical space between levels

type TreeDatum = { id: string; children: TreeDatum[] };

function computeTreeLayout(
  nodes: MindmapNode[],
  edges: MindmapEdge[],
): Map<string, { x: number; y: number }> {
  if (nodes.length === 0) return new Map();

  const childrenMap = new Map<string, string[]>();
  const hasParent = new Set<string>();
  for (const e of edges) {
    if (!childrenMap.has(e.source)) childrenMap.set(e.source, []);
    childrenMap.get(e.source)!.push(e.target);
    hasParent.add(e.target);
  }

  const rootNode =
    nodes.find(n => n.type === 'root') ??
    nodes.find(n => !hasParent.has(n.id)) ??
    nodes[0];
  if (!rootNode) return new Map();

  const visited = new Set<string>();
  const buildDatum = (id: string): TreeDatum => {
    visited.add(id);
    return {
      id,
      children: (childrenMap.get(id) ?? [])
        .filter(cid => !visited.has(cid))
        .map(cid => buildDatum(cid)),
    };
  };
  const root = hierarchy<TreeDatum>(buildDatum(rootNode.id));

  d3tree<TreeDatum>().nodeSize([TREE_NODE_W, TREE_NODE_H])(root as any);

  const pos = new Map<string, { x: number; y: number }>();
  (root as any).each((d: any) => pos.set(d.data.id, { x: d.x, y: d.y }));

  return pos;
}

const NODE_TYPE_OPTIONS = ['root', 'section', 'concept', 'fact'] as const;
type NodeType = typeof NODE_TYPE_OPTIONS[number];

function drawMindmapNode(
  node: any,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  isActive: boolean,
) {
  const nx = node.x as number;
  const ny = node.y as number;
  const isRoot = node.type === 'root';
  const isSection = node.type === 'section';
  const isConcept = node.type === 'concept';

  const fillColors: Record<string, string> = { root: '#6366f1', section: '#0ea5e9', concept: '#d1fae5', fact: '#fef3c7' };
  const borderColors: Record<string, string> = { root: '#4338ca', section: '#0369a1', concept: '#10b981', fact: '#f59e0b' };
  const textColors: Record<string, string> = { root: '#ffffff', section: '#ffffff', concept: '#065f46', fact: '#92400e' };

  const rr = (x: number, y: number, w: number, h: number, rad: number) => {
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  };

  // ── Compact (LOD) mode — fixed graph-coordinate boxes so nodes never visually collide
  const LOD_THRESHOLD = 0.65;
  if (globalScale < LOD_THRESHOLD) {
    const cw = isRoot ? 190 : isSection ? 160 : 130;
    const ch = isRoot ? 44 : isSection ? 38 : 32;
    const cr = isRoot ? 10 : 6;
    const cx = nx - cw / 2;
    const cy = ny - ch / 2;
    if (isRoot || isSection) {
      ctx.save();
      ctx.shadowColor = isRoot ? 'rgba(99,102,241,0.35)' : 'rgba(14,165,233,0.25)';
      ctx.shadowBlur = isRoot ? 12 : 7;
    }
    rr(cx, cy, cw, ch, cr);
    ctx.fillStyle = fillColors[node.type] ?? '#e2e8f0';
    ctx.fill();
    ctx.strokeStyle = isActive ? '#f43f5e' : (borderColors[node.type] ?? '#94a3b8');
    ctx.lineWidth = isActive ? 2 : 1.2;
    ctx.stroke();
    if (isRoot || isSection) ctx.restore();

    // Render label with fixed graph-coordinate font so text scales down with zoom
    const cfs = isRoot ? 13 : isSection ? 11 : 9.5;
    ctx.font = `${isRoot || isSection ? '700' : '500'} ${cfs}px Inter,sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = textColors[node.type] ?? '#1e293b';
    const rawLbl = String(node.label ?? '');
    const maxCw = cw - 16;
    const lodLines: string[] = [];
    let lc = '';
    for (const w of rawLbl.split(' ')) {
      const t = lc ? `${lc} ${w}` : w;
      if (ctx.measureText(t).width > maxCw && lc) { lodLines.push(lc); lc = w; }
      else lc = t;
    }
    if (lc) lodLines.push(lc);
    const clh = cfs * 1.3;
    const ty0 = ny - (lodLines.length - 1) * clh / 2;
    lodLines.forEach((line, i) => ctx.fillText(line, nx, ty0 + i * clh));

    node.__bw = cw;
    node.__bh = ch;
    return;
  }

  // ── Full label mode (globalScale ≥ LOD_THRESHOLD)
  const baseFontSize = isRoot ? 13 : isSection ? 10.5 : isConcept ? 9 : 8.5;
  const fs = Math.max(baseFontSize / globalScale, 2.5);
  ctx.font = `${isRoot || isSection ? '700' : '500'} ${fs}px Inter,sans-serif`;

  const rawLabel = node.label as string;
  const maxLineW = (isRoot ? 108 : isSection ? 88 : 76) / globalScale;
  const wrappedLines: string[] = [];
  let cur = '';
  for (const w of rawLabel.split(' ')) {
    const t = cur ? `${cur} ${w}` : w;
    if (ctx.measureText(t).width > maxLineW && cur) { wrappedLines.push(cur); cur = w; }
    else cur = t;
  }
  if (cur) wrappedLines.push(cur);

  const lh = fs * 1.4;
  const padX = (isRoot ? 14 : 10) / globalScale;
  const padY = (isRoot ? 10 : 7) / globalScale;
  const longestW = Math.max(...wrappedLines.map(l => ctx.measureText(l).width));
  const bw = longestW + padX * 2;
  const bh = wrappedLines.length * lh + padY * 2;
  const bx = nx - bw / 2;
  const by = ny - bh / 2;
  const r = isRoot ? Math.min(bh * 0.45, 12 / globalScale) : Math.min(6 / globalScale, bh * 0.35);

  if (isRoot || isSection) {
    ctx.save();
    ctx.shadowColor = isRoot ? 'rgba(99,102,241,0.4)' : 'rgba(14,165,233,0.3)';
    ctx.shadowBlur = (isRoot ? 14 : 8) / globalScale;
  }

  rr(bx, by, bw, bh, r);
  ctx.fillStyle = fillColors[node.type] ?? '#e2e8f0';
  ctx.fill();
  ctx.strokeStyle = isActive ? '#f43f5e' : (borderColors[node.type] ?? '#94a3b8');
  ctx.lineWidth = (isActive ? 3 : isRoot ? 2.5 : 1.5) / globalScale;
  ctx.stroke();

  if (isRoot || isSection) ctx.restore();

  ctx.fillStyle = textColors[node.type] ?? '#1e293b';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const textY0 = ny - (wrappedLines.length - 1) * lh / 2;
  wrappedLines.forEach((line, i) => ctx.fillText(line, nx, textY0 + i * lh));

  node.__bw = bw;
  node.__bh = bh;
}

function KnowledgeGraphPanel({ initialDisease, onDismissInitial }: { initialDisease?: string | null; onDismissInitial?: () => void }) {
  const [stases, setStases] = useState<LibraryStase[]>([]);
  const [staseSlug, setStaseSlug] = useState('ipd');
  const [diseases, setDiseases] = useState<LibraryDiseaseRow[]>([]);
  const [search, setSearch] = useState('');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [localMindmap, setLocalMindmap] = useState<MindmapData | null>(null);
  const [loadingMindmap, setLoadingMindmap] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [hasUnsaved, setHasUnsaved] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [activeNode, setActiveNode] = useState<MindmapNode | null>(null);
  const [editLabel, setEditLabel] = useState('');
  const [editSummary, setEditSummary] = useState('');
  const [editType, setEditType] = useState<NodeType>('concept');
  const [showVisuals, setShowVisuals] = useState(false);
  const [graphKey, setGraphKey] = useState(0);
  const [kgListOpen, setKgListOpen] = useState(false); // mobile: togglable list
  const containerRef = useRef<HTMLDivElement>(null);
  const graphPanelRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [isGraphFullscreen, setIsGraphFullscreen] = useState(false);
  const graphRef = useRef<any>(null);
  const { isMobile: kgMobile } = useScreenSize();

  const toggleGraphFullscreen = useCallback(async () => {
    const el = graphPanelRef.current;
    if (!el) return;
    try {
      if (!document.fullscreenElement) await el.requestFullscreen();
      else await document.exitFullscreen();
    } catch {
      /* fullscreen may be blocked */
    }
  }, []);

  useEffect(() => {
    const sync = () => {
      const panel = graphPanelRef.current;
      const isNow = !!panel && document.fullscreenElement === panel;
      setIsGraphFullscreen(isNow);
    };
    document.addEventListener('fullscreenchange', sync);
    return () => document.removeEventListener('fullscreenchange', sync);
  }, []);

  useEffect(() => {
    axios.get<{ stases: LibraryStase[] }>(`${API_URL}/library/stases`)
      .then((r) => setStases(r.data.stases || []))
      .catch(() => setStases([]));
  }, []);

  useEffect(() => {
    setLoadingList(true);
    axios.get(`${API_URL}/library/stases/${staseSlug}/diseases`)
      .then((r) => setDiseases(r.data.diseases || []))
      .catch(() => setDiseases([]))
      .finally(() => setLoadingList(false));
  }, [staseSlug]);

  useEffect(() => {
    if (!initialDisease || diseases.length === 0) return;
    const name = initialDisease.toLowerCase();
    const match = diseases.find((d) => d.name.toLowerCase().includes(name) || name.includes(d.name.toLowerCase()));
    if (match) { handleSelectDisease(match); onDismissInitial?.(); }
  }, [initialDisease, diseases]);

  const handleSelectDisease = async (disease: LibraryDiseaseRow) => {
    if (!disease.status || disease.status === 'missing') return;
    setSelectedId(disease.id);
    setLocalMindmap(null);
    setActiveNode(null);
    setEditMode(false);
    setHasUnsaved(false);
    setLoadingMindmap(true);
    try {
      const r = await axios.get<MindmapData>(`${API_URL}/library/stases/${staseSlug}/diseases/${disease.id}/mindmap`);
      setLocalMindmap(r.data);
    } catch (e: any) {
      setLocalMindmap({ disease: disease.name, nodes: [], edges: [], visual_refs: [], error: e?.message });
    } finally {
      setLoadingMindmap(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedId) return;
    setGenerating(true);
    setActiveNode(null);
    try {
      const r = await axios.post<MindmapData>(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/mindmap/generate`);
      setLocalMindmap(r.data);
      setHasUnsaved(false);
      setGraphKey(k => k + 1);
    } catch (e: any) {
      setSaveMsg(`Generate gagal: ${e?.response?.data?.detail ?? e.message}`);
      setTimeout(() => setSaveMsg(null), 4000);
    } finally {
      setGenerating(false);
    }
  };

  const handleSave = async () => {
    if (!selectedId || !localMindmap) return;
    setSaving(true);
    try {
      await axios.patch(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/mindmap`, {
        nodes: localMindmap.nodes,
        edges: localMindmap.edges,
        visual_refs: localMindmap.visual_refs ?? [],
        key_takeaways: localMindmap.key_takeaways ?? [],
        summary_root: localMindmap.summary_root ?? '',
      });
      setHasUnsaved(false);
      setSaveMsg('Tersimpan');
      setTimeout(() => setSaveMsg(null), 2500);
    } catch {
      setSaveMsg('Gagal menyimpan');
      setTimeout(() => setSaveMsg(null), 3000);
    } finally {
      setSaving(false);
    }
  };

  // ── Edit helpers ───────────────────────────────────────────────────────────
  const openEditForNode = (node: MindmapNode) => {
    setActiveNode(node);
    setEditLabel(node.label);
    setEditSummary(node.summary);
    setEditType(node.type as NodeType);
  };

  const handleNodeClick = (node: any) => {
    if (editMode) openEditForNode(node as MindmapNode);
    else setActiveNode(node as MindmapNode);
  };

  const typeGroupVal: Record<NodeType, { group: number; val: number }> = {
    root: { group: 0, val: 20 }, section: { group: 1, val: 12 },
    concept: { group: 2, val: 7 }, fact: { group: 3, val: 4 },
  };

  const applyNodeEdit = () => {
    if (!activeNode || !localMindmap) return;
    const tv = typeGroupVal[editType];
    setLocalMindmap(prev => prev ? {
      ...prev,
      nodes: prev.nodes.map(n => n.id === activeNode.id
        ? { ...n, label: editLabel.trim() || n.label, summary: editSummary, type: editType, group: tv.group, val: tv.val }
        : n),
    } : prev);
    setHasUnsaved(true);
    setActiveNode(null);
  };

  const handleDeleteNode = (nodeId: string) => {
    if (!localMindmap) return;
    if (!window.confirm('Hapus node ini? Edge yang terhubung juga akan dihapus.')) return;
    // Collect all descendant IDs recursively
    const toRemove = new Set<string>([nodeId]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const e of localMindmap.edges) {
        if (toRemove.has(e.source) && !toRemove.has(e.target)) {
          toRemove.add(e.target); changed = true;
        }
      }
    }
    setLocalMindmap(prev => prev ? {
      ...prev,
      nodes: prev.nodes.filter(n => !toRemove.has(n.id)),
      edges: prev.edges.filter(e => !toRemove.has(e.source) && !toRemove.has(e.target)),
    } : prev);
    setHasUnsaved(true);
    setGraphKey(k => k + 1);
    setActiveNode(null);
  };

  const handleAddChild = (parentId: string) => {
    if (!localMindmap) return;
    const parent = localMindmap.nodes.find(n => n.id === parentId);
    if (!parent) return;
    const childLevel = Math.min((parent.level ?? 0) + 1, 3);
    const childType: NodeType = childLevel === 1 ? 'section' : childLevel === 2 ? 'concept' : 'fact';
    const tv = typeGroupVal[childType];
    const newId = `custom_${Date.now()}`;
    const newNode: MindmapNode = {
      id: newId, label: 'Node Baru', summary: '', type: childType,
      level: childLevel, group: tv.group, val: tv.val,
    };
    setLocalMindmap(prev => prev ? {
      ...prev,
      nodes: [...prev.nodes, newNode],
      edges: [...prev.edges, { source: parentId, target: newId }],
    } : prev);
    setHasUnsaved(true);
    setGraphKey(k => k + 1);
    // Open edit panel for new node immediately
    setTimeout(() => openEditForNode(newNode), 50);
  };

  // ── Derived data ───────────────────────────────────────────────────────────
  const filtered = useMemo(() => {
    if (!search.trim()) return diseases;
    const q = search.toLowerCase();
    return diseases.filter((d) => d.name.toLowerCase().includes(q) || String(d.catalog_no).includes(q));
  }, [diseases, search]);

  const grouped = useMemo(() => {
    const m = new Map<string, LibraryDiseaseRow[]>();
    for (const d of filtered) {
      const g = d.group_label || 'Umum';
      if (!m.has(g)) m.set(g, []);
      m.get(g)!.push(d);
    }
    return m;
  }, [filtered]);

  const graphData = useMemo(() => {
    if (!localMindmap || localMindmap.nodes.length === 0) return { nodes: [], links: [] };
    const pos = computeTreeLayout(localMindmap.nodes, localMindmap.edges);
    const nodes = localMindmap.nodes.map(n => {
      const p = pos.get(n.id);
      return p ? { ...n, x: p.x, y: p.y, fx: p.x, fy: p.y } : { ...n };
    });
    return {
      nodes,
      links: localMindmap.edges.map(e => ({ source: e.source, target: e.target })),
    };
  }, [localMindmap]);

  const hasGraph = localMindmap && !localMindmap.not_generated && localMindmap.nodes.length > 0;

  useLayoutEffect(() => {
    if (!hasGraph) return;
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      if (el.offsetWidth > 0 && el.offsetHeight > 0) {
        // #region agent log
        console.log('[DBG-087f38] measure', { w: el.offsetWidth, h: el.offsetHeight, isGraphFullscreen });
        // #endregion
        setDimensions(prev => {
          if (prev.width === el.offsetWidth && prev.height === el.offsetHeight) return prev;
          // After dimension change, re-fit the graph on next frame
          requestAnimationFrame(() => graphRef.current?.zoomToFit(350, 40));
          return { width: el.offsetWidth, height: el.offsetHeight };
        });
      }
    };
    const obs = new ResizeObserver(measure);
    obs.observe(el);
    requestAnimationFrame(measure);
    return () => obs.disconnect();
  }, [hasGraph, isGraphFullscreen]);

  return (
    <div className="flex-1 flex flex-col md:flex-row min-h-0 overflow-hidden">
      {/* ── Mobile: toggle button for disease list ── */}
      {kgMobile && (
        <button
          type="button"
          onClick={() => setKgListOpen(!kgListOpen)}
          className="flex items-center gap-2 px-4 py-2.5 bg-slate-50 dark:bg-slate-900 border-b border-slate-100 dark:border-slate-800 shrink-0"
        >
          <span className="material-symbols-outlined text-[18px] text-slate-500">{kgListOpen ? 'expand_less' : 'format_list_bulleted'}</span>
          <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
            {selectedId ? diseases.find(d => d.id === selectedId)?.name || 'Pilih Penyakit' : 'Pilih Penyakit'}
          </span>
          <span className="material-symbols-outlined text-[16px] text-slate-400 ml-auto">{kgListOpen ? 'close' : 'chevron_right'}</span>
        </button>
      )}

      {/* ── Left: Disease List ── */}
      <div className={`${
        kgMobile
          ? (kgListOpen ? 'flex flex-col max-h-[50vh] border-b border-slate-200 dark:border-slate-700' : 'hidden')
          : 'w-56 lg:w-72 shrink-0 flex flex-col border-r border-slate-100 dark:border-slate-800'
      } bg-slate-50/60 dark:bg-slate-900/60`}>
        <div className={`p-3 md:p-4 border-b border-slate-100 dark:border-slate-800 ${kgMobile ? '' : ''}`}>
          <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest mb-1.5">Stase</p>
          <select
            value={staseSlug}
            onChange={(e) => { setStaseSlug(e.target.value); setSelectedId(null); setLocalMindmap(null); }}
            className="w-full text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            {stases.map((s) => <option key={s.slug} value={s.slug}>{s.display_name}</option>)}
            {stases.length === 0 && <option value="ipd">Stase IPD</option>}
          </select>
        </div>
        <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2">
            <span className="material-symbols-outlined text-slate-400 text-[16px]">search</span>
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Cari penyakit..."
              className="flex-1 text-sm bg-transparent text-slate-700 dark:text-slate-200 placeholder-slate-400 focus:outline-none" />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto py-2">
          {loadingList ? (
            <div className="flex items-center justify-center h-20">
              <div className="w-5 h-5 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin" />
            </div>
          ) : Array.from(grouped.entries()).map(([group, rows]) => (
            <div key={group} className="mb-1">
              <p className="px-4 py-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-widest">{group}</p>
              {rows.map((d) => {
                const hasArticle = d.status && d.status !== 'missing';
                const isSelected = selectedId === d.id;
                const hasMindmap = isSelected && localMindmap && !localMindmap.not_generated && localMindmap.nodes.length > 0;
                return (
                  <button key={d.id} onClick={() => handleSelectDisease(d)} disabled={!hasArticle}
                    className={`w-full text-left px-4 py-2.5 flex items-center gap-2.5 transition-all ${
                      isSelected ? 'bg-indigo-50 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300'
                      : hasArticle ? 'text-slate-700 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-800/50'
                      : 'text-slate-400 dark:text-slate-600 cursor-not-allowed opacity-60'
                    }`}
                  >
                    <span className={`shrink-0 w-4 h-4 rounded-full border-2 flex items-center justify-center ${isSelected ? 'border-indigo-500 bg-indigo-500' : hasArticle ? 'border-slate-300' : 'border-slate-200'}`}>
                      {isSelected && <span className="w-1.5 h-1.5 bg-white rounded-full" />}
                    </span>
                    <span className="flex-1 text-xs leading-snug line-clamp-2">
                      <span className="text-slate-400 mr-1">#{d.catalog_no}</span>{d.name}
                    </span>
                    <div className="flex items-center gap-1 shrink-0">
                      {hasMindmap && <span className="material-symbols-outlined text-[11px] text-indigo-400" title="Mindmap tersedia">account_tree</span>}
                      {d.competency_level && (
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full ${
                          d.competency_level === '4' ? 'bg-emerald-100 text-emerald-700' :
                          d.competency_level?.startsWith('3') ? 'bg-sky-100 text-sky-700' : 'bg-slate-100 text-slate-500'
                        }`}>{d.competency_level}</span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* ── Right: Mindmap Canvas ── */}
      <div
        ref={graphPanelRef}
        className={`flex-1 flex flex-col min-w-0 min-h-0 ${isGraphFullscreen ? 'bg-white dark:bg-slate-950' : ''}`}
      >

        {/* Empty: nothing selected */}
        {!selectedId && !loadingMindmap && (
          <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center p-8 opacity-60">
            <span className="material-symbols-outlined text-6xl text-slate-300">account_tree</span>
            <div>
              <p className="font-semibold text-slate-500 mb-1">Pilih Penyakit</p>
              <p className="text-sm text-slate-400">Pilih penyakit yang sudah memiliki artikel di Medical Library.</p>
            </div>
          </div>
        )}

        {/* Loading mindmap from server */}
        {loadingMindmap && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3">
            <div className="w-10 h-10 border-2 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
            <p className="text-sm text-slate-500">Memuat mindmap...</p>
          </div>
        )}

        {/* Generating via Copilot */}
        {generating && (
          <div className="flex-1 flex flex-col items-center justify-center gap-4">
            <div className="w-12 h-12 border-2 border-violet-200 border-t-violet-600 rounded-full animate-spin" />
            <div className="text-center">
              <p className="font-semibold text-slate-600 dark:text-slate-300">Copilot sedang membangun mindmap...</p>
              <p className="text-sm text-slate-400 mt-1">Membaca artikel & menyusun node secara sistematis</p>
            </div>
          </div>
        )}

        {/* Not generated yet */}
        {selectedId && !loadingMindmap && !generating && localMindmap?.not_generated && (
          <div className="flex-1 flex flex-col items-center justify-center gap-5 p-8">
            <span className="material-symbols-outlined text-6xl text-slate-200">hub</span>
            <div className="text-center max-w-sm">
              <p className="font-semibold text-slate-600 dark:text-slate-300 mb-1">Mindmap belum dibuat</p>
              <p className="text-sm text-slate-400 leading-relaxed">Klik tombol di bawah untuk membuat mindmap otomatis menggunakan Copilot dari artikel Medical Library.</p>
            </div>
            <button onClick={handleGenerate}
              className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-2xl hover:bg-indigo-700 transition-all font-semibold shadow-lg shadow-indigo-200">
              <span className="material-symbols-outlined text-[20px]">auto_awesome</span>
              Generate Mindmap
            </button>
          </div>
        )}

        {/* Mindmap loaded */}
        {selectedId && !loadingMindmap && !generating && hasGraph && (
          <>
            {/* Header */}
            <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 shrink-0">
              <div className="flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h2 className="text-base font-headline font-bold text-slate-800 dark:text-slate-100 truncate">{localMindmap!.disease}</h2>
                    {localMindmap!.competency_level && (
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full shrink-0 ${
                        localMindmap!.competency_level === '4' ? 'bg-emerald-100 text-emerald-700' :
                        localMindmap!.competency_level?.startsWith('3') ? 'bg-sky-100 text-sky-700' : 'bg-slate-100 text-slate-500'
                      }`}>Level {localMindmap!.competency_level}</span>
                    )}
                    <span className="text-[10px] text-slate-400">{localMindmap!.nodes.length} node</span>
                    {hasUnsaved && <span className="text-[10px] text-amber-500 font-medium">● Belum disimpan</span>}
                    {saveMsg && <span className={`text-[10px] font-medium ${saveMsg === 'Tersimpan' ? 'text-emerald-500' : 'text-red-500'}`}>{saveMsg}</span>}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1.5 flex-wrap mt-2">
                {(localMindmap!.visual_refs?.length ?? 0) > 0 && (
                  <button onClick={() => setShowVisuals(!showVisuals)}
                    className="flex items-center gap-1 px-2.5 py-1.5 text-[11px] bg-amber-50 text-amber-600 border border-amber-100 rounded-full hover:bg-amber-100 transition-all">
                    <span className="material-symbols-outlined text-[13px]">imagesmode</span>
                    {localMindmap!.visual_refs!.length} Visual
                  </button>
                )}
                {/* Edit Mode Toggle */}
                <button
                  type="button"
                  onClick={toggleGraphFullscreen}
                  className="flex items-center gap-1 px-2.5 py-1.5 text-[11px] bg-slate-50 text-slate-600 border border-slate-200 rounded-full hover:bg-slate-100 transition-all font-medium"
                  title={isGraphFullscreen ? 'Keluar layar penuh' : 'Layar penuh'}
                >
                  <span className="material-symbols-outlined text-[13px]">{isGraphFullscreen ? 'fullscreen_exit' : 'fullscreen'}</span>
                  {isGraphFullscreen ? 'Keluar' : 'Layar penuh'}
                </button>
                <button
                  onClick={() => { setEditMode(m => !m); setActiveNode(null); }}
                  className={`flex items-center gap-1 px-2.5 py-1.5 text-[11px] rounded-full border transition-all font-medium ${
                    editMode ? 'bg-rose-500 text-white border-rose-600 shadow-sm' : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
                  }`}
                >
                  <span className="material-symbols-outlined text-[13px]">{editMode ? 'edit_off' : 'edit'}</span>
                  {editMode ? 'Edit ON' : 'Edit'}
                </button>
                {/* Save */}
                {hasUnsaved && (
                  <button onClick={handleSave} disabled={saving}
                    className="flex items-center gap-1 px-2.5 py-1.5 text-[11px] bg-indigo-600 text-white rounded-full hover:bg-indigo-700 transition-all font-medium disabled:opacity-60">
                    <span className="material-symbols-outlined text-[13px]">{saving ? 'sync' : 'save'}</span>
                    {saving ? 'Menyimpan...' : 'Simpan'}
                  </button>
                )}
                {/* Regenerate */}
                <button onClick={() => { if (window.confirm('Regenerate mindmap? Data node yang sudah diedit akan ditimpa.')) handleGenerate(); }}
                  className="flex items-center gap-1 px-2.5 py-1.5 text-[11px] bg-violet-50 text-violet-600 border border-violet-100 rounded-full hover:bg-violet-100 transition-all">
                  <span className="material-symbols-outlined text-[13px]">auto_awesome</span>
                  Regenerate
                </button>
              </div>
            </div>

            {/* Visual Refs Strip */}
            {showVisuals && (localMindmap!.visual_refs?.length ?? 0) > 0 && (
              <div className="px-5 py-3 border-b border-slate-100 dark:border-slate-800 bg-amber-50/60 flex gap-4 overflow-x-auto shrink-0">
                {localMindmap!.visual_refs!.map((vref, i) => (
                  <div key={i} className="shrink-0 flex flex-col items-center gap-1">
                    {vref.image_url ? (
                      <img src={`${API_URL}${vref.image_url}`} alt={vref.heading}
                        className="h-24 w-auto rounded-lg border border-amber-100 object-cover"
                        onError={(e) => { e.currentTarget.style.display = 'none'; }} />
                    ) : (
                      <div className="h-24 w-28 bg-amber-100 rounded-lg flex items-center justify-center">
                        <span className="material-symbols-outlined text-amber-400">image</span>
                      </div>
                    )}
                    <p className="text-[10px] text-center text-amber-700 font-medium max-w-28 line-clamp-2">{vref.heading}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Legend + edit hint */}
            <div className="px-5 py-1.5 border-b border-slate-100 dark:border-slate-800 flex flex-wrap gap-2 items-center shrink-0 bg-white/50 dark:bg-slate-900/50">
              {[
                { label: 'Root', bg: '#6366f1', text: '#fff', border: '#4338ca' },
                { label: 'Section', bg: '#0ea5e9', text: '#fff', border: '#0369a1' },
                { label: 'Konsep', bg: '#d1fae5', text: '#065f46', border: '#10b981' },
                { label: 'Fakta', bg: '#fef3c7', text: '#92400e', border: '#f59e0b' },
              ].map(item => (
                <div key={item.label} className="flex items-center">
                  <div className="px-1.5 py-0.5 rounded text-[9px] font-semibold border" style={{ backgroundColor: item.bg, color: item.text, borderColor: item.border }}>{item.label}</div>
                </div>
              ))}
              <span className="text-[10px] text-slate-400 ml-auto">
                {editMode ? '✏️ Klik node untuk edit · Klik canvas kosong untuk deselect' : 'Klik node untuk ringkasan · Scroll zoom'}
              </span>
            </div>

            {/* Canvas + edit panel */}
            <div className="flex-1 flex min-h-0 min-w-0 relative">
              {/* Canvas */}
              <div className="absolute inset-0" ref={containerRef} onClick={(e) => { if ((e.target as HTMLElement).tagName === 'CANVAS') setActiveNode(null); }}>
                <ForceGraph2D
                  key={graphKey}
                  ref={graphRef}
                  graphData={graphData as any}
                  width={dimensions.width}
                  height={dimensions.height}
                  cooldownTicks={0}
                  warmupTicks={0}
                  d3AlphaDecay={1}
                  onEngineStop={() => {
                    if (!graphRef.current) return;
                    // #region agent log
                    console.log('[DBG-087f38] onEngineStop', { dimW: dimensions.width, dimH: dimensions.height, containerW: containerRef.current?.offsetWidth, containerH: containerRef.current?.offsetHeight });
                    // #endregion
                    graphRef.current.zoomToFit(350, 40);
                  }}
                  linkColor={() => 'rgba(148,163,184,0.45)'}
                  linkWidth={1.4}
                  linkCurvature={0}
                  linkDirectionalArrowLength={5}
                  linkDirectionalArrowRelPos={1}
                  enableNodeDrag={editMode}
                  onNodeClick={handleNodeClick}
                  nodeCanvasObjectMode={() => 'replace' as const}
                  nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) =>
                    drawMindmapNode(node, ctx, globalScale, activeNode?.id === node.id)
                  }
                  nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
                    ctx.fillStyle = color;
                    ctx.fillRect((node.x as number) - ((node.__bw as number) ?? 60) / 2, (node.y as number) - ((node.__bh as number) ?? 22) / 2, (node.__bw as number) ?? 60, (node.__bh as number) ?? 22);
                  }}
                />

                {/* View mode popup */}
                {!editMode && activeNode && (
                  <div className="absolute bottom-2 left-2 right-2 md:left-auto md:bottom-4 md:right-4 max-w-full md:max-w-xs bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-100 dark:border-slate-800 p-3 md:p-4 z-10">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border ${
                        activeNode.type === 'root' ? 'bg-indigo-600 text-white border-indigo-700' :
                        activeNode.type === 'section' ? 'bg-sky-500 text-white border-sky-600' :
                        activeNode.type === 'concept' ? 'bg-emerald-50 text-emerald-700 border-emerald-300' :
                        'bg-amber-50 text-amber-700 border-amber-300'
                      }`}>{activeNode.type}</span>
                      <button onClick={() => setActiveNode(null)} className="text-slate-400 hover:text-slate-600 shrink-0">
                        <span className="material-symbols-outlined text-[16px]">close</span>
                      </button>
                    </div>
                    <p className="font-headline font-bold text-sm md:text-base text-slate-800 dark:text-slate-100 mb-1.5 md:mb-2 leading-snug">{activeNode.label}</p>
                    <p className="text-xs md:text-sm text-slate-600 dark:text-slate-300 leading-relaxed line-clamp-4 md:line-clamp-none">{activeNode.summary}</p>
                  </div>
                )}
              </div>

              {/* Edit Panel — bottom sheet on mobile, side panel on desktop */}
              {editMode && activeNode && (
                <>
                  {/* Mobile overlay */}
                  {kgMobile && <div className="fixed inset-0 bg-black/40 z-30" onClick={() => setActiveNode(null)} />}
                  <div className={`${
                    kgMobile
                      ? 'fixed bottom-0 left-0 right-0 z-40 rounded-t-2xl max-h-[80vh] shadow-2xl'
                      : 'w-72 shrink-0 border-l border-rose-100 dark:border-slate-700 z-20'
                  } bg-white dark:bg-slate-900 flex flex-col overflow-y-auto`}>
                  {/* Panel header */}
                  <div className="px-4 py-3 border-b border-rose-100 dark:border-slate-800 flex items-center justify-between shrink-0 bg-rose-50 dark:bg-rose-900/20">
                    <div className="flex items-center gap-2">
                      <span className="material-symbols-outlined text-rose-500 text-[18px]">edit_note</span>
                      <span className="text-sm font-semibold text-rose-700 dark:text-rose-400">Edit Node</span>
                    </div>
                    <button onClick={() => setActiveNode(null)} className="text-rose-400 hover:text-rose-600">
                      <span className="material-symbols-outlined text-[16px]">close</span>
                    </button>
                  </div>

                  <div className="flex-1 p-4 flex flex-col gap-4">
                    {/* Type selector */}
                    <div>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Tipe Node</label>
                      <div className="flex gap-1.5 flex-wrap">
                        {NODE_TYPE_OPTIONS.map(t => (
                          <button key={t} onClick={() => setEditType(t)}
                            className={`text-[10px] font-bold px-2.5 py-1 rounded-lg border transition-all ${editType === t
                              ? t === 'root' ? 'bg-indigo-600 text-white border-indigo-700'
                              : t === 'section' ? 'bg-sky-500 text-white border-sky-600'
                              : t === 'concept' ? 'bg-emerald-500 text-white border-emerald-600'
                              : 'bg-amber-400 text-white border-amber-500'
                              : 'bg-slate-50 text-slate-500 border-slate-200 hover:border-slate-300'
                            }`}
                          >{t}</button>
                        ))}
                      </div>
                    </div>

                    {/* Label */}
                    <div>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Label Node</label>
                      <input
                        value={editLabel}
                        onChange={(e) => setEditLabel(e.target.value)}
                        className="w-full text-sm bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2.5 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-rose-300"
                        placeholder="Nama node..."
                      />
                    </div>

                    {/* Summary / Note */}
                    <div>
                      <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">Catatan / Ringkasan</label>
                      <textarea
                        value={editSummary}
                        onChange={(e) => setEditSummary(e.target.value)}
                        rows={6}
                        className="w-full text-sm bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2.5 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-rose-300 resize-none leading-relaxed"
                        placeholder="Tulis ringkasan atau catatan belajar..."
                      />
                    </div>

                    {/* Apply edits */}
                    <button onClick={applyNodeEdit}
                      className="w-full py-2.5 bg-rose-500 hover:bg-rose-600 text-white rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2">
                      <span className="material-symbols-outlined text-[16px]">check</span>
                      Terapkan Perubahan
                    </button>

                    <div className="border-t border-slate-100 dark:border-slate-800 pt-3 flex flex-col gap-2">
                      {/* Add child */}
                      <button onClick={() => handleAddChild(activeNode.id)}
                        className="w-full py-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-100 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2">
                        <span className="material-symbols-outlined text-[15px]">add_circle</span>
                        Tambah Node Anak
                      </button>
                      {/* Delete */}
                      {activeNode.type !== 'root' && (
                        <button onClick={() => handleDeleteNode(activeNode.id)}
                          className="w-full py-2 bg-red-50 hover:bg-red-100 text-red-600 border border-red-100 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2">
                          <span className="material-symbols-outlined text-[15px]">delete</span>
                          Hapus Node & Anak
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                </>
              )}
            </div>

          </>
        )}
      </div>
    </div>
  );
}

// ─── MEDICAL LIBRARY PANEL ───────────────────────────────────────────────────
function MedicalLibraryPanel({ components }: { components: typeof mdComponents }) {
  const [stases, setStases] = useState<LibraryStase[]>([]);
  const [staseSlug, setStaseSlug] = useState('ipd');
  const [diseases, setDiseases] = useState<LibraryDiseaseRow[]>([]);
  const [progress, setProgress] = useState({ filled: 0, total: 0, percent: 0 });
  const [search, setSearch] = useState('');
  const [onlyMissing, setOnlyMissing] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<LibraryDiseaseDetail | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [busy, setBusy] = useState(false);
  const [extraPrompt, setExtraPrompt] = useState('');
  const [refineOpen, setRefineOpen] = useState(false);
  const [refineText, setRefineText] = useState('');
  const [editOpen, setEditOpen] = useState(false);
  const [editMarkdown, setEditMarkdown] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewBase, setPreviewBase] = useState('');
  const [previewCandidate, setPreviewCandidate] = useState('');
  const [previewCombinedEdit, setPreviewCombinedEdit] = useState('');
  const [previewNote, setPreviewNote] = useState('');
  const [mergeAiLoading, setMergeAiLoading] = useState(false);
  const [visualOpen, setVisualOpen] = useState(false);
  const [visualBusy, setVisualBusy] = useState(false);
  const [visualSuggestions, setVisualSuggestions] = useState<ImageItem[]>([]);
  const [visualSelected, setVisualSelected] = useState<ImageItem[]>([]);
  const [mobileDetailView, setMobileDetailView] = useState(false);
  const { isMobile: libMobile } = useScreenSize();

  useEffect(() => {
    axios
      .get<{ stases: LibraryStase[] }>(`${API_URL}/library/stases`)
      .then((r) => setStases(r.data.stases || []))
      .catch(() => setStases([]));
  }, []);

  const loadDiseases = useCallback(async () => {
    setLoadingList(true);
    setErr(null);
    try {
      const r = await axios.get(`${API_URL}/library/stases/${staseSlug}/diseases`);
      setDiseases(r.data.diseases || []);
      setProgress(r.data.progress || { filled: 0, total: 0, percent: 0 });
    } catch {
      setErr('Gagal memuat daftar penyakit. Pastikan API berjalan.');
    } finally {
      setLoadingList(false);
    }
  }, [staseSlug]);

  useEffect(() => {
    void loadDiseases();
  }, [loadDiseases]);

  const loadDetail = useCallback(
    async (id: number) => {
      setLoadingDetail(true);
      setSelectedId(id);
      setErr(null);
      try {
        const r = await axios.get<LibraryDiseaseDetail>(`${API_URL}/library/stases/${staseSlug}/diseases/${id}`);
        setDetail(r.data);
        setEditMarkdown(r.data.markdown || '');
        setMobileDetailView(true); // on mobile, switch to detail view
      } catch {
        setDetail(null);
        setErr('Gagal memuat detail penyakit.');
      } finally {
        setLoadingDetail(false);
      }
    },
    [staseSlug]
  );

  const filtered = useMemo(() => {
    let rows = diseases;
    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter((d) => d.name.toLowerCase().includes(q) || String(d.catalog_no).includes(q));
    }
    if (onlyMissing) {
      rows = rows.filter((d) => !d.status || d.status === 'missing');
    }
    return rows;
  }, [diseases, search, onlyMissing]);

  const grouped = useMemo(() => {
    const m = new Map<string, LibraryDiseaseRow[]>();
    for (const d of filtered) {
      const g = d.group_label || 'Umum';
      if (!m.has(g)) m.set(g, []);
      m.get(g)!.push(d);
    }
    return m;
  }, [filtered]);

  const handlePreview = async () => {
    if (!selectedId) return;
    setPreviewLoading(true);
    setErr(null);
    try {
      const r = await axios.post<LibraryPreviewResponse>(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/preview`, {
        extra_prompt: extraPrompt.trim() || null,
        combine_with_existing: false,
        combine_mode: 'replace' as const,
        persist: false,
      });
      setPreviewBase(r.data.markdown_base || '');
      setPreviewCandidate(r.data.markdown_candidate || '');
      setPreviewCombinedEdit(r.data.markdown_combined || '');
      setPreviewNote(r.data.preview_note || '');
      setPreviewOpen(true);
      await loadDiseases();
      if (selectedId) await loadDetail(selectedId);
    } catch {
      setErr('Pratinjau regenerate gagal. Periksa API atau GITHUB_TOKEN.');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleApplyPreview = async () => {
    if (!selectedId) return;
    setBusy(true);
    setErr(null);
    try {
      await axios.patch(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/content`, {
        markdown: previewCombinedEdit,
        preview_commit: true,
      });
      setPreviewOpen(false);
      await loadDiseases();
      await loadDetail(selectedId);
    } catch {
      setErr('Gagal menyimpan hasil gabungan.');
    } finally {
      setBusy(false);
    }
  };

  const handleAiMergePreview = async () => {
    if (!previewCandidate.trim()) {
      setErr('Tidak ada kandidat baru untuk digabung.');
      return;
    }
    setMergeAiLoading(true);
    setErr(null);
    try {
      const r = await axios.post<{ markdown_merged: string }>(`${API_URL}/library/merge_markdown_copilot`, {
        markdown_base: previewBase,
        markdown_candidate: previewCandidate,
      });
      setPreviewCombinedEdit(r.data.markdown_merged);
    } catch {
      setErr('Gabung dengan AI gagal. Pastikan GITHUB_TOKEN pada server dan coba lagi.');
    } finally {
      setMergeAiLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!selectedId) return;
    if (detail?.markdown && !window.confirm('Timpa artikel yang ada langsung ke disk (tanpa pratinjau)?')) {
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await axios.post(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/generate`, {
        extra_prompt: extraPrompt.trim() || null,
      });
      setExtraPrompt('');
      await loadDiseases();
      await loadDetail(selectedId);
    } catch {
      setErr('Generate gagal. Periksa GITHUB_TOKEN atau koneksi API.');
    } finally {
      setBusy(false);
    }
  };

  const handleRefine = async () => {
    if (!selectedId || !refineText.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      await axios.post(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/refine`, {
        instruction: refineText.trim(),
      });
      setRefineOpen(false);
      setRefineText('');
      await loadDiseases();
      await loadDetail(selectedId);
    } catch {
      setErr('Perbarui dengan instruksi gagal (perlu GITHUB_TOKEN).');
    } finally {
      setBusy(false);
    }
  };

  const handleSaveEdit = async () => {
    if (!selectedId) return;
    setBusy(true);
    setErr(null);
    try {
      await axios.patch(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/content`, {
        markdown: editMarkdown,
      });
      setEditOpen(false);
      await loadDiseases();
      await loadDetail(selectedId);
    } catch {
      setErr('Simpan suntingan gagal.');
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedId) return;
    if (!window.confirm('Hapus artikel permanen untuk penyakit ini?')) return;
    setBusy(true);
    setErr(null);
    try {
      await axios.delete(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/article`);
      setDetail(null);
      setSelectedId(null);
      await loadDiseases();
    } catch {
      setErr('Penghapusan gagal.');
    } finally {
      setBusy(false);
    }
  };

  const openVisualManager = async () => {
    if (!selectedId || !detail) return;
    setErr(null);
    setVisualOpen(true);
    setVisualSelected(detail.images || []);
    setVisualSuggestions([]);

    const diseaseName = (detail.disease.name as string) || '';
    if (!diseaseName.trim()) return;
    try {
      const r = await axios.post<{ images: ImageItem[] }>(`${API_URL}/get_related_images`, {
        disease_name: diseaseName,
        limit: 10,
      });
      setVisualSuggestions(r.data.images || []);
    } catch {
      // non-fatal
    }
  };

  const saveVisualRefs = async () => {
    if (!selectedId) return;
    const bad = visualSelected.filter((img) => !imageItemHasStableRef(img));
    if (bad.length > 0) {
      setErr(
        'Sebagian gambar tidak punya referensi yang bisa disimpan (path lokal, image_ref, storage_url, atau URL gambar). Pilih ulang dari daftar atau rekomendasi.',
      );
      return;
    }
    const body = {
      images: visualSelected.map((img) => ({
        image_abs_path: (img.image_abs_path || '').trim() || undefined,
        image_ref: (img.image_ref || '').trim() || undefined,
        storage_url: (img.storage_url || '').trim() || undefined,
        image_url: (img.image_url || '').trim() || undefined,
        heading: img.heading || '',
        source_name: img.source_name || '',
        page_no: img.page_no || 0,
      })),
    };
    setVisualBusy(true);
    setErr(null);
    try {
      await axios.patch(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/visual_refs`, body);
      setVisualOpen(false);
      await loadDetail(selectedId);
    } catch {
      setErr('Gagal menyimpan referensi visual.');
    } finally {
      setVisualBusy(false);
    }
  };

  const staseOptions = stases.length
    ? stases
    : [{ id: 0, slug: 'ipd', display_name: 'Stase IPD', sort_order: 0, disease_count: 0, filled_count: 0 }];

  return (
    <div className="flex-1 flex flex-col min-h-0 h-full bg-background">
      <div className="flex-1 flex flex-col lg:flex-row min-h-0 gap-0">
        {/* List column — hidden on mobile when detail is open */}
        <aside className={`${
          libMobile && mobileDetailView ? 'hidden'
          : 'flex flex-col'
        } w-full md:w-[min(320px,40vw)] lg:w-[min(420px,40vw)] shrink-0 border-r border-slate-200/60 dark:border-slate-800 bg-slate-50/40 dark:bg-slate-950/30`}>
          <div className="p-3 md:p-4 border-b border-slate-200/50 space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-lg font-headline font-bold text-slate-800 dark:text-slate-100">Medical Library</h2>
              <button
                type="button"
                onClick={() => void loadDiseases()}
                className="p-2 rounded-xl text-slate-500 hover:bg-white/60 dark:hover:bg-slate-800/60"
                title="Muat ulang"
              >
                <span className="material-symbols-outlined text-[20px]">refresh</span>
              </button>
            </div>
            <label className="block text-xs font-label text-slate-500">Stase</label>
            <select
              value={staseSlug}
              onChange={(e) => {
                setStaseSlug(e.target.value);
                setSelectedId(null);
                setDetail(null);
              }}
              className="w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 px-3 py-2 text-sm"
            >
              {staseOptions.map((s) => (
                <option key={s.slug} value={s.slug}>
                  {s.display_name}
                </option>
              ))}
            </select>
            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>Progres penjelasan</span>
                <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                  {progress.filled}/{progress.total} ({progress.percent}%)
                </span>
              </div>
              <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                <div
                  className="h-full rounded-full bg-linear-to-r from-indigo-500 to-violet-500 transition-all duration-500"
                  style={{ width: `${Math.min(100, progress.percent)}%` }}
                />
              </div>
            </div>
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Cari nama atau nomor..."
              className="w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-white/80 px-3 py-2 text-sm"
            />
            <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
              <input
                type="checkbox"
                checked={onlyMissing}
                onChange={(e) => setOnlyMissing(e.target.checked)}
                className="rounded border-slate-300"
              />
              Hanya yang belum ada penjelasan
            </label>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {loadingList && <p className="text-center text-sm text-slate-400 py-6">Memuat...</p>}
            {err && <p className="text-xs text-amber-600 px-2 py-2">{err}</p>}
            {!loadingList &&
              Array.from(grouped.entries()).map(([group, rows]) => (
                <div key={group} className="mb-4">
                  <div className="sticky top-0 z-10 bg-slate-100/90 dark:bg-slate-900/90 backdrop-blur px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500 border-b border-slate-200/60">
                    {group}
                  </div>
                  <ul className="mt-1 space-y-0.5">
                    {rows.map((d) => {
                      const done = d.status === 'draft' || d.status === 'published';
                      const active = selectedId === d.id;
                      return (
                        <li key={d.id}>
                          <button
                            type="button"
                            onClick={() => void loadDetail(d.id)}
                            className={`w-full text-left px-3 py-2.5 rounded-xl flex items-start gap-2 transition-colors ${
                              active
                                ? 'bg-indigo-50 dark:bg-indigo-900/40 text-indigo-800 dark:text-indigo-200'
                                : 'hover:bg-white/60 dark:hover:bg-slate-800/50'
                            }`}
                          >
                            <span
                              className={`material-symbols-outlined text-[18px] shrink-0 mt-0.5 ${
                                done ? 'text-emerald-500' : 'text-slate-300'
                              }`}
                              style={{ fontVariationSettings: "'FILL' 1" }}
                            >
                              {done ? 'check_circle' : 'radio_button_unchecked'}
                            </span>
                            <span className="flex-1 min-w-0">
                              <span className="text-[10px] text-slate-400 font-mono">#{d.catalog_no}</span>
                              <span className="block text-sm font-medium leading-snug">{d.name}</span>
                            </span>
                            {d.competency_level && (
                              <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded-md bg-slate-200/80 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
                                {d.competency_level}
                              </span>
                            )}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
          </div>
        </aside>

        {/* Detail column — hidden on mobile unless detail selected */}
        <section className={`${
          libMobile && !mobileDetailView ? 'hidden'
          : 'flex flex-col'
        } flex-1 min-w-0 min-h-0 overflow-y-auto bg-white/40 dark:bg-slate-900/20`}>
          {!selectedId && (
            <div className="m-auto text-center max-w-sm p-6 md:p-8 text-slate-400">
              <span className="material-symbols-outlined text-4xl md:text-5xl mb-3 opacity-40">menu_book</span>
              <p className="text-sm">Pilih penyakit di daftar untuk melihat atau membuat penjelasan.</p>
              {libMobile && (
                <button
                  type="button"
                  onClick={() => setMobileDetailView(false)}
                  className="mt-4 flex items-center gap-1.5 mx-auto px-4 py-2 text-sm text-indigo-600 bg-indigo-50 rounded-full font-medium"
                >
                  <span className="material-symbols-outlined text-[16px]">arrow_back</span>
                  Kembali ke daftar
                </button>
              )}
            </div>
          )}
          {selectedId && loadingDetail && (
            <div className="m-auto flex flex-col items-center gap-2 text-slate-400">
              <span className="material-symbols-outlined animate-spin">progress_activity</span>
              <span className="text-sm">Memuat artikel...</span>
            </div>
          )}
          {selectedId && !loadingDetail && detail && (
            <div className="p-3 md:p-4 lg:p-8 max-w-4xl mx-auto w-full space-y-4 md:space-y-6 pb-24">
              {/* Mobile back button */}
              {libMobile && (
                <button
                  type="button"
                  onClick={() => { setMobileDetailView(false); setSelectedId(null); setDetail(null); }}
                  className="flex items-center gap-1.5 text-sm text-indigo-600 font-medium mb-2"
                >
                  <span className="material-symbols-outlined text-[18px]">arrow_back</span>
                  Daftar penyakit
                </button>
              )}
              <div className="flex flex-wrap items-start justify-between gap-3 md:gap-4">
                <div>
                  <p className="text-xs font-label text-slate-500 uppercase tracking-widest">Penyakit terpilih</p>
                  <h2 className="text-2xl font-headline font-bold text-slate-800 dark:text-slate-100">
                    {(detail.disease.name as string) || '—'}
                  </h2>
                  <p className="text-xs text-slate-500 mt-1">
                    Status:{' '}
                    <span className="font-semibold text-indigo-600">
                      {(detail.disease.status as string) || 'missing'}
                    </span>
                  </p>
                </div>
                <div className="grid grid-cols-2 md:flex md:flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={busy || previewLoading}
                    onClick={() => void handlePreview()}
                    className="px-4 py-2 rounded-full bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
                  >
                    {previewLoading ? '…' : 'Pratinjau regenerate'}
                  </button>
                  <button
                    type="button"
                    disabled={busy || previewLoading}
                    onClick={() => void handleGenerate()}
                    className="px-4 py-2 rounded-full border border-indigo-200 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-800 dark:text-indigo-200 text-sm font-medium hover:bg-indigo-100 disabled:opacity-50"
                    title="Menimpa artikel di disk tanpa pratinjau"
                  >
                    {busy ? '…' : 'Simpan langsung'}
                  </button>
                  <button
                    type="button"
                    disabled={busy || !detail.markdown}
                    onClick={() => {
                      setEditMarkdown(detail.markdown || '');
                      setEditOpen(true);
                    }}
                    className="px-4 py-2 rounded-full border border-slate-200 dark:border-slate-600 text-sm"
                  >
                    Sunting
                  </button>
                  <button
                    type="button"
                    disabled={busy || !detail.markdown}
                    onClick={() => setRefineOpen(true)}
                    className="px-4 py-2 rounded-full border border-violet-200 bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300 text-sm"
                  >
                    Instruksi AI
                  </button>
                  <button
                    type="button"
                    disabled={busy || !detail.markdown}
                    onClick={() => void handleDelete()}
                    className="px-4 py-2 rounded-full border border-red-200 text-red-600 text-sm hover:bg-red-50"
                  >
                    Hapus
                  </button>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200/80 dark:border-slate-700 p-4 bg-white/60 dark:bg-slate-900/40 space-y-3">
                <label className="text-xs font-semibold text-slate-500">Prompt tambahan (opsional, untuk regenerate)</label>
                <textarea
                  value={extraPrompt}
                  onChange={(e) => setExtraPrompt(e.target.value)}
                  placeholder="Contoh: tekankan diagnosis banding dan red flag..."
                  className="w-full rounded-xl border border-slate-200 dark:border-slate-600 bg-transparent px-3 py-2 text-sm min-h-18"
                />
              </div>

              {detail.images && detail.images.length > 0 && (
                <div>
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <h3 className="font-headline text-sm font-bold text-slate-500 flex items-center gap-2 uppercase tracking-widest">
                      <span className="material-symbols-outlined text-[18px]">imagesmode</span>
                      Referensi visual
                    </h3>
                    <button
                      type="button"
                      onClick={() => void openVisualManager()}
                      className="text-xs px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700 hover:bg-white/60"
                    >
                      Kelola
                    </button>
                  </div>
                  <div className="flex gap-4 overflow-x-auto pb-2 snap-x">
                    {detail.images.map((img, iIdx) => (
                      <a
                        key={iIdx}
                        href={resolveImageUrl(img.image_url)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="shrink-0 border border-slate-100 dark:border-slate-800 rounded-2xl overflow-hidden w-48 shadow-sm hover:border-indigo-400 transition-colors snap-center"
                      >
                        <img
                          src={resolveImageUrl(img.image_url)}
                          alt={img.heading}
                          className="w-full h-32 object-cover"
                          onError={(e) => {
                            e.currentTarget.onerror = null;
                            e.currentTarget.src = PLACEHOLDER_IMG;
                          }}
                        />
                        <div className="p-2 bg-white/90 dark:bg-slate-900/90">
                          <p className="text-[11px] text-center truncate text-slate-700 dark:text-slate-300">{img.heading}</p>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>
              )}
              {(!detail.images || detail.images.length === 0) && (
                <div className="flex items-center justify-between gap-2">
                  <h3 className="font-headline text-sm font-bold text-slate-500 flex items-center gap-2 uppercase tracking-widest">
                    <span className="material-symbols-outlined text-[18px]">imagesmode</span>
                    Referensi visual
                  </h3>
                  <button
                    type="button"
                    onClick={() => void openVisualManager()}
                    className="text-xs px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700 hover:bg-white/60"
                  >
                    Tambah
                  </button>
                </div>
              )}

              <div className="rounded-4xl border border-white/40 bg-surface-container-low/50 p-6 md:p-10 glass-border">
                {detail.markdown ? (
                  <article className="prose-slate prose max-w-none text-on-surface">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                      {detail.markdown}
                    </ReactMarkdown>
                  </article>
                ) : (
                  <p className="text-slate-500 text-sm italic">Belum ada artikel. Gunakan Generate untuk membuat dari RAG.</p>
                )}
              </div>
            </div>
          )}
        </section>
      </div>

      {refineOpen && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4" onClick={() => setRefineOpen(false)}>
          <div className="bg-white dark:bg-slate-900 rounded-2xl max-w-lg w-full p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="font-headline font-bold text-lg mb-2">Perbarui dengan instruksi</h3>
            <p className="text-xs text-slate-500 mb-3">Membutuhkan GITHUB_TOKEN di server.</p>
            <textarea
              value={refineText}
              onChange={(e) => setRefineText(e.target.value)}
              className="w-full rounded-xl border border-slate-200 dark:border-slate-600 px-3 py-2 text-sm min-h-30"
              placeholder="Contoh: ringkas bagian etiologi, tambahkan tabel dosis..."
            />
            <div className="flex justify-end gap-2 mt-4">
              <button type="button" className="px-4 py-2 text-sm rounded-full" onClick={() => setRefineOpen(false)}>
                Batal
              </button>
              <button
                type="button"
                disabled={busy || refineText.trim().length < 3}
                className="px-4 py-2 text-sm rounded-full bg-indigo-600 text-white disabled:opacity-50"
                onClick={() => void handleRefine()}
              >
                Terapkan
              </button>
            </div>
          </div>
        </div>
      )}

      {previewOpen && (
        <div
          className="fixed inset-0 z-60 bg-black/50 flex items-center justify-center p-4 overflow-y-auto"
          onClick={() => setPreviewOpen(false)}
        >
          <div
            className="bg-white dark:bg-slate-900 rounded-2xl max-w-[min(96rem,100%)] w-full p-4 md:p-6 shadow-xl my-4 max-h-[95vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start gap-2 mb-3">
              <div>
                <h3 className="font-headline font-bold text-lg">Pratinjau regenerate</h3>
                <p className="text-xs text-slate-500 mt-1">{previewNote}</p>
                {err && <p className="text-xs text-amber-700 dark:text-amber-400 mt-2 max-w-xl">{err}</p>}
              </div>
              <button
                type="button"
                className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800"
                onClick={() => setPreviewOpen(false)}
                aria-label="Tutup"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 flex-1 min-h-0">
              <div className="flex flex-col min-h-0 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
                <div className="text-xs font-semibold px-3 py-2 bg-slate-100 dark:bg-slate-800 text-slate-600">Artikel utama (saat ini)</div>
                <div className="overflow-y-auto p-3 max-h-[min(40vh,320px)] lg:max-h-[min(70vh,480px)] prose prose-sm dark:prose-invert max-w-none text-sm">
                  {previewBase.trim() ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                      {previewBase}
                    </ReactMarkdown>
                  ) : (
                    <p className="text-slate-400 italic text-sm">(Kosong)</p>
                  )}
                </div>
              </div>
              <div className="flex flex-col min-h-0 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
                <div className="text-xs font-semibold px-3 py-2 bg-indigo-50 dark:bg-indigo-950/50 text-indigo-800 dark:text-indigo-200">
                  Kandidat baru
                </div>
                <div className="overflow-y-auto p-3 max-h-[min(40vh,320px)] lg:max-h-[min(70vh,480px)] prose prose-sm dark:prose-invert max-w-none text-sm">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                    {previewCandidate}
                  </ReactMarkdown>
                </div>
              </div>
              <div className="flex flex-col min-h-0 border border-violet-200 dark:border-violet-800 rounded-xl overflow-hidden lg:col-span-1">
                <div className="text-xs font-semibold px-3 py-2 bg-violet-50 dark:bg-violet-950/40 text-violet-800 dark:text-violet-200 flex flex-wrap items-center justify-between gap-2">
                  <span>Hasil gabungan (bisa diedit)</span>
                  <button
                    type="button"
                    disabled={mergeAiLoading || !previewCandidate.trim() || busy}
                    onClick={() => void handleAiMergePreview()}
                    className="shrink-0 inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-violet-600 text-white text-[11px] font-medium hover:bg-violet-700 disabled:opacity-50"
                    title="Menggabungkan artikel utama dan kandidat dengan Copilot"
                  >
                    <span className="material-symbols-outlined text-[14px]">auto_awesome</span>
                    {mergeAiLoading ? 'Menggabung…' : 'Gabung dengan AI'}
                  </button>
                </div>
                <textarea
                  value={previewCombinedEdit}
                  onChange={(e) => setPreviewCombinedEdit(e.target.value)}
                  className="flex-1 w-full min-h-[min(40vh,280px)] lg:min-h-[min(70vh,440px)] p-3 text-xs font-mono bg-slate-50/80 dark:bg-slate-950/40 border-0 resize-y focus:ring-2 focus:ring-violet-400 outline-none"
                  spellCheck={false}
                />
              </div>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">
              Gunakan <strong className="font-medium">Gabung dengan AI</strong> untuk menyatukan kolom kiri dan tengah secara detail (Copilot), lalu sunting bila perlu. Perubahan di kolom kanan disimpan ke server saat Anda menekan <strong className="font-medium">Terapkan ke artikel utama</strong>.
            </p>
            <div className="flex justify-end gap-2 mt-4 flex-wrap">
              <button type="button" className="px-4 py-2 text-sm rounded-full" onClick={() => setPreviewOpen(false)}>
                Batal
              </button>
              <button
                type="button"
                disabled={busy || !previewCombinedEdit.trim()}
                className="px-4 py-2 text-sm rounded-full bg-indigo-600 text-white disabled:opacity-50"
                onClick={() => void handleApplyPreview()}
              >
                Terapkan ke artikel utama
              </button>
            </div>
          </div>
        </div>
      )}

      {visualOpen && (
        <div
          className="fixed inset-0 z-55 bg-black/50 flex items-center justify-center p-4 overflow-y-auto"
          onClick={() => setVisualOpen(false)}
        >
          <div
            className="bg-white dark:bg-slate-900 rounded-2xl max-w-5xl w-full p-4 md:p-6 shadow-xl my-4 max-h-[92vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start gap-2 mb-3">
              <div>
                <h3 className="font-headline font-bold text-lg">Kelola Referensi Visual</h3>
                <p className="text-xs text-slate-500 mt-1">Tambah/hapus gambar yang ditampilkan pada artikel ini (disimpan ke meta.json).</p>
                {err && <p className="text-xs text-amber-700 dark:text-amber-400 mt-2">{err}</p>}
              </div>
              <button
                type="button"
                className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800"
                onClick={() => setVisualOpen(false)}
                aria-label="Tutup"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
              <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden flex flex-col min-h-0">
                <div className="px-3 py-2 text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-600">
                  Dipilih ({visualSelected.length})
                </div>
                <div className="p-3 overflow-y-auto flex-1 min-h-0">
                  {visualSelected.length === 0 ? (
                    <p className="text-sm text-slate-400 italic">Belum ada gambar dipilih.</p>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {visualSelected.map((img, idx) => (
                        <div key={`${img.image_url}-${idx}`} className="border border-slate-200 rounded-xl overflow-hidden relative">
                          <img
                            src={resolveImageUrl(img.image_url)}
                            alt={img.heading}
                            className="w-full h-24 object-cover"
                            onError={(e) => {
                              e.currentTarget.onerror = null;
                              e.currentTarget.src = PLACEHOLDER_IMG;
                            }}
                          />
                          <button
                            type="button"
                            onClick={() => setVisualSelected((prev) => prev.filter((_, i) => i !== idx))}
                            className="absolute top-2 right-2 bg-black/60 text-white rounded-full p-1 hover:bg-black/70"
                            title="Hapus dari pilihan"
                          >
                            <span className="material-symbols-outlined text-[16px]">delete</span>
                          </button>
                          <div className="p-2 bg-white/90">
                            <p className="text-[11px] truncate text-slate-700">{img.heading || 'Gambar'}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden flex flex-col min-h-0">
                <div className="px-3 py-2 text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/40 text-indigo-800 dark:text-indigo-200">
                  Rekomendasi dari RAG ({visualSuggestions.length})
                </div>
                <div className="p-3 overflow-y-auto flex-1 min-h-0">
                  {visualSuggestions.length === 0 ? (
                    <p className="text-sm text-slate-400 italic">Tidak ada rekomendasi (atau API belum siap).</p>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {visualSuggestions.map((img, idx) => {
                        const exists = visualSelected.some((s) => imageItemStableKey(s) === imageItemStableKey(img));
                        return (
                          <button
                            key={`${img.image_url}-${idx}`}
                            type="button"
                            disabled={exists}
                            onClick={() => setVisualSelected((prev) => [...prev, img])}
                            className={`border rounded-xl overflow-hidden text-left transition-colors ${exists ? 'opacity-60 border-slate-200' : 'hover:border-indigo-400 border-slate-200'}`}
                            title={exists ? 'Sudah dipilih' : 'Tambah'}
                          >
                            <img
                              src={resolveImageUrl(img.image_url)}
                              alt={img.heading}
                              className="w-full h-24 object-cover"
                              onError={(e) => {
                                e.currentTarget.onerror = null;
                                e.currentTarget.src = PLACEHOLDER_IMG;
                              }}
                            />
                            <div className="p-2 bg-white/90">
                              <p className="text-[11px] truncate text-slate-700">{img.heading || 'Gambar'}</p>
                              <p className="text-[10px] text-slate-400 truncate">{img.source_name} • Hal {img.page_no}</p>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-4 flex-wrap">
              <button type="button" className="px-4 py-2 text-sm rounded-full" onClick={() => setVisualOpen(false)}>
                Batal
              </button>
              <button
                type="button"
                disabled={visualBusy}
                className="px-4 py-2 text-sm rounded-full bg-indigo-600 text-white disabled:opacity-50"
                onClick={() => void saveVisualRefs()}
              >
                {visualBusy ? 'Menyimpan…' : 'Simpan referensi visual'}
              </button>
            </div>
          </div>
        </div>
      )}

      {editOpen && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4" onClick={() => setEditOpen(false)}>
          <div className="bg-white dark:bg-slate-900 rounded-2xl max-w-3xl w-full p-6 shadow-xl max-h-[90vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <h3 className="font-headline font-bold text-lg mb-2">Sunting Markdown</h3>
            <textarea
              value={editMarkdown}
              onChange={(e) => setEditMarkdown(e.target.value)}
              className="flex-1 w-full rounded-xl border border-slate-200 dark:border-slate-600 px-3 py-2 text-sm font-mono min-h-80 overflow-y-auto"
            />
            <div className="flex justify-end gap-2 mt-4">
              <button type="button" className="px-4 py-2 text-sm rounded-full" onClick={() => setEditOpen(false)}>
                Batal
              </button>
              <button
                type="button"
                disabled={busy}
                className="px-4 py-2 text-sm rounded-full bg-indigo-600 text-white"
                onClick={() => void handleSaveEdit()}
              >
                Simpan
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── COLLAPSIBLE SECTION COMPONENT ──────────────────────────────────────────
function CollapsibleSection({ title, children, icon }: { title: string; children: React.ReactNode; icon: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [shouldCollapse, setShouldCollapse] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (contentRef.current) {
      if (contentRef.current.scrollHeight > 500) {
        setShouldCollapse(true);
      } else {
        setShouldCollapse(false);
      }
    }
  }, [children]);

  return (
    <div className="group border-b border-slate-50 dark:border-slate-800/50 pb-6 mb-6 last:border-0">
      <h3 className="font-headline text-base md:text-lg font-bold text-indigo-600 dark:text-indigo-400 mb-4 flex items-center gap-2">
        <span className="material-symbols-outlined text-indigo-400/60 group-hover:text-indigo-500 transition-colors text-[22px]" style={{ fontVariationSettings: "'FILL' 1" }}>
          {icon}
        </span>
        {title}
      </h3>
      
      <div 
        ref={contentRef}
        className={`relative transition-all duration-500 ease-in-out overflow-hidden ${
          shouldCollapse && !isExpanded ? 'max-h-100' : 'max-h-none'
        }`}
      >
        <div className="prose-slate dark:prose-invert text-on-surface text-base leading-relaxed pl-1">
          {children}
        </div>

        {shouldCollapse && !isExpanded && (
          <div className="absolute bottom-0 left-0 right-0 h-32 bg-linear-to-t from-white dark:from-slate-900 to-transparent pointer-events-none flex items-end justify-center pb-2">
            <button 
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setIsExpanded(true); }}
              className="pointer-events-auto flex items-center gap-2 px-6 py-2.5 bg-indigo-600 text-white rounded-full text-sm font-semibold shadow-lg shadow-indigo-200 dark:shadow-indigo-900/40 hover:bg-indigo-700 transition-all transform hover:scale-105 active:scale-95"
            >
              <span className="material-symbols-outlined text-[18px]">expand_more</span>
              Tampilkan Selengkapnya
            </button>
          </div>
        )}
      </div>

      {shouldCollapse && isExpanded && (
        <div className="flex justify-center mt-4">
          <button 
            onClick={(e) => { e.preventDefault(); e.stopPropagation(); setIsExpanded(false); }}
            className="flex items-center gap-2 px-4 py-2 text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/20 rounded-full text-xs font-semibold hover:bg-indigo-100 transition-all"
          >
            <span className="material-symbols-outlined text-[16px]">expand_less</span>
            Sembunyikan Sebagian
          </button>
        </div>
      )}
    </div>
  );
}

// ─── MAIN APP ────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem('medrag_chat_v3');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [isLoading, setIsLoading] = useState(false);
  const [evidenceModal, setEvidenceModal] = useState<{ citation: string; content: EvidenceItem[] } | null>(null);
  const [imageModal, setImageModal] = useState<ImageItem | null>(null);
  const [isSidebarMinimized, setIsSidebarMinimized] = useState(false);
  const [kgInitialDisease, setKgInitialDisease] = useState<string | null>(null); // auto-select in KG panel
  const [activeView, setActiveView] = useState<ActiveView>('chat');
  const [activeStase, setActiveStase] = useState<string>('ipd');
  const { isMobile, isTablet } = useScreenSize();
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    localStorage.setItem('medrag_chat_v3', JSON.stringify(messages));
  }, [messages]);

  // Ide 11: Build chat_history from current messages for multi-turn
  const buildChatHistory = useCallback(() => {
    try {
      const history: { role: string; content: string }[] = [];
      for (const msg of messages) {
        if (msg.role === 'user') {
          history.push({ role: 'user', content: msg.content || '' });
        } else if (msg.role === 'bot' && msg.data?.draft_answer) {
          const sections = msg.data.draft_answer.sections || [];
          const summary = `Analisis ${msg.data.draft_answer.disease || 'Kondisi'}: ${sections.map(s => s.title).join(', ')}`;
          history.push({ role: 'assistant', content: summary });
        }
      }
      return history.slice(-8); // Keep last 4 exchanges
    } catch (e) {
      console.error("Error building chat history:", e);
      return [];
    }
  }, [messages]);

  const submitQuery = async () => {
    if (!query.trim() || isLoading) return;

    const nextQuery = query.trim();
    setMessages((prev) => [...prev, { role: 'user', content: nextQuery }]);
    setQuery('');
    setIsLoading(true);

    try {
      const payload = {
        disease_name: nextQuery,
        detail_level: 'detail',
        top_k: 8,
        include_images: true,
        chat_history: buildChatHistory(),
        stase_slug: activeStase,
      };

      const response = await axios.post<ApiResponse>(`${API_URL}/search_disease_context`, payload);
      setMessages((prev) => [...prev, { role: 'bot', data: response.data }]);
    } catch (error: any) {
      const errMsg = error?.response?.data?.error || error.message || 'Terjadi kesalahan saat memproses data.';
      setMessages((prev) => [...prev, { role: 'bot', error: true, content: errMsg }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleOpenEvidence = (citation: string, allEvidence: EvidenceItem[]) => {
    const cleanStr = (s: string) =>
      s.replace(/^\(Sumber\)\s*/i, '').replace(/^\(Source\)\s*/i, '').toLowerCase().replace(/\s+/g, ' ').trim();

    const cleanCitation = cleanStr(citation);
    let matchingEvidence = allEvidence.filter(
      (e) => cleanStr(`${e.source_name} p.${e.page_no}`) === cleanCitation
    );

    if (matchingEvidence.length === 0) {
      const pageMatch = citation.match(/p\.(\d+)/i);
      const pageNum = pageMatch ? parseInt(pageMatch[1]) : -1;
      matchingEvidence = allEvidence.filter((e) => {
        if (pageNum > 0 && e.page_no !== pageNum) return false;
        const cleanSrc = cleanStr(e.source_name);
        const cleanCit = cleanCitation.replace(`p.${e.page_no}`, '').trim();
        const citWords = cleanCit.split(' ').filter((w) => w.length > 2);
        const matchCount = citWords.filter((w) => cleanSrc.includes(w)).length;
        return matchCount > 0 && matchCount >= Math.ceil(citWords.length * 0.5);
      });
    }

    setEvidenceModal({ citation, content: matchingEvidence });
  };

  const clearChat = () => setMessages([]);

  // Open KG panel and auto-select matching disease
  const handleOpenKG = (disease: string) => {
    setKgInitialDisease(disease);
    setActiveView('kg');
  };

  // ─── Render section content ──
  const renderSection = (section: DraftSection, sIdx: number, confidence?: number) => {
    const icon = SECTION_ICONS[section.title] || 'article';
    const content = (section.markdown ?? (section.points?.join('\n\n') ?? '')).trim();

    if (!content) return null;

    return (
      <div key={sIdx} className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 px-3 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
            <span className="material-symbols-outlined text-[14px]">{icon}</span>
            {section.title}
          </span>
          {typeof confidence === 'number' && (
            <span className={`inline-flex items-center rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide ${
              confidenceTone(confidence) === 'high'
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : confidenceTone(confidence) === 'medium'
                  ? 'bg-amber-50 text-amber-700 border border-amber-200'
                  : 'bg-rose-50 text-rose-700 border border-rose-200'
            }`}>
              Confidence {formatConfidenceLabel(confidence)}
            </span>
          )}
        </div>
        <CollapsibleSection title={section.title} icon={icon}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
            {content}
          </ReactMarkdown>
        </CollapsibleSection>
      </div>
    );
  };


// ─── ADMIN PANEL ─────────────────────────────────────────────────────────────

type AdminSource = {
  source_name: string;
  page_count: number;
  chunk_count: number;
  indexed: boolean;
  path: string;
};

type AdminStase = {
  slug: string;
  display_name: string;
  dirname: string;
  materi_dir: string;
  materi_exists: boolean;
  is_builtin: boolean;
  disease_count?: number;
};

function AdminPanel() {
  const [tab, setTab] = useState<'sources' | 'stases'>('sources');

  // Source Manager state
  const [staseSlug, setStaseSlug] = useState('ipd');
  const [staseList, setStaseList] = useState<AdminStase[]>([]);
  const [sources, setSources] = useState<AdminSource[]>([]);
  const [loadingSrc, setLoadingSrc] = useState(false);
  const [newSourceName, setNewSourceName] = useState('(Sumber) ');
  const [createMsg, setCreateMsg] = useState<string | null>(null);

  // Upload ZIP state
  const [uploadSourceName, setUploadSourceName] = useState('(Sumber) ');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [uploadMode, setUploadMode] = useState<'legacy' | 'cloud'>(DEFAULT_UPLOAD_MODE);
  const [cloudUploadJob, setCloudUploadJob] = useState<CloudUploadJob | null>(null);

  // Per-page upload state
  const [pageSourceName, setPageSourceName] = useState('');
  const [pageNo, setPageNo] = useState(1);
  const [pageContent, setPageContent] = useState('');
  const [savingPage, setSavingPage] = useState(false);
  const [pageMsg, setPageMsg] = useState<string | null>(null);

  // Reindex state
  const [reindexing, setReindexing] = useState(false);
  const [reindexMsg, setReindexMsg] = useState<string | null>(null);

  // Stase Manager state
  const [newSlug, setNewSlug] = useState('');
  const [newDisplayName, setNewDisplayName] = useState('');
  const [creatingStase, setCreatingStase] = useState(false);
  const [staseMsg, setStaseMsg] = useState<string | null>(null);

  // Load stases + sources
  useEffect(() => {
    axios.get<{ stases: AdminStase[] }>(`${API_URL}/admin/stases`)
      .then(r => setStaseList(r.data.stases || []))
      .catch(() => {});
  }, []);

  const loadSources = (slug: string) => {
    setLoadingSrc(true);
    axios.get<{ sources: AdminSource[] }>(`${API_URL}/admin/stases/${slug}/sources`)
      .then(r => setSources(r.data.sources || []))
      .catch(() => setSources([]))
      .finally(() => setLoadingSrc(false));
  };

  useEffect(() => { loadSources(staseSlug); }, [staseSlug]);

  const uploadProgressPercent = cloudUploadJob
    ? Math.round((cloudUploadJob.next_batch_index / cloudUploadJob.total_batches) * 100)
    : 0;

  const uploadProgressLabel = cloudUploadJob
    ? `Sedang memproses... Batch ${Math.min(cloudUploadJob.next_batch_index + 1, cloudUploadJob.total_batches)} dari ${cloudUploadJob.total_batches} (${Math.max(uploadProgressPercent, 0)}%)`
    : '';

  const runCloudUploadJob = async (job: CloudUploadJob, startBatchIndex = job.next_batch_index) => {
    setUploading(true);
    let nextIndex = startBatchIndex;

    try {
      for (nextIndex = startBatchIndex; nextIndex < job.total_batches; nextIndex += 1) {
        const batch = job.batches[nextIndex];
        const progressPercent = Math.round((nextIndex / job.total_batches) * 100);
        setCloudUploadJob({ ...job, next_batch_index: nextIndex });
        setUploadMsg(`Sedang memproses... Batch ${nextIndex + 1} dari ${job.total_batches} (${progressPercent}%)`);

        await axios.post(
          `${API_URL}/admin/stases/${job.stase_slug}/sources/${encodeURIComponent(job.source_name)}/batch_upload`,
          {
            pages: batch.pages,
            reset_source: nextIndex === 0 && startBatchIndex === 0,
            batch_index: nextIndex,
            total_batches: job.total_batches,
            source_name: job.source_name,
          },
          { timeout: 15000 },
        );

        setCloudUploadJob({ ...job, next_batch_index: nextIndex + 1 });
      }

      setUploadMsg(`✅ ${job.total_pages} halaman berhasil diupload dan diindeks.`);
      setCloudUploadJob(null);
      setUploadFile(null);
      loadSources(job.stase_slug);
    } catch (e: any) {
      const detail = e?.response?.data?.detail ?? e?.response?.data?.error ?? e.message;
      setUploadMsg(`⚠️ Sinkronisasi terhenti di batch ${nextIndex + 1}. ${detail}`);
      setCloudUploadJob({ ...job, next_batch_index: nextIndex });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadMsg(null), 7000);
    }
  };

  const startCloudUpload = async () => {
    if (!uploadFile || !uploadSourceName.startsWith('(Sumber) ')) {
      setUploadMsg('Pilih file ZIP dan nama sumber yang valid');
      return;
    }

    setUploading(true);
    try {
      const pages = await readZipMarkdownPages(uploadFile);
      if (pages.length === 0) {
        throw new Error('ZIP tidak berisi page-*/markdown.md');
      }

      const batches = splitIntoBatches(pages, CLOUD_BATCH_SIZE).map((batch, index) => ({
        batch_index: index,
        pages: batch,
      }));

      const job: CloudUploadJob = {
        source_name: uploadSourceName.trim(),
        stase_slug: staseSlug,
        batches,
        total_pages: pages.length,
        total_batches: batches.length,
        next_batch_index: 0,
        started_at: Date.now(),
      };

      setCloudUploadJob(job);
      setUploadMsg(`ZIP diproses di browser. Mulai sinkronisasi ${job.total_batches} batch...`);
      await runCloudUploadJob(job, 0);
    } catch (e: any) {
      setUploadMsg(`❌ ${e?.response?.data?.detail ?? e.message}`);
      setCloudUploadJob(null);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadMsg(null), 7000);
    }
  };

  const resumeCloudUpload = async () => {
    if (!cloudUploadJob) return;
    await runCloudUploadJob(cloudUploadJob, cloudUploadJob.next_batch_index);
  };

  const handleCreateSource = async () => {
    if (!newSourceName.startsWith('(Sumber) ') || newSourceName.trim().length < 12) {
      setCreateMsg('Nama harus: (Sumber) XX Nama'); return;
    }
    try {
      await axios.post(`${API_URL}/admin/stases/${staseSlug}/sources`, { source_name: newSourceName });
      setCreateMsg('✅ Berhasil dibuat!');
      loadSources(staseSlug);
      setNewSourceName('(Sumber) ');
    } catch (e: any) {
      setCreateMsg(`❌ ${e?.response?.data?.detail ?? e.message}`);
    }
    setTimeout(() => setCreateMsg(null), 4000);
  };

  const handleUploadZip = async () => {
    if (uploadMode === 'cloud') {
      await startCloudUpload();
      return;
    }

    if (!uploadFile || !uploadSourceName.startsWith('(Sumber) ')) {
      setUploadMsg('Pilih file ZIP dan nama sumber yang valid'); return;
    }
    setUploading(true);
    const form = new FormData();
    form.append('file', uploadFile);
    try {
      const r = await axios.post(`${API_URL}/admin/stases/${staseSlug}/sources/${encodeURIComponent(uploadSourceName)}/upload_zip`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUploadMsg(`✅ ${r.data.pages_uploaded} halaman diupload! Re-index dimulai...`);
      setTimeout(() => loadSources(staseSlug), 3000);
      setUploadFile(null);
    } catch (e: any) {
      setUploadMsg(`❌ ${e?.response?.data?.detail ?? e.message}`);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadMsg(null), 5000);
    }
  };

  const handleUploadPage = async () => {
    if (!pageSourceName || !pageContent) { setPageMsg('Isi semua field'); return; }
    setSavingPage(true);
    try {
      await axios.post(`${API_URL}/admin/stases/${staseSlug}/sources/${encodeURIComponent(pageSourceName)}/pages/${pageNo}`, {
        page_no: pageNo, markdown: pageContent,
      });
      setPageMsg(`✅ Halaman ${pageNo} tersimpan!`);
    } catch (e: any) {
      setPageMsg(`❌ ${e?.response?.data?.detail ?? e.message}`);
    } finally {
      setSavingPage(false);
      setTimeout(() => setPageMsg(null), 4000);
    }
  };

  const handleDeleteSource = async (sourceName: string) => {
    if (!window.confirm(`Hapus "${sourceName}"? Tindakan ini tidak dapat dibatalkan.`)) return;
    try {
      await axios.delete(`${API_URL}/admin/stases/${staseSlug}/sources/${encodeURIComponent(sourceName)}`);
      loadSources(staseSlug);
    } catch (e: any) { alert(`Gagal: ${e?.response?.data?.detail ?? e.message}`); }
  };

  const handleReindex = async (sourceName?: string) => {
    setReindexing(true);
    const params: Record<string, string> = { slug: staseSlug };
    if (sourceName) params['source_name'] = sourceName;
    try {
      await axios.post(`${API_URL}/admin/reindex`, null, { params });
      setReindexMsg(`✅ Re-index ${sourceName ?? 'semua sumber'} dimulai!`);
      setTimeout(() => loadSources(staseSlug), 5000);
    } catch (e: any) {
      setReindexMsg(`❌ ${e?.response?.data?.detail ?? e.message}`);
    } finally {
      setReindexing(false);
      setTimeout(() => setReindexMsg(null), 5000);
    }
  };

  const handleCreateStase = async () => {
    if (!newSlug || !newDisplayName) { setStaseMsg('Isi slug dan nama stase'); return; }
    setCreatingStase(true);
    try {
      await axios.post(`${API_URL}/admin/stases`, { slug: newSlug, display_name: newDisplayName });
      setStaseMsg(`✅ Stase "${newDisplayName}" berhasil dibuat!`);
      const r = await axios.get<{ stases: AdminStase[] }>(`${API_URL}/admin/stases`);
      setStaseList(r.data.stases || []);
      setNewSlug(''); setNewDisplayName('');
    } catch (e: any) {
      setStaseMsg(`❌ ${e?.response?.data?.detail ?? e.message}`);
    } finally {
      setCreatingStase(false);
      setTimeout(() => setStaseMsg(null), 4000);
    }
  };

  const handleDeleteStase = async (slug: string) => {
    if (!window.confirm(`Hapus stase "${slug}" dari registry? Folder materi tidak akan dihapus.`)) return;
    try {
      await axios.delete(`${API_URL}/admin/stases/${slug}`);
      const r = await axios.get<{ stases: AdminStase[] }>(`${API_URL}/admin/stases`);
      setStaseList(r.data.stases || []);
    } catch (e: any) { alert(`Gagal: ${e?.response?.data?.detail ?? e.message}`); }
  };

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-800 shrink-0 bg-linear-to-r from-violet-50 to-indigo-50 dark:from-violet-950/20 dark:to-indigo-950/20">
        <h2 className="text-lg font-headline font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
          <span className="material-symbols-outlined text-violet-500">admin_panel_settings</span>
          Admin Panel
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">Kelola sumber materi dan stase knowledge base</p>
      </div>

      {/* Tabs */}
      <div className="px-6 pt-4 shrink-0 flex gap-2 border-b border-slate-100 dark:border-slate-800">
        {[
          { id: 'sources' as const, icon: 'folder_open', label: 'Source Manager' },
          { id: 'stases' as const, icon: 'school', label: 'Stase Manager' },
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium rounded-t-xl transition-all border-b-2 ${
              tab === t.id
                ? 'border-violet-500 text-violet-600 bg-violet-50 dark:bg-violet-900/20'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}>
            <span className="material-symbols-outlined text-[16px]">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* ── SOURCE MANAGER TAB ── */}
        {tab === 'sources' && (
          <>
            {/* Stase selector */}
            <div className="flex items-center gap-3">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest shrink-0">Stase:</label>
              <select value={staseSlug} onChange={e => setStaseSlug(e.target.value)}
                className="text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-violet-400">
                {staseList.length > 0
                  ? staseList.map(s => <option key={s.slug} value={s.slug}>{s.display_name} ({s.slug})</option>)
                  : <option value="ipd">Stase IPD (ipd)</option>
                }
              </select>
            </div>

            {/* Sources list */}
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-100 dark:border-slate-800 overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2">
                  <span className="material-symbols-outlined text-[16px] text-violet-500">source</span>
                  Sumber Terindeks ({sources.length})
                </h3>
                <button onClick={() => handleReindex()} disabled={reindexing}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-violet-600 text-white rounded-xl hover:bg-violet-700 transition-all font-medium disabled:opacity-60">
                  <span className="material-symbols-outlined text-[13px]">{reindexing ? 'sync' : 'refresh'}</span>
                  {reindexing ? 'Indexing...' : 'Re-index Semua'}
                </button>
              </div>
              {reindexMsg && (
                <div className="px-4 py-2 text-xs font-medium bg-violet-50 text-violet-700 border-b border-violet-100">{reindexMsg}</div>
              )}
              {loadingSrc ? (
                <div className="flex items-center justify-center py-10"><div className="w-6 h-6 border-2 border-violet-300 border-t-violet-600 rounded-full animate-spin" /></div>
              ) : sources.length === 0 ? (
                <div className="text-center py-10 text-slate-400 text-sm">Belum ada sumber di stase ini</div>
              ) : (
                <div className="divide-y divide-slate-50 dark:divide-slate-800">
                  {sources.map(src => (
                    <div key={src.source_name} className="px-4 py-3 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                      <span className={`material-symbols-outlined text-[20px] shrink-0 ${src.indexed ? 'text-emerald-500' : 'text-slate-300'}`} style={{ fontVariationSettings: "'FILL' 1" }}>
                        {src.indexed ? 'check_circle' : 'radio_button_unchecked'}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-700 dark:text-slate-200 truncate">{src.source_name}</p>
                        <p className="text-xs text-slate-400 mt-0.5">
                          {src.page_count} halaman
                          {src.indexed && <span className="ml-2 text-emerald-600 font-medium">· {src.chunk_count} chunks ✅</span>}
                          {!src.indexed && <span className="ml-2 text-amber-500">· Belum diindeks</span>}
                        </p>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button onClick={() => handleReindex(src.source_name)} disabled={reindexing}
                          title="Re-index sumber ini"
                          className="p-1.5 text-slate-400 hover:text-violet-600 hover:bg-violet-50 rounded-lg transition-all">
                          <span className="material-symbols-outlined text-[16px]">refresh</span>
                        </button>
                        <button onClick={() => handleDeleteSource(src.source_name)}
                          title="Hapus sumber"
                          className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all">
                          <span className="material-symbols-outlined text-[16px]">delete</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Create source */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-100 dark:border-slate-800 p-4">
                <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2 mb-3">
                  <span className="material-symbols-outlined text-[16px] text-violet-500">create_new_folder</span>
                  Buat Sumber Baru
                </h3>
                <div className="space-y-2">
                  <input value={newSourceName} onChange={e => setNewSourceName(e.target.value)}
                    placeholder="(Sumber) A3 Nama Buku"
                    className="w-full text-sm border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 bg-slate-50 dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-400" />
                  <p className="text-[10px] text-slate-400">Format: (Sumber) {'{Kode}'} {'{Nama Singkat}'}</p>
                  <button onClick={handleCreateSource}
                    className="w-full py-2 bg-violet-600 text-white text-sm font-medium rounded-xl hover:bg-violet-700 transition-all">
                    Buat Folder Sumber
                  </button>
                  {createMsg && <p className="text-xs font-medium text-center" style={{ color: createMsg.startsWith('✅') ? '#16a34a' : '#dc2626' }}>{createMsg}</p>}
                </div>
              </div>

              {/* Upload ZIP */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-100 dark:border-slate-800 p-4 shadow-sm">
                <div className="flex items-start justify-between gap-3 mb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2">
                      <span className="material-symbols-outlined text-[16px] text-violet-500">upload_file</span>
                      Upload ZIP (Bulk Pages)
                    </h3>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Cloud mode membongkar ZIP di browser lalu mengirim batch 5 halaman per request.
                    </p>
                  </div>
                  <div className="inline-flex rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-1">
                    <button
                      type="button"
                      onClick={() => setUploadMode('legacy')}
                      className={`px-3 py-1.5 text-xs rounded-lg transition-all ${uploadMode === 'legacy' ? 'bg-white dark:bg-slate-900 text-slate-700 shadow-sm' : 'text-slate-500'}`}
                    >
                      Legacy
                    </button>
                    <button
                      type="button"
                      onClick={() => setUploadMode('cloud')}
                      className={`px-3 py-1.5 text-xs rounded-lg transition-all ${uploadMode === 'cloud' ? 'bg-violet-600 text-white shadow-sm' : 'text-slate-500'}`}
                    >
                      Cloud
                    </button>
                  </div>
                </div>

                <div className="space-y-3">
                  <input
                    value={uploadSourceName}
                    onChange={e => setUploadSourceName(e.target.value)}
                    placeholder="(Sumber) A3 Nama Buku"
                    className="w-full text-sm border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 bg-slate-50 dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-400"
                  />
                  <input
                    type="file"
                    accept=".zip"
                    onChange={e => setUploadFile(e.target.files?.[0] ?? null)}
                    className="w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-violet-50 file:text-violet-700 file:font-medium file:text-xs hover:file:bg-violet-100 cursor-pointer"
                  />

                  {uploadMode === 'cloud' && cloudUploadJob && (
                    <div className="space-y-2 rounded-2xl border border-violet-100 bg-violet-50/70 p-3 dark:border-violet-900/40 dark:bg-violet-950/20">
                      <div className="flex items-center justify-between text-xs font-medium text-violet-700 dark:text-violet-300">
                        <span>{uploadProgressLabel}</span>
                        <span>{cloudUploadJob.total_pages} halaman</span>
                      </div>
                      <div className="h-2 rounded-full bg-violet-100 dark:bg-violet-900/40 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-linear-to-r from-violet-500 via-indigo-500 to-cyan-400 transition-all duration-300"
                          style={{ width: `${Math.min(uploadProgressPercent, 100)}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-violet-600/80 dark:text-violet-300/70">
                        <span>Batch size: {CLOUD_BATCH_SIZE}</span>
                        <span>{cloudUploadJob.next_batch_index}/{cloudUploadJob.total_batches}</span>
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleUploadZip}
                    disabled={uploading || !uploadFile}
                    className="w-full py-2 bg-violet-600 text-white text-sm font-medium rounded-xl hover:bg-violet-700 transition-all disabled:opacity-60 flex items-center justify-center gap-1.5"
                  >
                    <span className="material-symbols-outlined text-[15px]">{uploading ? 'sync' : 'cloud_upload'}</span>
                    {uploading
                      ? (uploadMode === 'cloud' ? 'Menyinkronkan...' : 'Mengupload...')
                      : (uploadMode === 'cloud' ? 'Upload & Sinkronisasi Batch' : 'Upload & Index Otomatis')}
                  </button>

                  {uploadMode === 'cloud' && cloudUploadJob && !uploading && cloudUploadJob.next_batch_index < cloudUploadJob.total_batches && (
                    <button
                      type="button"
                      onClick={resumeCloudUpload}
                      className="w-full py-2.5 border border-violet-200 bg-white text-violet-700 text-sm font-medium rounded-xl hover:bg-violet-50 transition-all"
                    >
                      Lanjutkan Sinkronisasi yang Gagal
                    </button>
                  )}

                  {uploadMsg && (
                    <p
                      className="text-xs font-medium text-center leading-relaxed"
                      style={{ color: uploadMsg.startsWith('✅') ? '#16a34a' : uploadMsg.startsWith('⚠️') ? '#d97706' : '#dc2626' }}
                    >
                      {uploadMsg}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Per-page upload */}
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-100 dark:border-slate-800 p-4">
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2 mb-3">
                <span className="material-symbols-outlined text-[16px] text-violet-500">edit_note</span>
                Upload Per Halaman
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-2">
                  <select value={pageSourceName} onChange={e => setPageSourceName(e.target.value)}
                    className="w-full text-sm border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 bg-slate-50 dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-400">
                    <option value="">Pilih sumber...</option>
                    {sources.map(s => <option key={s.source_name} value={s.source_name}>{s.source_name}</option>)}
                  </select>
                  <div className="flex gap-2 items-center">
                    <label className="text-xs text-slate-500 shrink-0">Hal.</label>
                    <input type="number" value={pageNo} min={1} onChange={e => setPageNo(parseInt(e.target.value) || 1)}
                      className="w-20 text-sm border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 bg-slate-50 dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-400" />
                  </div>
                  <button onClick={handleUploadPage} disabled={savingPage || !pageSourceName || !pageContent}
                    className="w-full py-2 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-all disabled:opacity-60">
                    {savingPage ? 'Menyimpan...' : 'Simpan Halaman'}
                  </button>
                  {pageMsg && <p className="text-xs font-medium" style={{ color: pageMsg.startsWith('✅') ? '#16a34a' : '#dc2626' }}>{pageMsg}</p>}
                </div>
                <textarea value={pageContent} onChange={e => setPageContent(e.target.value)}
                  placeholder="# Judul Halaman&#10;&#10;Konten markdown halaman di sini..."
                  rows={6}
                  className="w-full text-sm border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2.5 bg-slate-50 dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-400 resize-none font-mono" />
              </div>
            </div>
          </>
        )}

        {/* ── STASE MANAGER TAB ── */}
        {tab === 'stases' && (
          <>
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-100 dark:border-slate-800 overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800">
                <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2">
                  <span className="material-symbols-outlined text-[16px] text-violet-500">school</span>
                  Stase Terdaftar ({staseList.length})
                </h3>
              </div>
              {staseList.length === 0 ? (
                <div className="text-center py-10 text-slate-400 text-sm">Memuat data stase...</div>
              ) : (
                <div className="divide-y divide-slate-50 dark:divide-slate-800">
                  {staseList.map(s => (
                    <div key={s.slug} className="px-4 py-3 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                      <span className={`material-symbols-outlined text-[20px] shrink-0 ${s.materi_exists ? 'text-emerald-500' : 'text-amber-400'}`} style={{ fontVariationSettings: "'FILL' 1" }}>
                        {s.materi_exists ? 'folder' : 'folder_off'}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-medium text-slate-700 dark:text-slate-200">{s.display_name}</p>
                          {s.is_builtin && <span className="text-[9px] font-bold px-1.5 py-0.5 bg-indigo-50 text-indigo-600 rounded-full border border-indigo-100">builtin</span>}
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">
                          slug: <code className="font-mono text-violet-600">{s.slug}</code>
                          {' · '}folder: <code className="font-mono text-slate-500">{s.dirname}/Materi/</code>
                          {!s.materi_exists && <span className="ml-1 text-amber-500">⚠ folder tidak ada</span>}
                        </p>
                      </div>
                      {!s.is_builtin && (
                        <button onClick={() => handleDeleteStase(s.slug)}
                          title="Hapus dari registry"
                          className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all shrink-0">
                          <span className="material-symbols-outlined text-[16px]">delete</span>
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-100 dark:border-slate-800 p-4">
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200 flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-[16px] text-violet-500">add_circle</span>
                Tambah Stase Baru
              </h3>
              <div className="space-y-3 max-w-sm">
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-widest block mb-1">Slug</label>
                  <input value={newSlug} onChange={e => setNewSlug(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
                    placeholder="saraf"
                    className="w-full text-sm border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 bg-slate-50 dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-400 font-mono" />
                  {newSlug && <p className="text-[10px] text-slate-400 mt-1">Folder: <code className="text-violet-600">{newSlug.charAt(0).toUpperCase() + newSlug.slice(1)}/Materi/</code></p>}
                </div>
                <div>
                  <label className="text-xs font-bold text-slate-500 uppercase tracking-widest block mb-1">Nama Stase</label>
                  <input value={newDisplayName} onChange={e => setNewDisplayName(e.target.value)}
                    placeholder="Stase Neurologi"
                    className="w-full text-sm border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 bg-slate-50 dark:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-violet-400" />
                </div>
                <button onClick={handleCreateStase} disabled={creatingStase || !newSlug || !newDisplayName}
                  className="w-full py-2.5 bg-violet-600 text-white text-sm font-medium rounded-xl hover:bg-violet-700 transition-all disabled:opacity-60 flex items-center justify-center gap-1.5">
                  <span className="material-symbols-outlined text-[16px]">{creatingStase ? 'sync' : 'add'}</span>
                  {creatingStase ? 'Membuat...' : 'Buat Stase'}
                </button>
                {staseMsg && <p className="text-xs font-medium text-center" style={{ color: staseMsg.startsWith('✅') ? '#16a34a' : '#dc2626' }}>{staseMsg}</p>}
              </div>
              <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-800 rounded-xl">
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  <strong className="text-slate-600 dark:text-slate-300">Konvensi folder:</strong> Slug <code className="text-violet-600">saraf</code> → folder{' '}
                  <code className="text-violet-600">E:\Coas\Saraf\Materi\</code><br />
                  File CSV katalog penyakit akan dibuat otomatis. Upload sumber materi via tab Source Manager.
                </p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─── END ADMIN PANEL ─────────────────────────────────────────────────────────

  return (
    <div className="h-dvh md:h-screen w-full overflow-hidden flex bg-background">
      {/* ── SideNavBar — hidden on mobile, icon-only on tablet, full on desktop ── */}
      <nav className={`fixed left-0 top-0 h-full z-40 flex-col p-4 bg-slate-50/40 dark:bg-slate-950/40 backdrop-blur-2xl transition-all duration-300 ease-in-out ${isSidebarMinimized || isTablet ? 'w-20' : 'w-72'} rounded-r-3xl tonal-layering no-border shadow-[40px_0_60px_-10px_rgba(0,0,0,0.04)] font-[Inter] text-sm hidden md:flex overflow-x-hidden`}>
        <div className="flex items-center gap-3 mb-10 px-2 mt-4">
          <div className="h-10 w-10 min-w-10 rounded-full bg-primary-container flex items-center justify-center shrink-0">
            <span className="material-symbols-outlined text-secondary" style={{ fontVariationSettings: "'FILL' 1" }}>medical_services</span>
          </div>
          <div className={`transition-opacity duration-300 whitespace-nowrap ${isSidebarMinimized || isTablet ? 'opacity-0 pointer-events-none w-0' : 'opacity-100'}`}>
            <h2 className="text-lg font-black text-slate-800 dark:text-slate-200">Clinical Assistant</h2>
            <p className="text-xs text-on-surface-variant font-label">AI-Powered Insights</p>
          </div>
        </div>

        <button
          onClick={() => {
            if (activeView === 'chat') clearChat();
            else setActiveView('chat');
          }}
          className={`gradient-btn w-full rounded-full py-3 px-3 text-white font-medium mb-8 flex items-center justify-center gap-2 hover:opacity-90 transition-all ${isSidebarMinimized || isTablet ? 'px-0' : ''}`}
        >
          <span className="material-symbols-outlined text-[20px] shrink-0">add</span>
          <span className={`whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized || isTablet ? 'opacity-0 hidden' : 'opacity-100'}`}>
            {activeView === 'chat' ? 'New Consultation' : 'Kembali ke Chat'}
          </span>
        </button>

        <div className="flex-1 flex flex-col gap-2 overflow-y-auto overflow-x-hidden pr-2">
          {(
            [
              { icon: 'chat_bubble', label: 'Recent Chats', id: 'chat' as const },
              { icon: 'book', label: 'Medical Library', id: 'library' as const },
              { icon: 'hub', label: 'Knowledge Graph', id: 'kg' as const },
              { icon: 'query_stats', label: 'Analytics', id: 'analytics' as const },
              { icon: 'admin_panel_settings', label: 'Admin Panel', id: 'admin' as const },
            ] as const
          ).map((item) => (
            <button
              type="button"
              key={item.label}
              onClick={() => setActiveView(item.id)}
              className={`flex items-center gap-3 px-3 py-3 rounded-xl transition-all flex-row cursor-pointer text-left w-full ${
                activeView === item.id
                  ? 'bg-indigo-50/50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300'
                  : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-white/20 dark:hover:bg-slate-800/20'
              }`}
            >
              <span className="material-symbols-outlined shrink-0 text-center w-6">{item.icon}</span>
              <span className={`font-medium whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized || isTablet ? 'opacity-0 w-0 hidden' : 'opacity-100'}`}>
                {item.label}
              </span>
            </button>
          ))}
        </div>

        <div className="mt-auto pt-4 border-t border-slate-200/50 flex flex-col gap-1 overflow-x-hidden">
          {[
            { icon: 'shield', label: 'Privacy' },
            { icon: 'contact_support', label: 'Support' },
          ].map((item) => (
            <a key={item.label} className="flex items-center gap-3 px-3 py-2 text-slate-600 dark:text-slate-400 hover:text-indigo-500 rounded-xl transition-all flex-row cursor-pointer">
              <span className="material-symbols-outlined text-[20px] shrink-0 text-center w-6">{item.icon}</span>
              <span className={`whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized || isTablet ? 'opacity-0 w-0 hidden' : 'opacity-100'}`}>{item.label}</span>
            </a>
          ))}
        </div>
      </nav>

      {/* ── Mobile Bottom Nav ── */}
      {isMobile && <MobileBottomNav activeView={activeView} onChangeView={setActiveView} />}

      {/* ── Main Content ── */}
      <main className={`flex-1 flex flex-col h-full min-h-0 relative transition-all duration-300 ml-0 ${isTablet ? 'md:ml-20' : ''} ${!isTablet ? (isSidebarMinimized ? 'md:ml-20' : 'md:ml-72') : ''}`}>
        {/* Header */}
        <header className="flex justify-between items-center px-3 md:px-8 h-14 md:h-20 w-full sticky top-0 z-30 bg-white/60 dark:bg-slate-900/60 backdrop-blur-3xl tracking-tighter border-b border-white/30 shrink-0">
          <div className="flex items-center gap-2 md:gap-4">
            {/* Mobile: app icon; Tablet+Desktop: sidebar toggle */}
            <div className="md:hidden w-8 h-8 rounded-full bg-linear-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shrink-0 shadow-sm">
              <span className="material-symbols-outlined text-white text-[16px]" style={{ fontVariationSettings: "'FILL' 1" }}>medical_services</span>
            </div>
            <button onClick={() => setIsSidebarMinimized(!isSidebarMinimized)} className="hidden md:flex p-2 text-on-surface-variant hover:bg-surface-container rounded-full transition-colors">
              <span className="material-symbols-outlined">{isSidebarMinimized || isTablet ? 'menu' : 'menu_open'}</span>
            </button>
            <h1 className="text-base md:text-xl lg:text-2xl font-headline font-bold bg-linear-to-r from-indigo-600 to-cyan-600 bg-clip-text text-transparent">
              Medical RAG
              {activeView === 'library' && (
                <span className="hidden md:block text-xs font-normal text-slate-500 mt-0.5">Medical Library</span>
              )}
            </h1>
          </div>
          <div className="flex items-center gap-2 md:gap-4">
            {activeView === 'chat' && (
              <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 dark:bg-emerald-900/20 rounded-full border border-emerald-100 dark:border-emerald-800">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">Multi-turn AI</span>
              </div>
            )}
            {activeView === 'chat' && (
              <button onClick={clearChat} className="hidden md:flex items-center gap-2 px-4 py-2 bg-surface-container-lowest glass-border rounded-full text-secondary hover:bg-surface-container-high transition-colors font-medium text-sm">
                <span className="material-symbols-outlined text-[18px]">delete</span>
                Bersihkan Chat
              </button>
            )}
            <button className="p-2 text-slate-500 hover:bg-white/40 transition-all rounded-full scale-95 active:duration-150">
              <span className="material-symbols-outlined">settings</span>
            </button>
          </div>
        </header>

        {activeView === 'library' && <MedicalLibraryPanel components={mdComponents} />}

        {activeView === 'kg' && (
          <KnowledgeGraphPanel
            initialDisease={kgInitialDisease}
            onDismissInitial={() => setKgInitialDisease(null)}
          />
        )}

        {activeView === 'analytics' && (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-slate-500">
            <span className="material-symbols-outlined text-5xl mb-4 opacity-40">query_stats</span>
            <p className="text-sm">Analytics akan tersedia di iterasi berikutnya.</p>
          </div>
        )}

        {activeView === 'admin' && <AdminPanel />}
        {/* Chat Canvas */}
        {activeView === 'chat' && (
        <>
        <div className="flex-1 overflow-y-auto px-3 md:px-12 lg:px-24 py-4 md:py-8 pb-48 md:pb-40 flex flex-col gap-4 md:gap-8 min-h-0">
          {messages.length === 0 && (
            <div className="m-auto text-center max-w-md pt-10 md:pt-20 flex flex-col items-center opacity-70 px-4">
              <span className="material-symbols-outlined text-5xl md:text-6xl text-secondary mb-3 md:mb-4 opacity-50" style={{ fontVariationSettings: "'FILL' 1" }}>biotech</span>
              <h2 className="text-xl md:text-2xl font-headline font-bold text-slate-700">Mulai Konsultasi RAG</h2>
              <p className="text-sm mt-2 text-slate-500 leading-relaxed font-body">
                Tanyakan kondisi klinis yang spesifik. Basis pengetahuan ini bersumber dari panduan medis terkini dan literatur terverifikasi.
              </p>
              <div className="mt-4 md:mt-6 grid grid-cols-1 sm:grid-cols-2 gap-2 w-full">
                {['Diabetes Mellitus Tipe 2', 'Hipertensi', 'Pneumonia', 'Gagal Jantung'].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setQuery(suggestion)}
                    className="text-xs text-left bg-white/60 glass-border px-3 py-2.5 rounded-xl hover:bg-indigo-50 hover:border-indigo-200 transition-all text-slate-600 hover:text-indigo-700"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} gap-1.5 md:gap-2 w-full ${msg.role === 'user' ? 'max-w-[90%] md:max-w-3xl ml-auto' : 'max-w-full md:max-w-4xl mr-auto mt-3 md:mt-6'} view-fade-in`}>
              {/* Avatar & Name */}
              <div className="flex items-center gap-2 mb-1 px-2">
                {msg.role === 'user' ? (
                  <>
                    <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">Anda</span>
                    <div className="w-6 h-6 rounded-full bg-secondary-container flex items-center justify-center overflow-hidden shrink-0 text-secondary">
                      <span className="text-[10px] font-bold">DR</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-8 h-8 rounded-full bg-linear-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shrink-0 shadow-sm text-white">
                      <span className="material-symbols-outlined text-[18px]" style={{ fontVariationSettings: "'FILL' 1" }}>medical_services</span>
                    </div>
                    <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">Clinical Assistant</span>
                  </>
                )}
              </div>

              {/* Message Bubble */}
              <div className={`${
                msg.role === 'user'
                  ? 'bg-surface-container-lowest glass-panel glass-border p-3 md:p-5 rounded-2xl md:rounded-3xl rounded-tr-sm ambient-shadow text-on-surface max-w-full'
                  : 'bg-surface-container-low/50 backdrop-blur-xl border border-white/40 p-3 md:p-6 lg:p-8 rounded-2xl md:rounded-[2.5rem] rounded-tl-sm shadow-[0_8px_32px_-12px_rgba(0,0,0,0.05)] w-full'
              }`}>
                {msg.role === 'user' ? (
                  <p className="font-body text-sm md:text-base leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                ) : msg.error ? (
                  <div className="bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-800 p-4 rounded-3xl">
                    <p className="text-red-600 dark:text-red-400 text-sm font-medium flex items-center gap-2">
                      <span className="material-symbols-outlined text-[18px]">error</span>
                      {msg.content}
                    </p>
                  </div>
                ) : msg.data ? (
                  <div className="prose prose-slate max-w-none text-on-surface font-body">
                    <div className="bg-white/80 dark:bg-slate-900/80 rounded-2xl md:rounded-[2.5rem] p-4 md:p-6 lg:p-10 glass-border ambient-shadow space-y-6 md:space-y-10">
                      {/* Disease Title */}
                      {msg.data.draft_answer.disease && (
                        <div className="border-b border-slate-100 dark:border-slate-800 pb-6">
                          <p className="text-sm font-label text-slate-500 uppercase tracking-widest mb-1">Diagnosis Analysis</p>
                          <h2 className="text-xl md:text-2xl lg:text-3xl font-headline font-black text-slate-800 dark:text-slate-100">
                            {msg.data.draft_answer.disease}
                          </h2>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {msg.data.evidence_quality && (
                              <span className={`inline-flex items-center rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide border ${
                                msg.data.evidence_quality === 'ok'
                                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                  : 'bg-amber-50 text-amber-700 border-amber-200'
                              }`}>
                                Evidence {msg.data.evidence_quality === 'ok' ? 'cukup' : 'rendah'}
                              </span>
                            )}
                            {typeof msg.data.draft_answer.answer_confidence === 'number' && (
                              <span className={`inline-flex items-center rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide border ${
                                confidenceTone(msg.data.draft_answer.answer_confidence) === 'high'
                                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                  : confidenceTone(msg.data.draft_answer.answer_confidence) === 'medium'
                                    ? 'bg-sky-50 text-sky-700 border-sky-200'
                                    : 'bg-rose-50 text-rose-700 border-rose-200'
                              }`}>
                                Confidence {formatConfidenceLabel(msg.data.draft_answer.answer_confidence)}
                              </span>
                            )}
                            {typeof msg.data.draft_answer.retrieval_passes === 'number' && (
                              <span className="inline-flex items-center rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide border bg-slate-50 text-slate-600 border-slate-200">
                                Retrieval {msg.data.draft_answer.retrieval_passes}x
                              </span>
                            )}
                            {typeof msg.data.draft_answer.detection_method === 'string' && (
                              <span className="inline-flex items-center rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide border bg-indigo-50 text-indigo-700 border-indigo-200">
                                Resolver {msg.data.draft_answer.detection_method}
                              </span>
                            )}
                          </div>
                          {/* Ide 11 indicator */}
                          {idx > 0 && (
                            <span className="inline-flex items-center gap-1 mt-2 text-[11px] bg-indigo-50 dark:bg-indigo-900/30 text-indigo-500 px-2 py-0.5 rounded-full border border-indigo-100">
                              <span className="material-symbols-outlined text-[12px]">history</span>
                              Konteks multi-turn
                            </span>
                          )}
                        </div>
                      )}

                      {/* Ide 2: Rich Markdown Sections */}
                      <div className="space-y-5 md:space-y-10">
                        {msg.data.draft_answer.sections.map((section, sIdx) => renderSection(section, sIdx, msg.data?.draft_answer.section_confidence_map?.[section.title]))}
                      </div>

                      {/* Citations */}
                      {msg.data.draft_answer.citations?.length > 0 && (
                        <div className="border-t border-slate-100 dark:border-slate-800 pt-8 mt-6">
                          <p className="text-xs font-semibold text-slate-500 mb-4 flex items-center gap-2 uppercase tracking-widest">
                            <span className="material-symbols-outlined text-[18px]">menu_book</span>
                            Sumber Referensi Terkait
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {msg.data.draft_answer.citations.map((cit, cIdx) => (
                              <button
                                key={cIdx}
                                onClick={() => handleOpenEvidence(cit, msg.data!.evidence)}
                                className="text-xs bg-indigo-50 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 px-4 py-2 rounded-full hover:bg-indigo-600 hover:text-white transition-all font-medium border border-indigo-100 dark:border-indigo-800"
                              >
                                [{cIdx + 1}] {cit.replace(/^\(Sumber\)\s*/i, '').replace(/^\(Source\)\s*/i, '')}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Images */}
                      {msg.data.images?.length > 0 && (
                        <div className="pt-8 border-t border-slate-100 dark:border-slate-800">
                          <h3 className="font-headline text-sm font-bold text-slate-500 mb-4 flex items-center gap-2 uppercase tracking-widest">
                            <span className="material-symbols-outlined text-[18px]">imagesmode</span>
                            Referensi Visual
                          </h3>
                          <div className="flex gap-3 md:gap-4 overflow-x-auto pb-4 snap-x -mx-1 px-1">
                            {msg.data.images.map((img, iIdx) => (
                              <div
                                key={iIdx}
                                onClick={() => setImageModal(img)}
                                className="cursor-pointer shrink-0 border border-slate-100 dark:border-slate-800 rounded-2xl overflow-hidden hover:border-indigo-400 transition-all snap-center group relative shadow-sm"
                              >
                                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center z-10">
                                  <span className="material-symbols-outlined text-white opacity-0 group-hover:opacity-100 transition-opacity scale-125">zoom_in</span>
                                </div>
                                <img
                                  src={resolveImageUrl(img.image_url)}
                                  alt={img.heading}
                                  className="w-36 h-24 md:w-48 md:h-32 object-cover transition-transform duration-500 group-hover:scale-110"
                                  onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = PLACEHOLDER_IMG; }}
                                />
                                <div className="p-3 bg-white/90 dark:bg-slate-900/90 backdrop-blur-sm w-full absolute bottom-0 z-20">
                                  <p className="text-[10px] md:text-[11px] text-center font-medium truncate w-32 md:w-40 text-slate-700 dark:text-slate-300">{img.heading}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : null}
              </div>

              {/* Bot Action Bar */}
              {msg.role === 'bot' && !msg.error && msg.data && (
                <div className="flex items-center gap-1.5 md:gap-2 mt-1.5 md:mt-2 ml-2 md:ml-4 flex-wrap">
                  <button aria-label="Copy" className="p-2 text-on-surface-variant hover:text-secondary hover:bg-secondary-container rounded-full transition-colors">
                    <span className="material-symbols-outlined text-[18px]">content_copy</span>
                  </button>
                  <button aria-label="Thumbs Up" className="p-2 text-on-surface-variant hover:text-secondary hover:bg-secondary-container rounded-full transition-colors">
                    <span className="material-symbols-outlined text-[18px]">thumb_up</span>
                  </button>
                  <button aria-label="Thumbs Down" className="p-2 text-on-surface-variant hover:text-secondary hover:bg-secondary-container rounded-full transition-colors">
                    <span className="material-symbols-outlined text-[18px]">thumb_down</span>
                  </button>
                  {/* Ide 18: Open Knowledge Graph */}
                  <button
                    onClick={() => handleOpenKG(msg.data!.draft_answer.disease)}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-violet-50 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400 border border-violet-100 dark:border-violet-800 rounded-full hover:bg-violet-600 hover:text-white transition-all font-medium"
                  >
                    <span className="material-symbols-outlined text-[15px]">hub</span>
                    Knowledge Graph
                  </button>
                </div>
              )}
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex flex-col items-start gap-2 w-full max-w-4xl mr-auto mt-6">
              <div className="flex items-center gap-2 mb-1 px-2 opacity-70">
                <div className="w-8 h-8 rounded-full bg-linear-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shrink-0 shadow-sm text-white">
                  <span className="material-symbols-outlined text-[18px] animate-pulse">medical_services</span>
                </div>
                <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">Clinical Assistant</span>
              </div>
              <div className="bg-surface-container-lowest glass-border glass-panel px-6 py-4 rounded-4xl rounded-tl-sm flex items-center gap-3 text-secondary animate-pulse">
                <span className="material-symbols-outlined animate-spin" style={{ animationDuration: '1.5s' }}>refresh</span>
                <span className="text-sm font-medium">Menganalisis dokumen referensi klinis...</span>
              </div>
            </div>
          )}

          <div className="h-10" ref={messagesEndRef} />
        </div>

        {/* Floating Input Area */}
        <div className={`absolute left-0 w-full p-3 md:p-6 lg:p-8 bg-linear-to-t from-background via-background/90 to-transparent pointer-events-none flex justify-center z-20 ${isMobile ? 'bottom-(--bottom-nav-h)' : 'bottom-0'}`}>
          <div className="w-full max-w-4xl pointer-events-auto">
            {/* Stase selector */}
            <div className="flex items-center gap-2 mb-2 px-1">
              <span className="text-[11px] font-label text-on-surface-variant/70 uppercase tracking-wide shrink-0">Stase:</span>
              {[
                { slug: 'ipd', label: 'IPD' },
                { slug: 'saraf', label: 'Saraf' },
                { slug: 'anak', label: 'Anak' },
                { slug: 'obgyn', label: 'ObGyn' },
              ].map(({ slug, label }) => (
                <button
                  key={slug}
                  onClick={() => setActiveStase(slug)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-all border ${
                    activeStase === slug
                      ? 'bg-primary text-white border-primary shadow-sm'
                      : 'bg-surface-container text-on-surface-variant border-surface-variant hover:bg-secondary-container hover:text-secondary hover:border-secondary-container'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className={`bg-surface-container-lowest/80 backdrop-blur-2xl rounded-2xl md:rounded-4xl glass-border p-1.5 md:p-2 flex items-end gap-1.5 md:gap-2 ambient-shadow transition-all ${isLoading ? 'opacity-80' : 'shadow-[0_-10px_40px_rgba(0,0,0,0.03)] focus-within:ring-2 focus-within:ring-primary-container'}`}>
              <button disabled={isLoading} className="p-2 md:p-3 text-on-surface-variant hover:text-secondary hover:bg-secondary-container/50 rounded-full transition-colors shrink-0 mb-0.5 md:mb-1 hidden md:flex">
                <span className="material-symbols-outlined">attach_file</span>
              </button>
              <div className="flex-1 relative">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      submitQuery();
                    }
                  }}
                  disabled={isLoading}
                  className="w-full bg-transparent border-none text-on-surface placeholder-on-surface-variant/60 resize-none py-3 md:py-4 px-2 focus:ring-0 font-body text-sm md:text-base max-h-32 min-h-11 md:min-h-14 block outline-none disabled:opacity-70"
                  placeholder={isLoading ? 'Memproses...' : (isMobile ? 'Tanya kondisi medis...' : `Tanya seputar kondisi medis ${activeStase.toUpperCase()}, gejala, atau tindak lanjut...`)}
                  rows={1}
                />
              </div>
              <button
                onClick={submitQuery}
                disabled={isLoading || !query.trim()}
                className="p-2.5 md:p-3 mb-0.5 md:mb-1 bg-primary-container text-on-primary-container hover:bg-secondary hover:text-white rounded-full transition-all shrink-0 flex items-center justify-center group shadow-sm disabled:opacity-50 disabled:hover:bg-primary-container disabled:hover:text-on-primary-container"
              >
                <span className="material-symbols-outlined group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform">send</span>
              </button>
            </div>
            <div className="text-center mt-3 pointer-events-none">
              <p className="text-[10px] md:text-xs text-on-surface-variant/70 font-label">
                Medical RAG dapat membuat kesalahan. Verifikasi informasi klinis penting sebelum tindakan.
              </p>
            </div>
          </div>
        </div>
        </>
        )}

        {/* ── Evidence Modal ── */}
        {evidenceModal && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-end md:items-center justify-center z-50 p-0 md:p-4" onClick={() => setEvidenceModal(null)}>
            <div className="bg-surface-container-lowest glass-border glass-panel rounded-t-2xl md:rounded-4xl max-w-2xl w-full max-h-[90vh] md:max-h-[85vh] flex flex-col shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <div className="bottom-sheet-handle md:hidden" />
              <div className="flex justify-between items-center p-4 md:p-6 border-b border-surface-variant/60">
                <h2 className="text-sm md:text-lg font-headline font-bold text-secondary flex items-center gap-2 min-w-0">
                  <span className="material-symbols-outlined text-[18px] md:text-[20px] shrink-0">library_books</span>
                  <span className="truncate">Sitasi: {evidenceModal.citation}</span>
                </h2>
                <button onClick={() => setEvidenceModal(null)} className="p-2 text-on-surface-variant hover:text-secondary hover:bg-secondary-container rounded-full transition-colors">
                  <span className="material-symbols-outlined text-[20px]">close</span>
                </button>
              </div>
              <div className="p-3 md:p-6 overflow-y-auto flex-1 space-y-3 md:space-y-4">
                {evidenceModal.content.length > 0 ? (
                  evidenceModal.content.map((ev, idx) => (
                    <div key={idx} className="bg-surface-container-low/50 rounded-2xl p-5 border border-white/40">
                      {ev.section_category && (
                        <span className="inline-block text-[10px] bg-indigo-50 text-indigo-500 px-2 py-0.5 rounded-full font-medium mb-2">
                          {ev.section_category.replace('_', ' ')}
                        </span>
                      )}
                      <h4 className="font-semibold text-sm text-tertiary mb-2 font-headline">{ev.heading}</h4>
                      <p className="text-sm text-on-surface-variant leading-relaxed font-body">{ev.content}</p>
                    </div>
                  ))
                ) : (
                  <p className="text-on-surface-variant italic text-center py-8">Detail chunk tidak ditemukan untuk sitasi ini.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Image Modal ── */}
        {imageModal && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-end md:items-center justify-center z-50 p-0 md:p-4" onClick={() => setImageModal(null)}>
            <div className="relative max-w-4xl w-full flex flex-col items-center bg-black/40 md:bg-transparent rounded-t-2xl md:rounded-none" onClick={(e) => e.stopPropagation()}>
              <div className="bottom-sheet-handle md:hidden mt-2" />
              <button onClick={() => setImageModal(null)} className="absolute top-3 right-3 md:-top-14 md:right-0 text-white hover:text-gray-300 flex items-center gap-2 font-medium bg-white/10 hover:bg-white/20 px-3 py-1.5 md:px-4 md:py-2 rounded-full backdrop-blur-sm transition-colors z-10">
                <span className="material-symbols-outlined text-[18px]">close</span> Tutup
              </button>
              <img
                src={resolveImageUrl(imageModal.image_url)}
                alt={imageModal.heading}
                className="w-full h-auto max-h-[60vh] md:max-h-[75vh] object-contain rounded-none md:rounded-2xl shadow-2xl"
                onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = PLACEHOLDER_IMG; }}
              />
              <div className="bg-surface-container-lowest glass-border glass-panel p-3 md:p-5 mt-2 md:mt-4 rounded-xl md:rounded-2xl text-center shadow-lg w-full max-w-xl mx-3 md:mx-auto">
                <p className="font-headline text-lg font-bold text-secondary mb-1 truncate">{imageModal.heading}</p>
                <p className="text-sm text-on-surface-variant font-label flex justify-center items-center gap-2">
                  <span className="material-symbols-outlined text-[16px]">menu_book</span>
                  Sumber: {imageModal.source_name} - Hal {imageModal.page_no}
                </p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
