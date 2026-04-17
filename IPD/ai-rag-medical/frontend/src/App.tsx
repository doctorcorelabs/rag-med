import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
// @ts-ignore: missing type declarations
import ForceGraph2D from 'react-force-graph-2d';

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
  evidence: EvidenceItem[];
  draft_answer: {
    disease: string;
    sections: DraftSection[];
    citations: string[];
    grounded: boolean;
  };
  images: ImageItem[];
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

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8010';

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

type ActiveView = 'chat' | 'library' | 'kg' | 'analytics';

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

  const fillColors: Record<string, string> = { root: '#6366f1', section: '#0ea5e9', concept: '#d1fae5', fact: '#fef3c7' };
  const textColors: Record<string, string> = { root: '#ffffff', section: '#ffffff', concept: '#065f46', fact: '#92400e' };
  const borderColors: Record<string, string> = { root: '#4338ca', section: '#0369a1', concept: '#10b981', fact: '#f59e0b' };

  if (isRoot || isSection) {
    ctx.save();
    ctx.shadowColor = isRoot ? 'rgba(99,102,241,0.4)' : 'rgba(14,165,233,0.3)';
    ctx.shadowBlur = (isRoot ? 14 : 8) / globalScale;
  }

  const rr = (x: number, y: number, w: number, h: number, rad: number) => {
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  };

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
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const graphRef = useRef<any>(null);

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

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(() => {
      if (containerRef.current)
        setDimensions({ width: containerRef.current.offsetWidth, height: containerRef.current.offsetHeight });
    });
    obs.observe(containerRef.current);
    setDimensions({ width: containerRef.current.offsetWidth, height: containerRef.current.offsetHeight });
    return () => obs.disconnect();
  }, []);

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
    return {
      nodes: localMindmap.nodes,
      links: localMindmap.edges.map(e => ({ source: e.source, target: e.target })),
    };
  }, [localMindmap]);

  const hasGraph = localMindmap && !localMindmap.not_generated && localMindmap.nodes.length > 0;

  return (
    <div className="flex-1 flex min-h-0 overflow-hidden">
      {/* ── Left: Disease List ── */}
      <div className="w-72 shrink-0 flex flex-col border-r border-slate-100 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-900/60">
        <div className="p-4 border-b border-slate-100 dark:border-slate-800">
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
      <div className="flex-1 flex flex-col min-w-0">

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
            <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 flex items-center gap-3 shrink-0 flex-wrap">
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
              <div className="flex items-center gap-1.5 shrink-0 flex-wrap">
                {(localMindmap!.visual_refs?.length ?? 0) > 0 && (
                  <button onClick={() => setShowVisuals(!showVisuals)}
                    className="flex items-center gap-1 px-2.5 py-1.5 text-[11px] bg-amber-50 text-amber-600 border border-amber-100 rounded-full hover:bg-amber-100 transition-all">
                    <span className="material-symbols-outlined text-[13px]">imagesmode</span>
                    {localMindmap!.visual_refs!.length} Visual
                  </button>
                )}
                {/* Edit Mode Toggle */}
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
                    <p className="text-[10px] text-center text-amber-700 font-medium max-w-[7rem] line-clamp-2">{vref.heading}</p>
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
            <div className="flex-1 flex min-h-0 relative overflow-hidden">
              {/* Canvas */}
              <div className="flex-1 relative" ref={containerRef} onClick={(e) => { if ((e.target as HTMLElement).tagName === 'CANVAS') setActiveNode(null); }}>
                <ForceGraph2D
                  key={graphKey}
                  ref={graphRef}
                  graphData={graphData as any}
                  width={dimensions.width - (editMode && activeNode ? 300 : 0)}
                  height={dimensions.height}
                  dagMode="radialout"
                  dagLevelDistance={110}
                  nodeRelSize={1}
                  nodeVal={(node: any) => {
                    const t = node.type;
                    return t === 'root' ? 60 : t === 'section' ? 28 : t === 'concept' ? 18 : 12;
                  }}
                  linkColor={() => 'rgba(148,163,184,0.45)'}
                  linkWidth={1.4}
                  linkCurvature={0.25}
                  linkDirectionalArrowLength={5}
                  linkDirectionalArrowRelPos={1}
                  enableNodeDrag
                  cooldownTicks={150}
                  d3AlphaDecay={0.02}
                  d3VelocityDecay={0.3}
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
                  <div className="absolute bottom-4 right-4 max-w-xs bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-100 dark:border-slate-800 p-4 z-10">
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
                    <p className="font-headline font-bold text-slate-800 dark:text-slate-100 mb-2 leading-snug">{activeNode.label}</p>
                    <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{activeNode.summary}</p>
                  </div>
                )}
              </div>

              {/* Edit Panel (slide in from right) */}
              {editMode && activeNode && (
                <div className="w-72 shrink-0 border-l border-rose-100 dark:border-slate-700 bg-white dark:bg-slate-900 flex flex-col shadow-xl z-20 overflow-y-auto">
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
              )}
            </div>

            {/* Key Takeaways */}
            {(localMindmap!.key_takeaways?.length ?? 0) > 0 && (
              <div className="shrink-0 border-t border-slate-100 dark:border-slate-800 px-5 py-2.5 bg-violet-50/50 dark:bg-violet-900/10 flex items-start gap-2 flex-wrap">
                <span className="material-symbols-outlined text-violet-400 text-[14px] mt-0.5">stars</span>
                {localMindmap!.key_takeaways!.map((t, i) => (
                  <span key={i} className="text-xs bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 px-2.5 py-1 rounded-full border border-violet-100">{t}</span>
                ))}
              </div>
            )}
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
    setVisualBusy(true);
    setErr(null);
    try {
      await axios.patch(`${API_URL}/library/stases/${staseSlug}/diseases/${selectedId}/visual_refs`, {
        images: visualSelected.map((img) => ({
          image_abs_path: (img.image_abs_path || '').trim() || undefined,
          image_ref: (img.image_ref || '').trim() || undefined,
          storage_url: (img.storage_url || '').trim() || undefined,
          image_url: (img.image_url || '').trim() || undefined,
          heading: img.heading || '',
          source_name: img.source_name || '',
          page_no: img.page_no || 0,
        })),
      });
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
        {/* List column */}
        <aside className="w-full lg:w-[min(420px,40vw)] shrink-0 border-r border-slate-200/60 dark:border-slate-800 flex flex-col bg-slate-50/40 dark:bg-slate-950/30">
          <div className="p-4 border-b border-slate-200/50 space-y-3">
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
                  className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-500"
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

        {/* Detail column */}
        <section className="flex-1 flex flex-col min-w-0 min-h-0 overflow-y-auto bg-white/40 dark:bg-slate-900/20">
          {!selectedId && (
            <div className="m-auto text-center max-w-sm p-8 text-slate-400">
              <span className="material-symbols-outlined text-5xl mb-3 opacity-40">menu_book</span>
              <p className="text-sm">Pilih penyakit di daftar untuk melihat atau membuat penjelasan.</p>
            </div>
          )}
          {selectedId && loadingDetail && (
            <div className="m-auto flex flex-col items-center gap-2 text-slate-400">
              <span className="material-symbols-outlined animate-spin">progress_activity</span>
              <span className="text-sm">Memuat artikel...</span>
            </div>
          )}
          {selectedId && !loadingDetail && detail && (
            <div className="p-4 md:p-8 max-w-4xl mx-auto w-full space-y-6 pb-24">
              <div className="flex flex-wrap items-start justify-between gap-4">
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
                <div className="flex flex-wrap gap-2">
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
                  className="w-full rounded-xl border border-slate-200 dark:border-slate-600 bg-transparent px-3 py-2 text-sm min-h-[72px]"
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

              <div className="rounded-[2rem] border border-white/40 bg-surface-container-low/50 p-6 md:p-10 glass-border">
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
              className="w-full rounded-xl border border-slate-200 dark:border-slate-600 px-3 py-2 text-sm min-h-[120px]"
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
          className="fixed inset-0 z-[60] bg-black/50 flex items-center justify-center p-4 overflow-y-auto"
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
          className="fixed inset-0 z-[55] bg-black/50 flex items-center justify-center p-4 overflow-y-auto"
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
              className="flex-1 w-full rounded-xl border border-slate-200 dark:border-slate-600 px-3 py-2 text-sm font-mono min-h-[320px] overflow-y-auto"
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

  // ── Render section content – Ide 2 (Markdown) ──
  const renderSection = (section: DraftSection, sIdx: number) => {
    const icon = SECTION_ICONS[section.title] || 'article';
    const content = section.markdown ?? (section.points?.join('\n\n') ?? '');

    return (
      <div key={sIdx} className="group">
        <h3 className="font-headline text-base font-bold text-indigo-600 dark:text-indigo-400 mb-3 flex items-center gap-2">
          <span className="material-symbols-outlined text-indigo-400/60 group-hover:text-indigo-500 transition-colors text-[20px]" style={{ fontVariationSettings: "'FILL' 1" }}>
            {icon}
          </span>
          {section.title}
        </h3>
        <div className="prose-slate text-on-surface text-base leading-relaxed pl-1">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
            {content}
          </ReactMarkdown>
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen w-full overflow-hidden flex bg-background">
      {/* ── SideNavBar ── */}
      <nav className={`fixed left-0 top-0 h-full z-40 flex-col p-4 bg-slate-50/40 dark:bg-slate-950/40 backdrop-blur-2xl transition-all duration-300 ease-in-out ${isSidebarMinimized ? 'w-20' : 'w-72'} rounded-r-3xl tonal-layering no-border shadow-[40px_0_60px_-10px_rgba(0,0,0,0.04)] font-[Inter] text-sm hidden md:flex overflow-x-hidden`}>
        <div className="flex items-center gap-3 mb-10 px-2 mt-4">
          <div className="h-10 w-10 min-w-[2.5rem] rounded-full bg-primary-container flex items-center justify-center shrink-0">
            <span className="material-symbols-outlined text-secondary" style={{ fontVariationSettings: "'FILL' 1" }}>medical_services</span>
          </div>
          <div className={`transition-opacity duration-300 whitespace-nowrap ${isSidebarMinimized ? 'opacity-0 pointer-events-none w-0' : 'opacity-100'}`}>
            <h2 className="text-lg font-black text-slate-800 dark:text-slate-200">Clinical Assistant</h2>
            <p className="text-xs text-on-surface-variant font-label">AI-Powered Insights</p>
          </div>
        </div>

        <button
          onClick={() => {
            if (activeView === 'chat') clearChat();
            else setActiveView('chat');
          }}
          className={`gradient-btn w-full rounded-full py-3 px-3 text-white font-medium mb-8 flex items-center justify-center gap-2 hover:opacity-90 transition-all ${isSidebarMinimized ? 'px-0' : ''}`}
        >
          <span className="material-symbols-outlined text-[20px] shrink-0">add</span>
          <span className={`whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized ? 'opacity-0 hidden' : 'opacity-100'}`}>
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
              <span className={`font-medium whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized ? 'opacity-0 w-0 hidden' : 'opacity-100'}`}>
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
              <span className={`whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized ? 'opacity-0 w-0 hidden' : 'opacity-100'}`}>{item.label}</span>
            </a>
          ))}
        </div>
      </nav>

      {/* ── Main Content ── */}
      <main className={`flex-1 flex flex-col h-full min-h-0 relative transition-all duration-300 ml-0 ${isSidebarMinimized ? 'md:ml-20' : 'md:ml-72'}`}>
        {/* Header */}
        <header className="flex justify-between items-center px-4 md:px-8 h-20 w-full sticky top-0 z-30 bg-white/60 dark:bg-slate-900/60 backdrop-blur-3xl tracking-tighter border-b border-white/30 shrink-0">
          <div className="flex items-center gap-4">
            <button onClick={() => setIsSidebarMinimized(!isSidebarMinimized)} className="hidden md:flex p-2 text-on-surface-variant hover:bg-surface-container rounded-full transition-colors">
              <span className="material-symbols-outlined">{isSidebarMinimized ? 'menu' : 'menu_open'}</span>
            </button>
            <h1 className="text-xl md:text-2xl font-headline font-bold bg-gradient-to-r from-indigo-600 to-cyan-600 bg-clip-text text-transparent">
              Medical RAG
              {activeView === 'library' && (
                <span className="block text-xs font-normal text-slate-500 mt-0.5">Medical Library</span>
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

        {/* Chat Canvas */}
        {activeView === 'chat' && (
        <>
        <div className="flex-1 overflow-y-auto px-4 md:px-12 lg:px-24 py-8 pb-40 flex flex-col gap-8 min-h-0">
          {messages.length === 0 && (
            <div className="m-auto text-center max-w-md pt-20 flex flex-col items-center opacity-70">
              <span className="material-symbols-outlined text-6xl text-secondary mb-4 opacity-50" style={{ fontVariationSettings: "'FILL' 1" }}>biotech</span>
              <h2 className="text-2xl font-headline font-bold text-slate-700">Mulai Konsultasi RAG</h2>
              <p className="text-sm mt-2 text-slate-500 leading-relaxed font-body">
                Tanyakan kondisi klinis yang spesifik. Basis pengetahuan ini bersumber dari panduan medis terkini dan literatur terverifikasi.
              </p>
              <div className="mt-6 grid grid-cols-2 gap-2 w-full">
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
            <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} gap-2 w-full ${msg.role === 'user' ? 'max-w-3xl ml-auto' : 'max-w-4xl mr-auto mt-6'}`}>
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
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shrink-0 shadow-sm text-white">
                      <span className="material-symbols-outlined text-[18px]" style={{ fontVariationSettings: "'FILL' 1" }}>medical_services</span>
                    </div>
                    <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">Clinical Assistant</span>
                  </>
                )}
              </div>

              {/* Message Bubble */}
              <div className={`${
                msg.role === 'user'
                  ? 'bg-surface-container-lowest glass-panel glass-border p-5 rounded-3xl rounded-tr-sm ambient-shadow text-on-surface max-w-full'
                  : 'bg-surface-container-low/50 backdrop-blur-xl border border-white/40 p-6 md:p-8 rounded-[2.5rem] rounded-tl-sm shadow-[0_8px_32px_-12px_rgba(0,0,0,0.05)] w-full'
              }`}>
                {msg.role === 'user' ? (
                  <p className="font-body text-base leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                ) : msg.error ? (
                  <div className="bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-800 p-4 rounded-3xl">
                    <p className="text-red-600 dark:text-red-400 text-sm font-medium flex items-center gap-2">
                      <span className="material-symbols-outlined text-[18px]">error</span>
                      {msg.content}
                    </p>
                  </div>
                ) : msg.data ? (
                  <div className="prose prose-slate max-w-none text-on-surface font-body">
                    <div className="bg-white/80 dark:bg-slate-900/80 rounded-[2.5rem] p-6 md:p-10 glass-border ambient-shadow space-y-10">
                      {/* Disease Title */}
                      {msg.data.draft_answer.disease && (
                        <div className="border-b border-slate-100 dark:border-slate-800 pb-6">
                          <p className="text-sm font-label text-slate-500 uppercase tracking-widest mb-1">Diagnosis Analysis</p>
                          <h2 className="text-2xl md:text-3xl font-headline font-black text-slate-800 dark:text-slate-100">
                            {msg.data.draft_answer.disease}
                          </h2>
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
                      <div className="space-y-10">
                        {msg.data.draft_answer.sections.map((section, sIdx) => renderSection(section, sIdx))}
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
                          <div className="flex gap-4 overflow-x-auto pb-4 snap-x">
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
                                  className="w-48 h-32 object-cover transition-transform duration-500 group-hover:scale-110"
                                  onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = PLACEHOLDER_IMG; }}
                                />
                                <div className="p-3 bg-white/90 dark:bg-slate-900/90 backdrop-blur-sm w-full absolute bottom-0 z-20">
                                  <p className="text-[11px] text-center font-medium truncate w-40 text-slate-700 dark:text-slate-300">{img.heading}</p>
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
                <div className="flex items-center gap-2 mt-2 ml-4 flex-wrap">
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
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-cyan-500 flex items-center justify-center shrink-0 shadow-sm text-white">
                  <span className="material-symbols-outlined text-[18px] animate-pulse">medical_services</span>
                </div>
                <span className="text-xs font-label text-on-surface-variant uppercase tracking-wider">Clinical Assistant</span>
              </div>
              <div className="bg-surface-container-lowest glass-border glass-panel px-6 py-4 rounded-[2rem] rounded-tl-sm flex items-center gap-3 text-secondary animate-pulse">
                <span className="material-symbols-outlined animate-spin" style={{ animationDuration: '1.5s' }}>refresh</span>
                <span className="text-sm font-medium">Menganalisis dokumen referensi klinis...</span>
              </div>
            </div>
          )}

          <div className="h-10" ref={messagesEndRef} />
        </div>

        {/* Floating Input Area */}
        <div className="absolute bottom-0 left-0 w-full p-4 md:p-8 bg-gradient-to-t from-background via-background/90 to-transparent pointer-events-none flex justify-center z-20">
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
            <div className={`bg-surface-container-lowest/80 backdrop-blur-2xl rounded-[2rem] glass-border p-2 flex items-end gap-2 ambient-shadow transition-all ${isLoading ? 'opacity-80' : 'shadow-[0_-10px_40px_rgba(0,0,0,0.03)] focus-within:ring-2 focus-within:ring-primary-container'}`}>
              <button disabled={isLoading} className="p-3 text-on-surface-variant hover:text-secondary hover:bg-secondary-container/50 rounded-full transition-colors shrink-0 mb-1">
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
                  className="w-full bg-transparent border-none text-on-surface placeholder-on-surface-variant/60 resize-none py-4 px-2 focus:ring-0 font-body text-base max-h-32 min-h-[56px] block outline-none disabled:opacity-70"
                  placeholder={isLoading ? 'Sedang memproses...' : `Tanya seputar kondisi medis ${activeStase.toUpperCase()}, gejala, atau tindak lanjut...`}
                  rows={1}
                />
              </div>
              <button
                onClick={submitQuery}
                disabled={isLoading || !query.trim()}
                className="p-3 mb-1 bg-primary-container text-on-primary-container hover:bg-secondary hover:text-white rounded-full transition-all shrink-0 flex items-center justify-center group shadow-sm disabled:opacity-50 disabled:hover:bg-primary-container disabled:hover:text-on-primary-container"
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
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setEvidenceModal(null)}>
            <div className="bg-surface-container-lowest glass-border glass-panel rounded-[2rem] max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <div className="flex justify-between items-center p-6 border-b border-surface-variant/60">
                <h2 className="text-lg font-headline font-bold text-secondary flex items-center gap-2">
                  <span className="material-symbols-outlined text-[20px]">library_books</span>
                  Sitasi: {evidenceModal.citation}
                </h2>
                <button onClick={() => setEvidenceModal(null)} className="p-2 text-on-surface-variant hover:text-secondary hover:bg-secondary-container rounded-full transition-colors">
                  <span className="material-symbols-outlined text-[20px]">close</span>
                </button>
              </div>
              <div className="p-6 overflow-y-auto flex-1 space-y-4">
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
          <div className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center z-50 p-4" onClick={() => setImageModal(null)}>
            <div className="relative max-w-4xl w-full flex flex-col items-center" onClick={(e) => e.stopPropagation()}>
              <button onClick={() => setImageModal(null)} className="absolute -top-14 right-0 text-white hover:text-gray-300 flex items-center gap-2 font-medium bg-white/10 hover:bg-white/20 px-4 py-2 rounded-full backdrop-blur-sm transition-colors">
                <span className="material-symbols-outlined text-[18px]">close</span> Tutup
              </button>
              <img
                src={resolveImageUrl(imageModal.image_url)}
                alt={imageModal.heading}
                className="w-full h-auto max-h-[75vh] object-contain rounded-2xl shadow-2xl"
                onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = PLACEHOLDER_IMG; }}
              />
              <div className="bg-surface-container-lowest glass-border glass-panel p-5 mt-4 rounded-2xl text-center shadow-lg w-full max-w-xl">
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
