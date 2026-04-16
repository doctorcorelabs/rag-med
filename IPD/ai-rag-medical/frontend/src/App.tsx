import { useState, useEffect, useRef, useCallback } from 'react';
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

type KGNode = { id: string; label: string; type: string; group: number; val: number };
type KGEdge = { source: string; target: string; relation: string };
type KnowledgeGraph = { nodes: KGNode[]; edges: KGEdge[]; disease: string };

const API_URL = 'http://127.0.0.1:8010';

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

// Node color mapping for knowledge graph
const NODE_GROUP_COLORS: Record<number, string> = {
  0: '#6366f1', // disease – indigo
  1: '#06b6d4', // Definisi – cyan
  2: '#f59e0b', // Etiologi – amber
  3: '#ef4444', // Manifestasi – red
  4: '#8b5cf6', // Diagnosis – violet
  5: '#10b981', // Tatalaksana – emerald
  6: '#f97316', // Komplikasi – orange
  7: '#64748b', // Prognosis – slate
  8: '#a78bfa', // Ringkasan – light violet
  9: '#94a3b8', // concept – gray
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

// ─── KNOWLEDGE GRAPH MODAL ───────────────────────────────────────────────────
function KnowledgeGraphModal({ disease, onClose }: { disease: string; onClose: () => void }) {
  const [graphData, setGraphData] = useState<KnowledgeGraph | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

  useEffect(() => {
    if (containerRef.current) {
      const { offsetWidth, offsetHeight } = containerRef.current;
      setDimensions({ width: offsetWidth || 800, height: offsetHeight || 500 });
    }
  }, []);

  useEffect(() => {
    setIsLoading(true);
    axios.get<KnowledgeGraph>(`${API_URL}/knowledge_graph/${encodeURIComponent(disease)}`)
      .then((res) => setGraphData(res.data))
      .catch((err) => setError(`Gagal memuat graph: ${err.message}`))
      .finally(() => setIsLoading(false));
  }, [disease]);

  const graphStructure = graphData
    ? {
        nodes: graphData.nodes,
        links: graphData.edges.map((e) => ({ source: e.source, target: e.target, relation: e.relation })),
      }
    : { nodes: [], links: [] };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-white dark:bg-slate-900 rounded-[2rem] w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl border border-white/20 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <span className="material-symbols-outlined text-white text-[20px]">hub</span>
            </div>
            <div>
              <h2 className="text-lg font-headline font-bold text-slate-800 dark:text-slate-100">
                Knowledge Graph
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">Peta relasi konsep: <span className="text-indigo-600 font-semibold">{disease}</span></p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors">
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        {/* Legend */}
        <div className="px-5 py-2 flex flex-wrap gap-2 border-b border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50">
          {[
            { label: 'Penyakit', color: '#6366f1' },
            { label: 'Definisi', color: '#06b6d4' },
            { label: 'Etiologi', color: '#f59e0b' },
            { label: 'Manifestasi', color: '#ef4444' },
            { label: 'Diagnosis', color: '#8b5cf6' },
            { label: 'Tatalaksana', color: '#10b981' },
            { label: 'Komplikasi', color: '#f97316' },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
              <span className="text-[11px] text-slate-500 dark:text-slate-400">{item.label}</span>
            </div>
          ))}
        </div>

        {/* Graph Canvas */}
        <div className="flex-1 relative" ref={containerRef}>
          {isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <div className="w-10 h-10 rounded-full border-2 border-indigo-200 border-t-indigo-600 animate-spin" />
              <p className="text-sm text-slate-500">Membangun grafik relasi...</p>
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="text-sm text-red-500">{error}</p>
            </div>
          )}
          {!isLoading && !error && graphData && graphData.nodes.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 opacity-60">
              <span className="material-symbols-outlined text-5xl text-slate-300">lan</span>
              <p className="text-sm text-slate-400">Belum ada data graph. Rebuild index terlebih dahulu.</p>
            </div>
          )}
          {!isLoading && !error && graphData && graphData.nodes.length > 0 && (
            <ForceGraph2D
              graphData={graphStructure as any}
              width={dimensions.width}
              height={dimensions.height}
              nodeLabel={(node: any) => node.label}
              nodeColor={(node: any) => NODE_GROUP_COLORS[node.group] ?? '#94a3b8'}
              nodeVal={(node: any) => node.val ?? 6}
              linkLabel={(link: any) => link.relation}
              linkColor={() => 'rgba(148,163,184,0.4)'}
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={1}
              linkDirectionalParticles={1}
              linkDirectionalParticleSpeed={0.004}
              enableNodeDrag
              cooldownTicks={80}
              nodeCanvasObjectMode={() => 'after'}
              nodeCanvasObject={(node: any, ctx, globalScale) => {
                const label = node.label as string;
                const fontSize = Math.max(10 / globalScale, 3);
                ctx.font = `${fontSize}px Inter, sans-serif`;
                ctx.fillStyle = '#334155';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label.length > 20 ? label.slice(0, 20) + '…' : label, node.x, (node.y as number) + (node.val ?? 6) + fontSize + 1);
              }}
            />
          )}
        </div>
      </div>
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
  const [kgDisease, setKgDisease] = useState<string | null>(null); // Ide 18: KG modal trigger
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    localStorage.setItem('medrag_chat_v3', JSON.stringify(messages));
  }, [messages]);

  // Ide 11: Build chat_history from current messages for multi-turn
  const buildChatHistory = useCallback(() => {
    const history: { role: string; content: string }[] = [];
    for (const msg of messages) {
      if (msg.role === 'user') {
        history.push({ role: 'user', content: msg.content });
      } else if (msg.role === 'bot' && msg.data) {
        // Represent bot response as assistant message summary
        const summary = `Analisis ${msg.data.draft_answer.disease}: ${msg.data.draft_answer.sections.map(s => s.title).join(', ')}`;
        history.push({ role: 'assistant', content: summary });
      }
    }
    return history.slice(-8); // Keep last 4 exchanges
  }, [messages]);

  const submitQuery = async () => {
    if (!query.trim() || isLoading) return;

    const nextQuery = query.trim();
    setMessages((prev) => [...prev, { role: 'user', content: nextQuery }]);
    setQuery('');
    setIsLoading(true);

    try {
      const response = await axios.post<ApiResponse>(`${API_URL}/search_disease_context`, {
        disease_name: nextQuery,
        detail_level: 'detail',
        top_k: 8,
        include_images: true,
        chat_history: buildChatHistory(), // Ide 11: Multi-turn history
      });
      setMessages((prev) => [...prev, { role: 'bot', data: response.data }]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, { role: 'bot', error: true, content: 'Terjadi kesalahan saat memproses data.' }]);
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

  // Ide 18: open KG from the most recent bot answer
  const handleOpenKG = (disease: string) => setKgDisease(disease);

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
      {/* ── Knowledge Graph Modal (Ide 18) ── */}
      {kgDisease && <KnowledgeGraphModal disease={kgDisease} onClose={() => setKgDisease(null)} />}

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

        <button onClick={clearChat} className={`gradient-btn w-full rounded-full py-3 px-3 text-white font-medium mb-8 flex items-center justify-center gap-2 hover:opacity-90 transition-all ${isSidebarMinimized ? 'px-0' : ''}`}>
          <span className="material-symbols-outlined text-[20px] shrink-0">add</span>
          <span className={`whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized ? 'opacity-0 hidden' : 'opacity-100'}`}>New Consultation</span>
        </button>

        <div className="flex-1 flex flex-col gap-2 overflow-y-auto overflow-x-hidden pr-2">
          {[
            { icon: 'chat_bubble', label: 'Recent Chats', active: true },
            { icon: 'book', label: 'Medical Library' },
            { icon: 'hub', label: 'Knowledge Graph' },
            { icon: 'query_stats', label: 'Analytics' },
          ].map((item) => (
            <a
              key={item.label}
              className={`flex items-center gap-3 px-3 py-3 rounded-xl transition-all flex-row cursor-pointer ${
                item.active
                  ? 'bg-indigo-50/50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300'
                  : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-white/20 dark:hover:bg-slate-800/20'
              }`}
            >
              <span className="material-symbols-outlined shrink-0 text-center w-6">{item.icon}</span>
              <span className={`font-medium whitespace-nowrap transition-opacity duration-300 ${isSidebarMinimized ? 'opacity-0 w-0 hidden' : 'opacity-100'}`}>
                {item.label}
              </span>
            </a>
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
      <main className={`flex-1 flex flex-col h-full relative transition-all duration-300 ml-0 ${isSidebarMinimized ? 'md:ml-20' : 'md:ml-72'}`}>
        {/* Header */}
        <header className="flex justify-between items-center px-4 md:px-8 h-20 w-full sticky top-0 z-30 bg-white/60 dark:bg-slate-900/60 backdrop-blur-3xl tracking-tighter border-b border-white/30">
          <div className="flex items-center gap-4">
            <button onClick={() => setIsSidebarMinimized(!isSidebarMinimized)} className="hidden md:flex p-2 text-on-surface-variant hover:bg-surface-container rounded-full transition-colors">
              <span className="material-symbols-outlined">{isSidebarMinimized ? 'menu' : 'menu_open'}</span>
            </button>
            <h1 className="text-xl md:text-2xl font-headline font-bold bg-gradient-to-r from-indigo-600 to-cyan-600 bg-clip-text text-transparent">Medical RAG</h1>
          </div>
          <div className="flex items-center gap-2 md:gap-4">
            <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 dark:bg-emerald-900/20 rounded-full border border-emerald-100 dark:border-emerald-800">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">Multi-turn AI</span>
            </div>
            <button onClick={clearChat} className="hidden md:flex items-center gap-2 px-4 py-2 bg-surface-container-lowest glass-border rounded-full text-secondary hover:bg-surface-container-high transition-colors font-medium text-sm">
              <span className="material-symbols-outlined text-[18px]">delete</span>
              Bersihkan Chat
            </button>
            <button className="p-2 text-slate-500 hover:bg-white/40 transition-all rounded-full scale-95 active:duration-150">
              <span className="material-symbols-outlined">settings</span>
            </button>
          </div>
        </header>

        {/* Chat Canvas */}
        <div className="flex-1 overflow-y-auto px-4 md:px-12 lg:px-24 py-8 pb-40 flex flex-col gap-8">
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
                                  src={`${API_URL}${img.image_url}`}
                                  alt={img.heading}
                                  className="w-48 h-32 object-cover transition-transform duration-500 group-hover:scale-110"
                                  onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = 'https://via.placeholder.com/150?text=Image+Not+Found'; }}
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
                  placeholder={isLoading ? 'Sedang memproses...' : 'Tanya seputar kondisi medis, gejala, atau tindak lanjut...'}
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
                src={`${API_URL}${imageModal.image_url}`}
                alt={imageModal.heading}
                className="w-full h-auto max-h-[75vh] object-contain rounded-2xl shadow-2xl"
                onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = 'https://via.placeholder.com/800?text=Image+Not+Found'; }}
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
