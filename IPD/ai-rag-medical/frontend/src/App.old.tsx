import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import {
  BookOpen,
  ClipboardList,
  FolderOpen,
  Loader2,
  Menu,
  MessageSquare,
  Plus,
  Search,
  Send,
  Settings,
  Shield,
  Sparkles,
  Trash2,
  X,
  BarChart3,
  FolderSearch,
} from 'lucide-react'

type EvidenceItem = {
  source_name: string
  page_no: number
  heading: string
  content: string
  markdown_path?: string
}

type ImageItem = {
  source_name: string
  page_no: number
  heading: string
  image_url: string
}

type ApiResponse = {
  query: string
  detail_level: string
  evidence_count: number
  evidence: EvidenceItem[]
  draft_answer: {
    disease: string
    sections: { title: string; points: string[] }[]
    citations: string[]
    grounded: boolean
  }
  images: ImageItem[]
}

type ChatMessage =
  | { role: 'user'; content: string }
  | { role: 'bot'; data?: ApiResponse; error?: boolean; content?: string }

type ConsultationSummary = {
  title: string
  lastQuery: string
  updatedAt: string
}

const API_URL = 'http://127.0.0.1:8010'
const STORAGE_KEY = 'medrag_chat'
const THREADS_KEY = 'medrag_threads'

export default function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? (JSON.parse(saved) as ChatMessage[]) : []
  })
  const [threads, setThreads] = useState<ConsultationSummary[]>(() => {
    const saved = localStorage.getItem(THREADS_KEY)
    return saved ? (JSON.parse(saved) as ConsultationSummary[]) : []
  })
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [evidenceModal, setEvidenceModal] = useState<{ citation: string; content: EvidenceItem[] } | null>(null)
  const [imageModal, setImageModal] = useState<ImageItem | null>(null)

  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    localStorage.setItem(THREADS_KEY, JSON.stringify(threads))
  }, [threads])

  const addThread = (title: string, queryText: string) => {
    setThreads((prev) => {
      const item: ConsultationSummary = {
        title,
        lastQuery: queryText,
        updatedAt: new Date().toLocaleString('id-ID', {
          day: '2-digit',
          month: 'short',
          hour: '2-digit',
          minute: '2-digit',
        }),
      }
      const next = [item, ...prev.filter((entry) => entry.title !== title)].slice(0, 6)
      return next
    })
  }

  const submitQuery = async () => {
    if (!query.trim() || isLoading) return

    const nextQuery = query.trim()
    const userMessage: ChatMessage = { role: 'user', content: nextQuery }
    setMessages((prev) => [...prev, userMessage])
    addThread(nextQuery.slice(0, 28), nextQuery)
    setQuery('')
    setIsLoading(true)

    try {
      const response = await axios.post<ApiResponse>(`${API_URL}/search_disease_context`, {
        disease_name: nextQuery,
        detail_level: 'detail',
        top_k: 8,
        include_images: true,
      })

      setMessages((prev) => [...prev, { role: 'bot', data: response.data }])
    } catch (error) {
      console.error(error)
      setMessages((prev) => [...prev, { role: 'bot', error: true, content: 'Terjadi kesalahan saat memproses data.' }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleOpenEvidence = (citation: string, allEvidence: EvidenceItem[]) => {
    const matchingEvidence = allEvidence.filter((item) => `${item.source_name} p.${item.page_no}` === citation)
    setEvidenceModal({ citation, content: matchingEvidence })
  }

  const handleNewConsultation = () => {
    setMessages([])
    setQuery('')
    setEvidenceModal(null)
    setImageModal(null)
    setSidebarOpen(false)
  }

  const recentThreads = threads.length ? threads : [{ title: 'Influenza', lastQuery: 'Buatan pertama', updatedAt: 'Baru' }]

  return (
    <div className="min-h-screen bg-[#f7f9fb] text-[#2a3439]">
      <div className="flex min-h-screen overflow-hidden">
        <aside className="hidden w-72 shrink-0 border-r border-white/60 bg-white/55 p-4 backdrop-blur-2xl md:flex md:flex-col glass-border">
          <div className="mb-10 flex items-center gap-3 px-3 pt-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#dae2fd] text-[#004eaa] shadow-sm">
              <BookOpen size={20} />
            </div>
            <div>
              <h2 className="font-headline text-lg font-black text-slate-800">Medical RAG</h2>
              <p className="font-label text-xs text-slate-500">Clinical Ethereal</p>
            </div>
          </div>

          <button
            onClick={handleNewConsultation}
            className="gradient-btn mb-8 flex w-full items-center justify-center gap-2 rounded-full px-6 py-3 font-medium text-white transition-opacity hover:opacity-90"
          >
            <Plus size={18} />
            New Consultation
          </button>

          <div className="flex-1 overflow-y-auto pr-2">
            <div className="mb-3 px-3 text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-400">
              Recent Chats
            </div>
            <div className="space-y-2">
              {recentThreads.map((thread) => (
                <button
                  key={`${thread.title}-${thread.updatedAt}`}
                  className="flex w-full items-start gap-3 rounded-2xl border border-transparent bg-white/55 px-4 py-3 text-left shadow-sm transition-all hover:border-white/80 hover:bg-white/80 hover:shadow-md"
                >
                  <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-full bg-[#e8eff3] text-[#004eaa]">
                    <MessageSquare size={16} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold text-slate-800">{thread.title}</div>
                    <div className="mt-0.5 truncate text-xs text-slate-500">{thread.lastQuery}</div>
                    <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-slate-400">{thread.updatedAt}</div>
                  </div>
                </button>
              ))}
            </div>

            <div className="mt-8 px-3 text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-400">
              Navigation
            </div>
            <div className="mt-2 space-y-2">
              {[
                { icon: ClipboardList, label: 'Medical Library' },
                { icon: FolderOpen, label: 'Patient Records' },
                { icon: BarChart3, label: 'Analytics' },
              ].map((item) => (
                <button
                  key={item.label}
                  className="flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-sm text-slate-600 transition-all hover:bg-white/70 hover:text-[#005bc4]"
                >
                  <item.icon size={18} />
                  <span className="font-medium">{item.label}</span>
                </button>
              ))}
            </div>

            <div className="mt-8 border-t border-slate-200/60 pt-4">
              <button className="flex w-full items-center gap-3 rounded-2xl px-4 py-2.5 text-sm text-slate-600 transition-colors hover:text-[#005bc4]">
                <Shield size={18} />
                <span>Privacy</span>
              </button>
              <button className="mt-1 flex w-full items-center gap-3 rounded-2xl px-4 py-2.5 text-sm text-slate-600 transition-colors hover:text-[#005bc4]">
                <FolderSearch size={18} />
                <span>Support</span>
              </button>
            </div>
          </div>
        </aside>

        <main className="relative flex min-h-screen flex-1 flex-col overflow-hidden">
          <header className="sticky top-0 z-30 flex h-20 items-center justify-between bg-white/60 px-4 backdrop-blur-3xl md:px-8 glass-border border-b border-white/50">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSidebarOpen(true)}
                className="rounded-full p-2 text-slate-500 transition-colors hover:bg-white/60 md:hidden"
              >
                <Menu size={20} />
              </button>
              <div>
                <h1 className="font-headline text-2xl font-bold tracking-tight bg-linear-to-r from-[#005bc4] to-[#006592] bg-clip-text text-transparent">
                  Medical RAG
                </h1>
                <p className="text-xs text-slate-500">Clinical insights, clean retrieval, elegant interface.</p>
              </div>
            </div>

            <div className="flex items-center gap-2 md:gap-3">
              <button
                onClick={handleNewConsultation}
                className="hidden items-center gap-2 rounded-full border border-white/70 bg-white/70 px-4 py-2 text-sm font-medium text-[#005bc4] shadow-sm transition-colors hover:bg-white md:flex"
              >
                <Trash2 size={16} />
                Bersihkan Chat
              </button>
              <button className="rounded-full p-2 text-slate-500 transition-colors hover:bg-white/70 hover:text-[#005bc4]">
                <Settings size={18} />
              </button>
            </div>
          </header>

          <div className="flex-1 overflow-y-auto px-4 py-8 pb-36 md:px-12 lg:px-24">
            <div className="mx-auto flex w-full max-w-5xl flex-col gap-8">
              {messages.length === 0 ? (
                <section className="mx-auto flex w-full max-w-3xl flex-col items-center justify-center rounded-4xl border border-white/70 bg-white/70 px-6 py-12 text-center shadow-[0_40px_60px_-10px_rgba(42,52,57,0.04)] backdrop-blur-2xl md:px-10">
                  <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-[#dae2fd] text-[#004eaa] shadow-sm">
                    <Sparkles size={28} />
                  </div>
                  <h2 className="font-headline text-3xl font-black tracking-tight text-slate-800 md:text-4xl">
                    Ask clinical questions with grounded answers.
                  </h2>
                  <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-500 md:text-base">
                    Mulai dengan penyakit, gejala, diagnosis, atau gambar. Jawaban ditampilkan terstruktur dengan sitasi dan
                    evidence yang bisa dibuka langsung.
                  </p>

                  <div className="mt-8 grid w-full gap-3 md:grid-cols-3">
                    {['Influenza', 'DBD', 'Bronkitis akut'].map((item) => (
                      <button
                        key={item}
                        onClick={() => setQuery(item)}
                        className="rounded-2xl border border-white/80 bg-white/80 px-4 py-3 text-left text-sm font-medium text-slate-600 transition-all hover:-translate-y-0.5 hover:shadow-md"
                      >
                        <div className="mb-1 text-[11px] uppercase tracking-[0.2em] text-slate-400">Quick prompt</div>
                        {item}
                      </button>
                    ))}
                  </div>
                </section>
              ) : null}

              <div className="space-y-7">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex w-full ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`flex w-full max-w-4xl flex-col gap-2 ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                      {message.role === 'user' ? (
                        <div className="mr-2 flex items-center gap-2 px-2">
                          <span className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Dr. Smith</span>
                          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#dae2fd] text-[10px] font-bold text-[#005bc4] shadow-sm">
                            DS
                          </div>
                        </div>
                      ) : (
                        <div className="ml-2 flex items-center gap-2 px-2">
                          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-linear-to-br from-[#005bc4] to-[#34b5fa] text-white shadow-sm">
                            <BookOpen size={16} />
                          </div>
                          <span className="text-xs uppercase tracking-[0.22em] text-slate-500">Clinical Assistant</span>
                        </div>
                      )}

                      <div
                        className={`glass-panel glass-border ambient-shadow w-full rounded-4xl p-5 md:p-8 ${
                          message.role === 'user'
                            ? 'rounded-tr-lg bg-white/85 text-slate-800'
                            : 'rounded-tl-lg bg-white/70 text-slate-800'
                        }`}
                      >
                        {message.role === 'user' ? (
                          <p className="text-base leading-relaxed text-slate-700">{message.content}</p>
                        ) : message.error ? (
                          <p className="text-sm text-red-500">{message.content}</p>
                        ) : (
                          <div className="space-y-6">
                            <p className="text-base leading-relaxed text-slate-700 md:text-lg">
                              Jawaban terstruktur berdasarkan materi yang terindeks.
                            </p>

                            {message.data?.draft_answer.sections.map((section, sectionIndex) => (
                              <section key={section.title} className="rounded-2xl border border-white/70 bg-white/75 p-5 glass-border">
                                <h3 className="mb-3 flex items-center gap-2 font-headline text-lg font-bold text-[#005bc4] md:text-xl">
                                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-[#dae2fd] text-[#004eaa]">
                                    {sectionIndex + 1}
                                  </span>
                                  {section.title}
                                </h3>
                                <ul className="space-y-2 text-sm leading-7 text-slate-700 md:text-base">
                                  {section.points.map((point) => (
                                    <li key={point} className="flex gap-2">
                                      <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-[#34b5fa]" />
                                      <span>{point}</span>
                                    </li>
                                  ))}
                                </ul>
                              </section>
                            ))}

                            {message.data?.draft_answer.citations.length ? (
                              <div className="rounded-2xl border border-white/70 bg-white/75 p-4 glass-border">
                                <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                                  <Search size={14} />
                                  Sumber Referensi
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {message.data.draft_answer.citations.map((citation, citationIndex) => (
                                    <button
                                      key={citation}
                                      onClick={() => handleOpenEvidence(citation, message.data?.evidence ?? [])}
                                      className="rounded-full border border-[#dae2fd] bg-[#f7f9fb] px-3 py-1.5 text-xs font-medium text-[#005bc4] transition-all hover:-translate-y-0.5 hover:bg-[#dae2fd]"
                                    >
                                      [{citationIndex + 1}] {citation}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            ) : null}

                            {message.data?.images.length ? (
                              <div className="rounded-2xl border border-white/70 bg-white/75 p-4 glass-border">
                                <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">
                                  <FolderOpen size={14} />
                                  Gambar Terkait
                                </div>
                                <div className="flex gap-3 overflow-x-auto pb-2">
                                  {message.data.images.map((image) => (
                                    <button
                                      key={`${image.source_name}-${image.page_no}-${image.image_url}`}
                                      onClick={() => setImageModal(image)}
                                      className="group shrink-0 overflow-hidden rounded-2xl border border-white/80 bg-white/80 p-1 text-left transition-all hover:-translate-y-0.5 hover:shadow-md"
                                    >
                                      <img
                                        src={`${API_URL}${image.image_url}`}
                                        alt={image.heading}
                                        className="h-28 w-40 rounded-xl object-cover"
                                        onError={(event) => {
                                          event.currentTarget.src = 'https://via.placeholder.com/400x240?text=Image+Not+Found'
                                        }}
                                      />
                                      <div className="w-40 p-2">
                                        <p className="truncate text-xs font-semibold text-slate-700">{image.heading}</p>
                                        <p className="mt-0.5 text-[10px] text-slate-400">
                                          {image.source_name} · Hal {image.page_no}
                                        </p>
                                      </div>
                                    </button>
                                  ))}
                                </div>
                              </div>
                            ) : null}

                            <div className="flex flex-wrap items-center gap-2 border-t border-slate-200/60 pt-4">
                              <button className="rounded-full p-2 text-slate-500 transition-colors hover:bg-[#f1f5f9] hover:text-[#005bc4]" aria-label="Copy">
                                <ClipboardList size={18} />
                              </button>
                              <button className="rounded-full p-2 text-slate-500 transition-colors hover:bg-[#f1f5f9] hover:text-[#005bc4]" aria-label="Thumbs Up">
                                👍
                              </button>
                              <button className="rounded-full p-2 text-slate-500 transition-colors hover:bg-[#f1f5f9] hover:text-[#005bc4]" aria-label="Thumbs Down">
                                👎
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}

                {isLoading ? (
                  <div className="flex justify-start">
                    <div className="glass-panel glass-border ambient-shadow flex items-center gap-3 rounded-3xl px-5 py-4 text-slate-600">
                      <Loader2 className="h-4 w-4 animate-spin text-[#005bc4]" />
                      <span className="text-sm">Mencari evidence dan menyusun jawaban...</span>
                    </div>
                  </div>
                ) : null}
                <div ref={messagesEndRef} />
              </div>
            </div>
          </div>

          <div className="pointer-events-none absolute bottom-0 left-0 z-20 flex w-full justify-center bg-linear-to-t from-[#f7f9fb] via-[#f7f9fb]/90 to-transparent p-4 md:p-8">
            <div className="pointer-events-auto w-full max-w-5xl">
              <div className="glass-panel glass-border ambient-shadow flex items-end gap-2 rounded-4xl p-2 shadow-[0_-10px_40px_rgba(0,0,0,0.03)] transition-all focus-within:ring-2 focus-within:ring-[#dae2fd]">
                <button className="mb-1 shrink-0 rounded-full p-3 text-slate-500 transition-colors hover:bg-[#f1f5f9] hover:text-[#005bc4]">
                  <Plus size={18} />
                </button>
                <div className="relative flex-1">
                  <textarea
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault()
                        if (!isLoading && query.trim()) {
                          void submitQuery()
                        }
                      }
                    }}
                    disabled={isLoading}
                    placeholder="Tanya seputar kondisi medis, gejala, atau referensi..."
                    rows={1}
                    className="block min-h-14 max-h-32 w-full resize-none border-none bg-transparent px-2 py-4 font-body text-base text-slate-700 placeholder:text-slate-400 focus:ring-0"
                  />
                </div>
                <button
                  onClick={() => void submitQuery()}
                  disabled={isLoading || !query.trim()}
                  className="gradient-btn mb-1 flex shrink-0 items-center justify-center rounded-full p-3 text-white transition-all disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Send size={18} />
                </button>
              </div>
              <div className="mt-3 text-center">
                <p className="text-[10px] font-label text-slate-400 md:text-xs">
                  Medical RAG can make mistakes. Verify important clinical information.
                </p>
              </div>
            </div>
          </div>

          {sidebarOpen ? (
            <div className="fixed inset-0 z-40 bg-slate-950/30 backdrop-blur-sm md:hidden" onClick={() => setSidebarOpen(false)}>
              <aside
                className="h-full w-72 bg-white/80 p-4 backdrop-blur-2xl glass-border"
                onClick={(event) => event.stopPropagation()}
              >
                <div className="mb-8 flex items-center justify-between px-2 pt-2">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#dae2fd] text-[#004eaa]">
                      <BookOpen size={20} />
                    </div>
                    <div>
                      <h2 className="font-headline text-lg font-black text-slate-800">Medical RAG</h2>
                      <p className="text-xs text-slate-500">Clinical Ethereal</p>
                    </div>
                  </div>
                  <button onClick={() => setSidebarOpen(false)} className="rounded-full p-2 text-slate-500 hover:bg-white/80">
                    <X size={18} />
                  </button>
                </div>
                <button
                  onClick={handleNewConsultation}
                  className="gradient-btn mb-6 flex w-full items-center justify-center gap-2 rounded-full px-6 py-3 font-medium text-white"
                >
                  <Plus size={18} />
                  New Consultation
                </button>
                <div className="space-y-2">
                  {recentThreads.map((thread) => (
                    <button
                      key={`${thread.title}-${thread.updatedAt}`}
                      className="flex w-full items-start gap-3 rounded-2xl border border-transparent bg-white/70 px-4 py-3 text-left shadow-sm"
                    >
                      <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-full bg-[#e8eff3] text-[#004eaa]">
                        <MessageSquare size={16} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-semibold text-slate-800">{thread.title}</div>
                        <div className="mt-0.5 truncate text-xs text-slate-500">{thread.lastQuery}</div>
                        <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-slate-400">{thread.updatedAt}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </aside>
            </div>
          ) : null}

        </main>

      {evidenceModal ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm"
          onClick={() => setEvidenceModal(null)}
        >
          <div
            className="max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-4xl bg-white/90 p-6 shadow-[0_40px_80px_-20px_rgba(42,52,57,0.2)] glass-border backdrop-blur-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="font-headline text-lg font-bold text-slate-800">Sitasi: {evidenceModal.citation}</h2>
              <button onClick={() => setEvidenceModal(null)} className="rounded-full p-2 text-slate-500 hover:bg-[#f1f5f9] hover:text-black">
                <X size={18} />
              </button>
            </div>
            {evidenceModal.content.length ? (
              <div className="space-y-4">
                {evidenceModal.content.map((item) => (
                  <div key={`${item.source_name}-${item.page_no}-${item.heading}`} className="rounded-2xl border border-white/70 bg-white/75 p-4 glass-border">
                    <h4 className="mb-2 text-sm font-semibold text-[#005bc4]">{item.heading}</h4>
                    <p className="text-sm leading-7 text-slate-600">{item.content}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="italic text-slate-500">Detail chunk tidak ditemukan.</p>
            )}
          </div>
        </div>
      ) : null}

      {imageModal ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm" onClick={() => setImageModal(null)}>
          <div className="relative w-full max-w-4xl" onClick={(event) => event.stopPropagation()}>
            <button onClick={() => setImageModal(null)} className="absolute -top-12 right-0 rounded-full bg-white/10 px-3 py-2 text-sm text-white backdrop-blur-md hover:bg-white/20">
              Tutup
            </button>
            <img
              src={`${API_URL}${imageModal.image_url}`}
              alt={imageModal.heading}
              className="max-h-[85vh] w-full rounded-4xl object-contain shadow-2xl"
              onError={(event) => {
                event.currentTarget.src = 'https://via.placeholder.com/800?text=Image+Not+Found'
              }}
            />
            <div className="glass-panel glass-border ambient-shadow mt-3 rounded-3xl p-4">
              <p className="font-headline font-bold text-slate-800">{imageModal.heading}</p>
              <p className="text-sm text-slate-600">
                Sumber: {imageModal.source_name} - Hal {imageModal.page_no}
              </p>
            </div>
          </div>
        </div>
      ) : null}

      </div>
    </div>
  )
}