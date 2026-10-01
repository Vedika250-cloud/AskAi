'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Send,
  Plus,
  Sparkles,
  BookOpen,
  GraduationCap,
  HelpCircle,
  FileText,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  AlertCircle,
  Loader2,
  MessageSquare,
  Menu,
  X,
  ArrowRight,
  Check,
  BookMarked,
  Library,
} from 'lucide-react';
import Link from 'next/link';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ─── Types ────────────────────────────────────────────────────────────────────

export type AnswerMode = 'simple' | 'detailed' | 'exam' | 'viva';

interface ModeOption {
  id: AnswerMode;
  label: string;
  short: string;
  description: string;
  icon: React.ElementType;
}

export type SourceRef = {
  source: string;
  page: number;
  similarity: number;
  chunk_id: string;
  excerpt: string;
};

type Message = {
  id: string;
  role: 'user' | 'ai';
  content: string;
  mode?: AnswerMode;
  sources?: SourceRef[];
  found?: boolean;
  followups?: string[]; // suggested follow-up questions from Gemini
};

type ChatSession = {
  id: string;
  title: string;
  messages: Message[];
};

// ─── Constants ────────────────────────────────────────────────────────────────

const ANSWER_MODES: ModeOption[] = [
  {
    id: 'simple',
    label: 'Simple',
    short: 'Beginner-friendly',
    description: 'Plain language with everyday analogies',
    icon: Sparkles,
  },
  {
    id: 'detailed',
    label: 'Detailed',
    short: 'In-depth',
    description: 'Step-by-step with examples',
    icon: BookOpen,
  },
  {
    id: 'exam',
    label: 'Exam',
    short: 'Exam format',
    description: 'Structured for university answers',
    icon: GraduationCap,
  },
  {
    id: 'viva',
    label: 'Viva',
    short: 'Oral Q&A',
    description: 'Core concept + 3 viva questions',
    icon: HelpCircle,
  },
];

const COURSE_TOPICS = [
  'Artificial Intelligence',
  'Deep Learning',
  'Natural Language Processing',
  'Generative AI',
  'Reinforcement Learning',
];

const SUGGESTED_PROMPTS: { text: string; mode: AnswerMode }[] = [
  { text: 'Explain CNN in simple words', mode: 'simple' },
  { text: 'Give me an exam answer for overfitting', mode: 'exam' },
  { text: 'Prepare me for an NLP viva', mode: 'viva' },
  { text: 'Explain reinforcement learning with an example', mode: 'detailed' },
];

// ─── Source Panel ─────────────────────────────────────────────────────────────

function SourcePanel({ sources, found }: { sources: SourceRef[]; found: boolean }) {
  const [expanded, setExpanded] = useState(false);

  if (!found || sources.length === 0) {
    return (
      <div className="mt-4 flex items-start gap-2.5 px-3.5 py-2.5 rounded-lg border border-amber-200 bg-amber-50 text-amber-800">
        <AlertTriangle size={13} className="flex-shrink-0 mt-0.5 text-amber-500" />
        <p className="text-xs leading-relaxed">
          <span className="font-semibold">No course material matched</span> — this answer draws on general knowledge.
          Your indexed course documents did not contain a sufficiently relevant passage.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-4 rounded-lg border border-neutral-200 bg-neutral-50 overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 hover:bg-neutral-100 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <FileText size={12} className="text-neutral-500 flex-shrink-0" />
          <span className="text-[11px] font-semibold text-neutral-600 uppercase tracking-wide">
            Based on your course material
          </span>
          <span className="text-[11px] text-neutral-400">
            · {sources.length} source{sources.length !== 1 ? 's' : ''}
          </span>
        </div>
        {expanded
          ? <ChevronUp size={12} className="text-neutral-400 flex-shrink-0" />
          : <ChevronDown size={12} className="text-neutral-400 flex-shrink-0" />
        }
      </button>

      <div className="px-3.5 pb-3 space-y-2">
        {sources.map((src, i) => (
          <div key={src.chunk_id} className="flex items-start gap-2">
            <span className="flex-shrink-0 mt-0.5 w-4 h-4 rounded-full bg-neutral-200 text-neutral-600 text-[9px] font-bold flex items-center justify-center">
              {i + 1}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-1.5 flex-wrap">
                <span className="text-[12px] font-medium text-neutral-800 break-all leading-snug">
                  {src.source}
                </span>
                <span className="text-[11px] text-neutral-500 whitespace-nowrap">
                  — p.{src.page}
                </span>
              </div>
              {expanded && src.excerpt && (
                <p className="mt-1 text-[11px] text-neutral-500 leading-relaxed italic border-l-2 border-neutral-300 pl-2">
                  "{src.excerpt}{src.excerpt.length >= 150 ? '…' : ''}"
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── AskAI Logo Mark ──────────────────────────────────────────────────────────

function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <div
      style={{ width: size, height: size }}
      className="rounded-lg bg-neutral-900 flex items-center justify-center flex-shrink-0"
    >
      <span
        style={{ fontSize: size * 0.38 }}
        className="text-white font-bold tracking-tight select-none"
      >
        A
      </span>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedMode, setSelectedMode] = useState<AnswerMode>('detailed');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const isBusy = isLoading || isStreaming;

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, isStreaming]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 180) + 'px';
    }
  }, [input]);

  // Close mobile sidebar on outside click
  useEffect(() => {
    if (!sidebarOpen) return;
    const handler = (e: MouseEvent) => {
      const sidebar = document.getElementById('askai-sidebar');
      if (sidebar && !sidebar.contains(e.target as Node)) setSidebarOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [sidebarOpen]);

  const saveCurrentSession = useCallback(() => {
    if (messages.length === 0) return;
    const firstUser = messages.find(m => m.role === 'user');
    const title = firstUser
      ? firstUser.content.slice(0, 52) + (firstUser.content.length > 52 ? '…' : '')
      : 'Conversation';
    setSessions(prev => [
      { id: Date.now().toString(), title, messages },
      ...prev.slice(0, 19), // keep last 20
    ]);
  }, [messages]);

  const startNewChat = () => {
    abortControllerRef.current?.abort();
    saveCurrentSession();
    setMessages([]);
    setError(null);
    setInput('');
    setIsLoading(false);
    setIsStreaming(false);
    setSidebarOpen(false);
    setTimeout(() => inputRef.current?.focus(), 80);
  };

  const loadSession = (session: ChatSession) => {
    abortControllerRef.current?.abort();
    saveCurrentSession();
    setMessages(session.messages);
    setSessions(prev => prev.filter(s => s.id !== session.id));
    setError(null);
    setInput('');
    setIsLoading(false);
    setIsStreaming(false);
    setSidebarOpen(false);
  };

  const currentModeConfig = ANSWER_MODES.find(m => m.id === selectedMode) ?? ANSWER_MODES[1];

  const sendMessage = async (overrideText?: string, overrideMode?: AnswerMode) => {
    const textToSend = (overrideText ?? input).trim();
    const modeToSend = overrideMode ?? selectedMode;

    if (!textToSend || isBusy) return;

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const currentHistory = messages.map(m => ({ role: m.role, content: m.content }));

    setInput('');
    setError(null);
    setIsLoading(true);
    setIsStreaming(false);

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: textToSend };
    const aiMsgId = (Date.now() + 1).toString();

    setMessages(prev => [...prev, userMsg]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: textToSend, history: currentHistory, mode: modeToSend }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error((data as { error?: string }).error ?? `Server error ${response.status}`);
      }
      if (!response.body) throw new Error('No response body received.');

      setMessages(prev => [...prev, { id: aiMsgId, role: 'ai', content: '', mode: modeToSend }]);
      setIsLoading(false);
      setIsStreaming(true);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let lineBuffer = '';

      const processLine = (rawLine: string) => {
        const trimmed = rawLine.trim();
        if (!trimmed) return;
        let parsed: { type: string; text?: string; sources?: SourceRef[]; found?: boolean; message?: string; followups?: string[] };
        try { parsed = JSON.parse(trimmed); } catch { return; }

        if (parsed.type === 'chunk' && parsed.text) {
          setMessages(prev =>
            prev.map(m => m.id === aiMsgId ? { ...m, content: m.content + parsed.text! } : m)
          );
        } else if (parsed.type === 'meta') {
          setMessages(prev =>
            prev.map(m =>
              m.id === aiMsgId
                ? {
                    ...m,
                    sources: parsed.sources ?? [],
                    found: parsed.found ?? false,
                    followups: parsed.followups ?? [],
                  }
                : m
            )
          );
        } else if (parsed.type === 'error') {
          setError(parsed.message ?? 'An error occurred.');
          setMessages(prev => prev.filter(m => m.id !== aiMsgId));
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        lineBuffer += decoder.decode(value, { stream: true });
        const idx = lineBuffer.lastIndexOf('\n');
        if (idx !== -1) {
          const lines = lineBuffer.slice(0, idx + 1).split('\n');
          lineBuffer = lineBuffer.slice(idx + 1);
          for (const l of lines) processLine(l);
        }
      }
      if (lineBuffer.trim()) processLine(lineBuffer);

    } catch (err: unknown) {
      if ((err as { name?: string }).name === 'AbortError') return;
      console.error('Chat error:', err);
      setError((err as Error).message ?? 'Something went wrong. Please try again.');
      setMessages(prev => {
        const placeholder = prev.find(m => m.id === aiMsgId);
        return placeholder && !placeholder.content ? prev.filter(m => m.id !== aiMsgId) : prev;
      });
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ─── Sidebar Content (reused for desktop + mobile) ─────────────────────────

  const SidebarContent = () => (
    <div className="flex flex-col h-full">

      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-4 border-b border-neutral-200">
        <LogoMark size={28} />
        <div>
          <p className="text-sm font-semibold text-neutral-900 leading-none">AskAI</p>
          <p className="text-[10px] text-neutral-400 mt-0.5 leading-none">Academic Assistant</p>
        </div>
      </div>

      {/* New Chat */}
      <div className="px-3 pt-3">
        <button
          onClick={startNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium text-neutral-700 bg-white border border-neutral-200 hover:bg-neutral-50 hover:border-neutral-300 transition-colors"
        >
          <Plus size={14} className="text-neutral-500" />
          New Chat
        </button>
      </div>

      {/* Recent Conversations */}
      <div className="px-3 pt-4 flex-1 overflow-y-auto min-h-0">
        <p className="text-[10px] font-semibold text-neutral-400 uppercase tracking-widest mb-1.5 px-1">
          Recent
        </p>
        {sessions.length === 0 ? (
          <p className="text-xs text-neutral-400 px-1 py-1">No recent chats yet</p>
        ) : (
          <ul className="space-y-0.5">
            {sessions.map(session => (
              <li key={session.id}>
                <button
                  onClick={() => loadSession(session)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left text-xs text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900 transition-colors group"
                >
                  <MessageSquare size={12} className="text-neutral-400 flex-shrink-0 group-hover:text-neutral-600" />
                  <span className="truncate">{session.title}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Response Mode */}
      <div className="px-3 pt-3 pb-2 border-t border-neutral-200">
        <p className="text-[10px] font-semibold text-neutral-400 uppercase tracking-widest mb-2 px-1">
          Response Mode
        </p>
        <ul className="space-y-0.5">
          {ANSWER_MODES.map(mode => {
            const Icon = mode.icon;
            const active = selectedMode === mode.id;
            return (
              <li key={mode.id}>
                <button
                  onClick={() => setSelectedMode(mode.id)}
                  className={cn(
                    'w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-left text-xs transition-colors',
                    active
                      ? 'bg-neutral-900 text-white'
                      : 'text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900'
                  )}
                >
                  <Icon size={12} className={active ? 'text-white' : 'text-neutral-400'} />
                  <span className="font-medium">{mode.label}</span>
                  {active && <Check size={10} className="ml-auto text-neutral-300" />}
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      {/* Course Topics */}
      <div className="px-3 pt-3 pb-4 border-t border-neutral-200">
        <p className="text-[10px] font-semibold text-neutral-400 uppercase tracking-widest mb-2 px-1">
          Course Topics
        </p>
        <ul className="space-y-0.5">
          {COURSE_TOPICS.map(topic => (
            <li key={topic}>
              <div className="flex items-center gap-2 px-2 py-1 rounded-md text-xs text-neutral-500">
                <BookMarked size={11} className="text-neutral-300 flex-shrink-0" />
                <span className="truncate">{topic}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );

  // ─── JSX ───────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen overflow-hidden bg-white text-neutral-900 antialiased">

      {/* ── Desktop Sidebar ─────────────────────────────────────────────── */}
      <aside className="hidden md:flex flex-col w-60 shrink-0 border-r border-neutral-200 bg-stone-50">
        <SidebarContent />
      </aside>

      {/* ── Mobile Sidebar Overlay ──────────────────────────────────────── */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-black/30" onClick={() => setSidebarOpen(false)} />
          <aside
            id="askai-sidebar"
            className="absolute left-0 top-0 bottom-0 w-64 bg-stone-50 border-r border-neutral-200 z-50 flex flex-col"
          >
            <div className="flex items-center justify-between px-4 py-3.5 border-b border-neutral-200">
              <div className="flex items-center gap-2">
                <LogoMark size={24} />
                <span className="text-sm font-semibold">AskAI</span>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1 rounded-md hover:bg-neutral-100 text-neutral-500"
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              <SidebarContent />
            </div>
          </aside>
        </div>
      )}

      {/* ── Main Column ─────────────────────────────────────────────────── */}
      <div className="flex flex-col flex-1 min-w-0">

        {/* Mobile top bar */}
        <header className="md:hidden flex items-center justify-between px-4 py-3 border-b border-neutral-200 bg-white">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-1.5 rounded-md hover:bg-neutral-100 text-neutral-600"
            aria-label="Open sidebar"
          >
            <Menu size={18} />
          </button>
          <div className="flex items-center gap-2">
            <LogoMark size={22} />
            <span className="text-sm font-semibold">AskAI</span>
          </div>
          <button
            onClick={startNewChat}
            className="p-1.5 rounded-md hover:bg-neutral-100 text-neutral-600"
            aria-label="New chat"
          >
            <Plus size={18} />
          </button>
        </header>

        {/* ── Scroll Area ─────────────────────────────────────────────── */}
        <main className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (

            /* ── Welcome Screen ─────────────────────────────────────── */
            <div className="flex flex-col items-center justify-center min-h-full px-6 py-16 text-center">
              <LogoMark size={44} />

              <h1 className="mt-5 text-2xl font-semibold text-neutral-900 tracking-tight">
                How can AskAI help you today?
              </h1>
              <p className="mt-2 text-sm text-neutral-500 max-w-sm leading-relaxed">
                Ask anything from your course. AskAI retrieves answers directly from your lecture materials.
              </p>

              {/* Suggested prompts */}
              <div className="mt-8 w-full max-w-lg grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {SUGGESTED_PROMPTS.map(({ text, mode }) => {
                  const modeConf = ANSWER_MODES.find(m => m.id === mode)!;
                  const Icon = modeConf.icon;
                  return (
                    <button
                      key={text}
                      onClick={() => {
                        setSelectedMode(mode);
                        sendMessage(text, mode);
                      }}
                      className="group flex items-start gap-3 p-3.5 rounded-xl border border-neutral-200 bg-white hover:border-neutral-300 hover:bg-neutral-50 text-left transition-colors"
                    >
                      <Icon size={15} className="text-neutral-400 flex-shrink-0 mt-0.5 group-hover:text-neutral-600" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-neutral-700 leading-snug">{text}</p>
                        <p className="text-[11px] text-neutral-400 mt-0.5">{modeConf.short}</p>
                      </div>
                      <ArrowRight size={13} className="text-neutral-300 flex-shrink-0 mt-0.5 group-hover:text-neutral-500 self-center" />
                    </button>
                  );
                })}
              </div>

              {/* Current mode indicator */}
              <p className="mt-6 text-xs text-neutral-400">
                Mode:{' '}
                <span className="font-medium text-neutral-600">{currentModeConfig.label}</span>
                {' '}— change it in the sidebar
              </p>
            </div>

          ) : (

            /* ── Message Thread ──────────────────────────────────────── */
            <div className="max-w-2xl mx-auto w-full px-4 py-8 space-y-8">
              {messages.map(msg => {
                const modeMeta = msg.mode ? ANSWER_MODES.find(m => m.id === msg.mode) : null;

                if (msg.role === 'user') {
                  return (
                    <div key={msg.id} className="flex justify-end">
                      <div className="max-w-[80%] px-4 py-2.5 rounded-2xl bg-neutral-100 text-neutral-800 text-sm leading-relaxed">
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    </div>
                  );
                }

                // AI message
                return (
                  <div key={msg.id} className="flex flex-col gap-1">
                    {/* Mode label */}
                    {modeMeta && (
                      <div className="flex items-center gap-1.5 mb-1">
                        <modeMeta.icon size={12} className="text-neutral-400" />
                        <span className="text-[11px] font-medium text-neutral-400 uppercase tracking-wide">
                          {modeMeta.label}
                        </span>
                      </div>
                    )}

                    {/* Answer text */}
                    {msg.content ? (
                      <div className={cn(
                        'prose prose-sm max-w-none text-neutral-800',
                        'prose-headings:font-semibold prose-headings:text-neutral-900 prose-headings:mt-4 prose-headings:mb-1',
                        'prose-p:leading-relaxed prose-p:my-1.5',
                        'prose-li:my-0.5',
                        'prose-strong:font-semibold prose-strong:text-neutral-900',
                        'prose-code:text-neutral-700 prose-code:bg-neutral-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-[13px] prose-code:font-mono prose-code:before:content-none prose-code:after:content-none',
                        'prose-pre:bg-neutral-900 prose-pre:text-neutral-100 prose-pre:rounded-xl prose-pre:p-4 prose-pre:text-[13px]',
                        'prose-blockquote:border-l-neutral-300 prose-blockquote:text-neutral-600',
                        'prose-hr:border-neutral-200',
                      )}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      /* Empty content while streaming starts */
                      <div className="h-5 w-5 border-2 border-neutral-200 border-t-neutral-600 rounded-full animate-spin" />
                    )}

                    {/* Source panel */}
                    {msg.sources !== undefined && (
                      <SourcePanel sources={msg.sources} found={msg.found ?? false} />
                    )}
                  </div>
                );
              })}

              {/* Thinking indicator (retrieval phase) */}
              {isLoading && (
                <div className="flex items-center gap-2.5 text-neutral-400">
                  <Loader2 size={14} className="animate-spin" />
                  <span className="text-xs">Searching course materials…</span>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="flex items-start gap-2.5 px-4 py-3 rounded-lg border border-red-200 bg-red-50 text-red-700 text-sm">
                  <AlertCircle size={15} className="flex-shrink-0 mt-0.5" />
                  <p className="leading-relaxed">{error}</p>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {/* ── Input Area ───────────────────────────────────────────────── */}
        <div className="shrink-0 border-t border-neutral-200 bg-white px-4 pt-3 pb-4">
          <div className="max-w-2xl mx-auto">

            {/* Active mode pill — quick switcher strip */}
            <div className="flex items-center gap-1 mb-2.5 flex-wrap">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider mr-1">Mode:</span>
              {ANSWER_MODES.map(mode => {
                const active = selectedMode === mode.id;
                return (
                  <button
                    key={mode.id}
                    onClick={() => setSelectedMode(mode.id)}
                    title={mode.description}
                    className={cn(
                      'px-2 py-0.5 rounded text-[11px] font-medium transition-colors',
                      active
                        ? 'bg-neutral-900 text-white'
                        : 'text-neutral-500 hover:text-neutral-800 hover:bg-neutral-100'
                    )}
                  >
                    {mode.label}
                  </button>
                );
              })}
            </div>

            {/* Textarea + Send */}
            <div className={cn(
              'flex items-end gap-2 rounded-xl border bg-white px-3 py-2.5 transition-colors',
              isBusy ? 'border-neutral-200' : 'border-neutral-300 focus-within:border-neutral-400'
            )}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  isLoading ? 'Searching course materials…' :
                  isStreaming ? 'Receiving response…' :
                  'Ask a question about your course…'
                }
                disabled={isBusy}
                rows={1}
                className="flex-1 resize-none bg-transparent text-sm text-neutral-900 placeholder:text-neutral-400 focus:outline-none leading-relaxed max-h-44 disabled:opacity-50"
              />
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim() || isBusy}
                aria-label="Send"
                className={cn(
                  'flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-lg transition-colors',
                  input.trim() && !isBusy
                    ? 'bg-neutral-900 text-white hover:bg-neutral-700'
                    : 'bg-neutral-100 text-neutral-300 cursor-not-allowed'
                )}
              >
                {isStreaming
                  ? <Loader2 size={14} className="animate-spin" />
                  : <Send size={14} />
                }
              </button>
            </div>

            <p className="mt-1.5 text-[10px] text-neutral-400 text-center">
              Enter to send · Shift+Enter for new line · answers grounded in your course materials
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
