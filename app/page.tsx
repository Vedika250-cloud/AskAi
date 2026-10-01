'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Send, 
  Plus, 
  Bot, 
  User, 
  AlertCircle, 
  Loader2, 
  Sparkles, 
  BookOpen, 
  GraduationCap, 
  HelpCircle,
  Check,
  FileText,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type AnswerMode = 'simple' | 'detailed' | 'exam' | 'viva';

interface ModeOption {
  id: AnswerMode;
  label: string;
  badge: string;
  description: string;
  examplePrompt: string;
  icon: React.ElementType;
}

const ANSWER_MODES: ModeOption[] = [
  {
    id: 'simple',
    label: 'Simple Explanation',
    badge: 'Beginner-Friendly',
    description: 'Plain language with an intuitive, everyday example',
    examplePrompt: 'Explain what overfitting is in simple terms with an everyday analogy.',
    icon: Sparkles,
  },
  {
    id: 'detailed',
    label: 'Detailed Explanation',
    badge: 'In-Depth',
    description: 'Step-by-step walkthrough covering principles and mechanisms',
    examplePrompt: 'Explain gradient descent and backpropagation step-by-step.',
    icon: BookOpen,
  },
  {
    id: 'exam',
    label: 'Exam Answer',
    badge: 'University Format',
    description: 'Concise, structured points optimized for college exams',
    examplePrompt: 'What is overfitting? Give a structured 5-mark answer for exams.',
    icon: GraduationCap,
  },
  {
    id: 'viva',
    label: 'Viva Preparation',
    badge: 'Oral Exam Q&A',
    description: 'Core spoken summary + 3 likely viva questions with model answers',
    examplePrompt: 'Viva preparation for Artificial Intelligence and the Turing test.',
    icon: HelpCircle,
  },
];

// Source reference exactly as returned by the retrieval system
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
  // Retrieval-system sources — undefined until response arrives
  sources?: SourceRef[];
  // Whether retrieval found relevant course material
  found?: boolean;
};

// ─── Source Panel Component ────────────────────────────────────────────────────
// Renders the retrieval-system sources for one AI response.
// Sources come exclusively from the retrieval layer — never invented by Gemini.

function SourcePanel({
  sources,
  found,
}: {
  sources: SourceRef[];
  found: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  // "Not found" case — course material was searched but nothing was relevant
  if (!found || sources.length === 0) {
    return (
      <div className="mt-3 flex items-start gap-2 px-3.5 py-2.5 rounded-xl border border-amber-200 bg-amber-50/60 text-amber-800">
        <AlertTriangle size={14} className="flex-shrink-0 mt-0.5 text-amber-500" />
        <p className="text-xs leading-relaxed">
          <span className="font-semibold">No course material found</span> — this answer is based on general knowledge. 
          Your indexed course documents did not contain a sufficiently relevant match for this question.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-3 rounded-xl border border-emerald-200 bg-emerald-50/50 overflow-hidden">
      {/* Header row */}
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2 hover:bg-emerald-100/60 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <FileText size={13} className="text-emerald-600 flex-shrink-0" />
          <span className="text-[11px] font-bold text-emerald-800 uppercase tracking-wide">
            Based on your course material
          </span>
          <span className="text-[11px] text-emerald-600 font-normal">
            · {sources.length} source{sources.length !== 1 ? 's' : ''} retrieved
          </span>
        </div>
        {expanded
          ? <ChevronUp size={13} className="text-emerald-600 flex-shrink-0" />
          : <ChevronDown size={13} className="text-emerald-600 flex-shrink-0" />
        }
      </button>

      {/* Always-visible compact source list */}
      <div className="px-3.5 pb-2.5 space-y-1.5">
        {sources.map((src, i) => (
          <div key={src.chunk_id} className="flex items-start gap-2">
            <span className="flex-shrink-0 mt-0.5 w-4 h-4 rounded-full bg-emerald-200 text-emerald-800 text-[9px] font-bold flex items-center justify-center">
              {i + 1}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-1.5 flex-wrap">
                <span className="text-[12px] font-semibold text-gray-800 break-all leading-snug">
                  {src.source}
                </span>
                <span className="text-[11px] text-emerald-700 font-medium whitespace-nowrap">
                  — Page {src.page}
                </span>
              </div>

              {/* Excerpt — shown when panel is expanded */}
              {expanded && src.excerpt && (
                <p className="mt-1 text-[11px] text-gray-500 leading-relaxed italic border-l-2 border-emerald-300 pl-2">
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

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedMode, setSelectedMode] = useState<AnswerMode>('detailed');
  const [isLoading, setIsLoading] = useState(false);    // true = waiting for first token
  const [isStreaming, setIsStreaming] = useState(false); // true = tokens arriving
  const [error, setError] = useState<string | null>(null);

  // Ref to abort the active fetch when user starts a new question or unmounts
  const abortControllerRef = useRef<AbortController | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, isStreaming]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const startNewChat = () => {
    // Cancel any in-progress stream
    abortControllerRef.current?.abort();
    setMessages([]);
    setError(null);
    setInput('');
    setIsLoading(false);
    setIsStreaming(false);
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const currentModeConfig = ANSWER_MODES.find(m => m.id === selectedMode) || ANSWER_MODES[1];

  const sendMessage = async (overrideText?: string, overrideMode?: AnswerMode) => {
    const textToSend = (overrideText || input).trim();
    const modeToSend = overrideMode || selectedMode;

    if (!textToSend || isLoading || isStreaming) return;

    // Cancel any previous in-flight request
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Capture the current conversation history to send to backend
    const currentHistory = messages.map(m => ({ role: m.role, content: m.content }));

    setInput('');
    setError(null);
    setIsLoading(true);
    setIsStreaming(false);

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: textToSend,
    };

    // Unique ID for the AI placeholder — we update it in place as chunks arrive
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
        // Non-streaming error path (e.g. 400/500 before the stream starts)
        const data = await response.json().catch(() => ({}));
        throw new Error((data as any).error || `Server error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('No response body received.');
      }

      // ── Insert empty AI placeholder so it appears immediately ──────────────
      setMessages(prev => [
        ...prev,
        { id: aiMsgId, role: 'ai', content: '', mode: modeToSend },
      ]);
      setIsLoading(false);
      setIsStreaming(true);

      // ── Read NDJSON lines from the stream ──────────────────────────────────
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let lineBuffer = '';

      const processLine = (rawLine: string) => {
        const trimmed = rawLine.trim();
        if (!trimmed) return;

        let parsed: { type: string; text?: string; sources?: SourceRef[]; found?: boolean; mode?: string; message?: string };
        try {
          parsed = JSON.parse(trimmed);
        } catch {
          // Malformed line — ignore
          return;
        }

        if (parsed.type === 'chunk' && parsed.text) {
          // Append delta to the placeholder message
          setMessages(prev =>
            prev.map(m =>
              m.id === aiMsgId ? { ...m, content: m.content + parsed.text! } : m
            )
          );
        } else if (parsed.type === 'meta') {
          // Apply sources once generation is complete
          setMessages(prev =>
            prev.map(m =>
              m.id === aiMsgId
                ? { ...m, sources: parsed.sources ?? [], found: parsed.found ?? false }
                : m
            )
          );
        } else if (parsed.type === 'error') {
          setError(parsed.message || 'An AI error occurred.');
          // Remove the empty placeholder
          setMessages(prev => prev.filter(m => m.id !== aiMsgId));
        }
      };

      while (true) {
        const { value, done } = await reader.read();

        if (done) break;

        lineBuffer += decoder.decode(value, { stream: true });

        // Process all complete lines in the buffer
        const newlineIdx = lineBuffer.lastIndexOf('\n');
        if (newlineIdx !== -1) {
          const completeLines = lineBuffer.slice(0, newlineIdx + 1).split('\n');
          lineBuffer = lineBuffer.slice(newlineIdx + 1);
          for (const l of completeLines) processLine(l);
        }
      }

      // Flush any remaining partial line
      if (lineBuffer.trim()) processLine(lineBuffer);

    } catch (err: any) {
      if (err.name === 'AbortError') {
        // User cancelled — leave whatever text was streamed in place
        return;
      }
      console.error('Chat error:', err);
      setError(err.message || 'An error occurred while communicating with the AI.');
      // Clean up empty placeholder if nothing was streamed
      setMessages(prev => {
        const placeholder = prev.find(m => m.id === aiMsgId);
        if (placeholder && !placeholder.content) {
          return prev.filter(m => m.id !== aiMsgId);
        }
        return prev;
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

  return (
    <div className="flex flex-col h-screen bg-gray-50/50 text-gray-900 font-sans selection:bg-blue-100 selection:text-blue-900">
      
      {/* Header */}
      <header className="sticky top-0 z-10 flex items-center justify-between px-4 sm:px-6 py-3 bg-white/80 backdrop-blur-md border-b border-gray-200/60 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-blue-600 text-white shadow-sm">
            <Bot size={22} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-gray-900 tracking-tight leading-tight">AskAI</h1>
              <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-blue-50 text-blue-700 border border-blue-200/60">
                Grounded RAG
              </span>
            </div>
            <p className="text-xs font-medium text-gray-500">AI-Powered Academic Assistant for Students</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Active Mode indicator in header */}
          <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-100/80 border border-gray-200 text-xs text-gray-700">
            <span className="text-gray-400 font-normal">Active Style:</span>
            <currentModeConfig.icon size={13} className="text-blue-600" />
            <span className="font-semibold text-gray-800">{currentModeConfig.label}</span>
          </div>

          <button
            onClick={startNewChat}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors border border-gray-200/50 shadow-sm"
          >
            <Plus size={16} />
            <span className="hidden sm:inline">New Chat</span>
          </button>
        </div>
      </header>

      {/* Main Conversation Area */}
      <main className="flex-1 overflow-y-auto px-4 py-6 sm:py-8">
        <div className="max-w-3xl mx-auto space-y-6 pb-12">
          
          {messages.length === 0 ? (
            /* Empty State / Welcome Screen */
            <div className="flex flex-col items-center justify-center min-h-[50vh] text-center px-4 animate-in fade-in duration-500">
              <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mb-5 shadow-sm">
                <Bot size={32} />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to AskAI</h2>
              <p className="text-gray-600 max-w-xl leading-relaxed text-sm sm:text-base">
                Your course assistant grounded directly in your course lecture materials. Choose an <strong>Answer Style</strong> below to customize how AskAI explains each concept to you.
              </p>

              {/* Mode Selection Grid on Welcome Screen */}
              <div className="mt-6 w-full max-w-2xl">
                <div className="text-left mb-2.5">
                  <span className="text-xs font-bold uppercase tracking-wider text-gray-400">Select Answer Style</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {ANSWER_MODES.map((mode) => {
                    const IconComponent = mode.icon;
                    const isSelected = selectedMode === mode.id;
                    return (
                      <button
                        key={mode.id}
                        type="button"
                        onClick={() => setSelectedMode(mode.id)}
                        className={cn(
                          "flex items-start gap-3 p-3.5 rounded-xl border text-left transition-all",
                          isSelected
                            ? "bg-blue-50/70 border-blue-400 ring-2 ring-blue-500/20 shadow-sm"
                            : "bg-white border-gray-200 hover:border-gray-300 hover:bg-gray-50/60 shadow-xs"
                        )}
                      >
                        <div className={cn(
                          "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5",
                          isSelected ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-600"
                        )}>
                          <IconComponent size={16} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-1">
                            <span className={cn("text-sm font-semibold", isSelected ? "text-blue-950" : "text-gray-900")}>
                              {mode.label}
                            </span>
                            {isSelected && <Check size={14} className="text-blue-600 flex-shrink-0" />}
                          </div>
                          <p className="text-xs text-gray-500 mt-0.5 line-clamp-2 leading-relaxed">
                            {mode.description}
                          </p>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Quick Prompt Cards */}
              <div className="mt-6 w-full max-w-2xl">
                <div className="text-left mb-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-gray-400">Try an example question</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {ANSWER_MODES.map((mode) => (
                    <button 
                      key={mode.id}
                      onClick={() => {
                        setSelectedMode(mode.id);
                        sendMessage(mode.examplePrompt, mode.id);
                      }}
                      className="p-2.5 text-xs text-left text-gray-700 bg-white border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50/40 transition-all shadow-xs flex items-center justify-between gap-2"
                    >
                      <span className="truncate">"{mode.examplePrompt}"</span>
                      <span className="text-[10px] text-gray-400 font-medium px-1.5 py-0.5 bg-gray-100 rounded flex-shrink-0">
                        {mode.label.split(' ')[0]}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

            </div>
          ) : (
            /* Message List */
            <div className="space-y-6">
              {messages.map((msg) => {
                const modeMeta = msg.mode ? ANSWER_MODES.find(m => m.id === msg.mode) : null;
                const ModeIcon = modeMeta?.icon;

                  return (
                  <div
                    key={msg.id}
                    className={cn(
                      "flex gap-3 sm:gap-4",
                      msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                    )}
                  >
                    <div className={cn(
                      "flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full shadow-sm mt-1",
                      msg.role === 'user' ? "bg-gray-800 text-white" : "bg-blue-600 text-white"
                    )}>
                      {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                    </div>

                    {/* Bubble + Source Panel stacked vertically for AI */}
                    <div className={cn(
                      "flex flex-col min-w-0",
                      msg.role === 'user' ? "max-w-[90%] sm:max-w-[80%]" : "max-w-[90%] sm:max-w-[80%]"
                    )}>
                      {/* The answer bubble */}
                      <div className={cn(
                        "rounded-2xl px-5 py-4 shadow-sm",
                        msg.role === 'user'
                          ? "bg-gray-800 text-white rounded-tr-none"
                          : "bg-white border border-gray-200 text-gray-800 rounded-tl-none"
                      )}>
                        {/* Mode Badge on AI responses */}
                        {msg.role === 'ai' && modeMeta && (
                          <div className="flex items-center gap-1.5 mb-2.5 pb-2 border-b border-gray-100">
                            {ModeIcon && <ModeIcon size={13} className="text-blue-600" />}
                            <span className="text-[11px] font-semibold text-gray-700">
                              {modeMeta.label}
                            </span>
                            <span className="text-[10px] text-gray-400">·</span>
                            <span className="text-[10px] text-gray-500 font-normal">
                              {modeMeta.badge}
                            </span>
                          </div>
                        )}

                        {msg.role === 'user' ? (
                          <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{msg.content}</p>
                        ) : (
                          <div className="prose prose-sm sm:prose-base prose-blue max-w-none break-words
                              prose-pre:bg-gray-50 prose-pre:text-gray-800 prose-pre:border prose-pre:border-gray-200
                              prose-code:text-blue-600 prose-code:bg-blue-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:before:content-none prose-code:after:content-none
                              prose-p:leading-relaxed prose-headings:font-bold prose-headings:text-gray-900"
                          >
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {msg.content}
                            </ReactMarkdown>
                          </div>
                        )}
                      </div>

                      {/* Source Panel — shown below every AI answer, visually separate */}
                      {msg.role === 'ai' && msg.sources !== undefined && (
                        <SourcePanel
                          sources={msg.sources}
                          found={msg.found ?? false}
                        />
                      )}
                    </div>
                  </div>
                );
              })}
              
              {/* Thinking indicator — shown while retrieval + first token are pending */}
              {isLoading && (
                <div className="flex gap-3 sm:gap-4 flex-row animate-in fade-in duration-300">
                  <div className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full shadow-sm mt-1 bg-blue-600 text-white">
                    <Loader2 size={16} className="animate-spin" />
                  </div>
                  <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-none px-5 py-4 shadow-sm flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-xs text-gray-500 font-medium">
                      Searching course materials · preparing {currentModeConfig.label.toLowerCase()}…
                    </span>
                  </div>
                </div>
              )}

              
              {/* Error State */}
              {error && (
                <div className="flex justify-center my-4 animate-in fade-in slide-in-from-bottom-2">
                  <div className="flex items-center gap-2 bg-red-50 text-red-700 px-4 py-3 rounded-xl border border-red-100 text-sm max-w-lg shadow-sm">
                    <AlertCircle size={18} className="flex-shrink-0" />
                    <p>{error}</p>
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="bg-white border-t border-gray-200 p-3 sm:p-5 shadow-[0_-10px_40px_rgba(0,0,0,0.04)]">
        <div className="max-w-3xl mx-auto space-y-2.5">
          
          {/* Answer Mode Selector Pills */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 max-w-full">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mr-1 hidden sm:inline">
                Style:
              </span>
              {ANSWER_MODES.map((mode) => {
                const IconComp = mode.icon;
                const isActive = selectedMode === mode.id;
                return (
                  <button
                    key={mode.id}
                    type="button"
                    onClick={() => setSelectedMode(mode.id)}
                    title={mode.description}
                    className={cn(
                      "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all cursor-pointer whitespace-nowrap",
                      isActive
                        ? "bg-blue-600 text-white shadow-xs"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200/80 hover:text-gray-900 border border-gray-200/60"
                    )}
                  >
                    <IconComp size={13} className={isActive ? "text-white" : "text-gray-500"} />
                    <span>{mode.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Active Mode Summary helper */}
            <span className="text-[11px] text-gray-500 hidden lg:inline-block italic">
              {currentModeConfig.description}
            </span>
          </div>

          {/* Text Input Box */}
          <div className="relative flex items-end gap-2 bg-gray-50 border border-gray-200 focus-within:border-blue-400 focus-within:ring-4 focus-within:ring-blue-500/10 rounded-2xl transition-all shadow-sm">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isLoading ? 'Searching course materials…' :
                isStreaming ? 'Receiving response…' :
                `Ask a course question in "${currentModeConfig.label}" style…`
              }
              disabled={isLoading || isStreaming}
              rows={1}
              className="w-full max-h-48 py-3.5 pl-4 pr-12 bg-transparent border-none focus:outline-none focus:ring-0 resize-none text-[15px] leading-relaxed disabled:opacity-50"
            />
            <div className="absolute right-2 bottom-2">
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim() || isLoading || isStreaming}
                className="flex items-center justify-center w-9 h-9 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 transition-colors shadow-sm cursor-pointer disabled:cursor-not-allowed"
                aria-label="Send message"
              >
                {isStreaming
                  ? <Loader2 size={16} className="animate-spin" />
                  : <Send size={16} className={input.trim() && !isLoading ? 'translate-x-0.5 -translate-y-0.5' : ''} />
                }
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between text-[11px] text-gray-400 px-1">
            <span>AskAI retrieves from course materials &amp; answers using prompt-engineered study styles.</span>
            <span className="hidden sm:inline">Press Enter to send, Shift+Enter for new line</span>
          </div>

        </div>
      </footer>
    </div>
  );
}
