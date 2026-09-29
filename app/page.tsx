'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, Plus, Bot, User, AlertCircle, Loader2 } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

type Message = {
  id: string;
  role: 'user' | 'ai';
  content: string;
};

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const startNewChat = () => {
    setMessages([]);
    setError(null);
    setInput('');
    setTimeout(() => inputRef.current?.focus(), 100);
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userText = input.trim();
    // Capture the current conversation history to send to the backend
    const currentHistory = messages.map(m => ({ role: m.role, content: m.content }));

    setInput('');
    setError(null);
    setIsLoading(true);

    const newUserMsg: Message = { id: Date.now().toString(), role: 'user', content: userText };
    setMessages((prev) => [...prev, newUserMsg]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Send both the new message and the conversation history
        body: JSON.stringify({ message: userText, history: currentHistory }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch response');
      }

      const aiMsg: Message = { id: (Date.now() + 1).toString(), role: 'ai', content: data.answer };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err: any) {
      console.error('Chat error:', err);
      setError(err.message || 'An error occurred while communicating with the AI.');
    } finally {
      setIsLoading(false);
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
      <header className="sticky top-0 z-10 flex items-center justify-between px-4 py-3 bg-white/80 backdrop-blur-md border-b border-gray-200/60 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-blue-600 text-white shadow-sm">
            <Bot size={22} />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900 tracking-tight leading-tight">AskAI</h1>
            <p className="text-xs font-medium text-gray-500">AI-Powered Student Academic Assistant</p>
          </div>
        </div>
        <button
          onClick={startNewChat}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors border border-gray-200/50 shadow-sm"
        >
          <Plus size={16} />
          <span className="hidden sm:inline">New Chat</span>
        </button>
      </header>

      {/* Main Conversation Area */}
      <main className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-3xl mx-auto space-y-8 pb-12">
          
          {messages.length === 0 ? (
            /* Empty State / Welcome Screen */
            <div className="flex flex-col items-center justify-center min-h-[50vh] text-center px-4 animate-in fade-in duration-500">
              <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mb-6 shadow-sm">
                <Bot size={32} />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-3">Welcome to AskAI</h2>
              <p className="text-gray-600 max-w-lg leading-relaxed">
                I am your dedicated academic assistant. You can ask me questions about <strong className="text-gray-900 font-semibold">Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, Reinforcement Learning,</strong> and related topics to help you with your coursework.
              </p>
              <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                <button 
                  onClick={() => setInput('Explain the difference between supervised and unsupervised learning.')}
                  className="p-3 text-sm text-left text-gray-700 bg-white border border-gray-200 rounded-xl hover:border-blue-300 hover:bg-blue-50/50 transition-all shadow-sm"
                >
                  "Explain the difference between supervised and unsupervised learning."
                </button>
                <button 
                  onClick={() => setInput('What is the vanishing gradient problem in Deep Learning?')}
                  className="p-3 text-sm text-left text-gray-700 bg-white border border-gray-200 rounded-xl hover:border-blue-300 hover:bg-blue-50/50 transition-all shadow-sm"
                >
                  "What is the vanishing gradient problem in Deep Learning?"
                </button>
              </div>
            </div>
          ) : (
            /* Message List */
            <div className="space-y-6">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex gap-4",
                    msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                  )}
                >
                  <div className={cn(
                    "flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full shadow-sm mt-1",
                    msg.role === 'user' ? "bg-gray-800 text-white" : "bg-blue-600 text-white"
                  )}>
                    {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  
                  <div className={cn(
                    "max-w-[85%] sm:max-w-[75%] rounded-2xl px-5 py-3.5 shadow-sm",
                    msg.role === 'user' 
                      ? "bg-gray-800 text-white rounded-tr-none" 
                      : "bg-white border border-gray-200 text-gray-800 rounded-tl-none"
                  )}>
                    {msg.role === 'user' ? (
                      <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{msg.content}</p>
                    ) : (
                      <div className="prose prose-sm sm:prose-base prose-blue max-w-none break-words
                          prose-pre:bg-gray-50 prose-pre:text-gray-800 prose-pre:border prose-pre:border-gray-200
                          prose-code:text-blue-600 prose-code:bg-blue-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:before:content-none prose-code:after:content-none
                          prose-p:leading-relaxed"
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {/* Loading State */}
              {isLoading && (
                <div className="flex gap-4 flex-row animate-in fade-in duration-300">
                  <div className="flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-full shadow-sm mt-1 bg-blue-600 text-white">
                    <Loader2 size={16} className="animate-spin" />
                  </div>
                  <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-none px-5 py-4 shadow-sm flex items-center gap-1.5 h-[52px]">
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
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
      <footer className="bg-white border-t border-gray-200 p-4 sm:p-6 shadow-[0_-10px_40px_rgba(0,0,0,0.04)]">
        <div className="max-w-3xl mx-auto relative">
          <div className="relative flex items-end gap-2 bg-gray-50 border border-gray-200 focus-within:border-blue-400 focus-within:ring-4 focus-within:ring-blue-500/10 rounded-2xl transition-all shadow-sm">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your course..."
              disabled={isLoading}
              rows={1}
              className="w-full max-h-48 py-3.5 pl-4 pr-12 bg-transparent border-none focus:outline-none focus:ring-0 resize-none text-[15px] leading-relaxed disabled:opacity-50"
            />
            <div className="absolute right-2 bottom-2">
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="flex items-center justify-center w-9 h-9 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 transition-colors shadow-sm"
                aria-label="Send message"
              >
                <Send size={16} className={input.trim() && !isLoading ? "translate-x-0.5 -translate-y-0.5" : ""} />
              </button>
            </div>
          </div>
          <div className="text-center mt-3">
            <p className="text-[11px] text-gray-400">
              AskAI may produce inaccurate information about people, places, or facts. Always verify with your course materials.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
