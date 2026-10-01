import { NextResponse } from 'next/server';
import { retrieveRelevantChunks } from '@/lib/retrieval';
import { streamGroundedAnswer, generateGroundedAnswer, type AnswerMode } from '@/lib/gemini';

// Default RAG retrieval parameters
const TOP_K = 3;
const SIMILARITY_THRESHOLD = 0.55;

const VALID_MODES: AnswerMode[] = ['simple', 'detailed', 'exam', 'viva'];

/**
 * Streaming NDJSON Chat API
 *
 * Protocol: newline-delimited JSON over text/plain.
 * Each line is one of:
 *   {"type":"chunk","text":"..."}       — a text delta from the Gemini stream
 *   {"type":"meta","sources":[...],"found":bool,"mode":"..."} — sent once, after the stream ends
 *   {"type":"error","message":"..."}    — fatal error (stream ends immediately after)
 *
 * The client reads lines, accumulates chunk.text deltas into the displayed answer,
 * then uses the meta object to render the source panel.
 *
 * Fallback: If streaming itself fails (e.g. the SDK stream errors before any token),
 * the route falls back to a single non-streaming generateGroundedAnswer call and
 * emits the full answer as a single chunk followed by meta — so the client still works.
 */
export async function POST(request: Request) {
  const encoder = new TextEncoder();

  // Helper: encode one NDJSON line
  const line = (obj: object) => encoder.encode(JSON.stringify(obj) + '\n');

  let body: { message?: string; history?: unknown[]; mode?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body.' }, { status: 400 });
  }

  const { message, history, mode } = body;

  if (!message || typeof message !== 'string' || !message.trim()) {
    return NextResponse.json({ error: 'Message is required.' }, { status: 400 });
  }

  const question = message.trim();
  const answerMode: AnswerMode = mode && VALID_MODES.includes(mode as AnswerMode)
    ? (mode as AnswerMode)
    : 'detailed';
  const safeHistory = Array.isArray(history)
    ? (history as { role: string; content: string }[])
    : [];

  // ── Step 1 & 2: Embed + Retrieve (happens before streaming starts) ─────────
  let retrievalResult: {
    found: boolean;
    chunks: { source: string; page: number; similarity: number; chunk_id: string; text: string }[];
    topScore: number;
  };

  try {
    retrievalResult = await retrieveRelevantChunks(question, TOP_K, SIMILARITY_THRESHOLD);
  } catch (retrievalErr: any) {
    console.warn('Retrieval warning (proceeding with empty context):', retrievalErr.message);
    retrievalResult = { found: false, chunks: [], topScore: 0 };
  }

  const sources = retrievalResult.chunks.map(chunk => ({
    source: chunk.source,
    page: chunk.page,
    similarity: chunk.similarity,
    chunk_id: chunk.chunk_id,
    excerpt: chunk.text.replace(/\s+/g, ' ').trim().slice(0, 150),
  }));

  const metaPayload = {
    type: 'meta' as const,
    sources,
    found: retrievalResult.found,
    topScore: retrievalResult.topScore,
    mode: answerMode,
  };

  // ── Step 3: Stream Gemini generation ──────────────────────────────────────
  const stream = new ReadableStream({
    async start(controller) {
      try {
        // Try streaming first
        const generator = streamGroundedAnswer(
          question,
          retrievalResult.chunks,
          safeHistory,
          answerMode
        );

        for await (const textDelta of generator) {
          controller.enqueue(line({ type: 'chunk', text: textDelta }));
        }
      } catch (streamErr: any) {
        // Stream failed — try non-streaming fallback
        console.warn('Streaming failed, falling back to non-streaming:', streamErr.message);
        try {
          const fullAnswer = await generateGroundedAnswer(
            question,
            retrievalResult.chunks,
            safeHistory,
            answerMode
          );
          controller.enqueue(line({ type: 'chunk', text: fullAnswer }));
        } catch (fallbackErr: any) {
          console.error('Both streaming and fallback generation failed:', fallbackErr.message);
          controller.enqueue(
            line({ type: 'error', message: 'The AI service is temporarily unavailable. Please try again.' })
          );
          controller.enqueue(line(metaPayload));
          controller.close();
          return;
        }
      }

      // Always emit the meta line last so the client knows sources
      controller.enqueue(line(metaPayload));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      // text/plain so Next.js does not try to JSON-parse the body
      'Content-Type': 'text/plain; charset=utf-8',
      // Prevent buffering by proxies / nginx
      'X-Content-Type-Options': 'nosniff',
      'Cache-Control': 'no-cache, no-store',
    },
  });
}
