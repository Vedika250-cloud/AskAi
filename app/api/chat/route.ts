import { NextResponse } from 'next/server';
import { retrieveRelevantChunks } from '@/lib/retrieval';
import {
  streamGroundedAnswer,
  generateGroundedAnswer,
  generateFollowups,
  type AnswerMode,
} from '@/lib/gemini';

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
 *   {"type":"meta","sources":[...],"found":bool,"mode":"...","followups":["Q1","Q2","Q3"]}
 *   {"type":"error","message":"..."}    — fatal error (stream ends after this)
 *
 * Follow-up questions are generated concurrently with the main answer stream
 * using a small separate Gemini call, adding zero perceived latency.
 * They are included in the meta line and rendered as clickable chips by the client.
 */
export async function POST(request: Request) {
  const encoder = new TextEncoder();
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

  // ── Step 1 & 2: Embed + Retrieve ───────────────────────────────────────────
  let retrievalResult: {
    found: boolean;
    chunks: { source: string; page: number; similarity: number; chunk_id: string; text: string }[];
    topScore: number;
  };

  try {
    retrievalResult = await retrieveRelevantChunks(question, TOP_K, SIMILARITY_THRESHOLD);
  } catch (retrievalErr: unknown) {
    console.warn('Retrieval warning (proceeding with empty context):', (retrievalErr as Error).message);
    retrievalResult = { found: false, chunks: [], topScore: 0 };
  }

  const sources = retrievalResult.chunks.map(chunk => ({
    source: chunk.source,
    page: chunk.page,
    similarity: chunk.similarity,
    chunk_id: chunk.chunk_id,
    excerpt: chunk.text.replace(/\s+/g, ' ').trim().slice(0, 150),
  }));

  // ── Step 3: Stream answer + generate follow-ups concurrently ───────────────
  const stream = new ReadableStream({
    async start(controller) {

      // Kick off follow-up generation immediately — it runs while we stream the answer.
      // generateFollowups() never throws; returns [] on any failure.
      const followupsPromise = generateFollowups(question);

      let streamFailed = false;

      try {
        const generator = streamGroundedAnswer(
          question,
          retrievalResult.chunks,
          safeHistory,
          answerMode
        );
        for await (const textDelta of generator) {
          controller.enqueue(line({ type: 'chunk', text: textDelta }));
        }
      } catch (streamErr: unknown) {
        streamFailed = true;
        console.warn('Streaming failed, falling back:', (streamErr as Error).message);
        try {
          const fullAnswer = await generateGroundedAnswer(
            question,
            retrievalResult.chunks,
            safeHistory,
            answerMode
          );
          controller.enqueue(line({ type: 'chunk', text: fullAnswer }));
        } catch (fallbackErr: unknown) {
          console.error('Both streaming and fallback failed:', (fallbackErr as Error).message);
          controller.enqueue(
            line({ type: 'error', message: 'The AI service is temporarily unavailable. Please try again.' })
          );
          // Still resolve followups so meta is consistent
          const followups = await followupsPromise;
          controller.enqueue(line({
            type: 'meta', sources, found: retrievalResult.found,
            topScore: retrievalResult.topScore, mode: answerMode, followups,
          }));
          controller.close();
          return;
        }
      }

      // By now follow-ups are almost certainly ready (generation took longer)
      const followups = await followupsPromise;

      controller.enqueue(line({
        type: 'meta',
        sources,
        found: retrievalResult.found,
        topScore: retrievalResult.topScore,
        mode: answerMode,
        followups,
        _streamFailed: streamFailed, // debug flag, stripped by client
      }));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'X-Content-Type-Options': 'nosniff',
      'Cache-Control': 'no-cache, no-store',
    },
  });
}
