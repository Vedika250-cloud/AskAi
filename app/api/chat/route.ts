import { NextResponse } from 'next/server';
import { retrieveRelevantChunks } from '@/lib/retrieval';
import { generateGroundedAnswer, type AnswerMode } from '@/lib/gemini';

// Default RAG retrieval parameters
const TOP_K = 3;
const SIMILARITY_THRESHOLD = 0.55;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { message, history, mode } = body;

    // Handle empty user messages gracefully
    if (!message || message.trim() === '') {
      return NextResponse.json(
        { error: 'Message is required.' },
        { status: 400 }
      );
    }

    const question = message.trim();

    // Validate requested answer mode, defaulting to 'detailed'
    const validModes: AnswerMode[] = ['simple', 'detailed', 'exam', 'viva'];
    const answerMode: AnswerMode = mode && validModes.includes(mode) ? mode : 'detailed';

    // ── Step 1 & 2: Embed question & Retrieve top relevant chunks ─────────────
    let retrievalResult;
    try {
      retrievalResult = await retrieveRelevantChunks(question, TOP_K, SIMILARITY_THRESHOLD);
    } catch (retrievalErr: any) {
      console.warn('Retrieval warning (proceeding with fallback):', retrievalErr.message);
      retrievalResult = {
        question,
        found: false,
        chunks: [],
        topScore: 0,
        threshold: SIMILARITY_THRESHOLD,
      };
    }

    // ── Step 3 & 4: Prompt Construction & Gemini Grounded Generation ──────────
    const answer = await generateGroundedAnswer(
      question,
      retrievalResult.chunks,
      history || [],
      answerMode
    );

    // ── Step 5: Return Grounded Answer, Sources & Active Mode ─────────────────
    const sources = retrievalResult.chunks.map(chunk => ({
      source: chunk.source,
      page: chunk.page,
      similarity: chunk.similarity,
      chunk_id: chunk.chunk_id,
    }));

    return NextResponse.json({
      answer,
      sources,
      found: retrievalResult.found,
      topScore: retrievalResult.topScore,
      mode: answerMode,
    });
  } catch (error: any) {
    console.error('RAG Pipeline API Route Error:', error);
    return NextResponse.json(
      { 
        error: 'Failed to process academic query',
        status: error.status || 500,
        message: error.message || 'Unknown error occurred'
      },
      { status: 500 }
    );
  }
}
