import fs from 'fs';
import path from 'path';
import { GoogleGenAI } from '@google/genai';

// ─── Configuration ────────────────────────────────────────────────────────────
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'gemini-embedding-001';
const EMBEDDINGS_FILE = path.join(process.cwd(), 'data', 'embeddings.json');

const DEFAULT_TOP_K    = 3;
const DEFAULT_THRESHOLD = 0.55;

// Module-level singleton — avoids creating a new client on every request
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── Stored chunk shape (mirrors generate-embeddings.js output) ───────────────

/** Shape of each record in data/embeddings.json */
export type EmbeddingRecord = {
  chunk_id:  string;
  source:    string;
  page:      number;
  text:      string;
  embedding: number[];
};

// ─── Public types ─────────────────────────────────────────────────────────────

/** A single chunk returned by the retrieval engine, with its similarity score. */
export type RetrievedChunk = {
  chunk_id:   string;
  source:     string;    // Source filename e.g. "Deep_Learning.pdf"
  page:       number;    // Page number within the source document
  text:       string;    // The actual course-material text
  similarity: number;    // Cosine similarity score [0.0 – 1.0]
};

/** The full result returned by retrieveRelevantChunks(). */
export type RetrievalResult = {
  question:  string;
  found:     boolean;          // false if no chunk met the threshold
  chunks:    RetrievedChunk[]; // Top-K relevant chunks (empty when not found)
  topScore:  number;           // Highest similarity seen (even if below threshold)
  threshold: number;           // Threshold that was applied
};

// ─── Cosine Similarity ────────────────────────────────────────────────────────

/**
 * Calculates the cosine similarity between two equal-length vectors.
 * Returns a value between 0 (unrelated) and 1 (identical meaning).
 *
 * Formula:  cos(θ) = (A · B) / (|A| × |B|)
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA      += vecA[i] * vecA[i];
    normB      += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ─── Cached vector store ──────────────────────────────────────────────────────
// Loaded once on first call; stays in memory for the server process lifetime.

let _corpusCache: EmbeddingRecord[] | null = null;

function loadCorpus(): EmbeddingRecord[] {
  if (_corpusCache) return _corpusCache;

  if (!fs.existsSync(EMBEDDINGS_FILE)) {
    throw new Error(
      `Embeddings file not found: ${EMBEDDINGS_FILE}\n` +
      'Run: npm run process-docs && npm run generate-embeddings'
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(EMBEDDINGS_FILE, 'utf-8'));
  } catch (e) {
    throw new Error(
      `Failed to parse embeddings file (it may be corrupted): ${(e as Error).message}`
    );
  }

  if (!Array.isArray(parsed)) {
    throw new Error('embeddings.json must contain a JSON array.');
  }

  _corpusCache = parsed as EmbeddingRecord[];
  return _corpusCache;
}

// ─── Retrieval Engine ─────────────────────────────────────────────────────────

/**
 * Core RAG retrieval function.
 *
 * 1. Generates a semantic embedding for the question via Gemini.
 * 2. Loads the pre-built local vector store (data/embeddings.json).
 * 3. Scores every chunk with cosine similarity.
 * 4. Returns the top-K chunks that exceed the similarity threshold.
 *
 * Intentionally SEPARATE from Gemini answer-generation — retrieves only.
 *
 * @param question            The student's question string
 * @param topK                How many chunks to return (default: 3)
 * @param similarityThreshold Minimum cosine similarity required (default: 0.55)
 */
export async function retrieveRelevantChunks(
  question: string,
  topK: number = DEFAULT_TOP_K,
  similarityThreshold: number = DEFAULT_THRESHOLD
): Promise<RetrievalResult> {

  // ── Step 1: Embed the question (with one retry on transient errors) ─────────
  let questionVector: number[];
  try {
    const embResponse = await ai.models.embedContent({
      model: EMBEDDING_MODEL,
      contents: question,
    });
    const values = embResponse.embeddings?.[0]?.values;
    if (!values || values.length === 0) {
      throw new Error('Gemini returned an empty embedding vector.');
    }
    questionVector = values;
  } catch (firstErr: unknown) {
    // One retry on transient 429 / 503
    const msg = (firstErr as Error).message ?? '';
    const isRetryable = msg.includes('429') || msg.includes('503');
    if (!isRetryable) throw firstErr;

    await new Promise(r => setTimeout(r, 2000));
    const retryResp = await ai.models.embedContent({
      model: EMBEDDING_MODEL,
      contents: question,
    });
    const values = retryResp.embeddings?.[0]?.values;
    if (!values || values.length === 0) {
      throw new Error('Gemini returned an empty embedding vector on retry.');
    }
    questionVector = values;
  }

  // ── Step 2: Load local vector store (cached after first call) ──────────────
  const corpus = loadCorpus();

  // ── Step 3: Score every chunk ───────────────────────────────────────────────
  const scored: RetrievedChunk[] = corpus.map(chunk => ({
    chunk_id:   chunk.chunk_id,
    source:     chunk.source,
    page:       chunk.page,
    text:       chunk.text,
    similarity: Array.isArray(chunk.embedding)
      ? cosineSimilarity(questionVector, chunk.embedding)
      : 0,
  }));

  scored.sort((a, b) => b.similarity - a.similarity);
  const topScore = scored[0]?.similarity ?? 0;

  // ── Step 4: Filter by threshold and cap at topK ─────────────────────────────
  const relevant = scored
    .filter(c => c.similarity >= similarityThreshold)
    .slice(0, topK);

  return {
    question,
    found:     relevant.length > 0,
    chunks:    relevant,
    topScore,
    threshold: similarityThreshold,
  };
}
