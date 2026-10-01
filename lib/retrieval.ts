import fs from 'fs';
import path from 'path';
import { GoogleGenAI } from '@google/genai';

// ─── Configuration ────────────────────────────────────────────────────────────
// These values can be overridden via environment variables.
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'gemini-embedding-001';
const EMBEDDINGS_FILE = path.join(process.cwd(), 'data', 'embeddings.json');

// Default retrieval settings
const DEFAULT_TOP_K = 3;
const DEFAULT_THRESHOLD = 0.55;

// ─── Types ─────────────────────────────────────────────────────────────────────

/** A single chunk returned by the retrieval engine, including similarity score. */
export type RetrievedChunk = {
  chunk_id: string;
  source: string;     // Source filename (e.g. "Deep_Learning.pdf")
  page: number;       // Page number within the source document
  text: string;       // The actual course-material text
  similarity: number; // Cosine similarity score [0.0 – 1.0]
};

/** The full result object returned by retrieveRelevantChunks(). */
export type RetrievalResult = {
  question: string;
  found: boolean;                 // false if no chunk met the similarity threshold
  chunks: RetrievedChunk[];       // Top-K relevant chunks (empty if not found)
  topScore: number;               // Highest similarity score seen (even if below threshold)
  threshold: number;              // Threshold that was applied
};

// ─── Cosine Similarity ─────────────────────────────────────────────────────────

/**
 * Calculates the cosine similarity between two equal-length vectors.
 * Returns a value between 0 (completely unrelated) and 1 (identical meaning).
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

// ─── Retrieval Engine ──────────────────────────────────────────────────────────

/**
 * Core RAG retrieval function.
 *
 * Given a student's question it:
 * 1. Generates a semantic embedding for the question using Gemini.
 * 2. Loads the pre-built local vector store (data/embeddings.json).
 * 3. Computes cosine similarity between the question and every stored chunk.
 * 4. Returns the top-K chunks that exceed the similarity threshold.
 *
 * This function is intentionally SEPARATE from Gemini answer-generation.
 * It only retrieves; it does NOT call generateContent().
 *
 * @param question           The student's question string
 * @param topK               How many chunks to return (default: 3)
 * @param similarityThreshold Minimum cosine similarity required (default: 0.55)
 * @returns                  A RetrievalResult object
 */
export async function retrieveRelevantChunks(
  question: string,
  topK: number = DEFAULT_TOP_K,
  similarityThreshold: number = DEFAULT_THRESHOLD
): Promise<RetrievalResult> {

  // Guard: embeddings must exist
  if (!fs.existsSync(EMBEDDINGS_FILE)) {
    throw new Error(
      `Embeddings file not found at: ${EMBEDDINGS_FILE}\n` +
      'Please run: node scripts/generate-embeddings.js'
    );
  }

  // ── Step 1: Embed the question ─────────────────────────────────────────────
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  const embResponse = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: question,
  });

  const questionVector = embResponse.embeddings?.[0]?.values;
  if (!questionVector || questionVector.length === 0) {
    throw new Error('Failed to generate an embedding for the question.');
  }

  // ── Step 2: Load local vector store ───────────────────────────────────────
  const rawData = fs.readFileSync(EMBEDDINGS_FILE, 'utf-8');
  const storedChunks: Array<{
    chunk_id: string;
    source: string;
    page: number;
    text: string;
    embedding: number[];
  }> = JSON.parse(rawData);

  // ── Step 3: Score every chunk with cosine similarity ──────────────────────
  const scoredChunks: RetrievedChunk[] = storedChunks.map(chunk => ({
    chunk_id:   chunk.chunk_id,
    source:     chunk.source,
    page:       chunk.page,
    text:       chunk.text,
    similarity: Array.isArray(chunk.embedding)
      ? cosineSimilarity(questionVector, chunk.embedding)
      : 0,
  }));

  // Sort highest → lowest similarity
  scoredChunks.sort((a, b) => b.similarity - a.similarity);

  const topScore = scoredChunks[0]?.similarity ?? 0;

  // ── Step 4: Filter by threshold and take top K ────────────────────────────
  const relevantChunks = scoredChunks
    .filter(chunk => chunk.similarity >= similarityThreshold)
    .slice(0, topK);

  return {
    question,
    found:     relevantChunks.length > 0,
    chunks:    relevantChunks,
    topScore,
    threshold: similarityThreshold,
  };
}
