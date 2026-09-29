import fs from 'fs';
import path from 'path';
import { GoogleGenAI } from '@google/genai';

// Initialize the Gemini client specifically for embeddings
const ai = new GoogleGenAI({});
const EMBEDDING_MODEL = 'text-embedding-004';

// Helper function to calculate cosine similarity between two vectors
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export type RetrievedChunk = {
  chunk_id: string;
  source: string;
  page: number;
  text: string;
  similarity: number;
};

/**
 * Retrieves the most relevant course material chunks for a given question.
 * Note: This code is kept completely separate from the Gemini answer-generation code.
 *
 * @param question The user's question
 * @param topK Number of chunks to return (default: 3)
 * @param similarityThreshold Minimum similarity score (default: 0.55)
 * @returns Array of relevant chunks
 */
export async function retrieveRelevantChunks(
  question: string,
  topK: number = 3,
  similarityThreshold: number = 0.55
): Promise<RetrievedChunk[]> {
  const EMBEDDINGS_FILE = path.join(process.cwd(), 'data', 'embeddings.json');
  
  if (!fs.existsSync(EMBEDDINGS_FILE)) {
    throw new Error('Embeddings file not found. Please run the embedding script first.');
  }

  // 1. Generate an embedding for the student's question
  const response = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: question,
  });
  const questionVector = response.embeddings?.[0]?.values;
  
  if (!questionVector) {
    throw new Error('Failed to generate embedding for the question.');
  }

  // 2. Load stored course-material embeddings
  const rawData = fs.readFileSync(EMBEDDINGS_FILE, 'utf-8');
  const storedChunks = JSON.parse(rawData);

  // 3. Compare it with the stored course-material embeddings
  const scoredChunks: RetrievedChunk[] = storedChunks.map((chunk: any) => {
    return {
      chunk_id: chunk.chunk_id,
      source: chunk.source,
      page: chunk.page,
      text: chunk.text,
      similarity: chunk.embedding ? cosineSimilarity(questionVector, chunk.embedding) : 0
    };
  });

  // Sort by highest similarity first
  scoredChunks.sort((a, b) => b.similarity - a.similarity);

  // 4. Filter by threshold and take top K
  const relevantChunks = scoredChunks
    .filter(chunk => chunk.similarity >= similarityThreshold)
    .slice(0, topK);

  return relevantChunks;
}
