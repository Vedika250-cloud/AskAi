const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { GoogleGenAI } = require('@google/genai');

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("Error: GEMINI_API_KEY is not set.");
  console.error("Run with: node --env-file=.env.local scripts/test-retrieval.js");
  process.exit(1);
}

const ai = new GoogleGenAI({});
const EMBEDDING_MODEL = 'text-embedding-004';
const EMBEDDINGS_FILE = path.join(process.cwd(), 'data', 'embeddings.json');

// Mathematical function to measure the distance between two vectors
function cosineSimilarity(vecA, vecB) {
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

// The core retrieval logic (kept completely separate from answer-generation)
async function retrieveRelevantChunks(question, topK = 3, similarityThreshold = 0.6) {
  if (!fs.existsSync(EMBEDDINGS_FILE)) {
    throw new Error('Embeddings file not found. Please run the embedding script first.');
  }

  // 1. Generate an embedding for the student's question
  const response = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: question,
  });
  const questionVector = response.embeddings[0].values;

  // 2. Load stored embeddings
  const rawData = fs.readFileSync(EMBEDDINGS_FILE, 'utf-8');
  const storedChunks = JSON.parse(rawData);

  // 3. Compare it with the stored course-material embeddings
  const scoredChunks = storedChunks.map(chunk => {
    return {
      chunk_id: chunk.chunk_id,
      source: chunk.source,
      page: chunk.page,
      text: chunk.text,
      similarity: cosineSimilarity(questionVector, chunk.embedding)
    };
  });

  // Sort by highest similarity first
  scoredChunks.sort((a, b) => b.similarity - a.similarity);

  // 4. Find the most relevant chunks based on threshold
  const relevantChunks = scoredChunks
    .filter(chunk => chunk.similarity >= similarityThreshold)
    .slice(0, topK);

  return relevantChunks;
}

// Interactive Test Mode
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function promptQuestion() {
  rl.question('\nQuestion: ', async (question) => {
    if (question.toLowerCase() === 'exit' || question.toLowerCase() === 'quit') {
      rl.close();
      return;
    }

    if (!question.trim()) {
      promptQuestion();
      return;
    }

    try {
      console.log('Retrieving sources...\n');
      
      const threshold = 0.55; // Configurable similarity threshold
      const topChunks = await retrieveRelevantChunks(question, 3, threshold);

      if (topChunks.length === 0) {
        console.log('No relevant course material found.');
      } else {
        console.log('Retrieved sources:\n');
        topChunks.forEach((chunk, index) => {
          console.log(`${index + 1}. ${chunk.source} — Page ${chunk.page} (Similarity: ${(chunk.similarity * 100).toFixed(2)}%)`);
          // Uncomment to see the actual text chunk that was retrieved during debugging
          // console.log(`   Preview: "${chunk.text.substring(0, 100)}..."\n`);
        });
      }
    } catch (error) {
      console.error('Error during retrieval:', error.message);
    }

    // Loop again
    promptQuestion();
  });
}

console.log("=========================================");
console.log("AskAI Retrieval Test Mode");
console.log("Type 'exit' to quit.");
console.log("=========================================");
promptQuestion();
