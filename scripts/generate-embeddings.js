const fs = require('fs');
const path = require('path');
const { GoogleGenAI } = require('@google/genai');

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("Error: GEMINI_API_KEY is not set.");
  console.error("Please run this script by passing the key inline, e.g.:");
  console.error("GEMINI_API_KEY=your_key node scripts/generate-embeddings.js");
  console.error("Or ensure your environment is loaded.");
  process.exit(1);
}

const ai = new GoogleGenAI({});
const EMBEDDING_MODEL = 'text-embedding-004'; // The recommended embedding model in the Gemini ecosystem

const CHUNKS_FILE = path.join(process.cwd(), 'data', 'processed_chunks.json');
const EMBEDDINGS_FILE = path.join(process.cwd(), 'data', 'embeddings.json');

// A simple delay helper to prevent hitting rate limits
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function generateEmbeddings() {
  if (!fs.existsSync(CHUNKS_FILE)) {
    console.error(`No chunks file found at ${CHUNKS_FILE}. Please run process-documents.js first.`);
    return;
  }

  const rawChunks = fs.readFileSync(CHUNKS_FILE, 'utf-8');
  const chunks = JSON.parse(rawChunks);

  // Load existing embeddings to avoid redundant API calls
  let existingEmbeddings = {};
  if (fs.existsSync(EMBEDDINGS_FILE)) {
    try {
      const rawEmbeddings = fs.readFileSync(EMBEDDINGS_FILE, 'utf-8');
      const loaded = JSON.parse(rawEmbeddings);
      // Map by chunk_id for fast O(1) lookup
      loaded.forEach(item => {
        existingEmbeddings[item.chunk_id] = item;
      });
      console.log(`Loaded ${loaded.length} existing embeddings from local store.`);
    } catch (e) {
      console.log("Could not parse existing embeddings file. Starting fresh.");
    }
  }

  const finalEmbeddings = [];
  let newCount = 0;

  console.log(`Scanning ${chunks.length} chunks...`);

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    // If we already have the embedding for this specific chunk, skip the API call
    if (existingEmbeddings[chunk.chunk_id]) {
      finalEmbeddings.push(existingEmbeddings[chunk.chunk_id]);
      continue;
    }

    try {
      console.log(`Generating embedding for chunk ${i + 1}/${chunks.length} [${chunk.chunk_id}]...`);
      
      const response = await ai.models.embedContent({
        model: EMBEDDING_MODEL,
        contents: chunk.text,
      });

      // Extract the vector values (an array of floating point numbers)
      const vector = response.embeddings[0].values;

      // Preserve all metadata (source, page, text) and append the vector
      finalEmbeddings.push({
        ...chunk,
        embedding: vector
      });

      newCount++;
      
      // Small delay to gracefully respect Gemini API rate limits
      await delay(250);
      
    } catch (error) {
      console.error(`\nFailed to embed chunk ${chunk.chunk_id}:`, error.message);
      console.log('Saving progress so far and exiting. Run the script again to resume from where you left off.');
      break; 
    }
  }

  // Save the result back to disk as our simple local vector store
  fs.writeFileSync(EMBEDDINGS_FILE, JSON.stringify(finalEmbeddings, null, 2));
  
  console.log(`\n=========================================`);
  console.log(`Embeddings Complete!`);
  console.log(`Generated ${newCount} new embeddings.`);
  console.log(`Total embedded chunks in local store: ${finalEmbeddings.length}`);
  console.log(`Saved to: ${EMBEDDINGS_FILE}`);
  console.log(`=========================================`);
}

generateEmbeddings();
