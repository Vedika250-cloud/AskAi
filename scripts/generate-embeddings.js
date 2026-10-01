const fs = require('fs');
const path = require('path');
const { GoogleGenAI } = require('@google/genai');

// Auto-load environment variables from .env.local or .env if not already set
function loadEnv() {
  if (process.env.GEMINI_API_KEY) return;
  const envFiles = ['.env.local', '.env'];
  for (const file of envFiles) {
    const envPath = path.join(process.cwd(), file);
    if (fs.existsSync(envPath)) {
      const content = fs.readFileSync(envPath, 'utf-8');
      content.split('\n').forEach(line => {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#')) {
          const eqIdx = trimmed.indexOf('=');
          if (eqIdx !== -1) {
            const key = trimmed.substring(0, eqIdx).trim();
            const val = trimmed.substring(eqIdx + 1).trim().replace(/^["']|["']$/g, '');
            if (!process.env[key]) process.env[key] = val;
          }
        }
      });
      break;
    }
  }
}

loadEnv();

// Paths
const CHUNKS_FILE = process.env.CHUNKS_FILE || path.join(process.cwd(), 'data', 'processed_chunks.json');
const EMBEDDINGS_FILE = process.env.EMBEDDINGS_FILE || path.join(process.cwd(), 'data', 'embeddings.json');

// Supported Gemini Embedding model (verified in the Gemini v1beta ecosystem)
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'gemini-embedding-001';

// Batch configuration
const BATCH_SIZE = 15;        // Chunks per API request
const BATCH_DELAY_MS = 1000;  // Pacing delay between batches to stay under free-tier quota
const MAX_RETRIES = 5;        // Retries on transient API rate limits / network hiccups

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Extracts wait duration from Google's quota exhaustion message, or defaults to 60s.
 */
function extractRetrySeconds(errorMessage) {
  if (!errorMessage) return 60;
  const match = errorMessage.match(/retry in\s+([\d\.]+)s/i);
  if (match && match[1]) {
    return Math.ceil(parseFloat(match[1])) + 2; // +2s safety buffer
  }
  const delayMatch = errorMessage.match(/"retryDelay":\s*"(\d+)s"/i);
  if (delayMatch && delayMatch[1]) {
    return parseInt(delayMatch[1], 10) + 2;
  }
  return 60;
}

/**
 * Calls Gemini embedContent API with automatic retry on rate limits (429) or transient errors.
 */
async function embedBatchWithRetry(ai, texts, attempt = 1) {
  try {
    const response = await ai.models.embedContent({
      model: EMBEDDING_MODEL,
      contents: texts,
    });
    return response.embeddings.map(e => e.values);
  } catch (error) {
    const isRateLimit = error.status === 429 || (error.message && (error.message.includes('429') || error.message.includes('RESOURCE_EXHAUSTED')));
    const isTransient = error.status >= 500 || (error.message && error.message.includes('503'));

    if ((isRateLimit || isTransient) && attempt <= MAX_RETRIES) {
      let waitSeconds = attempt * 3;
      if (isRateLimit) {
        waitSeconds = extractRetrySeconds(error.message);
        console.warn(`\n  [Quota 429] Free-tier window limit reached. Waiting ${waitSeconds}s for quota to refresh (Attempt ${attempt}/${MAX_RETRIES})...`);
      } else {
        console.warn(`\n  [Transient ${error.status || 'Error'}] Retrying batch in ${waitSeconds}s (Attempt ${attempt}/${MAX_RETRIES})...`);
      }
      await delay(waitSeconds * 1000);
      return embedBatchWithRetry(ai, texts, attempt + 1);
    }
    throw error;
  }
}

/**
 * Main embedding generator function.
 * 
 * @param {Object} options
 * @param {boolean} options.rebuild - If true, ignores existing embeddings and rebuilds the index from scratch.
 * @returns {Promise<Array<Object>>} The complete array of embedded chunks.
 */
async function generateEmbeddings(options = {}) {
  const rebuild = options.rebuild || false;

  console.log('====================================================');
  console.log('AskAI RAG Pipeline — Stage 2: Embedding Generation');
  console.log('====================================================\n');

  // Verify API Key
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error('Error: GEMINI_API_KEY is not set.');
    console.error('Please configure your GEMINI_API_KEY in .env.local:');
    console.error('  GEMINI_API_KEY=your_key_here\n');
    process.exit(1);
  }

  const ai = new GoogleGenAI({ apiKey });

  // 1. Read processed course chunks
  if (!fs.existsSync(CHUNKS_FILE)) {
    console.error(`Error: Chunks file not found at: ${CHUNKS_FILE}`);
    console.error('Please run the document processing script first:');
    console.error('  node scripts/process-documents.js\n');
    process.exit(1);
  }

  const rawChunks = fs.readFileSync(CHUNKS_FILE, 'utf-8');
  let chunks = [];
  try {
    chunks = JSON.parse(rawChunks);
  } catch (err) {
    console.error(`Error: Could not parse ${CHUNKS_FILE}:`, err.message);
    process.exit(1);
  }

  if (!Array.isArray(chunks) || chunks.length === 0) {
    console.log(`No chunks found in ${CHUNKS_FILE}. Process your course documents first.`);
    return [];
  }

  console.log(`Loaded ${chunks.length} total chunks from: ${CHUNKS_FILE}`);
  console.log(`Using Embedding Model: ${EMBEDDING_MODEL}`);

  // 2. Load existing embeddings for caching / resume capability
  let existingStore = new Map();
  if (!rebuild && fs.existsSync(EMBEDDINGS_FILE)) {
    try {
      const existingRaw = fs.readFileSync(EMBEDDINGS_FILE, 'utf-8');
      const loaded = JSON.parse(existingRaw);
      if (Array.isArray(loaded)) {
        loaded.forEach(item => {
          if (item.chunk_id && Array.isArray(item.embedding)) {
            existingStore.set(item.chunk_id, item);
          }
        });
        console.log(`Found ${existingStore.size} cached embeddings in: ${EMBEDDINGS_FILE}`);
      }
    } catch {
      console.log('Could not parse existing embeddings file; starting fresh.');
      existingStore.clear();
    }
  } else if (rebuild) {
    console.log('Rebuild flag enabled: Rebuilding entire vector index from scratch...');
  }

  // 3. Identify chunks that actually need embeddings
  const chunksToProcess = [];
  const finalEmbeddings = [];

  for (const chunk of chunks) {
    if (!rebuild && existingStore.has(chunk.chunk_id)) {
      finalEmbeddings.push(existingStore.get(chunk.chunk_id));
    } else {
      chunksToProcess.push(chunk);
    }
  }

  if (chunksToProcess.length === 0) {
    console.log('\n✓ All chunks already have embeddings in local vector store!');
    console.log(`Total indexed chunks: ${finalEmbeddings.length}`);
    console.log('To force rebuild all embeddings from scratch, run:');
    console.log('  node scripts/generate-embeddings.js --rebuild\n');
    return finalEmbeddings;
  }

  console.log(`\nNeed to generate embeddings for ${chunksToProcess.length} new/updated chunk(s).`);
  console.log(`Processing in batches of ${BATCH_SIZE}...\n`);

  // Helper function to periodically checkpoint progress to disk
  const saveProgress = () => {
    const outputDir = path.dirname(EMBEDDINGS_FILE);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    fs.writeFileSync(EMBEDDINGS_FILE, JSON.stringify(finalEmbeddings, null, 2), 'utf-8');
  };

  let newlyEmbeddedCount = 0;
  const totalBatches = Math.ceil(chunksToProcess.length / BATCH_SIZE);

  for (let b = 0; b < totalBatches; b++) {
    const batchStart = b * BATCH_SIZE;
    const batchEnd = Math.min(batchStart + BATCH_SIZE, chunksToProcess.length);
    const batch = chunksToProcess.slice(batchStart, batchEnd);

    const progressStr = `[Batch ${b + 1}/${totalBatches}] Chunks ${batchStart + 1}–${batchEnd} (${batch[0].chunk_id})...`;
    process.stdout.write(progressStr);

    try {
      const texts = batch.map(c => c.text);
      const vectors = await embedBatchWithRetry(ai, texts);

      for (let i = 0; i < batch.length; i++) {
        finalEmbeddings.push({
          chunk_id: batch[i].chunk_id,
          source: batch[i].source,
          page: batch[i].page,
          text: batch[i].text,
          embedding: vectors[i]
        });
      }

      newlyEmbeddedCount += batch.length;
      console.log(' ✓');

      // Checkpoint progress to disk after every batch
      saveProgress();

      // Delay between batches to respect free-tier rate limits
      if (b < totalBatches - 1) {
        await delay(BATCH_DELAY_MS);
      }
    } catch (error) {
      console.error(`\n✗ Error generating embeddings for batch ${b + 1}:`, error.message);
      console.log('Safely saved progress so far.');
      console.log('Run the script again to resume from where you left off.\n');
      saveProgress();
      break;
    }
  }

  // Final summary
  const sampleDim = finalEmbeddings[0]?.embedding?.length || 0;
  console.log('\n====================================================');
  console.log('Embedding Generation Summary:');
  console.log(`  - Newly Generated Embeddings : ${newlyEmbeddedCount}`);
  console.log(`  - Total Vector Store Entries : ${finalEmbeddings.length}`);
  console.log(`  - Vector Dimensions          : ${sampleDim}`);
  console.log(`  - Saved Local Vector Store   : ${EMBEDDINGS_FILE}`);
  console.log('====================================================\n');

  return finalEmbeddings;
}

// Check for rebuild / force flags from CLI:
// node scripts/generate-embeddings.js --rebuild
// node scripts/generate-embeddings.js --force
const args = process.argv.slice(2);
const shouldRebuild = args.includes('--rebuild') || args.includes('-r') || args.includes('--force') || args.includes('-f');

if (require.main === module) {
  generateEmbeddings({ rebuild: shouldRebuild }).catch(err => {
    console.error('Fatal embedding process error:', err);
    process.exit(1);
  });
}

module.exports = { generateEmbeddings };
