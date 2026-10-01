/**
 * AskAI — RAG Pipeline Stage 3: Retrieval Test Mode
 *
 * This script lets you test the retrieval stage INDEPENDENTLY from
 * the Gemini answer-generation code. It will NEVER call generateContent().
 *
 * Usage:
 *   node scripts/test-retrieval.js               → interactive prompt loop
 *   node scripts/test-retrieval.js --query "..."  → single question, then exit
 *   node scripts/test-retrieval.js -q "..."       → same, shorthand
 *
 * Configuration flags (can also be set as env vars):
 *   --top    <number>    How many results to return (default: 3)
 *   --thresh <number>    Minimum similarity 0–1 (default: 0.55)
 *   --preview            Show a 200-char text preview for each result
 *   --debug              Show ALL scored chunks (ignores threshold), not just top K
 *
 * Examples:
 *   node scripts/test-retrieval.js -q "What is overfitting?"
 *   node scripts/test-retrieval.js -q "Explain backpropagation" --top 5 --preview
 *   node scripts/test-retrieval.js --thresh 0.60 --debug
 */

'use strict';

const fs       = require('fs');
const path     = require('path');
const readline = require('readline');
const { GoogleGenAI } = require('@google/genai');

// ─── Environment Loading ───────────────────────────────────────────────────────

function loadEnv() {
  if (process.env.GEMINI_API_KEY) return;
  for (const file of ['.env.local', '.env']) {
    const envPath = path.join(process.cwd(), file);
    if (!fs.existsSync(envPath)) continue;
    fs.readFileSync(envPath, 'utf-8').split('\n').forEach(line => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) return;
      const eqIdx = trimmed.indexOf('=');
      if (eqIdx === -1) return;
      const key = trimmed.substring(0, eqIdx).trim();
      const val = trimmed.substring(eqIdx + 1).trim().replace(/^["']|["']$/g, '');
      if (!process.env[key]) process.env[key] = val;
    });
    break;
  }
}

loadEnv();

// ─── CLI Argument Parsing ──────────────────────────────────────────────────────

function parseArgs(argv) {
  const args = argv.slice(2);
  const opts = {
    query:   null,
    topK:    3,
    thresh:  0.55,
    preview: false,
    debug:   false,
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if ((a === '--query' || a === '-q') && args[i + 1]) { opts.query   = args[++i]; }
    else if (a === '--top'    && args[i + 1])            { opts.topK   = Math.max(1, parseInt(args[++i], 10) || 3); }
    else if (a === '--thresh' && args[i + 1])            { opts.thresh = parseFloat(args[++i]) || 0.55; }
    else if (a === '--preview')                          { opts.preview = true; }
    else if (a === '--debug')                            { opts.debug   = true; }
  }
  return opts;
}

// ─── Configuration ─────────────────────────────────────────────────────────────

const OPTS           = parseArgs(process.argv);
const EMBEDDINGS_FILE = path.join(process.cwd(), 'data', 'embeddings.json');
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'gemini-embedding-001';

// ─── Guard: Verify pre-requisites ──────────────────────────────────────────────

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('✗ Error: GEMINI_API_KEY is not set.');
  console.error('  Add it to .env.local:  GEMINI_API_KEY=your_key_here');
  process.exit(1);
}

if (!fs.existsSync(EMBEDDINGS_FILE)) {
  console.error(`✗ Error: Embeddings file not found at: ${EMBEDDINGS_FILE}`);
  console.error('  Run first:  node scripts/generate-embeddings.js');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });

// ─── Cosine Similarity ─────────────────────────────────────────────────────────

function cosineSimilarity(vecA, vecB) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot   += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ─── Core Retrieval (no answer generation) ────────────────────────────────────

/**
 * Retrieves the most semantically relevant course-material chunks for a question.
 *
 * Steps:
 * 1. Generate an embedding for the question using the Gemini Embeddings API.
 * 2. Load the local vector store (data/embeddings.json).
 * 3. Score every stored chunk with cosine similarity.
 * 4. Apply the similarity threshold and return the top-K results.
 *
 * NOTE: This function is COMPLETELY SEPARATE from generateContent().
 *       It only retrieves — it does NOT generate answers.
 *
 * @param {string}  question          The student's question
 * @param {number}  topK              Max number of chunks to return
 * @param {number}  similarityThreshold  Minimum required similarity (0–1)
 * @param {boolean} debugAll          If true, return all scored chunks ignoring threshold
 * @returns {Promise<object>}         RetrievalResult object
 */
async function retrieveRelevantChunks(question, topK, similarityThreshold, debugAll = false) {
  // Step 1 — Embed the question
  const embResponse = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: question,
  });
  const questionVector = embResponse.embeddings?.[0]?.values;
  if (!questionVector || questionVector.length === 0) {
    throw new Error('Failed to generate an embedding for the question.');
  }

  // Step 2 — Load local vector store
  const storedChunks = JSON.parse(fs.readFileSync(EMBEDDINGS_FILE, 'utf-8'));

  // Step 3 — Score every chunk
  const scoredChunks = storedChunks.map(chunk => ({
    chunk_id:   chunk.chunk_id,
    source:     chunk.source,
    page:       chunk.page,
    text:       chunk.text,
    similarity: Array.isArray(chunk.embedding)
                  ? cosineSimilarity(questionVector, chunk.embedding)
                  : 0,
  }));

  // Sort highest similarity first
  scoredChunks.sort((a, b) => b.similarity - a.similarity);

  const topScore = scoredChunks[0]?.similarity ?? 0;

  // Step 4 — Filter and slice
  const relevantChunks = debugAll
    ? scoredChunks.slice(0, topK)                                      // debug: ignore threshold
    : scoredChunks.filter(c => c.similarity >= similarityThreshold).slice(0, topK);

  return {
    question,
    found:     relevantChunks.length > 0,
    chunks:    relevantChunks,
    topScore,
    threshold: similarityThreshold,
    totalScanned: storedChunks.length,
  };
}

// ─── Display Helpers ───────────────────────────────────────────────────────────

function similarityBar(score) {
  const filled = Math.round(score * 20);
  return '[' + '█'.repeat(filled) + '░'.repeat(20 - filled) + ']';
}

function printResult(result, showPreview, debugMode) {
  const sep = '─'.repeat(60);
  console.log();
  console.log(sep);
  console.log(`Question: "${result.question}"`);
  console.log(sep);

  if (!result.found) {
    console.log();
    console.log('  ✗ No relevant course material found.');
    console.log();
    console.log(`  Threshold : ${(result.threshold * 100).toFixed(0)}%`);
    console.log(`  Best match: ${(result.topScore * 100).toFixed(2)}% (below threshold)`);
    if (result.topScore > 0) {
      console.log();
      console.log('  Tip: Try lowering --thresh to see more results.');
      console.log(`       e.g.  --thresh ${Math.max(0, (result.topScore - 0.05)).toFixed(2)}`);
    }
    console.log();
    return;
  }

  console.log();
  if (debugMode) {
    console.log(`  Debug mode — showing top ${result.chunks.length} chunks (threshold ignored)`);
  } else {
    console.log(`  Retrieved sources (threshold ≥ ${(result.threshold * 100).toFixed(0)}%):`);
  }
  console.log();

  result.chunks.forEach((chunk, idx) => {
    const pct = (chunk.similarity * 100).toFixed(2);
    console.log(`  ${idx + 1}. ${chunk.source} — Page ${chunk.page}`);
    console.log(`     Similarity : ${pct}%  ${similarityBar(chunk.similarity)}`);
    console.log(`     Chunk ID   : ${chunk.chunk_id}`);
    if (showPreview) {
      const preview = chunk.text.replace(/\s+/g, ' ').trim().substring(0, 220);
      console.log(`     Preview    : "${preview}${chunk.text.length > 220 ? '…' : ''}"`);
    }
    console.log();
  });

  console.log(`  Scanned ${result.totalScanned} chunks total.`);
  console.log();
}

// ─── Interactive Loop ─────────────────────────────────────────────────────────

async function runSingleQuery(question) {
  process.stdout.write('  Retrieving…  ');
  try {
    const result = await retrieveRelevantChunks(
      question, OPTS.topK, OPTS.thresh, OPTS.debug
    );
    console.log('done.');
    printResult(result, OPTS.preview, OPTS.debug);
  } catch (err) {
    console.log();
    console.error(`  ✗ Retrieval error: ${err.message}`);
    console.log();
  }
}

function startInteractiveMode() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const banner = [
    '',
    '╔══════════════════════════════════════════════════════════╗',
    '║       AskAI — RAG Retrieval Test Mode (Stage 3)          ║',
    '╠══════════════════════════════════════════════════════════╣',
    `║  Embedding model : ${EMBEDDING_MODEL.padEnd(38)}║`,
    `║  Top-K results   : ${String(OPTS.topK).padEnd(38)}║`,
    `║  Threshold       : ${(OPTS.thresh * 100).toFixed(0).padEnd(36)}%  ║`,
    `║  Preview mode    : ${(OPTS.preview ? 'ON ' : 'OFF').padEnd(38)}║`,
    `║  Debug mode      : ${(OPTS.debug   ? 'ON ' : 'OFF').padEnd(38)}║`,
    '╠══════════════════════════════════════════════════════════╣',
    '║  Commands:  "exit" or "quit" to stop                     ║',
    '║             "help" for usage tips                         ║',
    '╚══════════════════════════════════════════════════════════╝',
    '',
  ];
  console.log(banner.join('\n'));

  const prompt = () => {
    rl.question('Question: ', async input => {
      const q = input.trim();

      if (!q) { prompt(); return; }

      if (q === 'exit' || q === 'quit') {
        console.log('\nGoodbye!\n');
        rl.close();
        return;
      }

      if (q === 'help') {
        console.log('\n  Tips:');
        console.log('  • Ask specific questions about your course topics.');
        console.log('  • If you get "no results", try rephrasing or lowering --thresh.');
        console.log('  • Run with --preview to see text snippets from matched chunks.');
        console.log('  • Run with --debug to see all top-K results ignoring the threshold.');
        console.log('  • Run with --top 5 to retrieve more results.\n');
        prompt(); return;
      }

      await runSingleQuery(q);
      prompt();
    });
  };

  prompt();
}

// ─── Entry Point ──────────────────────────────────────────────────────────────

if (OPTS.query) {
  // Single-shot mode: run one query and exit
  runSingleQuery(OPTS.query).then(() => process.exit(0)).catch(err => {
    console.error(err);
    process.exit(1);
  });
} else {
  // Interactive prompt loop
  startInteractiveMode();
}
