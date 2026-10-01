/**
 * AskAI Retrieval Evaluation Script
 *
 * Runs every question in data/eval_dataset.json through the live retrieval
 * pipeline (embed → cosine similarity → rank), checks whether the expected
 * source document appears in the top-K results, and writes a full results
 * JSON to data/eval_results.json.
 *
 * Usage:
 *   node scripts/eval-retrieval.js
 *   node scripts/eval-retrieval.js --top 5 --thresh 0.50
 *
 * Flags:
 *   --top   <n>   Number of results to retrieve (default 3)
 *   --thresh <f>  Similarity threshold for "retrieved" (default 0.55)
 */

'use strict';

const fs   = require('fs');
const path = require('path');

// ─── Load .env.local ────────────────────────────────────────────────────────

function loadEnv() {
  const envPath = path.join(process.cwd(), '.env.local');
  if (!fs.existsSync(envPath)) {
    throw new Error('.env.local not found. Create it with GEMINI_API_KEY=...');
  }
  const content = fs.readFileSync(envPath, 'utf-8');
  for (const raw of content.split('\n')) {
    const line = raw.trim(); // strips \r from CRLF files
    if (!line || line.startsWith('#')) continue;
    const eqIdx = line.indexOf('=');
    if (eqIdx < 1) continue;
    const key = line.slice(0, eqIdx).trim();
    const val = line.slice(eqIdx + 1).trim().replace(/^['"]|['"]$/g, '');
    if (key) process.env[key] = val;
  }
}

loadEnv();

if (!process.env.GEMINI_API_KEY) {
  console.error('ERROR: GEMINI_API_KEY not set in .env.local');
  process.exit(1);
}

const { GoogleGenAI } = require('@google/genai');

// ─── Config ──────────────────────────────────────────────────────────────────

const args         = process.argv.slice(2);
const TOP_K        = parseInt(args[args.indexOf('--top')   + 1] || '3', 10);
const THRESHOLD    = parseFloat(args[args.indexOf('--thresh') + 1] || '0.55');
const EMBED_MODEL  = process.env.GEMINI_EMBED_MODEL || 'gemini-embedding-001';
const RATE_DELAY   = 700; // ms between embed calls (free tier: 100/min)

const DATA_DIR     = path.join(process.cwd(), 'data');
const DATASET_FILE = path.join(DATA_DIR, 'eval_dataset.json');
const EMBEDDINGS_F = path.join(DATA_DIR, 'embeddings.json');
const RESULTS_FILE = path.join(DATA_DIR, 'eval_results.json');
const CHARTS_DIR   = path.join(DATA_DIR, 'eval_charts');

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── Maths ───────────────────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function round(n, dp = 4) {
  return Math.round(n * 10 ** dp) / 10 ** dp;
}

// ─── Embedding ───────────────────────────────────────────────────────────────

async function embedText(text) {
  const res = await ai.models.embedContent({
    model: EMBED_MODEL,
    contents: [{ parts: [{ text }] }],
  });
  return res.embeddings[0].values;
}

// ─── Retrieval ───────────────────────────────────────────────────────────────

function retrieve(queryEmbedding, corpus, topK) {
  const scored = corpus.map(rec => ({
    chunk_id:   rec.chunk_id,
    source:     rec.source,
    page:       rec.page,
    similarity: cosineSimilarity(queryEmbedding, rec.embedding),
  }));
  scored.sort((a, b) => b.similarity - a.similarity);
  return scored.slice(0, topK);
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log('='.repeat(60));
  console.log('  AskAI Retrieval Evaluation');
  console.log(`  top_k=${TOP_K}  threshold=${THRESHOLD}  model=${EMBED_MODEL}`);
  console.log('='.repeat(60));

  // Load dataset
  if (!fs.existsSync(DATASET_FILE)) {
    console.error('ERROR: data/eval_dataset.json not found.');
    process.exit(1);
  }
  const dataset = JSON.parse(fs.readFileSync(DATASET_FILE, 'utf-8'));
  console.log(`\nLoaded ${dataset.length} test questions.\n`);

  // Load corpus embeddings
  console.log('Loading embedding corpus...');
  const corpus = JSON.parse(fs.readFileSync(EMBEDDINGS_F, 'utf-8'));
  console.log(`Corpus: ${corpus.length} chunks loaded.\n`);

  // Ensure output dir
  if (!fs.existsSync(CHARTS_DIR)) fs.mkdirSync(CHARTS_DIR, { recursive: true });

  // ── Per-question evaluation ──────────────────────────────────────────────

  const results = [];

  for (let i = 0; i < dataset.length; i++) {
    const tc = dataset[i];
    const prefix = `[${String(i + 1).padStart(2)}/${dataset.length}]`;
    process.stdout.write(`${prefix} ${tc.question.slice(0, 55).padEnd(56)}`);

    let qEmbedding;
    try {
      qEmbedding = await embedText(tc.question);
    } catch (err) {
      console.log(`  ERROR: ${err.message}`);
      results.push({
        id:              tc.id,
        question:        tc.question,
        expected_topic:  tc.expected_topic,
        expected_source: tc.expected_source,
        error:           err.message,
        hit_at_1:        false,
        hit_at_k:        false,
        top_similarity:  0,
        avg_similarity:  0,
        retrieved:       false,
        top_chunks:      [],
      });
      await new Promise(r => setTimeout(r, RATE_DELAY));
      continue;
    }

    const topK   = retrieve(qEmbedding, corpus, TOP_K);
    const needle = tc.expected_source.toLowerCase();

    const hit1 = topK[0]?.source?.toLowerCase().includes(needle) ?? false;
    const hitK = topK.some(c => c.source?.toLowerCase().includes(needle));
    const topSim  = round(topK[0]?.similarity ?? 0);
    const avgSim  = round(topK.reduce((s, c) => s + c.similarity, 0) / (topK.length || 1));
    const fetched = topSim >= THRESHOLD;

    console.log(` hit@${TOP_K}=${hitK ? '✓' : '✗'}  sim=${topSim.toFixed(3)}  ${fetched ? 'RETRIEVED' : 'BELOW_THRESH'}`);

    results.push({
      id:              tc.id,
      question:        tc.question,
      expected_topic:  tc.expected_topic,
      expected_source: tc.expected_source,
      hit_at_1:        hit1,
      hit_at_k:        hitK,
      top_similarity:  topSim,
      avg_similarity:  avgSim,
      retrieved:       fetched,
      top_chunks:      topK.map(c => ({
        source:     c.source,
        page:       c.page,
        similarity: round(c.similarity),
      })),
    });

    await new Promise(r => setTimeout(r, RATE_DELAY));
  }

  // ── Aggregate Metrics ────────────────────────────────────────────────────

  const ok      = results.filter(r => !r.error);
  const n       = ok.length;

  const hitK     = ok.filter(r => r.hit_at_k).length;
  const hit1     = ok.filter(r => r.hit_at_1).length;
  const fetched  = ok.filter(r => r.retrieved).length;
  const failed   = n - fetched;
  const avgTop   = round(ok.reduce((s, r) => s + r.top_similarity, 0) / (n || 1));
  const avgAvg   = round(ok.reduce((s, r) => s + r.avg_similarity, 0) / (n || 1));

  // Per-topic breakdown
  const topics = {};
  for (const r of ok) {
    const t = r.expected_topic;
    if (!topics[t]) topics[t] = { total: 0, hit_k: 0, hit_1: 0, retrieved: 0, sim_sum: 0 };
    topics[t].total++;
    if (r.hit_at_k)  topics[t].hit_k++;
    if (r.hit_at_1)  topics[t].hit_1++;
    if (r.retrieved) topics[t].retrieved++;
    topics[t].sim_sum += r.top_similarity;
  }

  const perTopic = Object.entries(topics).map(([name, v]) => ({
    topic:         name,
    questions:     v.total,
    hit_at_k:      v.hit_k,
    hit_at_k_pct:  round(v.hit_k / v.total * 100, 1),
    hit_at_1:      v.hit_1,
    retrieved:     v.retrieved,
    avg_sim:       round(v.sim_sum / v.total),
  }));

  const summary = {
    total_questions:           results.length,
    successful_embed:          n,
    failed_embed:              results.filter(r => r.error).length,
    top_k:                     TOP_K,
    threshold:                 THRESHOLD,
    embedding_model:           EMBED_MODEL,
    hit_at_k:                  hitK,
    hit_at_k_rate_pct:         round(hitK / n * 100, 1),
    hit_at_1:                  hit1,
    hit_at_1_rate_pct:         round(hit1 / n * 100, 1),
    retrieved_above_threshold: fetched,
    failed_retrieval:          failed,
    retrieval_rate_pct:        round(fetched / n * 100, 1),
    avg_top_similarity:        avgTop,
    avg_avg_similarity:        avgAvg,
    per_topic:                 perTopic,
  };

  // ── Save Results ─────────────────────────────────────────────────────────

  const output = {
    meta: {
      timestamp:  new Date().toISOString(),
      top_k:      TOP_K,
      threshold:  THRESHOLD,
      embed_model: EMBED_MODEL,
    },
    summary,
    per_topic:  perTopic,
    results,
  };

  fs.writeFileSync(RESULTS_FILE, JSON.stringify(output, null, 2));

  // ── Print Summary ─────────────────────────────────────────────────────────

  console.log('\n' + '='.repeat(60));
  console.log('  RETRIEVAL EVALUATION RESULTS');
  console.log('='.repeat(60));
  console.log(`  Total questions        : ${results.length}`);
  console.log(`  Successful embeds      : ${n}`);
  console.log(`  Failed embeds          : ${results.filter(r => r.error).length}`);
  console.log('─'.repeat(60));
  console.log(`  Hit@${TOP_K} (expected doc in top ${TOP_K})  : ${hitK}/${n} = ${round(hitK/n*100,1)}%`);
  console.log(`  Hit@1 (expected doc is #1)     : ${hit1}/${n} = ${round(hit1/n*100,1)}%`);
  console.log(`  Retrieved (sim ≥ ${THRESHOLD})       : ${fetched}/${n} = ${round(fetched/n*100,1)}%`);
  console.log(`  Failed retrievals              : ${failed}/${n}`);
  console.log(`  Avg top-1 similarity           : ${avgTop}`);
  console.log('─'.repeat(60));
  console.log('  Per-topic breakdown:');
  for (const t of perTopic) {
    console.log(`    ${t.topic.padEnd(32)} hit@${TOP_K}=${t.hit_at_k}/${t.questions}  sim=${t.avg_sim}`);
  }
  console.log('='.repeat(60));
  console.log(`\nResults saved → data/eval_results.json`);
  console.log(`Run chart script → python scripts/eval-charts.py`);
}

main().catch(err => {
  console.error('\nFATAL:', err.message);
  process.exit(1);
});
