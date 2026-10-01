/**
 * AskAI — End-to-End RAG Pipeline CLI Test with Student-Friendly Answer Modes
 *
 * Full flow:
 * Student Question -> Embedding -> Retrieval -> Top Chunks -> Prompt Construction -> Gemini -> Grounded Answer + Sources
 *
 * Modes:
 *   1. simple   — Simple Explanation (plain language + small everyday example)
 *   2. detailed — Detailed Explanation (step-by-step walkthrough + examples)
 *   3. exam     — Exam Answer (concise, structured format for college exams)
 *   4. viva     — Viva Preparation (spoken summary + 3 viva questions & answers)
 *
 * Usage:
 *   node scripts/test-rag.js -q "What is overfitting?" -m simple
 *   node scripts/test-rag.js -q "What is overfitting?" -m exam
 *   node scripts/test-rag.js -q "What is overfitting?" -m viva
 *   node scripts/test-rag.js -q "What is overfitting?" -m detailed
 */

'use strict';

const fs = require('fs');
const path = require('path');
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

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('Error: GEMINI_API_KEY is not set in .env.local');
  process.exit(1);
}

const EMBEDDINGS_FILE = path.join(process.cwd(), 'data', 'embeddings.json');
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'gemini-embedding-001';
const CHAT_MODEL = process.env.GEMINI_CHAT_MODEL || 'gemini-3.5-flash';

const ai = new GoogleGenAI({ apiKey });

// ─── Mode Definitions ──────────────────────────────────────────────────────────
const MODE_INSTRUCTIONS = {
  simple: {
    label: 'Simple Explanation',
    instruction: `RESPONSE STYLE: Simple Explanation
- Explain this concept in very simple, plain language without heavy technical jargon.
- Include a clear, relatable everyday analogy or small example.
- Focus on intuition: explain what it means and why it matters as if explaining to a beginner student.`,
  },
  detailed: {
    label: 'Detailed Explanation',
    instruction: `RESPONSE STYLE: Detailed Explanation
- Explain the concept thoroughly, step-by-step, covering key mechanisms, principles, and nuances.
- Include concrete examples or practical use cases where useful.
- Break down technical terminology systematically so the student understands both the 'how' and the 'why'.`,
  },
  exam: {
    label: 'Exam Answer',
    instruction: `RESPONSE STYLE: Exam Answer
- Give a concise, structured answer suitable for a university or college examination.
- Organize with clear examination headings:
  1. **Definition / Overview** (concise, high-impact definition)
  2. **Key Points / Characteristics** (bullet points with bold terms)
  3. **Core Mechanism / Principle** (structured breakdown)
  4. **Summary / Key Takeaway**
- Optimize for clarity, precision, and quick memorization for exams.`,
  },
  viva: {
    label: 'Viva Preparation',
    instruction: `RESPONSE STYLE: Viva Preparation
- First, provide a crisp 2-to-3 sentence spoken summary of the concept that a student can speak confidently in front of an oral examiner.
- Then, provide exactly 3 likely viva/oral examination questions that professors commonly ask about this topic.
- For each question, provide a concise, direct model answer (1-2 sentences) that the student can memorize and state aloud.
- Structure clearly as:
  ### 🎯 Core Concept for Viva
  (Crisp verbal summary)

  ### ❓ Likely Viva Questions & Model Answers
  1. **Q:** [Likely examiner question]
     **A:** [Direct spoken model answer]
  2. **Q:** [Likely examiner question]
     **A:** [Direct spoken model answer]
  3. **Q:** [Likely examiner question]
     **A:** [Direct spoken model answer]`,
  },
};

// ─── Cosine Similarity ─────────────────────────────────────────────────────────
function cosineSimilarity(vecA, vecB) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ─── Step 1 & 2: Retrieval ────────────────────────────────────────────────────
async function retrieveChunks(question, topK = 3, threshold = 0.55) {
  if (!fs.existsSync(EMBEDDINGS_FILE)) {
    throw new Error(`Embeddings file not found: ${EMBEDDINGS_FILE}. Run node scripts/generate-embeddings.js first.`);
  }

  const embRes = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: question,
  });

  const qVector = embRes.embeddings?.[0]?.values;
  if (!qVector) throw new Error('Failed to embed question.');

  const stored = JSON.parse(fs.readFileSync(EMBEDDINGS_FILE, 'utf-8'));
  const scored = stored.map(c => ({
    chunk_id: c.chunk_id,
    source: c.source,
    page: c.page,
    text: c.text,
    similarity: Array.isArray(c.embedding) ? cosineSimilarity(qVector, c.embedding) : 0,
  }));

  scored.sort((a, b) => b.similarity - a.similarity);
  const relevant = scored.filter(c => c.similarity >= threshold).slice(0, topK);

  return {
    found: relevant.length > 0,
    chunks: relevant,
    topScore: scored[0]?.similarity || 0,
  };
}

// ─── Step 3: Prompt Construction ──────────────────────────────────────────────
const SYSTEM_INSTRUCTION = `You are AskAI, a knowledgeable, clear, and supportive academic assistant for students.

Your role is to help students learn by answering their questions using their course lecture slides and course documents.

CORE BEHAVIOR RULES:
1. Grounding in Course Context:
   - Answer primarily using the supplied RELEVANT COURSE MATERIAL CONTEXT.
   - Do NOT invent, assume, or hallucinate facts, algorithms, proofs, or claims not supported by the context.

2. Insufficient Context / Missing Information:
   - If the supplied course material does not contain enough information to answer the question, or if no relevant material was found, clearly state that the course material does not cover this topic in sufficient detail.

3. General Knowledge:
   - When helpful for educational clarity, or when course materials do not cover the topic, you may provide a simple, accurate explanation using general computer science/AI knowledge.
   - However, you MUST clearly distinguish general knowledge from course material (e.g. "Note: Beyond what is covered in your course slides...").

4. Citations & Sources:
   - NEVER fabricate, invent, or guess a source or page number.
   - Only cite sources and page numbers that appear in the provided context and directly supported your answer.
   - Whenever course material is used, conclude your response with a dedicated sources section formatted EXACTLY as follows:

📚 Sources:
* <filename> — Page <number>

   - If no course materials were relevant or used, do NOT output fake sources; state that no course sources were matched.

5. Student-Friendly Tone:
   - Keep answers well-structured, easy to understand, encouraging, and engaging.
   - Strictly follow the specific RESPONSE STYLE requested in the prompt.`;

function constructPrompt(question, chunks, mode = 'detailed') {
  const modeObj = MODE_INSTRUCTIONS[mode] || MODE_INSTRUCTIONS.detailed;

  if (!chunks || chunks.length === 0) {
    return `STUDENT QUESTION:
${question}

RELEVANT COURSE MATERIAL CONTEXT:
[No relevant course material was found in the indexed course documents for this question.]

TASK INSTRUCTIONS:
${modeObj.instruction}

ADDITIONAL RULE:
Inform the student that their course materials do not appear to cover this topic. You may provide a brief explanation using general knowledge, but clearly state that this explanation is based on general knowledge rather than their specific course materials. Do not cite any course documents.`;
  }

  const contextBlocks = chunks.map((c, i) => `---
[DOCUMENT #${i + 1}]
Source: ${c.source}
Page: ${c.page}
Excerpt:
${c.text}
`).join('\n');

  return `STUDENT QUESTION:
${question}

RELEVANT COURSE MATERIAL CONTEXT:
${contextBlocks}
---

TASK INSTRUCTIONS:
${modeObj.instruction}

GROUNDING & CITATION RULES:
1. Answer primarily using the excerpted course material above, tailored to the requested RESPONSE STYLE.
2. If course material is partial, supplement with general knowledge but explicitly distinguish it.
3. Conclude your response with the '📚 Sources:' block citing the specific sources and pages that were used.`;
}

// ─── Step 4: Full Pipeline Execution ──────────────────────────────────────────
async function runRAGPipeline(question, mode = 'detailed') {
  const modeObj = MODE_INSTRUCTIONS[mode] || MODE_INSTRUCTIONS.detailed;

  console.log('\n' + '═'.repeat(60));
  console.log(`Student Question : "${question}"`);
  console.log(`Selected Mode    : ${modeObj.label} (${mode})`);
  console.log('═'.repeat(60));

  // 1. Retrieval
  process.stdout.write('🔍 Retrieving course material chunks... ');
  const retrieval = await retrieveChunks(question, 3, 0.55);
  console.log(`done. (Found: ${retrieval.chunks.length} chunks, Top Score: ${(retrieval.topScore * 100).toFixed(2)}%)`);

  if (retrieval.chunks.length > 0) {
    console.log('\nRetrieved Chunks:');
    retrieval.chunks.forEach((c, i) => {
      console.log(`  ${i + 1}. ${c.source} — Page ${c.page} (${(c.similarity * 100).toFixed(2)}% similarity)`);
    });
  } else {
    console.log('\n(No chunks met the 55% similarity threshold)');
  }

  // 2. Prompt Construction with Mode Instructions
  const prompt = constructPrompt(question, retrieval.chunks, mode);

  // 3. Gemini Generation
  console.log(`\n🤖 Generating ${modeObj.label} with Gemini...\n`);
  
  let answer = '';
  const MAX_RETRIES = 3;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const res = await ai.models.generateContent({
        model: CHAT_MODEL,
        contents: prompt,
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
        },
      });
      answer = res.text || 'No answer generated.';
      break;
    } catch (err) {
      if ((err.status === 503 || err.status === 429) && attempt < MAX_RETRIES - 1) {
        const wait = Math.pow(2, attempt + 1) * 1000;
        console.warn(`  [Notice] Gemini API ${err.status} temporary spike. Retrying in ${wait / 1000}s...`);
        await new Promise(r => setTimeout(r, wait));
      } else {
        throw err;
      }
    }
  }

  console.log('─'.repeat(60));
  console.log(answer.trim());
  console.log('─'.repeat(60) + '\n');
}

// ─── CLI Entry ────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
let query = null;
let mode = 'detailed';

for (let i = 0; i < args.length; i++) {
  if ((args[i] === '-q' || args[i] === '--query') && args[i + 1]) {
    query = args[i + 1];
    i++;
  } else if ((args[i] === '-m' || args[i] === '--mode') && args[i + 1]) {
    const rawMode = args[i + 1].toLowerCase();
    if (['simple', 'detailed', 'exam', 'viva'].includes(rawMode)) {
      mode = rawMode;
    }
    i++;
  }
}

if (query) {
  runRAGPipeline(query, mode)
    .then(() => process.exit(0))
    .catch(err => {
      console.error('RAG Error:', err.message);
      process.exit(1);
    });
} else {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log('\nAskAI — Interactive End-to-End RAG Tester with Answer Modes');
  console.log('Modes available: simple, detailed, exam, viva');
  console.log('Type "exit" to quit.\n');

  const ask = () => {
    rl.question('Question: ', async input => {
      const q = input.trim();
      if (!q) { ask(); return; }
      if (q === 'exit' || q === 'quit') {
        rl.close();
        return;
      }
      try {
        await runRAGPipeline(q, mode);
      } catch (e) {
        console.error('Error:', e.message);
      }
      ask();
    });
  };
  ask();
}
