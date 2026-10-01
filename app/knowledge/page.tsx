import { readFileSync } from 'fs';
import { join } from 'path';
import Link from 'next/link';

// ─── Data Layer (server-side only, no API keys needed) ─────────────────────

type ChunkRecord = {
  chunk_id: string;
  source: string;
  page: number;
  text: string;
};

type DocStat = {
  name: string;
  pages: number;
  chunks: number;
};

type KnowledgeData = {
  totalDocs: number;
  totalChunks: number;
  embeddingsIndexed: number;
  embeddingDimensions: number;
  docs: DocStat[];
  topics: string[];
};

function loadKnowledgeData(): KnowledgeData | null {
  try {
    const chunksPath = join(process.cwd(), 'data', 'processed_chunks.json');
    const chunks: ChunkRecord[] = JSON.parse(readFileSync(chunksPath, 'utf-8'));

    // Per-document stats
    const docMap = new Map<string, { pages: Set<number>; chunks: number }>();
    for (const chunk of chunks) {
      if (!docMap.has(chunk.source)) {
        docMap.set(chunk.source, { pages: new Set(), chunks: 0 });
      }
      const entry = docMap.get(chunk.source)!;
      entry.pages.add(chunk.page);
      entry.chunks++;
    }

    const docs: DocStat[] = Array.from(docMap.entries())
      .map(([name, { pages, chunks: c }]) => ({ name, pages: pages.size, chunks: c }))
      .sort((a, b) => a.name.localeCompare(b.name));

    // Derive topic names from filenames (strip .pdf, replace _ with space)
    const topics = docs.map(d =>
      d.name.replace(/\.pdf$/i, '').replace(/_/g, ' ')
    );

    // Embedding stats
    let embeddingsIndexed = 0;
    let embeddingDimensions = 0;
    try {
      const embPath = join(process.cwd(), 'data', 'embeddings.json');
      const embs: { chunk_id: string; embedding: number[] }[] =
        JSON.parse(readFileSync(embPath, 'utf-8'));
      embeddingsIndexed = embs.length;
      embeddingDimensions = embs[0]?.embedding?.length ?? 0;
    } catch {
      // embeddings file optional for display
    }

    return {
      totalDocs: docs.length,
      totalChunks: chunks.length,
      embeddingsIndexed,
      embeddingDimensions,
      docs,
      topics,
    };
  } catch {
    return null;
  }
}

// ─── Stat Card ──────────────────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white border border-neutral-200 rounded-xl px-5 py-4">
      <p className="text-2xl font-semibold text-neutral-900 tabular-nums">{value}</p>
      <p className="text-sm text-neutral-600 mt-0.5">{label}</p>
      {sub && <p className="text-xs text-neutral-400 mt-0.5">{sub}</p>}
    </div>
  );
}

// ─── Page ───────────────────────────────────────────────────────────────────

export default function KnowledgePage() {
  const data = loadKnowledgeData();

  return (
    <div className="min-h-screen bg-stone-50 text-neutral-900 antialiased">
      <div className="max-w-3xl mx-auto px-6 py-12">

        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-800 mb-6 transition-colors"
          >
            ← Back to AskAI
          </Link>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-lg bg-neutral-900 flex items-center justify-center flex-shrink-0">
              <span className="text-white text-sm font-bold">A</span>
            </div>
            <h1 className="text-xl font-semibold text-neutral-900">Course Knowledge Base</h1>
          </div>
          <p className="text-sm text-neutral-500 ml-11">
            Documents indexed by the RAG pipeline and available for retrieval.
          </p>
        </div>

        {data === null ? (
          <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
            <strong>Knowledge base not found.</strong> Run{' '}
            <code className="font-mono bg-red-100 px-1 rounded">npm run process-docs</code> and{' '}
            <code className="font-mono bg-red-100 px-1 rounded">npm run generate-embeddings</code> to
            build the index.
          </div>
        ) : (
          <>
            {/* Summary stats */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
              <StatCard label="Documents" value={data.totalDocs} />
              <StatCard label="Text chunks" value={data.totalChunks.toLocaleString()} sub="sliding window, 800 chars" />
              <StatCard
                label="Embeddings"
                value={data.embeddingsIndexed.toLocaleString()}
                sub={data.embeddingDimensions ? `${data.embeddingDimensions.toLocaleString()}-dim vectors` : undefined}
              />
              <StatCard label="Topics" value={data.topics.length} />
            </div>

            {/* Topic tags */}
            <section className="mb-8">
              <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-widest mb-3">
                Available Topics
              </h2>
              <div className="flex flex-wrap gap-2">
                {data.topics.map(topic => (
                  <span
                    key={topic}
                    className="px-3 py-1 rounded-full text-sm font-medium bg-white border border-neutral-200 text-neutral-700"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            </section>

            {/* Document breakdown */}
            <section>
              <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-widest mb-3">
                Document Breakdown
              </h2>
              <div className="rounded-xl border border-neutral-200 overflow-hidden bg-white">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-neutral-200 bg-neutral-50">
                      <th className="text-left px-5 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wide">
                        Document
                      </th>
                      <th className="text-right px-5 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wide">
                        Pages
                      </th>
                      <th className="text-right px-5 py-3 text-xs font-semibold text-neutral-500 uppercase tracking-wide">
                        Chunks
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.docs.map((doc, i) => (
                      <tr
                        key={doc.name}
                        className={i < data.docs.length - 1 ? 'border-b border-neutral-100' : ''}
                      >
                        <td className="px-5 py-3.5 text-neutral-800 font-medium">
                          {doc.name}
                        </td>
                        <td className="px-5 py-3.5 text-right text-neutral-500 tabular-nums">
                          {doc.pages}
                        </td>
                        <td className="px-5 py-3.5 text-right text-neutral-500 tabular-nums">
                          {doc.chunks}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="border-t border-neutral-200 bg-neutral-50">
                      <td className="px-5 py-3 text-xs font-semibold text-neutral-500">Total</td>
                      <td className="px-5 py-3 text-right text-xs font-semibold text-neutral-700 tabular-nums">
                        {data.docs.reduce((s, d) => s + d.pages, 0)}
                      </td>
                      <td className="px-5 py-3 text-right text-xs font-semibold text-neutral-700 tabular-nums">
                        {data.totalChunks.toLocaleString()}
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </section>

            {/* Pipeline note */}
            <p className="mt-8 text-xs text-neutral-400 leading-relaxed border-t border-neutral-200 pt-5">
              <strong className="text-neutral-500">RAG pipeline:</strong>{' '}
              PDF → text extraction (pdf2json) → sliding-window chunking (800 chars, 150 overlap) →
              embedding generation (gemini-embedding-001, {data.embeddingDimensions.toLocaleString()} dimensions) →
              cosine-similarity retrieval → grounded generation (gemini-3.5-flash).
            </p>
          </>
        )}
      </div>
    </div>
  );
}
