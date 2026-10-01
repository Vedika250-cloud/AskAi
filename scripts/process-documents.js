const fs = require('fs');
const path = require('path');
const PDFParser = require('pdf2json');

// Directory paths
const MATERIAL_DIR = process.env.COURSE_MATERIAL_DIR || path.join(process.cwd(), 'data', 'course_material');
const OUTPUT_FILE = process.env.OUTPUT_FILE || path.join(process.cwd(), 'data', 'processed_chunks.json');

// Chunking configuration
const CHUNK_SIZE = 800;     // Target characters per chunk
const CHUNK_OVERLAP = 150;  // Characters of overlap between adjacent chunks

/**
 * Clean common extraction artifacts from raw PDF text:
 * - Ligatures (ﬁ, ﬂ, etc.)
 * - Unprintable / control characters
 * - Table of contents leader dots (....)
 * - Hyphenated words split across line breaks (e.g. "algo-\nrithm" -> "algorithm")
 * - Repeated and erratic whitespace
 */
function cleanExtractionArtifacts(text) {
  if (!text) return '';
  return text
    // Replace standard Unicode ligatures with normal characters
    .replace(/\uFB00/g, 'ff')
    .replace(/\uFB01/g, 'fi')
    .replace(/\uFB02/g, 'fl')
    .replace(/\uFB03/g, 'ffi')
    .replace(/\uFB04/g, 'ffl')
    // Remove control characters (keeping \t and \n)
    .replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, '')
    // Remove long runs of dots or underscores common in TOC / headers
    .replace(/\.{3,}/g, ' ')
    .replace(/_{3,}/g, ' ')
    // Rejoin words hyphenated at line-breaks (e.g. "net-\nwork" -> "network")
    .replace(/(\b\w+)-\s*\n\s*(\w+\b)/g, '$1$2')
    // Rejoin words with isolated hyphenation spaces (e.g. "con- nection" -> "connection")
    .replace(/(\b[a-zA-Z]{2,})-\s+([a-zA-Z]{2,}\b)/g, '$1$2')
    // Normalize excessive horizontal spaces and tabs into a single space
    .replace(/[ \t]+/g, ' ')
    // Normalize triple or more newlines to double newlines (paragraphs)
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * Extracts text from a PDF file page by page using pdf2json.
 * Silences internal parser warnings for clean terminal logging.
 * Resolves to an array of objects: { pageNumber: number, text: string }
 */
function extractPagesFromPdf(filePath) {
  return new Promise((resolve, reject) => {
    // Intercept pdf2json console warnings to keep CLI output clean
    const origWarn = console.warn;
    const origStderr = process.stderr.write;
    const origStdout = process.stdout.write;

    const filterOutput = (origFn) => function(chunk, encoding, callback) {
      const str = chunk ? chunk.toString() : '';
      if (
        str.startsWith('Warning:') ||
        str.includes('Setting up fake worker') ||
        str.includes('NOT valid form element') ||
        str.includes('undefined function')
      ) {
        if (typeof callback === 'function') callback();
        return true;
      }
      return origFn.apply(this, arguments);
    };

    process.stderr.write = filterOutput(origStderr);
    process.stdout.write = filterOutput(origStdout);
    console.warn = () => {};

    const restoreLogs = () => {
      process.stderr.write = origStderr;
      process.stdout.write = origStdout;
      console.warn = origWarn;
    };

    const pdfParser = new PDFParser(null, 1);

    pdfParser.on('pdfParser_dataError', (errData) => {
      restoreLogs();
      reject(new Error(errData?.parserError || 'Unknown PDF parsing error'));
    });

    pdfParser.on('pdfParser_dataReady', (pdfData) => {
      restoreLogs();
      const pages = [];
      const pdfPages = pdfData.Pages || [];

      for (let i = 0; i < pdfPages.length; i++) {
        const page = pdfPages[i];
        const pageNumber = i + 1; // 1-indexed page number
        let rawPageText = '';

        // In pdf2json, each page has a list of Texts, each containing text runs (R)
        for (const textItem of page.Texts || []) {
          for (const r of textItem.R || []) {
            try {
              rawPageText += decodeURIComponent(r.T) + ' ';
            } catch {
              rawPageText += (r.T || '') + ' ';
            }
          }
        }

        const cleanedPageText = cleanExtractionArtifacts(rawPageText);
        pages.push({
          pageNumber,
          text: cleanedPageText
        });
      }

      resolve(pages);
    });

    pdfParser.loadPDF(filePath);
  });
}

/**
 * Splits text into manageable, overlapping chunks while respecting word and sentence boundaries.
 * 
 * @param {string} text - Cleaned text from a page
 * @param {string} filename - Source filename for metadata and ID generation
 * @param {number} pageNum - Page number for metadata
 * @param {number} startChunkIndex - Starting chunk index for this document
 * @returns {Array<Object>} Array of chunk objects
 */
function createChunksFromPage(text, filename, pageNum, startChunkIndex) {
  const chunks = [];
  
  // Normalize source name for chunk IDs (e.g. "Deep Learning.pdf" -> "deep_learning")
  const baseName = path.basename(filename, path.extname(filename))
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');

  // If text is short enough to fit in a single chunk
  if (text.length <= CHUNK_SIZE) {
    if (text.length >= 30) {
      chunks.push({
        chunk_id: `${baseName}_${String(startChunkIndex).padStart(3, '0')}`,
        source: filename,
        page: pageNum,
        text: text
      });
    }
    return chunks;
  }

  let startIndex = 0;
  let chunkIndex = startChunkIndex;

  while (startIndex < text.length) {
    let endIndex = Math.min(startIndex + CHUNK_SIZE, text.length);

    // If not at the end of the text, seek a natural boundary (sentence end or space)
    if (endIndex < text.length) {
      const window = text.substring(startIndex, endIndex);
      
      // Look for a sentence boundary in the second half of the window
      const lastSentence = window.lastIndexOf('. ');
      if (lastSentence > CHUNK_SIZE * 0.5) {
        endIndex = startIndex + lastSentence + 1;
      } else {
        // Fallback: break at the last space so words are not cut in half
        const lastSpace = window.lastIndexOf(' ');
        if (lastSpace > CHUNK_SIZE * 0.5) {
          endIndex = startIndex + lastSpace;
        }
      }
    }

    const chunkContent = text.substring(startIndex, endIndex).trim();

    if (chunkContent.length >= 30) {
      chunks.push({
        chunk_id: `${baseName}_${String(chunkIndex).padStart(3, '0')}`,
        source: filename,
        page: pageNum,
        text: chunkContent
      });
      chunkIndex++;
    }

    // Stop if we have reached the end of the text
    if (endIndex >= text.length) {
      break;
    }

    // Advance sliding window with overlap
    const nextStart = endIndex - CHUNK_OVERLAP;
    if (nextStart <= startIndex) {
      startIndex = endIndex;
    } else {
      startIndex = nextStart;
    }
  }

  return chunks;
}

/**
 * Main document processing pipeline:
 * 1. Scans data/course_material for PDFs
 * 2. Extracts page-by-page text
 * 3. Cleans extraction artifacts
 * 4. Chunks text with metadata preservation
 * 5. Saves chunks locally to data/processed_chunks.json
 */
async function processDocuments() {
  console.log('====================================================');
  console.log('AskAI RAG Pipeline — Stage 1: Document Processing');
  console.log('====================================================\n');

  // Ensure course_material directory exists
  if (!fs.existsSync(MATERIAL_DIR)) {
    fs.mkdirSync(MATERIAL_DIR, { recursive: true });
    console.log(`Created course material directory: ${MATERIAL_DIR}`);
  }

  // Scan directory for PDF files
  const allFiles = fs.readdirSync(MATERIAL_DIR);
  const pdfFiles = allFiles.filter(file => file.toLowerCase().endsWith('.pdf'));

  if (pdfFiles.length === 0) {
    console.log(`No PDF files found in: ${MATERIAL_DIR}`);
    console.log('Please place your course PDF files inside that folder and run this script again.');
    return;
  }

  console.log(`Found ${pdfFiles.length} PDF file(s) in: ${MATERIAL_DIR}`);
  pdfFiles.forEach((file, idx) => console.log(`  ${idx + 1}. ${file}`));
  console.log('\nStarting text extraction and chunking...\n');

  const allProcessedChunks = [];
  let totalPagesProcessed = 0;

  for (const file of pdfFiles) {
    const filePath = path.join(MATERIAL_DIR, file);
    const startTime = Date.now();
    process.stdout.write(`Processing "${file}"... `);

    try {
      // 1. Extract text page-by-page
      const pages = await extractPagesFromPdf(filePath);
      totalPagesProcessed += pages.length;

      // 2. Clean & chunk each page
      let docChunkIndex = 1;
      let docChunksCount = 0;

      for (const page of pages) {
        if (!page.text || page.text.length < 20) {
          continue; // Skip blank or header-only pages
        }

        const pageChunks = createChunksFromPage(page.text, file, page.pageNumber, docChunkIndex);
        if (pageChunks.length > 0) {
          allProcessedChunks.push(...pageChunks);
          docChunkIndex += pageChunks.length;
          docChunksCount += pageChunks.length;
        }
      }

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`✓ (${pages.length} pages -> ${docChunksCount} chunks in ${elapsed}s)`);
    } catch (err) {
      console.log(`✗ Error: ${err.message}`);
    }
  }

  // Ensure output directory exists
  const outputDir = path.dirname(OUTPUT_FILE);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Save all chunks to local JSON file
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allProcessedChunks, null, 2), 'utf-8');

  console.log('\n====================================================');
  console.log('Processing Summary:');
  console.log(`  - Total Documents Processed : ${pdfFiles.length}`);
  console.log(`  - Total Pages Read          : ${totalPagesProcessed}`);
  console.log(`  - Total Chunks Created      : ${allProcessedChunks.length}`);
  console.log(`  - Output File Saved To      : ${OUTPUT_FILE}`);
  console.log('====================================================');

  if (allProcessedChunks.length > 0) {
    console.log('\nSample Chunk Output:');
    console.log(JSON.stringify(allProcessedChunks[0], null, 2));
  }
}

// Execute the pipeline
processDocuments().catch(err => {
  console.error('Fatal error during document processing:', err);
  process.exit(1);
});
