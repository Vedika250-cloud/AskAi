const fs = require('fs');
const path = require('path');
const pdf = require('pdf-parse');

const MATERIAL_DIR = path.join(process.cwd(), 'data', 'course_material');
const OUTPUT_FILE = path.join(process.cwd(), 'data', 'processed_chunks.json');

// Configuration for our simple chunking strategy
const CHUNK_SIZE = 1000; // Characters per chunk
const CHUNK_OVERLAP = 200; // Characters to overlap between chunks to maintain context

/**
 * Custom page renderer to extract text while keeping track of page numbers.
 * pdf-parse natively extracts all text into a single string, but we want 
 * page metadata for our RAG pipeline.
 */
function render_page(pageData) {
  let render_options = {
    normalizeWhitespace: false,
    disableCombineTextItems: false
  };

  return pageData.getTextContent(render_options).then(function(textContent) {
    let lastY, text = '';
    for (let item of textContent.items) {
      if (lastY == item.transform[5] || !lastY) {
        text += item.str;
      } else {
        text += '\n' + item.str;
      }
      lastY = item.transform[5];
    }
    // We inject a special delimiter so we can split the pages later
    return text + '\n---PAGE_BREAK---\n';
  });
}

async function processDocuments() {
  if (!fs.existsSync(MATERIAL_DIR)) {
    fs.mkdirSync(MATERIAL_DIR, { recursive: true });
    console.log(`Created directory: ${MATERIAL_DIR}`);
  }

  const files = fs.readdirSync(MATERIAL_DIR).filter(file => file.toLowerCase().endsWith('.pdf'));
  
  if (files.length === 0) {
    console.log(`No PDF files found in ${MATERIAL_DIR}.`);
    console.log('Please place your course PDFs in the directory and run this script again.');
    return;
  }

  let allChunks = [];
  console.log(`Found ${files.length} PDF(s). Starting extraction...`);

  for (const file of files) {
    const filePath = path.join(MATERIAL_DIR, file);
    console.log(`\nProcessing: ${file}`);
    
    try {
      const dataBuffer = fs.readFileSync(filePath);
      
      // Parse the PDF with our custom page renderer
      const data = await pdf(dataBuffer, { pagerender: render_page });
      
      // Split the text back into pages using our delimiter
      const pages = data.text.split('\n---PAGE_BREAK---\n');
      
      // Process each page
      for (let i = 0; i < pages.length; i++) {
        const pageNum = i + 1;
        const rawText = pages[i];
        
        // Clean obvious extraction artifacts
        // 1. Replace multiple spaces/newlines with a single space
        // 2. Remove null bytes or weird unprintable chars
        let cleanText = rawText
          .replace(/\s+/g, ' ') 
          .replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, '')
          .trim();
          
        if (cleanText.length < 10) continue; // Skip empty or nearly empty pages

        // Chunking strategy: Sliding window over the clean text
        let startIndex = 0;
        let chunkIndex = 0;
        
        while (startIndex < cleanText.length) {
          let endIndex = Math.min(startIndex + CHUNK_SIZE, cleanText.length);
          let chunkText = cleanText.substring(startIndex, endIndex);
          
          // Avoid cutting words in half by shrinking the chunk to the last space
          if (endIndex < cleanText.length) {
            const lastSpace = chunkText.lastIndexOf(' ');
            if (lastSpace > CHUNK_SIZE / 2) { 
              endIndex = startIndex + lastSpace;
              chunkText = cleanText.substring(startIndex, endIndex);
            }
          }

          allChunks.push({
            chunk_id: `${file.replace('.pdf', '')}_p${pageNum}_c${chunkIndex}`,
            source: file,
            page: pageNum,
            text: chunkText.trim()
          });
          
          // Advance the window, taking overlap into account
          startIndex = endIndex - CHUNK_OVERLAP;
          
          // Safeguard to prevent infinite loops if overlap logic fails
          if (startIndex >= endIndex) {
            startIndex = endIndex;
          }
          
          chunkIndex++;
        }
      }
      console.log(`✓ Successfully chunked ${file}`);
    } catch (error) {
      console.error(`✗ Error processing ${file}:`, error.message);
    }
  }

  // Save the result to a local JSON file
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allChunks, null, 2));
  console.log(`\n=========================================`);
  console.log(`Extraction Complete!`);
  console.log(`Generated ${allChunks.length} total chunks from ${files.length} document(s).`);
  console.log(`Saved chunks to: ${OUTPUT_FILE}`);
  console.log(`=========================================`);
}

processDocuments();
