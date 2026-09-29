const { GoogleGenAI } = require('@google/genai'); try { const ai = new GoogleGenAI({}); console.log('SDK initialized successfully.'); } catch (e) { console.error('Initialization error:', e); }
