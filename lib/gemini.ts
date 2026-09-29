import { GoogleGenAI } from '@google/genai';

// Initialize the Google Gen AI SDK.
// By default, it automatically reads the GEMINI_API_KEY environment variable.
const ai = new GoogleGenAI({});

// Use a currently available stable Flash model
export const GEMINI_MODEL = 'gemini-3.5-flash';

// Helper for exponential backoff
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Sends a message to the Gemini API and returns the generated response.
 * Includes safe retry logic for temporary 503/UNAVAILABLE errors.
 * @param message The user's input string.
 * @param history The conversation history.
 * @returns The generated response string from Gemini.
 */
export async function generateChatResponse(message: string, history: { role: string, content: string }[] = []): Promise<string> {
  if (!message || message.trim() === '') {
    throw new Error('Message cannot be empty.');
  }

  // Map our simple history format to the format required by the Gemini API
  const contents = history.map((msg) => ({
    role: msg.role === 'ai' ? 'model' : 'user',
    parts: [{ text: msg.content }],
  }));

  // Append the current user message to the end of the history
  contents.push({
    role: 'user',
    parts: [{ text: message }],
  });

  const MAX_RETRIES = 3;
  let attempt = 0;

  while (attempt < MAX_RETRIES) {
    try {
      const response = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents: contents,
      });
      return response.text || '';
    } catch (error: any) {
      // Check if it's a 503 UNAVAILABLE error
      if (error.status === 503 || (error.message && error.message.includes('503'))) {
        attempt++;
        if (attempt >= MAX_RETRIES) {
          console.error(`Gemini API failed after ${MAX_RETRIES} attempts due to high demand.`);
          throw new Error('The Gemini API is currently unavailable due to high demand. Please try again later.');
        }
        // Exponential backoff: 1s, 2s, 4s...
        const backoffMs = Math.pow(2, attempt - 1) * 1000;
        console.warn(`Gemini API 503 UNAVAILABLE. Retrying in ${backoffMs}ms... (Attempt ${attempt}/${MAX_RETRIES})`);
        await delay(backoffMs);
      } else {
        // If it's not a 503, throw immediately (e.g. invalid key, 404, bad request)
        throw error;
      }
    }
  }

  return '';
}
