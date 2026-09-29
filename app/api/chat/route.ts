import { NextResponse } from 'next/server';
import { generateChatResponse } from '@/lib/gemini';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { message, history } = body;

    // Handle empty user messages gracefully
    if (!message || message.trim() === '') {
      return NextResponse.json(
        { error: 'Message is required.' },
        { status: 400 }
      );
    }

    // Call our clean function and pass the conversation history
    const answer = await generateChatResponse(message, history || []);

    // Return the generated answer to the frontend
    return NextResponse.json({ answer });
  } catch (error: any) {
    console.error('API Route Error:', error);
    // Handle API errors gracefully and return useful debugging details
    return NextResponse.json(
      { 
        error: 'Gemini API request failed',
        status: error.status || 500,
        message: error.message || 'Unknown error occurred'
      },
      { status: 500 }
    );
  }
}
