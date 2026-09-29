# AskAI — AI-Powered Student Academic Assistant

This project is built using Next.js, React, Tailwind CSS, and the Google Gemini API. It serves as a modern AI academic assistant for students, currently in its first stage of development. 

## Folder Structure

The project has been structured minimally to avoid over-engineering while providing a solid foundation for future RAG (Retrieval-Augmented Generation) enhancements.

- `app/`: Contains Next.js App Router files, including the main page and API routes.
- `components/`: For reusable React components (e.g., UI elements, chat boxes).
- `lib/`: Utility functions, helper modules, and Gemini API setup files.
- `data/`: A directory designated for raw knowledge base files and course material (for future RAG processing).
- `public/`: Static assets such as images and fonts.

## Installation

1. Ensure you have Node.js installed.
2. Clone or download this project.
3. In the project root folder, run the following command to install all required dependencies:

```bash
npm install
```

## Running the Development Server

1. Duplicate the `.env.example` file and rename it to `.env.local`.
2. Add your Google Gemini API key to the `.env.local` file:
   `GEMINI_API_KEY=your_api_key_here`
3. Start the Next.js development server by running:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Files Created

During this initial setup, the following key directories and files were created/configured:
- `app/` (including `layout.tsx`, `page.tsx`, and `globals.css`)
- `components/` (empty, ready for components)
- `lib/` (empty, ready for utility functions)
- `data/` (empty, ready for course data)
- `public/` (contains static Next.js SVGs)
- `.env.example` (containing `GEMINI_API_KEY=`)
- `README.md` (this file)
- `package.json`, `tailwind.config.ts`, `tsconfig.json`, and other standard Next.js configuration files.

*Note: RAG functionalities such as text chunking, embeddings, and vector similarity search are not yet implemented in this phase.*
