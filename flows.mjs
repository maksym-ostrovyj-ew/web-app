import 'dotenv/config';
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import { defineFlow } from '@genkit-ai/flow';
import { z } from 'zod';


const ai = genkit({
  plugins: [googleAI()], // читає GEMINI_API_KEY або GOOGLE_API_KEY з .env
});

// Мінімальний флоу: ехо/відповідь від моделі
export const helloFlow = defineFlow(
  {
    name: 'helloFlow',
    inputSchema: z.object({ question: z.string() }),
    outputSchema: z.string(),
  },
  async ({ question }) => {
    const res = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `Answer briefly: ${question}`,
    });

    return res?.outputText ??
      (typeof res?.text === 'function' ? res.text() : res?.text) ??
      'No output';
  }
);

// Щоб процес не завершувався одразу й був видимий у Dev UI:
console.log('Flows loaded. Waiting for Genkit Dev UI… (press Ctrl+C to exit)');
setInterval(() => {}, 1 << 30);
