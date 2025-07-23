import {GoogleGenAI} from "@google/genai";
import type { RedditItem } from "./scrape";

const MODEL='gemini-2.5-flash'
const API_KEY = process.env.NEXT_PUBLIC_GEMINI_API_KEY
const ai = new GoogleGenAI({apiKey:API_KEY})

export function buildPrompt(data: RedditItem[], maxItems = 25): string {

    let prompt = `You are an expert behavioral analyst.
            Given the following Reddit activity (posts and comments), generate a detailed user persona.
            For each trait (interest, tone, profession guess, etc.), cite the ID or snippet of the specific post/comment used.
            Format clearly: list each trait followed by the citation.
  
            Reddit Activity:
            `;
  
    data.slice(0, maxItems).forEach((item, i) => {
        const snippet = item.text.length > 500 ? item.text.slice(0, 500) + '...' : item.text;
        prompt += `\n${i + 1}. ID: ${item.id}\n   Type: ${item.type}\n   Text: ${snippet}\n`;
        });
  
    return prompt;
}

export async function generatePersonaFromComments(items: RedditItem[]) {

    const demoPrompt = buildPrompt(items)
    const res = await ai.models.generateContent({
        model: MODEL,
        contents: demoPrompt,
        config:{
            temperature: 0.02,
        },
    });
  
    const reply = res.text || "Shit something went Wrong.."
    return reply
}
  