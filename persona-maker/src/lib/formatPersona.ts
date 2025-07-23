export function formatResponse(geminiOutput: string): string {
    let cleanText = geminiOutput;
    cleanText = cleanText.replace(/\*\*(.*?)\*\*/g, '$1');
    cleanText = cleanText.replace(/^\s*[\*\-]\s*/gm, '- ');
    cleanText = cleanText.replace(/\n{2,}/g, '\n\n');
    return cleanText.trim();
}