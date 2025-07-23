'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import GlassCard from '@/components/GlassCard';
import { parseRedditUrl } from '@/lib/parse';
import { scrapeUserComments, RedditItem } from '@/lib/scrape';
import { generatePersonaFromComments } from '@/lib/generatePersona';
import { formatResponse } from '@/lib/formatPersona';

export default function Home() {
    const [redditUrl, setRedditUrl] = useState('');
    const [jsonData, setJsonData] = useState<RedditItem[]>([]);
    const [persona, setPersona] = useState('');
    const [loading, setLoading] = useState(false);
    const [fileReady, setFileReady] = useState(false);
    const [filename, setFilename] = useState('');

    async function handleScrapeAndGenerate(e: React.FormEvent) {
        e.preventDefault();
        setLoading(true);
        setPersona('');
        setJsonData([]);

        const parsed = parseRedditUrl(redditUrl);
        if (parsed.type !== 'user' || !parsed.username) {
        alert('Only Reddit user profile URLs are supported for now.');
        setLoading(false);
        return;
        }

        const items = await scrapeUserComments(parsed.username);
        setJsonData(items);

        const personaText = await generatePersonaFromComments(items);
        setPersona(personaText);
        setFilename(`${parsed.username}_persona.txt`);
        setFileReady(true);
        setLoading(false);
    }

    const formatted_persona = formatResponse(persona);

    function handleDownload() {
        const blob = new Blob([formatted_persona], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    function handleDownloadJSON(data: RedditItem[]) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `reddit_data_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    return (
        
        <main className="relative flex flex-col items-center justify-center min-h-screen bg-gray-900 overflow-hidden text-gray-100">
        <div className="absolute inset-0 z-0 flex">
            <img src="/demon-bg-0.jpeg" alt="" className="flex-1 object-cover h-full w-full" />
            <img src="/demon-bg-3.jpeg" alt="" className="flex-1 object-cover h-full w-full" />
            <img src="/demon-bg-1.jpeg" alt="" className="flex-1 object-cover h-full w-full" />
            <img src="/demon-bg-2.jpeg" alt="" className="flex-1 object-cover h-full w-full" />
        </div>

        <div className="relative z-10 flex flex-col items-center w-full px-4">
        <motion.h1
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.7, ease: 'backOut' }}
        className="text-2xl font-extrabold mb-2"
        >
        Upload your reddit url below and get the user persona.
        </motion.h1>

        <form onSubmit={handleScrapeAndGenerate} className="w-full max-w-xl mb-10">
        <GlassCard>
            <div className="flex flex-col md:flex-row items-center gap-4">
            <input
                type="url"
                className="flex-1 bg-transparent border border-white/20 px-4 py-3 rounded-lg text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-white/30 transition"
                placeholder="Enter Reddit profile URL..."
                value={redditUrl}
                onChange={(e) => setRedditUrl(e.target.value)}
                required
            />
            <motion.button
                type="submit"
                className="mt-4 md:mt-0 bg-gray-700 hover:bg-gray-600 transition px-6 py-3 rounded-lg font-semibold"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
            >
                Go
            </motion.button>
            </div>
        </GlassCard>
        </form>

        {loading && (
        <motion.div
            className="flex flex-col items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
        >
        <motion.img
            src="/loading-img.jpeg"
            alt="Loading spinner"
            className="w-16 h-16 mb-3 rounded-full"
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
        />
        
            <p className="mt-4 text-white text-2xl">Generating persona...</p>
        </motion.div>
        )}

        {persona && (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="relative w-full max-w-2xl z-10"
        >
            <GlassCard>
            <h2 className="text-2xl font-semibold mb-4">User Persona</h2>

            {/* Scrollable content box inside the GlassCard */}
            <div className="flex-1 overflow-y-auto pr-2 max-h-[50vh]">
                <pre className="whitespace-pre-wrap text-gray-100 text-sm">
                {formatted_persona}
                </pre>
            </div>

            {/* Buttons are always going to be at the bottom of the card */}
            <div className="mt-5 flex flex-col sm:flex-row gap-4 justify-center">
                {fileReady && (
                <motion.button
                    className="bg-gray-700 hover:bg-gray-600 px-5 py-3 rounded-lg font-medium"
                    onClick={handleDownload}
                    whileHover={{ y: -2 }}
                    whileTap={{ scale: 0.95 }}
                >
                    Download Persona
                </motion.button>
                )}
                {jsonData.length > 0 && (
                <motion.button
                    className="bg-gray-700 hover:bg-gray-600 px-5 py-3 rounded-lg font-medium"
                    onClick={() => handleDownloadJSON(jsonData)}
                    whileHover={{ y: -2 }}
                    whileTap={{ scale: 0.95 }}
                >
                    Download Comments
                </motion.button>
                )}
            </div>
            </GlassCard>
            <div className='h-10'></div>
        </motion.div>
        )}
        </div>
        </main>
        )
    }