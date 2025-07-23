export function parseRedditUrl(url: string): { type: 'user' | null; username?: string } {
    try {
        const parsed = new URL(url);
        const path = parsed.pathname;
    
        if (path.startsWith("/user/")) {
            const username = path.split("/")[2];
            return { type: 'user', username };
        }
    
        return { type: null };
        } catch (err) {
            return { type: null };
        }
    }