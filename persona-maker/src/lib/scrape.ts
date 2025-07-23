export type RedditItem = {
    id: string;
    type: 'comment' | 'post';
    subreddit: string;
    url: string;
    label: string;
    text: string;
};
  
export async function scrapeUserComments(username: string): Promise<RedditItem[]> {
    const items: RedditItem[] = [];
  
    try {
        const res = await fetch(`https://www.reddit.com/user/${username}/comments.json`);
        const json = await res.json();
        const comments = json?.data?.children ?? [];
    
        comments.forEach((item: any) => {
            const d = item.data;
    
            items.push({
            id: d.id,
            type: 'comment',
            subreddit: d.subreddit,
            url: `https://www.reddit.com${d.permalink}`,
            label: `Comment on ${d.link_title || d.link_permalink || 'a post'}`,
            text: d.body,
            });
        });
  
        return items;
    } catch (err){
        console.error("Reddit scrape error:", err);
        return [];
    }
}  