import requests
from bs4 import BeautifulSoup

def scrape_article_text(url):
    """
    Extracts main textual content from a news article URL.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title (tries standard h1 or title tags)
        title = ""
        if soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)
            
        # Extract article body paragraphs
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
        
        if not article_text:
            return None, "Could not extract readable article text from the provided URL."
            
        full_content = f"{title} {article_text}".strip()
        return full_content, None

    except Exception as e:
        return None, f"Error fetching URL: {str(e)}"

# --- Quick Test ---
if __name__ == "__main__":
    test_url = input("Enter a news article URL to test: ")
    text, error = scrape_article_text(test_url)
    
    if error:
        print(f"\n❌ {error}")
    else:
        print("\n✅ Successfully Scraped Article!")
        print(f"Sample Text: {text[:300]}...")