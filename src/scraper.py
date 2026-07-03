import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    try:
        # Send an HTTP request to the webpage URL
        response = requests.get(url, timeout=10)
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from all paragraph (<p>) tags on the webpage
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])

        return article_text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Code block to test the scraper locally
if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Main_Page"
    
    print("Testing Scraper Output (Showing first 500 characters):")
    print(scrape_article(test_url)[:500])