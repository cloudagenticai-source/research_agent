import requests
from bs4 import BeautifulSoup

def fetch_page(url: str, timeout: int = 15) -> dict:
    """
    Fetch a URL and extract its main text content.
    
    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
        
    Returns:
        dict: Contains url, title, text, status_code, content_type.
              Returns status_code=None if connection failed.
    """
    headers = {
        'User-Agent': 'ResearchAgent/1.0 (Python/3)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        # Simple error handling for connection/http errors
        return {
            "url": url,
            "title": None,
            "text": None,
            "status_code": None,
            "content_type": None
        }

    status_code = response.status_code
    content_type = response.headers.get('Content-Type', '')

    content_type = response.headers.get('Content-Type', '')

    # Check for HTML (Header OR Content Sniffing)
    is_html = 'text/html' in content_type
    if not is_html and ('<html' in response.text.lower() or '<body' in response.text.lower()):
        is_html = True

    if not is_html:
         return {
            "url": url,
            "title": None,
            "text": None,
            "status_code": status_code,
            "content_type": content_type
        }
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
        element.decompose()

    # Extract title
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    # Helper to clean text
    def clean_text(raw_text):
        if not raw_text: return ""
        lines = (line.strip() for line in raw_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)

    # Strategy 1: Standard extraction
    text = soup.get_text(separator=' ')
    cleaned_text = clean_text(text)

    # Strategy 2: Body fallback if too short
    if len(cleaned_text) < 500 and soup.body:
        body_text = soup.body.get_text(separator=' ')
        cleaned_body = clean_text(body_text)
        if len(cleaned_body) > len(cleaned_text):
             cleaned_text = cleaned_body

    # Truncate
    if len(cleaned_text) > 8000:
        cleaned_text = cleaned_text[:8000] + "..."

    return {
        "url": url,
        "title": title,
        "text": cleaned_text,
        "status_code": status_code,
        "content_type": content_type
    }
