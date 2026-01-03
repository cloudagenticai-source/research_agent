import os
from serpapi import GoogleSearch

def search_web(query: str, num_results: int = 5) -> list[dict]:
    """
    Perform a web search using SerpAPI.
    
    Args:
        query: The search query string.
        num_results: Number of results to return (default: 5).
        
    Returns:
        List of dictionaries containing search results.
    
    Raises:
        RuntimeError: If SERPAPI_API_KEY is missing or if the API call fails.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY environment variable is not set.")

    params = {
        "q": query,
        "num": num_results,
        "api_key": api_key,
        "engine": "google"
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
    except Exception as e:
        raise RuntimeError(f"SerpAPI call failed: {str(e)}")

    if "error" in results:
        raise RuntimeError(f"SerpAPI returned error: {results['error']}")

    organic_results = results.get("organic_results", [])
    parsed_results = []

    for result in organic_results[:num_results]:
        parsed_results.append({
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("snippet"),
            "source": result.get("source"),
            "position": result.get("position")
        })

    return parsed_results
