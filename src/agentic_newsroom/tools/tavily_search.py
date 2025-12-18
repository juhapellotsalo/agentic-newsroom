"""
Tavily web search tool for the Agentic Newsroom.

This module provides a wrapper around the Tavily search API
for performing web searches during research.
"""

import os
from typing import List, Dict, Any, Union
from tavily import TavilyClient

def perform_search(queries: List[str]) -> List[dict]:
     """Search Tavily and return raw results (url + title + content snippet)."""
     api_key = os.getenv("TAVILY_API_KEY")
     if not api_key:
         raise Exception("TAVILY_API_KEY environment variable is not set")
     tavily = TavilyClient(api_key=api_key)

     print(f"🔎 Searching for: {queries}")
     aggregated= []
     for query in queries:
        try:
            resp = tavily.search(query, search_depth="advanced", max_results=3)
            results = resp.get("results", [])
            aggregated.extend(results)
        except Exception as e:
            print(f"Search failed for '{query}': {e}")
     
     return aggregated

def perform_extract(urls: List[str]) -> List[dict]:
    """Scrape full content from URLs."""
    if not urls:
        return []
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise Exception("TAVILY_API_KEY environment variable is not set")
    tavily = TavilyClient(api_key=api_key)
    
    print(f"🌐 Extracting from {len(urls)} URLs...")
    try:
        # extract_depth="advanced" handles popups/layouts better if available, standard is fine too
        resp = tavily.extract(urls=urls)
        return resp.get("results", [])
    except Exception as e:
        print(f"Extract failed: {e}")
        return []

if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Load environment variables from project root
    project_root = Path(__file__).parent.parent.parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        print("Warning: .env file not found.")

    print("--- Testing Search ---")
    results = perform_search(["Dyatlov Pass incident theories"])
    print(f"Found {len(results)} results")
    if results:
        print(f"First result title: {results[0].get('title')}")

    print("\n--- Testing Extract ---")
    if results:
        url = results[0].get("url")
        if url:
            extracted = perform_extract([url])
            print(f"Extracted {len(extracted)} items")
            if extracted:
                 print(f"Content snippet: {extracted[0].get('raw_content', '')[:100]}...")
