"""
Tavily web search tool for the Agentic Newsroom.

This module provides a wrapper around the Tavily search API
for performing web searches during research.
"""

import json
from typing import List, Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults


def format_tavily_results(search_docs: Any) -> str:
    """
    Format Tavily search results into a structured string.
    
    This function handles various return formats from the Tavily API
    and normalizes them into a consistent XML-like format for LLM consumption.
    
    Args:
        search_docs: Raw search results from Tavily (can be string, list, or dict)
    
    Returns:
        Formatted string with search results in <Document> tags
    """
    formatted_docs = []
    
    # Case 1: It's a string (maybe JSON?)
    if isinstance(search_docs, str):
        try:
            search_docs = json.loads(search_docs)
        except:
            # If it's just a raw string, wrap it
            formatted_docs.append(f"<Document>\n{search_docs}\n</Document>")
            search_docs = []  # Clear so we don't loop
    
    # Case 2: It's a list
    if isinstance(search_docs, list):
        for doc in search_docs:
            if isinstance(doc, dict):
                # Standard case: dict with url/content
                url = doc.get("url", "unknown")
                content = doc.get("content", str(doc))
                formatted_docs.append(f'<Document href="{url}"/>\n{content}\n</Document>')
            elif isinstance(doc, str):
                # Fallback: list of strings
                formatted_docs.append(f"<Document>\n{doc}\n</Document>")
            else:
                # Fallback: unknown object
                formatted_docs.append(f"<Document>\n{str(doc)}\n</Document>")
    
    return "\n\n---\n\n".join(formatted_docs)


def search_web(query: str, max_results: int = 3) -> str:
    """
    Perform a web search using Tavily and return formatted results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 3)
    
    Returns:
        Formatted search results as a string
    
    Raises:
        Exception: If the search fails
    """
    tavily_tool = TavilySearchResults(max_results=max_results)
    
    try:
        search_docs = tavily_tool.invoke({"query": query})
        return format_tavily_results(search_docs)
    except Exception as e:
        raise Exception(f"Error performing web search: {str(e)}")


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Load environment variables from project root
    # This file is in src/agentic_newsroom/tools/, so go up 3 levels to reach project root
    project_root = Path(__file__).parent.parent.parent.parent
    env_path = project_root / '.env'
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✓ Loaded .env from: {env_path}\n")
    else:
        print(f"⚠️  Warning: .env file not found at {env_path}\n")
    
    # Example search query for testing
    test_query = "ancient Mesopotamian clay tablet maps 1899 discovery"
    
    print(f"Searching for: {test_query}")
    print("=" * 80)
    
    try:
        results = search_web(test_query, max_results=3)
        print(results)
    except Exception as e:
        print(f"Search failed: {e}")
