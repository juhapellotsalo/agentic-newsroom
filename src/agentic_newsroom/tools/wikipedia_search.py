"""
Wikipedia search tool for the Agentic Newsroom.

This module provides a wrapper around the Wikipedia API
for retrieving encyclopedia articles during research.
"""

from typing import List
from langchain_community.document_loaders import WikipediaLoader


def format_wikipedia_results(search_docs: List) -> str:
    """
    Format Wikipedia search results into a structured string.
    
    This function normalizes Wikipedia documents into a consistent 
    XML-like format for LLM consumption.
    
    Args:
        search_docs: List of Document objects from WikipediaLoader
    
    Returns:
        Formatted string with search results in <Document> tags
    """
    formatted_docs = []
    
    for doc in search_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        content = doc.page_content
        
        formatted_docs.append(
            f'<Document source="{source}" page="{page}"/>\n{content}\n</Document>'
        )
    
    return "\n\n---\n\n".join(formatted_docs)


def search_wikipedia(query: str, max_docs: int = 2) -> str:
    """
    Search Wikipedia and return formatted results.
    
    Args:
        query: Search query string
        max_docs: Maximum number of documents to return (default: 2)
    
    Returns:
        Formatted search results as a string
    
    Raises:
        Exception: If the search fails
    """
    try:
        loader = WikipediaLoader(query=query, load_max_docs=max_docs)
        search_docs = loader.load()
        return format_wikipedia_results(search_docs)
    except Exception as e:
        raise Exception(f"Error performing Wikipedia search: {str(e)}")


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
    test_query = "Socotra dragon tree"
    
    print(f"Searching Wikipedia for: {test_query}")
    print("=" * 80)
    
    try:
        results = search_wikipedia(test_query, max_docs=2)
        print(results)
    except Exception as e:
        print(f"Search failed: {e}")
