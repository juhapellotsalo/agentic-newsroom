"""Utilities for generating article slugs."""

import re


def generate_article_slug(news_tip: str, max_words: int = 5) -> str:
    """Generate a clean filesystem-safe slug from a news tip.

    Args:
        news_tip: The news tip or article topic string
        max_words: Maximum number of words to include (default: 5)

    Returns:
        A filesystem-safe slug (lowercase, underscores, alphanumeric only)

    Example:
        >>> generate_article_slug("Cloudflare experienced a major global outage")
        'cloudflare_experienced_a_major_global'
    """
    # Take first N words
    words = news_tip.split()[:max_words]
    slug = "_".join(words).lower()
    # Remove non-alphanumeric characters except underscores
    slug = re.sub(r'[^a-z0-9_]', '', slug)
    return slug
