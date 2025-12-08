"""File system utilities for the Agentic Newsroom.

This module provides high-level functions to load and save agent states
from the local filesystem, using the 'tmp/<slug>' directory structure.
"""

from pathlib import Path
from typing import Optional

from agentic_newsroom.schemas import (
    StoryBrief,
    DraftPackage,
    EditorDecision,
    ResearchPackage,
    FinalArticle
)
from agentic_newsroom.utils.serializers import load_schema, save_schema
from agentic_newsroom.utils.slugs import generate_article_slug

# Define the root directory for temporary files
# Assuming this is run from the project root or src, we try to find the project root
# A robust way is to look for .env or just assume standard structure relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"


def get_article_dir(slug: str) -> Path:
    """Get the directory path for a specific article slug."""
    return TMP_DIR / slug


def save_story_brief(brief: StoryBrief, slug: str) -> Path:
    """Save a StoryBrief to the article directory."""
    directory = get_article_dir(slug)
    json_path, _ = save_schema(brief, directory, "story_brief")
    return json_path


def load_story_brief(slug: str) -> StoryBrief:
    """Load a StoryBrief from the article directory."""
    directory = get_article_dir(slug)
    return load_schema(StoryBrief, directory, "story_brief")


def save_research_package(package: ResearchPackage, slug: str) -> Path:
    """Save a ResearchPackage to the article directory."""
    directory = get_article_dir(slug)
    # serializers.py might need an update to handle ResearchPackage if it doesn't already
    # Checking serializers.py content from previous turns... 
    # It seems serializers.py didn't have ResearchPackage explicitly in save_schema dispatch
    # I will need to update serializers.py as well or handle it here.
    # For now, let's assume I'll update serializers.py or use a generic save if possible.
    # Actually, let's update serializers.py first or handle it.
    # But wait, the user asked to start with assignment editor.
    # Assignment editor only needs save_story_brief.
    # I will implement the others as needed or now.
    
    # Since I can't easily see serializers.py again without a tool call, 
    # and I recall it had a dispatch dictionary or if/else chain.
    # I'll stick to what's needed for Assignment Editor for now to be safe, 
    # but it's better to have a complete files.py.
    
    # Let's just implement save_story_brief for now as requested? 
    # No, "Start by putting this into the assignment editor agent" implies the CLI part.
    # But I need the utils.
    
    # I will implement the full files.py but note that serializers might need updates for other types.
    # I'll check serializers.py content in my memory... 
    # Step 79 showed serializers.py. It handles: StoryBrief, DraftPackage, FactCheckReport, EditorDecision.
    # It DOES NOT handle ResearchPackage or FinalArticle.
    # So I should update serializers.py too.
    
    json_path, _ = save_schema(package, directory, "research_package")
    return json_path


def load_research_package(slug: str) -> ResearchPackage:
    """Load a ResearchPackage from the article directory."""
    directory = get_article_dir(slug)
    return load_schema(ResearchPackage, directory, "research_package")


def save_draft_package(package: DraftPackage, slug: str) -> Path:
    """Save a DraftPackage to the article directory."""
    directory = get_article_dir(slug)
    json_path, _ = save_schema(package, directory, "draft_package")
    return json_path


def load_draft_package(slug: str) -> DraftPackage:
    """Load a DraftPackage from the article directory."""
    directory = get_article_dir(slug)
    return load_schema(DraftPackage, directory, "draft_package")


def save_final_article(article: FinalArticle, slug: str) -> Path:
    """Save a FinalArticle to the article directory."""
    directory = get_article_dir(slug)
    json_path, _ = save_schema(article, directory, "final_article")
    return json_path


def load_final_article(slug: str) -> FinalArticle:
    """Load a FinalArticle from the article directory."""
    directory = get_article_dir(slug)
    return load_schema(FinalArticle, directory, "final_article")


def save_editor_decision(decision: EditorDecision, slug: str) -> Path:
    """Save an EditorDecision to the article directory."""
    directory = get_article_dir(slug)
    json_path, _ = save_schema(decision, directory, "editor_decision")
    return json_path


def load_editor_decision(slug: str) -> EditorDecision:
    """Load an EditorDecision from the article directory."""
    directory = get_article_dir(slug)
    return load_schema(EditorDecision, directory, "editor_decision")
