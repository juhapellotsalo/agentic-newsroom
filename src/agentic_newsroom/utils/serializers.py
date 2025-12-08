"""Serialization utilities for saving Pydantic schemas to disk."""

import json
from pathlib import Path
from typing import Union

from agentic_newsroom.schemas import (
    StoryBrief,
    DraftPackage,
    EditorDecision,
    ResearchPackage,
    FinalArticle,
)


def format_story_brief_md(brief: StoryBrief) -> str:
    """Format a StoryBrief as markdown.

    Args:
        brief: The StoryBrief to format

    Returns:
        Markdown-formatted string
    """
    md_parts = []

    md_parts.append("# Story Brief")
    md_parts.append("")
    md_parts.append("---")
    md_parts.append("")

    md_parts.append("## Topic")
    md_parts.append("")
    md_parts.append(brief.topic)
    md_parts.append("")

    md_parts.append("## Article Type")
    md_parts.append("")
    md_parts.append(brief.article_type)
    md_parts.append("")

    md_parts.append("## Angle")
    md_parts.append("")
    md_parts.append(brief.angle)
    md_parts.append("")

    md_parts.append(f"## Key Questions ({len(brief.key_questions)})")
    md_parts.append("")
    if brief.key_questions:
        for i, question in enumerate(brief.key_questions, 1):
            md_parts.append(f"{i}. {question}")
    else:
        md_parts.append("*No key questions listed*")
    md_parts.append("")

    return "\n".join(md_parts)


def format_draft_package_md(draft_pkg: DraftPackage) -> str:
    """Format a DraftPackage as markdown.

    Args:
        draft_pkg: The DraftPackage to format

    Returns:
        Markdown-formatted string
    """
    md_parts = []

    md_parts.append("# Draft Article")
    md_parts.append("")
    md_parts.append(f"**Word Count:** {len(draft_pkg.full_draft.split())} words")
    md_parts.append("")
    md_parts.append("---")
    md_parts.append("")

    md_parts.append(draft_pkg.full_draft)
    md_parts.append("")
    md_parts.append("---")
    md_parts.append("")

    md_parts.append(f"## Sources ({len(draft_pkg.sources)})")
    md_parts.append("")
    if draft_pkg.sources:
        for i, source in enumerate(draft_pkg.sources, 1):
            md_parts.append(f"{i}. {source}")
    else:
        md_parts.append("*No sources listed*")
    md_parts.append("")

    md_parts.append(f"## Flagged Facts ({len(draft_pkg.flagged_facts)})")
    md_parts.append("")
    if draft_pkg.flagged_facts:
        for i, fact in enumerate(draft_pkg.flagged_facts, 1):
            md_parts.append(f"{i}. {fact}")
    else:
        md_parts.append("*No facts flagged for verification*")
    md_parts.append("")

    md_parts.append(f"## Structural Notes ({len(draft_pkg.structural_notes)})")
    md_parts.append("")
    if draft_pkg.structural_notes:
        for i, note in enumerate(draft_pkg.structural_notes, 1):
            md_parts.append(f"{i}. {note}")
    else:
        md_parts.append("*No structural notes*")
    md_parts.append("")

    return "\n".join(md_parts)





def format_editor_decision_md(decision: EditorDecision) -> str:
    """Format an EditorDecision as markdown.

    Args:
        decision: The EditorDecision to format

    Returns:
        Markdown-formatted string
    """
    md_parts = []

    if decision.decision == "approve":
        md_parts.append("# APPROVED FOR PUBLICATION")
    else:
        md_parts.append("# REVISION REQUIRED")

    md_parts.append("")
    md_parts.append("---")
    md_parts.append("")

    md_parts.append("## Decision")
    md_parts.append("")
    md_parts.append(f"**{decision.decision.upper()}**")
    md_parts.append("")

    if decision.editor_notes:
        md_parts.append("## Editor Notes")
        md_parts.append("")
        for i, note in enumerate(decision.editor_notes, 1):
            md_parts.append(f"{i}. {note}")
        md_parts.append("")

    if decision.rejection_reasons:
        md_parts.append("## Rejection Reasons")
        md_parts.append("")
        for i, reason in enumerate(decision.rejection_reasons, 1):
            md_parts.append(f"{i}. {reason}")
        md_parts.append("")

    return "\n".join(md_parts)


def save_schema(
    schema: Union[StoryBrief, DraftPackage, EditorDecision],
    directory: Path,
    filename_base: str,
    version: int = None
) -> tuple[Path, Path]:
    """Save a Pydantic schema to JSON and Markdown files.

    Args:
        schema: The schema instance to save
        directory: Directory to save files in
        filename_base: Base filename (e.g., "story_brief", "draft_package")
        version: Optional version number to append to filename

    Returns:
        Tuple of (json_path, md_path)

    Example:
        >>> save_schema(story_brief, Path("tmp/article"), "story_brief")
        (Path("tmp/article/story_brief.json"), Path("tmp/article/story_brief.md"))

        >>> save_schema(draft_pkg, Path("tmp/article"), "draft_package", version=1)
        (Path("tmp/article/draft_package_1.json"), Path("tmp/article/draft_package_1.md"))
    """
    # Ensure directory exists
    directory.mkdir(parents=True, exist_ok=True)

    # Build filenames
    if version is not None:
        json_filename = f"{filename_base}_{version}.json"
        md_filename = f"{filename_base}_{version}.md"
    else:
        json_filename = f"{filename_base}.json"
        md_filename = f"{filename_base}.md"

    json_path = directory / json_filename
    md_path = directory / md_filename

    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(schema.model_dump(), f, indent=2, ensure_ascii=False)

    # Format and save markdown based on schema type
    # Format and save markdown based on schema type
    if isinstance(schema, StoryBrief):
        md_content = format_story_brief_md(schema)
    elif isinstance(schema, DraftPackage):
        md_content = format_draft_package_md(schema)
    elif isinstance(schema, EditorDecision):
        md_content = format_editor_decision_md(schema)
    elif isinstance(schema, ResearchPackage):
        # Basic markdown for research package
        md_content = f"# Research Package\n\nTotal Items: {len(schema.results)}\n\n"
        for i, res in enumerate(schema.results, 1):
            md_content += f"## {i}. {res.source}\n\n{res.search_result}\n\n---\n\n"
    elif isinstance(schema, FinalArticle):
        # Final article is just the text
        md_content = schema.final_article
    else:
        raise ValueError(f"Unsupported schema type: {type(schema)}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    return json_path, md_path


def load_schema(
    schema_class: type,
    directory: Path,
    filename_base: str,
    version: int = None
):
    """Load a Pydantic schema from a JSON file.

    Args:
        schema_class: The schema class to instantiate (e.g., StoryBrief)
        directory: Directory containing the file
        filename_base: Base filename (e.g., "story_brief", "draft_package")
        version: Optional version number

    Returns:
        Instance of schema_class

    Example:
        >>> brief = load_schema(StoryBrief, Path("tmp/article"), "story_brief")
        >>> draft = load_schema(DraftPackage, Path("tmp/article"), "draft_package", version=1)
    """
    if version is not None:
        json_filename = f"{filename_base}_{version}.json"
    else:
        json_filename = f"{filename_base}.json"

    json_path = directory / json_filename

    if not json_path.exists():
        raise FileNotFoundError(f"No {json_filename} found in {directory}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return schema_class(**data)
