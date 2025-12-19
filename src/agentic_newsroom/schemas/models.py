from datetime import date
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from agentic_newsroom.schemas.base import NewsroomModel


class Category(str, Enum):
    """Article categories for The Agentic Newsroom."""
    SCIENCE = "Science"
    HISTORY = "History"
    PLANET_EARTH = "Planet Earth"
    MYSTERY = "Mystery"


class StoryBrief(NewsroomModel):

    topic: str = Field(..., description="Clear statement of what the story is about")
    angle: str = Field(..., description="The specific approach or perspective to take")
    category: Category = Field(..., description="The category that best fits the article")
    article_type: str = Field(..., description="The type of article and its target length")
    key_questions: List[str] = Field(..., description="3-5 questions the article should answer")
    slug: str = Field(..., description="Short but descriptive slug for the story")
    people_in_graphics: str = Field(
        ...,
        description="Instructions for people in the hero image. Default: 'Do not include any people in the hero image.' or custom instructions if explicitly requested."
    )

    def to_markdown(self) -> str:
        """Convert StoryBrief to markdown format."""
        lines = [
            f"# Story Brief: {self.topic}",
            "",
            f"**Angle:** {self.angle}",
            "",
            f"**Category:** {self.category.value}",
            "",
            f"**Article Type:** {self.article_type}",
            "",
            "## Key Questions",
            ""
        ]
        for i, question in enumerate(self.key_questions, 1):
            lines.append(f"{i}. {question}")

        lines.extend([
            "",
            f"**Slug:** {self.slug}",
            "",
            f"**People in Graphics:** {self.people_in_graphics}",
        ])

        return "\n".join(lines)


class SearchResult(BaseModel):
    """A single piece of gathered info."""
    source: str = Field(description="The URL or Title of the source")
    content: str = Field(description="The actual content text found")
    relevance: str = Field(description="Brief note on why this is relevant")

class ResearchPackage(NewsroomModel):
    results: List[SearchResult] = Field(..., description="List of collected search results")

    def to_markdown(self) -> str:
        md = f"# Research Package\n\n"
        md += f"**Total Items:** {len(self.results)}\n\n"

        for i, item in enumerate(self.results, 1):
                # Accessing item attributes works because they are SearchResult objects
                md += f"### {i}. {item.source}\n"
                md += f"**Relevance:** {item.relevance}\n\n"
                md += "```text\n"
                md += f"{item.content}\n"
                md += "```\n\n---\n\n"
        return md

class DraftPackage(NewsroomModel):
    """Draft package created by the Reporter.

    Contains the article draft along with metadata for review.
    """
    full_draft: str = Field(..., description="The complete article text")
    sources: List[str] = Field(..., description="List of sources (URLs, publications, experts)")
    sources_section: str = Field(..., description="Prose paragraph describing sources used for end of article")

    def to_markdown(self) -> str:
        """Format DraftPackage as markdown."""
        md_parts = []
        md_parts.append("# Draft Package")
        md_parts.append("")
        md_parts.append("## Full Draft")
        md_parts.append("")
        md_parts.append(self.full_draft)
        md_parts.append("")
        md_parts.append("## Sources")
        md_parts.append("")
        md_parts.append(self.sources_section)
        md_parts.append("")
        if self.sources:
            for s in self.sources:
                md_parts.append(f"- {s}")
        md_parts.append("")
        return "\n".join(md_parts)

class RevisionNotes(NewsroomModel):
    is_acceptable: bool = Field(..., description="Whether the revision is acceptable for publication")

    # Quantitative
    word_count: int = Field("Word count of the revised article")
    target_range: str = Field("Target range of the article length (words)")
    length_verdict: str = Field("Verdict how the article length compares to the target range")

    # Acceptance criteria breakdown
    blockers: List[str] = Field(..., description="Critical issues preventing publication: quote violations, factual errors, missing attribution, unanswered key questions")
    suggestions: List[str] = Field(..., description="Non-blocking improvements: style preferences, minor structural suggestions, word count fine-tuning")

    def to_markdown(self) -> str:
        md = f"# Revision Notes\n\n"
        md += f"**Is Acceptable:** {self.is_acceptable}\n\n"
        md += f"**Word Count:** {self.word_count}\n\n"
        md += f"**Target Range:** {self.target_range}\n\n"
        md += f"**Length Verdict:** {self.length_verdict}\n\n"

        md += f"## Blockers\n\n"
        if self.blockers:
            for blocker in self.blockers:
                md += f"- {blocker}\n"
        else:
            md += "*None - article is ready for publication*\n"
        md += "\n"

        md += f"## Suggestions\n\n"
        if self.suggestions:
            for suggestion in self.suggestions:
                md += f"- {suggestion}\n"
        else:
            md += "*No additional suggestions*\n"
        md += "\n"

        return md

    def save_snapshot(self, slug: str, draft_package: 'DraftPackage', revision: int):
        """Save versioned snapshots of draft and revision notes to reporter folder."""
        output_dir = self.get_serialization_path(slug) / "reporter"
        output_dir.mkdir(exist_ok=True)

        # Save draft package
        with open(output_dir / f"draft_package_version_{revision}.md", "w") as f:
            f.write(draft_package.to_markdown())

        # Save revision notes
        with open(output_dir / f"revision_notes_version_{revision}.md", "w") as f:
            f.write(self.to_markdown())



class ReviewRubric(BaseModel):
    """Quality rubric for recording (doesn't affect flow)"""
    accuracy: int = Field(..., ge=1, le=4, description="Facts match research: 1=POOR 2=FAIR 3=GOOD 4=EXCELLENT")
    attribution: int = Field(..., ge=1, le=4, description="Sources properly cited: 1=POOR 2=FAIR 3=GOOD 4=EXCELLENT")
    completeness: int = Field(..., ge=1, le=4, description="Key questions answered: 1=POOR 2=FAIR 3=GOOD 4=EXCELLENT")
    compliance: int = Field(..., ge=1, le=4, description="Follows style rules: 1=POOR 2=FAIR 3=GOOD 4=EXCELLENT")
    structure: int = Field(..., ge=1, le=4, description="Good lead, flow, pacing: 1=POOR 2=FAIR 3=GOOD 4=EXCELLENT")
    voice: int = Field(..., ge=1, le=4, description="Magazine tone, engaging: 1=POOR 2=FAIR 3=GOOD 4=EXCELLENT")

    def to_markdown(self) -> str:
        score_labels = {1: "POOR", 2: "FAIR", 3: "GOOD", 4: "EXCELLENT"}
        lines = [
            "| Dimension | Score |",
            "|-----------|-------|",
            f"| accuracy | {score_labels[self.accuracy]} ({self.accuracy}) |",
            f"| attribution | {score_labels[self.attribution]} ({self.attribution}) |",
            f"| completeness | {score_labels[self.completeness]} ({self.completeness}) |",
            f"| compliance | {score_labels[self.compliance]} ({self.compliance}) |",
            f"| structure | {score_labels[self.structure]} ({self.structure}) |",
            f"| voice | {score_labels[self.voice]} ({self.voice}) |",
        ]
        return "\n".join(lines)


class FactReview(BaseModel):
    """Output from fact reviewer - focuses on accuracy, attribution, completeness"""
    issues: List[str] = Field(..., description="Specific factual issues to fix (empty if none)")
    rubric: ReviewRubric = Field(..., description="Quality scores for recording")

    def to_markdown(self) -> str:
        md = "# Fact Review\n\n"
        md += "## Issues\n\n"
        if self.issues:
            for issue in self.issues:
                md += f"- {issue}\n"
        else:
            md += "*No factual issues found*\n"
        md += "\n## Rubric\n\n"
        md += self.rubric.to_markdown()
        return md


class StyleReview(BaseModel):
    """Output from style reviewer - focuses on compliance, structure, voice"""
    issues: List[str] = Field(..., description="Specific style issues to fix (empty if none)")
    rubric: ReviewRubric = Field(..., description="Quality scores for recording")

    def to_markdown(self) -> str:
        md = "# Style Review\n\n"
        md += "## Issues\n\n"
        if self.issues:
            for issue in self.issues:
                md += f"- {issue}\n"
        else:
            md += "*No style issues found*\n"
        md += "\n## Rubric\n\n"
        md += self.rubric.to_markdown()
        return md


class RevisedDraft(BaseModel):
    """Output from revision nodes - just the updated draft text"""
    full_draft: str = Field(..., description="The revised article text")


class FinalArticle(NewsroomModel):
    """The polished, publication-ready article from the Copy Editor."""
    title: str = Field(..., description="The article headline")
    subtitle: Optional[str] = Field(None, description="Optional subtitle or deck")
    article: str = Field(..., description="The article body in markdown (without title)")
    published_date: Optional[date] = Field(None, description="Publication date, set programmatically")

    def to_markdown(self) -> str:
        md = f"# {self.title}\n\n"
        if self.published_date:
            md += f"*Published: {self.published_date.strftime('%B %d, %Y')}*\n\n"
        if self.subtitle:
            md += f"*{self.subtitle}*\n\n"
        md += self.article
        return md


# --- Structured Outputs ---
class Queries(BaseModel):
    queries: List[str]

class UrlSelection(BaseModel):
    urls: List[str]
    reasoning: str

class ExtractedInfo(BaseModel):
    new_items: List[SearchResult]
    is_complete: bool
    summary_of_findings: str


class PublicationApproval(NewsroomModel):
    """Publication approval from the Editor in Chief.

    The Chief reviews the final article against magazine guardrails
    and provides approval for publication.
    """
    approved: bool = Field(..., description="Whether the article is approved for publication")
    notes: List[str] = Field(default_factory=list, description="Any notes from the Editor in Chief")

    def to_markdown(self) -> str:
        status = "APPROVED" if self.approved else "NOT APPROVED"
        md = f"# Publication Decision\n\n"
        md += f"**Status:** {status}\n\n"
        if self.notes:
            md += "## Notes\n\n"
            for note in self.notes:
                md += f"- {note}\n"
        return md