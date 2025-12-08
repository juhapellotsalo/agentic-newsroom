"""
Data models for the Agentic Newsroom.

These Pydantic models define the structure of data that flows through
the newsroom workflow, enabling structured outputs from LLMs.
"""

from typing import List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class SearchQuery(BaseModel):
    search_query: str = Field(..., description="The search query")


class StoryBrief(BaseModel):
    """Story brief created by the Assignment Editor.

    Defines the scope, angle, and approach for an article.
    """
    topic: str = Field(..., description="Clear statement of what the story is about")
    angle: str = Field(..., description="The specific approach or perspective to take")
    article_type: str = Field(..., description="The type of article and its target length")
    key_questions: List[str] = Field(..., description="3-5 questions the article should answer")


class SearchResult(BaseModel):
    search_result: str = Field(..., description="The search result content")
    source: str = Field(..., description="The source of the search result")


class ResearchPackage(BaseModel):
    """Research package created by the Research Assistant.

    Contains the collected research material.
    """
    results: List[SearchResult] = Field(..., description="List of collected search results")


class DraftPackage(BaseModel):
    """Draft package created by the Reporter.

    Contains the article draft along with metadata for review.
    """
    full_draft: str = Field(..., description="The complete article text")
    sources: List[str] = Field(..., description="List of sources (URLs, publications, experts)")
    flagged_facts: List[str] = Field(..., description="Facts/claims that need verification")
    structural_notes: List[str] = Field(..., description="Notes about the draft structure (e.g., 'lead may be too soft')")

class FinalArticle(BaseModel):
    final_article: str = Field(..., description="The final article text")

class EditorDecision(BaseModel):
    """Editorial decision from the Editor-in-Chief.
    
    Contains the decision on whether to approve or reject for publication.
    """
    decision: str = Field(..., description="Editorial decision: 'approve' or 'reject'")
    editor_notes: List[str] = Field(default_factory=list, description="General notes if approved (optional)")
    rejection_reasons: List[str] = Field(default_factory=list, description="Specific reasons if decision is 'reject'")

class NewsroomState(TypedDict):
    """State for the unified newsroom workflow.

    This state flows through the entire editorial pipeline from news tip to publication decision.
    """
    # Input
    article_idea: str

    # Agent outputs (updated as workflow progresses)
    story_brief: Optional[StoryBrief]
    research_package: Optional[ResearchPackage]
    draft_package: Optional[DraftPackage]
    final_article: Optional[FinalArticle]
    editor_decision: Optional[EditorDecision]
    feedback: Optional[str]


class ReporterState(TypedDict):
    """State for the Reporter subgraph.
    
    The reporter now receives pre-collected research from the Research Assistant
    and focuses solely on writing the draft.
    """
    # Inputs
    story_brief: StoryBrief
    research_package: ResearchPackage
    feedback: Optional[str]

    # Output
    draft_package: Optional[DraftPackage]


class ResearchState(TypedDict):
    """State for the Research Assistant subgraph."""
    story_brief: StoryBrief
    context: Annotated[List[BaseMessage], operator.add]
    search_results: Annotated[List[SearchResult], operator.add]
    max_num_turns: int
    current_turn: int
    is_research_complete: bool


class EditorState(TypedDict):
    """State for the Editor subgraph."""
    
    # Input
    draft_package: DraftPackage

    # Output
    final_article: Optional[FinalArticle]


class EditorInChiefState(TypedDict):
    """State for the Editor in Chief subgraph."""
    
    # Inputs
    story_brief: StoryBrief
    final_article: FinalArticle
    
    # Output
    editor_decision: Optional[EditorDecision]