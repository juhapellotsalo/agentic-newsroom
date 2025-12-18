
import operator
from typing import TypedDict, Optional, List, Annotated
from langchain_core.messages import BaseMessage
from agentic_newsroom.schemas.models import StoryBrief, SearchResult, ResearchPackage, DraftPackage, RevisionNotes, FinalArticle, FactReview, StyleReview, PublicationApproval


class NewsroomState(TypedDict):
    """State for the unified newsroom workflow.

    This state flows through the entire editorial pipeline from news tip to publication.
    """
    # Input
    article_idea: str

    # Agent outputs (updated as workflow progresses)
    story_brief: Optional[StoryBrief]
    research_package: Optional[ResearchPackage]
    draft_package: Optional[DraftPackage]
    final_article: Optional[FinalArticle]
    hero_image_path: Optional[str]
    approval: Optional[PublicationApproval]

class ResearchState(TypedDict):
    """The graph state for the research assistant."""
    story_brief: StoryBrief
    
    # The 'Memory' of the agent
    context: Annotated[List[BaseMessage], operator.add]
    
    # Working Memory for the current loop
    queries: List[str]
    raw_search_results: List[dict] # Snippets from Tavily Search
    urls_to_extract: List[str]     # URLs chosen by curation
    
    # Result Accumulator
    search_results: Annotated[List[SearchResult], operator.add]

    # Final trimmed package
    research_package: ResearchPackage 
    
    # Loop Control
    current_turn: int
    max_turns: int
    is_complete: bool


class ReporterState(TypedDict):
    """State for the Reporter subgraph.

    Flow: write_draft → review_facts → revise_facts → review_style → revise_style → finalize_draft
    """
    # Inputs
    story_brief: StoryBrief
    research_package: ResearchPackage

    # Reviews (for recording)
    fact_review: Optional[FactReview]
    style_review: Optional[StyleReview]

    # Output
    draft_package: Optional[DraftPackage]


class CopyEditorState(TypedDict):
    """State for the Copy Editor subgraph."""

    # Inputs
    story_brief: StoryBrief
    draft_package: DraftPackage

    # Output
    final_article: Optional[FinalArticle]


class GraphicDeskState(TypedDict):
    """State for the Graphic Desk subgraph."""

    # Inputs
    story_brief: StoryBrief
    final_article: FinalArticle

    # Outputs
    image_prompt: Optional[str]
    hero_image_path: Optional[str]


class EditorInChiefState(TypedDict):
    """State for the Editor in Chief subgraph.

    The Chief reviews the final article against magazine guardrails
    and approves it for publication.
    """

    # Inputs
    story_brief: StoryBrief
    final_article: FinalArticle

    # Output
    approval: Optional[PublicationApproval]