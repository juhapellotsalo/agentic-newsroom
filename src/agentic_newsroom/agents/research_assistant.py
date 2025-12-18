from dotenv import load_dotenv
load_dotenv()

import logging
from typing import Literal, List, Optional
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import START, END, StateGraph

logger = logging.getLogger(__name__)

from agentic_newsroom.schemas.states import ResearchState
from agentic_newsroom.schemas.models import SearchResult, ResearchPackage, StoryBrief
from agentic_newsroom.  tools.tavily_search import perform_search, perform_extract
from agentic_newsroom.llm.openai import get_mini_model

# --- Configuration ---
default_model = get_mini_model()

# You can override this via initial state, but this is the default cap
DEFAULT_MAX_TURNS = 5


# --- Prompts ---

generate_queries_prompt = """You are a senior researcher planning the next step for a deep-dive science magazine feature.

<Story Brief>
{story_brief}
</Story Brief>

<Current Info>
{context_str}
</Current Info>

<Task>
Your goal is to dig deeper to find not just facts, but **stories, characters, and scenes**.
1. Look at what we know (Current Info).
2. Identify **Narrative Gaps**: Do we lack the "smell" of the place? Do we need a specific character's voice? Do we need the turning point of the event?
3. Generate 3 targeted search queries to fill those gaps.
</Task>
"""

curate_sources_prompt = """You are a picky editor filtering research sources for a premium science magazine feature.

<Story Brief>
{story_brief}
</Story Brief>

<Task>
We have performed a broad search. Now we need to spend our extraction budget reading only the BEST sources.
Select up to 3 URLs that seem most likely to contain **First-Person Accounts, Primary Source Data, or Deep Analysis**.
</Task>
<Selection Criteria>
1. **Prioritize**:
   - Oral histories / Interviews / Diaries
   - Official Govt Reports / Archives (Factually dense)
   - Long-form journalism (The Atlantic, New Yorker style)
   - Academic monographs (Deep context)
2. **Avoid**:
   - Generic travel blogs
   - Shallow "Top 10 facts" lists
   - SEO-spam sites
   - Pinterest / Social Media aggregators
   - Reddit / Quora and similar unverified sources
   - PDF files (cannot be extracted)
   - Audio sources (cannot be extracted)
   - Video sources (cannot be extracted)
</Selection Criteria>

<Search Results>
{results_str}
</Search Results>
"""


extract_analyze_prompt = """You are a research analyst tasked to extract key facts and quotes from the sources.
A reporter of scientific magazine will use the facts you extract to write a full feature with strong narrative voices.


<Story Brief>
Here's the story brief:

{story_brief}
</Story Brief>

<Task>
We have extracted full text from several sources.
1. Read the text below.
2. Extract KEY FACTS, QUOTES, and DATA POINTS in verbatim that help write the story. (check the Important section for more details)
3. Ignore navigation, links, images, ads, or irrelevant fluff.
4. For every item you extract, you MUST preserve the Source URL.
</Task>

<Important>
Pay special attention to quotes the reporter may use in the article.
These quotes must be in verbatim and attributed back to the source.
</Important>

<Analysis Completion Criteria>
You're conducting a deep investigation. Do NOT mark research as complete unless:
1. You have at least 5 distinct high-quality sources (primary sources, interviews, or academic papers).
2. You have found specific personal quotes or anecdotes (narrative color).
3. You have covered the full timeline mentioned in the Brief.
If any of these are missing, set is_complete=False so we keep searching.
</Analysis Completion Criteria>

<Extracted Content>
{content_str}
</Extracted Content>
"""

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


# --- Nodes ---

from langchain_core.runnables import RunnableConfig

def generate_queries_node(state: ResearchState, config: RunnableConfig = None):
    current_turn = state.get("current_turn", 0) + 1
    logger.info(f"→ generate_queries (turn {current_turn})")

    brief = state["story_brief"]
    # Grab only 5 last messages from the context for short-term memory
    # Since this is called in a loop it's assumed that messages older than 5 are not relevant
    context_msgs = state.get("context", [])
    context_str = "\n".join([m.content for m in context_msgs[-5:]])

    # Handle configuration
    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", default_model)

    system_msg = generate_queries_prompt.format(
        story_brief=brief.model_dump_json(indent=2),
        context_str=context_str or "No research done yet."
    )

    # Generate
    structured_model = model.with_structured_output(Queries)
    res = structured_model.invoke([SystemMessage(content=system_msg)])

    logger.info(f"  Generated {len(res.queries)} queries")
    logger.debug(f"  Queries: {res.queries}")

    return {
        "queries": res.queries,
        "current_turn": current_turn
    }

def search_node(state: ResearchState):
    logger.info("→ search_web")
    queries = state.get("queries", [])
    logger.debug(f"  Searching for {len(queries)} queries")

    raw_results = perform_search(queries)

    logger.info(f"  Found {len(raw_results)} raw results")
    return {
        "raw_search_results": raw_results
    }

def curate_node(state: ResearchState, config: RunnableConfig = None):
    logger.info("→ curate_urls")

    brief = state["story_brief"]
    raw = state.get("raw_search_results", [])

    if not raw:
        logger.warning("  No raw results to curate")
        return {"urls_to_extract": []}

    logger.debug(f"  Curating from {len(raw)} raw results")

    # Format snippets for LLM
    results_str = ""
    for r in raw:
        # r is a dictionary representing one search result from Tavily e.g., {'url': '...', 'title': '...', 'content': '...'}).
        results_str += f"- URL: {r.get('url')}\n  Title: {r.get('title')}\n  Snippet: {r.get('content')}\n\n"

    # Handle configuration
    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", default_model)

    system_msg = curate_sources_prompt.format(
        story_brief=brief.model_dump_json(indent=2),
        results_str=results_str
    )

    structured_model = model.with_structured_output(UrlSelection)
    selection = structured_model.invoke([SystemMessage(content=system_msg)])

    logger.info(f"  Selected {len(selection.urls)} URLs for extraction")
    logger.debug(f"  Reasoning: {selection.reasoning}")

    return {
        "urls_to_extract": selection.urls
    }

def extract_analyze_node(state: ResearchState, config: RunnableConfig = None):
    logger.info("→ extract_analyze")

    brief = state["story_brief"]
    urls = state.get("urls_to_extract", [])

    logger.debug(f"  Extracting content from {len(urls)} URLs")
    extracted_data = perform_extract(urls)

    # Format for analysis
    content_str = ""
    for data in extracted_data:
        # data is dictionary representing one search result from Tavily e.g., {'url': '...', 'title': '...', 'content': '...'}).
        raw_content = data.get('raw_content', '')
        # Truncate massive pages to avoid token limits
        content_str += f"--- SOURCE: {data.get('url')} ---\n{raw_content[:10000]}\n\n"

    # Handle configuration
    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", default_model)

    system_msg = extract_analyze_prompt.format(
        story_brief=brief.model_dump_json(indent=2),
        content_str=content_str
    )

    structured_model = model.with_structured_output(ExtractedInfo)
    analysis = structured_model.invoke([SystemMessage(content=system_msg)])

    logger.info(f"  Extracted {len(analysis.new_items)} new items")
    logger.info(f"  Research complete: {analysis.is_complete}")
    logger.debug(f"  Summary: {analysis.summary_of_findings}")

    # Add context history so next query generations knows what we found
    update_msg = f"Turn {state['current_turn']} Findings: {analysis.summary_of_findings}"

    return {
        "search_results": analysis.new_items,
        "is_complete": analysis.is_complete,
        "context": [AIMessage(content=update_msg)],
        "is_research_complete": analysis.is_complete # Sync with state schema
    }

def finalize_research_node(state: ResearchState):
    """Deduplicate and wrap in ResearchPackage."""
    logger.info("→ finalize_research")

    raw = state.get("search_results", [])
    logger.debug(f"  Processing {len(raw)} raw search results")

    merged_map = {}

    for r in raw:
        if r.source not in merged_map:
            merged_map[r.source] = SearchResult(source=r.source, content=r.content, relevance=r.relevance)
        else:
            merged_map[r.source].content += "\n\n" + r.content

    final_list = list(merged_map.values())

    package = ResearchPackage(results=final_list)

    # Save artifact
    brief = state.get("story_brief")
    package.save(brief.slug)

    logger.info(f"  Finalized research package with {len(final_list)} unique sources")

    return {"research_package": package}

def check_loop(state: ResearchState) -> Literal["generate_queries", "finalize_research"]:
    """Decide if we should continue checking or stop."""
    current_turn = state.get("current_turn", 0)
    max_turns = state.get("max_turns", DEFAULT_MAX_TURNS)
    is_complete = state.get("is_complete") or state.get("is_research_complete")

    if is_complete or current_turn >= max_turns:
        reason = "complete" if is_complete else f"max turns ({max_turns})"
        logger.info(f"→ check_loop: Ending research ({reason})")
        return "finalize_research" # GO TO FINALIZER

    logger.info(f"→ check_loop: Continuing research (turn {current_turn}/{max_turns})")
    return "generate_queries"


def build_research_assistant_graph():
    builder = StateGraph(ResearchState)

    # Add Nodes
    builder.add_node("generate_queries", generate_queries_node)
    builder.add_node("search_web", search_node)
    builder.add_node("curate_urls", curate_node)
    builder.add_node("extract_analyze", extract_analyze_node)
    builder.add_node("finalize_research", finalize_research_node) 

    # Add Edges
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "search_web")
    builder.add_edge("search_web", "curate_urls")
    builder.add_edge("curate_urls", "extract_analyze")

    # Conditional Edge
    builder.add_conditional_edges(
        "extract_analyze", 
        check_loop,
        ["generate_queries", "finalize_research"] 
    )

    # Final Edge
    builder.add_edge("finalize_research", END)

    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.newsroom_logging import setup_logging

    setup_logging()

    def main():

        parser = argparse.ArgumentParser(description="Research Assistant Agent")
        parser.add_argument("slug", help="The article slug")
        parser.add_argument("--turns", type=int, default=DEFAULT_MAX_TURNS, help="Max research turns")
        args = parser.parse_args()

        slug = args.slug
        print(f"🔍 Research Assistant: Researching '{slug}'...")
        logger.info(f"Starting research assistant for slug: {slug}")

        try:
            story_brief = StoryBrief.load(slug)
            logger.info(f"Loaded story brief: {story_brief.topic}")
        except FileNotFoundError:
            print(f"❌ Story Brief not found for slug '{slug}'. Run assignment_editor.py first.")
            logger.error(f"Story brief not found for slug: {slug}")
            return

        # Create graph
        logger.info("Building research assistant graph")
        graph = build_research_assistant_graph()

        # Run graph
        initial_state = {
            "story_brief": story_brief,
            "max_turns": args.turns,
            "current_turn": 0,
            "context": [],
            "search_results": []
        }

        logger.info(f"Starting research workflow (max_turns={args.turns})")

        # Increase recursion limit to support many turns
        config = {"recursion_limit": 100}
        result = graph.invoke(initial_state, config=config)

        # Save package
        if result.get("research_package"):
            # package.save(slug) is handled in finalize_research_node
            print(f"✅ Research Complete!")
            print(f"   Items: {len(result['research_package'].results)}")
            print(f"   Saved to artifacts/{slug}/")
            logger.info(f"Research complete with {len(result['research_package'].results)} items")
        else:
            print("❌ Research failed to produce a package.")
            logger.error("Research workflow failed to produce a package")

    main()
