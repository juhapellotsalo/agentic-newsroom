from dotenv import load_dotenv
load_dotenv()

from typing import Literal, Annotated, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas import ResearchState, SearchResult
from agentic_newsroom.tools.tavily_search import search_web
from agentic_newsroom.tools.wikipedia_search import search_wikipedia
from agentic_newsroom.prompts.context import NewsRoomContext
from agentic_newsroom.llm import get_mini_model

from agentic_newsroom.schemas import ResearchState, SearchResult, SearchQuery
from agentic_newsroom.tools.wikipedia_search import search_wikipedia as wikipedia_search_tool
from agentic_newsroom.tools.tavily_search import search_web as tavily_search_tool

# Initialize model
model = get_mini_model()

# Agent-specific persona
research_assistant_profile = """
Your name is Sarah. You are the lead researcher for Agentic Newsroom.
You are a meticulous fact-finder who loves digging into complex topics.
Your job is to provide the reporter with the raw materials they need: verified facts, primary sources, expert quotes, and key context.
You never settle for the first result; you triangulate information to ensure accuracy.
"""

search_query_prompt = f"""You are a research assistant.

{NewsRoomContext.build(research_assistant_profile)}

<Story Brief>
{{story_brief}}
</Story Brief>

<Task>
Generate search queries to gather information for the story defined in the brief.
Pay attention to the **Article Type**:
- **Web Daily**: Focus on the specific recent event/study and immediate context.
- **Standard Feature**: Cast a wider net for historical context, side characters, and deeper analysis.
</Task>

<Context>
Previous queries and results (if any):
{{context}}
</Context>

<Instructions>
1. Analyze the brief and the current state of research.
2. Generate 3-5 targeted search queries.
3. If research seems sufficient for the Article Type, set 'is_research_complete' to True.
</Instructions>
"""


wikipedia_search_instructions = f"""You are an expert at creating Wikipedia search queries.

{NewsRoomContext.build(research_assistant_profile)}

<StoryBrief>
{{story_brief}}
</StoryBrief>

<Instructions>
1. Look at the latest research question or topic in the context.
2. Generate a **simple, direct query** targeting a Wikipedia article title or main topic.
3. Wikipedia search works best with:
   - Single entities or concepts: "Socotra", "Dracaena cinnabari", "Dragon blood tree"
   - Proper nouns: place names, species names, historical events
   - 1-4 words maximum
</Instructions>
"""

web_search_instructions = f"""You are an expert Search Query Optimizer.

{NewsRoomContext.build(research_assistant_profile)}

<StoryBrief>
{{story_brief}}
</StoryBrief>

<Instructions>
1. Look at the latest research question or topic provided in the context.
2. Formulate a specific, keyword-rich query that is likely to yield high-quality results.
3. Avoid conversational language (e.g., "Tell me about...") and focus on entities and keywords.
</Instructions>
"""

collect_material_instructions = f"""You are a Research Assistant analyzing search results.

{NewsRoomContext.build(research_assistant_profile)}

<StoryBrief>
{{story_brief}}
</StoryBrief>

<Task>
Your primary task is to process the raw search results and extract relevant information.
You must also critically evaluate if we have enough information to write the article based on its **Article Type**.
</Task>

<Sufficiency Criteria>
**Web Daily**:
- COMPLETE when you have the core event/study facts AND 1-2 distinct sources.
- Do NOT go down deep rabbit holes. Fast and factual.

**Standard Feature**:
- COMPLETE when you have the core facts + historical context + "color" (descriptive details/scenes) + diverse perspectives.
- We need enough material to build a narrative arc, not just a list of facts.
</Sufficiency Criteria>

<Instructions>
1. **Extract Material**: Review the raw search results. Extract key facts, quotes, and figures into structured items.
2. **Evaluate Sufficiency**: Compare the TOTAL information gathered so far (including previous turns) against the Story Brief and the **Sufficiency Criteria** above.
3. **Decision**:
   - If the material matches the Article Type requirements, mark as COMPLETE.
   - If important questions remain unanswered or the material is too thin for the type, mark as INCOMPLETE.
</Instructions>
"""

def generate_queries(state: ResearchState):
    """Generate search queries based on the story brief."""
    
    story_brief = state["story_brief"]
    context = state.get("context", [])
    current_turn = state.get("current_turn", 0)

    # Generate search queries
    system_msg = search_query_prompt.format(
        story_brief=story_brief.model_dump_json(),
        context="\n".join([msg.content for msg in context if isinstance(msg, AIMessage)])
    )
    
    response = model.invoke([SystemMessage(content=system_msg)] + context)

    return {
        "context": [response],
        "current_turn": current_turn + 1
    }

def search_web(state: ResearchState):
    """Search the web using Tavily based on the conversation context."""
    
    story_brief = state["story_brief"]
    context = state.get("context", [])
    
    # Generate query using general search instructions
    structured_model = model.with_structured_output(SearchQuery)
    system_msg = SystemMessage(
        content=web_search_instructions.format(story_brief=story_brief.model_dump_json())
    )
    search_query = structured_model.invoke([system_msg] + context)
    
    # Execute search using the tool (handles both search and formatting)
    try:
        formatted_docs = tavily_search_tool(search_query.search_query, max_results=3)
    except Exception as e:
        return {"context": [f"Error performing web search: {str(e)}"]}
    
    return {"context": [formatted_docs]}

def search_wikipedia(state: ResearchState):
    """Retrieve docs from Wikipedia based on the conversation context."""
    
    story_brief = state["story_brief"]
    context = state.get("context", [])
    
    # Generate a structured search query from the context
    # Use Wikipedia-specific instructions
    structured_model = model.with_structured_output(SearchQuery)
    system_msg = SystemMessage(
        content=wikipedia_search_instructions.format(story_brief=story_brief.model_dump_json())
    )
    search_query = structured_model.invoke([system_msg] + context)
    
    # Execute search using the tool
    try:
        formatted_search_docs = wikipedia_search_tool(search_query.search_query, max_docs=2)
    except Exception as e:
        return {"context": [f"Error performing Wikipedia search: {str(e)}"]}
    
    return {"context": [formatted_search_docs]}

def collect_material(state: ResearchState):
    """Process search results, extract structured data, and decide if research is complete."""
    
    story_brief = state["story_brief"]
    context = state.get("context", [])
    
    # Define the structure for the LLM's analysis
    class ResearchEvaluation(BaseModel):
        new_material: list[SearchResult] = Field(
            description="Relevant facts and information extracted from the latest search results"
        )
        is_complete: bool = Field(
            description="True if the gathered information is sufficient to answer the research questions, False if more research is needed"
        )
        reasoning: str = Field(
            description="Brief explanation of why research is complete or what is still missing"
        )

    system_msg = SystemMessage(
        content=collect_material_instructions.format(
            story_brief=story_brief.model_dump_json()
        )
    )
    
    structured_model = model.with_structured_output(ResearchEvaluation)
    evaluation = structured_model.invoke([system_msg] + context)
    
    # Create a summary message for the context
    summary_msg = (
        f"Collected {len(evaluation.new_material)} new items.\n"
        f"Status: {'Complete' if evaluation.is_complete else 'Continuing'}\n"
        f"Reasoning: {evaluation.reasoning}"
    )
    
    # The 'search_results' key will APPEND to the existing list because of operator.add in schema
    return {
        "search_results": evaluation.new_material,
        "context": [AIMessage(content=summary_msg)],
        "is_research_complete": evaluation.is_complete
    }

def should_continue(state: ResearchState) -> Literal["generate_queries", END]:
    """Decide whether to continue research or end."""
    
    # Check loop limits
    current_turn = state.get("current_turn", 0)
    max_turns = state.get("max_num_turns", 3)
    
    if current_turn >= max_turns:
        return END
    
    # Check LLM decision
    if state.get("is_research_complete", False):
        return END
        
    return "generate_queries"

def build_research_assistant_graph():
    builder = StateGraph(ResearchState)

    # Add Nodes
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("search_web", search_web)
    builder.add_node("search_wikipedia", search_wikipedia)
    builder.add_node("collect_material", collect_material)

    # Add Edges
    builder.add_edge(START, "generate_queries")

    # Parallel execution: generate_queries -> both search nodes
    builder.add_edge("generate_queries", "search_web")
    builder.add_edge("generate_queries", "search_wikipedia")

    # Fan-in: both search nodes -> collect_material
    builder.add_edge("search_web", "collect_material")
    builder.add_edge("search_wikipedia", "collect_material")

    # Conditional Loop
    builder.add_conditional_edges(
        "collect_material", 
        should_continue, 
        ["generate_queries", END]
    )

    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.files import load_story_brief, save_research_package
    from agentic_newsroom.schemas import ResearchPackage

    def main():
        parser = argparse.ArgumentParser(description="Research Assistant Agent")
        parser.add_argument("slug", help="The article slug")
        args = parser.parse_args()

        slug = args.slug
        print(f"🔍 Research Assistant: Researching '{slug}'...")

        try:
            story_brief = load_story_brief(slug)
        except FileNotFoundError:
            print(f"❌ Story Brief not found for slug '{slug}'. Run assignment_editor.py first.")
            return

        # Create graph
        graph = build_research_assistant_graph()

        # Run graph
        initial_state = {
            "story_brief": story_brief,
            "max_num_turns": 3,
            "current_turn": 0,
            "context": [],
            "search_results": []
        }
        
        result = graph.invoke(initial_state)
        
        # Extract research package
        # The graph returns 'search_results' list, we need to wrap it
        research_results = result.get("search_results", [])
        pkg = ResearchPackage(results=research_results)
        
        path = save_research_package(pkg, slug)
        
        print(f"✅ Research Complete!")
        print(f"   Items: {len(pkg.results)}")
        print(f"   Saved to: {path}")

    main()
