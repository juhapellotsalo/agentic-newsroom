from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas import ReporterState, DraftPackage
from agentic_newsroom.prompts.context import NewsRoomContext
from agentic_newsroom.llm import get_reasoning_model

# Initialize model
model = get_reasoning_model()

# Agent-specific persona
reporter_profile = """
Your name is Jennifer. You are the lead reporter for Agentic Newsroom.
You are an accomplished magazine journalist with more than twenty years of experience in various high profile scientific and geographical magazines.
Your deeply research reporting blends scientific rigor, clear narrative structure, and vivid descriptive writing.
You are endlessly curious about the planet, the natural world, history, science, and remote places. 
You always strive to uncover the most compelling angle and explore overlooked details.
You prioritize clarity, accuracy, and factual integrity. 
You rely only on verified information and credible sources.
Your research is meticulous and method-driven. 
You organize complex information, verify claims, resolve contradictions, and present findings in a structured and engaging narrative.
Your job is to produce well-written, deeply researched articles that reflect Agentic Newsroom's editorial standards: 
authentic curiosity, intellectual honesty, and a sense of wonder about the world.
"""

reporter_write_draft_prompt = f"""You are a reporter.

{NewsRoomContext.build(reporter_profile)}

<Story Brief>
This is the story brief you have been tasked to write:
{{story_brief}}
</Story Brief>

<Research Material>
You have access to a Research Package containing collected search results.
Each result includes the source URL and the content found.
Use this material as the factual basis for your article.
</Research Material>

{{examples}}

<Task>
Your task is to write a complete Draft Package.
This includes the full article text, a list of sources used, flagged facts that need verification, and structural notes.
</Task>

<Instructions>
1. **Analyze**: Read the Story Brief to understand the angle and **Article Type**.
2. **Plan**: Decide on the structure and flow of the article based on the defined **Article Type** (Web Daily or Standard Feature).
3. **Write**: Compose the article in your own voice.
   - **Do NOT copy-paste** text from sources. Rewrite in your own words.
   - **Attribute** every major fact or claim to its source (e.g., "According to National Geographic...").
4. **Cite**: Keep track of which sources you use for the `sources` list.
5. **Verify**: Flag any claims that seem dubious or contradictory in the `flagged_facts` list.
</Instructions>

<CRITICAL>
- **Attribution**: You must attribute information to its source within the text.
- **No Direct Quotes**: Do not use direct quotes unless you are quoting a public figure or official statement found in the research. You cannot interview people yourself.
- **Originality**: The writing must be yours. The facts come from the research.
- **Length**: Pay careful attention that your article aims for the target length defined in the story brief.
- **Tone**: Match the requested tone and target audience.
</CRITICAL>
"""

def write_draft(state: ReporterState):
    # Get inputs from state
    story_brief = state["story_brief"]
    research_package = state["research_package"]
    feedback = state.get("feedback", "")
    
    # Format inputs for the prompt
    brief_str = story_brief.model_dump_json(indent=2)
    
    # Format research results into a readable string for the LLM
    research_str = ""
    for i, item in enumerate(research_package.results, 1):
        research_str += f"Source {i}: {item.source}\nContent: {item.search_result}\n\n"

    # Prepare examples if available
    examples_str = ""
    # Assuming this file is in src/agentic_newsroom/agents/reporter.py
    project_root = Path(__file__).parent.parent.parent.parent
    examples_dir = project_root / "examples" / "articles"
    
    if examples_dir.exists():
        md_files = list(examples_dir.glob("*.md"))
        if md_files:
            examples_content = []
            for md_file in md_files:
                with open(md_file, "r") as f:
                    content = f.read()
                    examples_content.append(f"--- Example Article: {md_file.name} ---\n{content}\n")
            
            if examples_content:
                examples_str = "<Examples>\nHere are some example articles to guide your tone and style:\n\n" + "\n".join(examples_content) + "</Examples>\n"

    system_msg_content = reporter_write_draft_prompt.format(
        story_brief=brief_str,
        examples=examples_str
    )
           
    
    # Pass research as a HumanMessage context
    messages = [
        SystemMessage(content=system_msg_content),
        HumanMessage(content=f"Here is the collected research material:\n\n{research_str}")
    ]
    
    structured_model = model.with_structured_output(DraftPackage)
    response = structured_model.invoke(messages)
    
    return {
        "draft_package": response
    }

def build_reporter_graph():
    builder = StateGraph(ReporterState)
    builder.add_node("write_draft", write_draft)
    builder.add_edge(START, "write_draft")
    builder.add_edge("write_draft", END)
    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.files import load_story_brief, load_research_package, save_draft_package

    def main():
        parser = argparse.ArgumentParser(description="Reporter Agent")
        parser.add_argument("slug", help="The article slug")
        args = parser.parse_args()

        slug = args.slug
        print(f"✍️ Reporter: Writing draft for '{slug}'...")

        try:
            story_brief = load_story_brief(slug)
            research_package = load_research_package(slug)
        except FileNotFoundError as e:
            print(f"❌ Missing input files: {e}")
            return

        # Create graph
        graph = build_reporter_graph()

        # Run graph
        initial_state = {
            "story_brief": story_brief,
            "research_package": research_package,
            "feedback": None # Could load feedback from file if we wanted to support that loop via CLI
        }
        
        result = graph.invoke(initial_state)
        
        draft_package = result.get("draft_package")
        
        if draft_package:
            path = save_draft_package(draft_package, slug)
            
            print(f"✅ Draft Written!")
            print(f"   Words: {len(draft_package.full_draft.split())}")
            print(f"   Saved to: {path}")
        else:
            print("❌ Failed to write draft.")

    main()
