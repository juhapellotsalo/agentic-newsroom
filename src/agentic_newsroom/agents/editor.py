from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas import EditorState, FinalArticle
from agentic_newsroom.prompts.context import NewsRoomContext
from agentic_newsroom.llm import get_reasoning_model

# Initialize model
model = get_reasoning_model()

# Agent-specific persona
editor_profile = """
Your name is Anthony. You are the line editor for Agentic Newsroom.
You have more than twenty years of experience editing major science and geography magazines. 
"""

editor_write_prompt = f"""You are a newsroom line editor.

{NewsRoomContext.build(editor_profile)}

<Task>
You are given a draft article and your task is to polish it into a final piece.
</Task>

<Instructions>
- Improve clarity, coherence, pacing, and readability without altering meaning or introducing new unsupported claims.
- Fix grammar, syntax, factual inconsistencies, and stylistic issues while maintaining the reporter’s intended voice.
- Ensure the final article aligns with the magazine guardrails and the magazine's editorial tone.
- Do NOT remove attributions except for trivial ones (e.g., Wikipedia references). 
- Do NOT reduce attribution to vague phrases (“scientists say…”) if the draft names a specific study, synthesis, or research team.
</Instructions>

<Output Format>
- Write the final article in markdown format with correct headings, subheadings, and formatting.
- Write **only** the final article. 
- No commentary, explanations, or extra text of any kind.
</Output Format>
"""

def write_final_article(state: EditorState):

    draft_package_str =  state["draft_package"].model_dump_json()

    # Pass research as a HumanMessage context
    messages = [
        SystemMessage(content=editor_write_prompt),
        HumanMessage(content=f"Here is the draft package for you to edit:\n\n{draft_package_str}")
    ]

    structured_model = model.with_structured_output(FinalArticle)
    response = structured_model.invoke(messages)
    
    return {
        "final_article": response
    }

def build_editor_graph():
    builder = StateGraph(EditorState)
    builder.add_node("write_final_article", write_final_article)
    builder.add_edge(START, "write_final_article")
    builder.add_edge("write_final_article", END)
    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.files import load_draft_package, save_final_article

    def main():
        parser = argparse.ArgumentParser(description="Editor Agent")
        parser.add_argument("slug", help="The article slug")
        args = parser.parse_args()

        slug = args.slug
        print(f"📝 Editor: Polishing draft for '{slug}'...")

        try:
            draft_package = load_draft_package(slug)
        except FileNotFoundError as e:
            print(f"❌ Missing input files: {e}")
            return

        # Create graph
        graph = build_editor_graph()

        # Run graph
        initial_state = {
            "draft_package": draft_package
        }
        
        result = graph.invoke(initial_state)
        
        final_article = result.get("final_article")
        
        if final_article:
            path = save_final_article(final_article, slug)
            
            print(f"✅ Article Polished!")
            print(f"   Words: {len(final_article.final_article.split())}")
            print(f"   Saved to: {path}")
        else:
            print("❌ Failed to polish article.")

    main()
