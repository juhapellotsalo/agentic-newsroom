from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas import EditorInChiefState, EditorDecision
from agentic_newsroom.prompts.context import NewsRoomContext
from agentic_newsroom.llm import get_reasoning_model

# Initialize model
model = get_reasoning_model()

# Agent-specific persona
editor_in_chief_profile = """
Your name is Margaret. You are the Editor in Chief of Agentic Newsroom.
You have more than thirty years of experience leading major science and geography magazines.
Your role is to make the final publication decision on every article.
You are the last line of editorial judgment, ensuring that every published piece meets Agentic Newsroom's standards for accuracy, clarity, narrative quality, and editorial integrity.
You balance editorial vision with practical publishing considerations.
You are decisive but fair, and your feedback is always constructive and specific.
"""

editor_in_chief_prompt = f"""You are the Editor in Chief.

{NewsRoomContext.build(editor_in_chief_profile)}

<Task>
Review the provided Article Draft against the Story Brief and Editorial Standards.
Make a decision: APPROVE or REJECT.
</Task>

If you reject, you must provide clear, actionable revision instructions.
</Instructions>

<Story Brief>
{{story_brief}}
</Story Brief>
"""

def review_article(state: EditorInChiefState):
    """
    Editor in Chief reviews the final article against the story brief
    and makes the approval/rejection decision.
    """
    story_brief = state["story_brief"]
    final_article = state["final_article"]
    
    # Format story brief for the prompt
    brief_str = story_brief.model_dump_json(indent=2)
    
    # Build the system message
    system_msg_content = editor_in_chief_prompt.format(
        story_brief=brief_str
    )
    
    # Pass final article as HumanMessage
    messages = [
        SystemMessage(content=system_msg_content),
        HumanMessage(content=f"Here is the final article for your review:\n\n{final_article.final_article}")
    ]
    
    structured_model = model.with_structured_output(EditorDecision)
    response = structured_model.invoke(messages)
    
    return {
        "editor_decision": response
    }

def build_editor_in_chief_graph():
    builder = StateGraph(EditorInChiefState)
    builder.add_node("review_article", review_article)
    builder.add_edge(START, "review_article")
    builder.add_edge("review_article", END)
    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.files import load_story_brief, load_final_article, save_editor_decision

    def main():
        parser = argparse.ArgumentParser(description="Editor in Chief Agent")
        parser.add_argument("slug", help="The article slug")
        args = parser.parse_args()

        slug = args.slug
        print(f"👩‍⚖️ Editor in Chief: Reviewing '{slug}'...")

        try:
            story_brief = load_story_brief(slug)
            final_article = load_final_article(slug)
        except FileNotFoundError as e:
            print(f"❌ Missing input files: {e}")
            return

        # Create graph
        graph = build_editor_in_chief_graph()

        # Run graph
        initial_state = {
            "story_brief": story_brief,
            "final_article": final_article
        }
        
        result = graph.invoke(initial_state)
        
        editor_decision = result.get("editor_decision")
        
        if editor_decision:
            path = save_editor_decision(editor_decision, slug)
            
            print(f"✅ Decision Made: {editor_decision.decision.upper()}")
            print(f"   Saved to: {path}")
        else:
            print("❌ Failed to make decision.")

    main()
