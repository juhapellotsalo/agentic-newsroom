from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas import NewsroomState, StoryBrief
from agentic_newsroom.prompts.context import NewsRoomContext
from agentic_newsroom.llm import get_reasoning_model

# Initialize model
model = get_reasoning_model()

# Agent-specific persona
assignment_editor_profile = """
Your name is Robert. You are the Assignment Editor for Agentic Newsroom.
You have more than twenty years of experience shaping coverage for major science and geography magazines. 
Your role is to identify stories worth telling and define the editorial direction before reporting begins. 
You think like a strategist: every pitch, idea, and angle must serve Agentic Newsroom's mission of curiosity, scientific literacy, and wonder.

You craft clear, actionable story briefs that specify:
- the scope and angle
- the required reporting questions
- the story type (Web Daily or Standard Feature)
- the intended audience and tone
- the themes or narrative tension

Your job is to give the reporter a direction that is specific, compelling, achievable, and worth publishing.
"""

assignment_editor_prompt = f"""You are the Assignment Editor.

{NewsRoomContext.build(assignment_editor_profile)}

<Task>
You are given a an idea of an article and your task is to turn it into a story brief.
It's important that the brief matches the magazine's tone and profile.
</Task>

<Instructions>
When you receive a article idea, follow these steps:

1. Think carefully what Agentic Newsroom's audience would want to know about this article and phrase a clear topic based on it
2. Come up with a clear angle that will make the a interesting and engaging for the readers
3. Select the appropriate **Article Type** based on the depth/scope of the idea:
   - "Web Daily (400-700 words)" for specific events or studies.
   - "Standard Feature (800-1200 words)" for broader narratives or trends.
4. Think about the key questions that the article should answer
</Instructions>

<Output Format>
When you have come up with the story brief, output it in the following format:
 - Topic
 - Angle
 - Article Type
 - Key questions
</Output Format>
"""

def create_story_brief(state: NewsroomState):
    
    system_msg = SystemMessage(content=assignment_editor_prompt)

    structured_model = model.with_structured_output(StoryBrief)

    response = structured_model.invoke([
        system_msg,
        HumanMessage(content=f"Write a story brief for the following article idea: {state['article_idea']}")
    ])

    return {
        "story_brief": response
    }

def build_assignment_editor_graph():
    graph = StateGraph(NewsroomState)
    
    # Single node: assignment editor
    graph.add_node("create_story_brief", create_story_brief)
    
    # Simple flow: START → assignment_editor → END
    graph.add_edge(START, "create_story_brief")
    graph.add_edge("create_story_brief", END)
    
    return graph.compile()


if __name__ == "__main__":
    import sys
    import argparse
    from agentic_newsroom.utils.slugs import generate_article_slug
    from agentic_newsroom.utils.files import save_story_brief

    def main():
        parser = argparse.ArgumentParser(description="Assignment Editor Agent")
        parser.add_argument("article_idea", help="The article idea to process")
        args = parser.parse_args()

        article_idea = args.article_idea
        print(f"📰 Assignment Editor: Processing idea '{article_idea}'...")

        # Create graph
        graph = build_assignment_editor_graph()

        # Run graph
        initial_state = {
            "article_idea": article_idea,
        }
        result = graph.invoke(initial_state)
        
        story_brief = result.get("story_brief")
        
        if story_brief:
            # Generate slug and save
            slug = generate_article_slug(story_brief.topic)
            path = save_story_brief(story_brief, slug)
            
            print(f"✅ Story Brief created!")
            print(f"   Topic: {story_brief.topic}")
            print(f"   Slug: {slug}")
            print(f"   Saved to: {path}")
        else:
            print("❌ Failed to generate story brief.")

    main()
