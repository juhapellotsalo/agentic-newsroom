from dotenv import load_dotenv
load_dotenv()

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas.models import StoryBrief
from agentic_newsroom.schemas.states import NewsroomState
from agentic_newsroom.prompts.common import article_types
from agentic_newsroom.llm.openai import get_smart_model

logger = logging.getLogger(__name__)

# Initialize model (uses reasoning model in reasoning mode, mini in mini mode)
default_model = get_smart_model()

assignment_editor_prompt = """You are the Assignment Editor for The Agentic Newsroom.

<Task>
You are given a an idea of an article and your task is to turn it into a story brief.
</Task>

{article_types}

<Instructions>
When you receive a article idea, follow these steps:

1. Think carefully what the audience would want to know about this article and phrase a clear topic based on it
2  Come up with a clear angle that will make the a interesting and engaging for the readers
3. Think about the key questions that the article should answer
4. Estimate the length of the article using the article types definition defined above
</Instructions>

""".format(article_types=article_types)

def create_story_brief(state: NewsroomState, config: RunnableConfig = None):
    logger.info("→ create_story_brief")

    article_idea = state['article_idea']
    logger.debug(f"  Article idea: {article_idea[:100]}...")  # Truncate for logging

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", default_model)

    system_msg = SystemMessage(content=assignment_editor_prompt)
    structured_model = model.with_structured_output(StoryBrief)

    response = structured_model.invoke([
        system_msg,
        HumanMessage(content=f"Write a story brief for the following article idea: {article_idea}")
    ])

    story_brief = response
    story_brief.save(story_brief.slug)

    logger.info(f"  Created story brief: {story_brief.topic}")
    logger.info(f"  Article type: {story_brief.article_type}")
    logger.info(f"  Slug: {story_brief.slug}")

    return {
        "story_brief": story_brief
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
    import argparse
    from agentic_newsroom.utils.newsroom_logging import setup_logging

    setup_logging()

    def main():

        parser = argparse.ArgumentParser(description="Assignment Editor Agent")
        parser.add_argument("article_idea", help="The article idea to process")
        args = parser.parse_args()

        article_idea = args.article_idea
        print(f"📰 Assignment Editor: Processing idea '{article_idea}'...")
        logger.info(f"Starting assignment editor for article idea: {article_idea[:50]}...")

        # Create graph
        logger.info("Building assignment editor graph")
        graph = build_assignment_editor_graph()

        # Run graph
        initial_state = {
            "article_idea": article_idea,
        }

        logger.info("Starting assignment editor workflow")
        result = graph.invoke(initial_state)

        story_brief = result.get("story_brief")

        if story_brief:
            print(f"✅ Story Brief created!")
            print(f"   Topic: {story_brief.topic}")
            print(f"   Slug: {story_brief.slug}")
            print(f"   Saved to artifacts/{story_brief.slug}/")
            logger.info(f"Assignment editor complete. Created story brief: {story_brief.slug}")
        else:
            print("❌ Failed to generate story brief.")
            logger.error("Assignment editor workflow failed to produce story brief")

    main()
