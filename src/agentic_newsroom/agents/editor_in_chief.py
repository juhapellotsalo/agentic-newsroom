from dotenv import load_dotenv
load_dotenv()

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas.models import StoryBrief, FinalArticle, PublicationApproval
from agentic_newsroom.schemas.states import EditorInChiefState
from agentic_newsroom.llm.openai import get_smart_model, get_mini_model
from agentic_newsroom.prompts.common import magazine_guardrails

logger = logging.getLogger(__name__)

# Default model
default_model = get_smart_model()

# =============================================================================
# PROMPT
# =============================================================================

review_prompt = f"""You are the Editor in Chief. Your task is to review the final article against the magazine guardrails and approve it for publication.

{magazine_guardrails}

<Task>
Review the article against the guardrails above. If the article complies with all guardrails, approve it.
Provide brief notes if you have any observations about the article.
</Task>

<Output>
- approved: true if the article complies with guardrails
- notes: Any observations (can be empty if none)
</Output>
"""


# =============================================================================
# NODES
# =============================================================================

def review_and_approve(state: EditorInChiefState, config: RunnableConfig = None):
    """Review final article against guardrails and approve for publication."""
    logger.info("-> review_and_approve")

    story_brief = state["story_brief"]
    final_article = state["final_article"]

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", default_model)

    logger.info(f"  Article: {final_article.title}")
    logger.info(f"  Topic: {story_brief.topic}")

    # Build article context for review
    article_md = final_article.to_markdown()

    messages = [
        SystemMessage(content=review_prompt),
        HumanMessage(content=f"Review this article:\n\n{article_md}")
    ]

    logger.info("  Reviewing against guardrails...")
    structured_model = model.with_structured_output(PublicationApproval)
    approval = structured_model.invoke(messages)

    status = "APPROVED" if approval.approved else "NOT APPROVED"
    logger.info(f"  Decision: {status}")
    if approval.notes:
        logger.info(f"  Notes: {len(approval.notes)} items")

    # Save approval to slug directory
    approval.save(story_brief.slug)
    logger.info(f"  Saved: artifacts/{story_brief.slug}/publication_approval.json")

    return {"approval": approval}


# =============================================================================
# GRAPH
# =============================================================================

def build_editor_in_chief_graph():
    """
    Build editor in chief graph:
    START -> review_and_approve -> END
    """
    builder = StateGraph(EditorInChiefState)
    builder.add_node("review_and_approve", review_and_approve)
    builder.add_edge(START, "review_and_approve")
    builder.add_edge("review_and_approve", END)
    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.newsroom_logging import setup_logging

    setup_logging()

    def main():
        parser = argparse.ArgumentParser(description="Editor in Chief Agent")
        parser.add_argument("slug", help="The article slug")
        parser.add_argument("--mini", action="store_true", help="Use mini model instead of smart model")
        args = parser.parse_args()

        slug = args.slug
        model = get_mini_model() if args.mini else get_smart_model()
        model_name = "mini" if args.mini else "smart"

        print(f"Editor in Chief: Reviewing '{slug}' (model: {model_name})...")
        logger.info(f"Starting editor in chief for slug: {slug} with {model_name} model")

        try:
            story_brief = StoryBrief.load(slug)
            final_article = FinalArticle.load(slug)
            logger.info(f"Loaded story brief: {story_brief.topic}")
            logger.info(f"Loaded final article: {final_article.title}")
        except FileNotFoundError as e:
            print(f"Missing input files: {e}")
            logger.error(f"Missing input files: {e}")
            return

        # Build and run graph
        logger.info("Building editor in chief graph")
        graph = build_editor_in_chief_graph()

        initial_state = {
            "story_brief": story_brief,
            "final_article": final_article,
        }
        config = {"configurable": {"model": model}}

        logger.info("Starting review")
        result = graph.invoke(initial_state, config)

        approval = result.get("approval")

        if approval:
            status = "APPROVED" if approval.approved else "NOT APPROVED"
            print(f"Decision: {status}")
            if approval.notes:
                print("Notes:")
                for note in approval.notes:
                    print(f"  - {note}")
            print(f"Saved to: artifacts/{slug}/publication_approval.json")
            logger.info(f"Editor in Chief complete. Decision: {status}")
        else:
            print("Failed to review article.")
            logger.error("Editor in Chief workflow failed")

    main()