from dotenv import load_dotenv
load_dotenv()

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas.states import CopyEditorState
from agentic_newsroom.schemas.models import FinalArticle, DraftPackage, StoryBrief
from agentic_newsroom.llm.openai import get_smart_model

logger = logging.getLogger(__name__)

default_model = get_smart_model()

copy_editor_prompt = """You are a copy editor. Polish the draft into a publication-ready article.

<Primary Task>
Weave source citations naturally into prose:
- "(Bligh's Narrative; court transcripts)" → "as Bligh recorded in his Narrative"
- "(Smith 2020)" → "Smith's research shows" or "according to Smith"
- "(Wikipedia)" → remove, this is common knowledge
</Primary Task>

<Secondary>
- Fix grammar/syntax
- Improve sentence flow
- Maintain reporter's voice
- Keep same structure and length
- Preserve section headings as markdown (## Heading)
</Secondary>

<Output Format>
- title: A compelling headline (not the same as the draft's first heading)
- subtitle: Optional deck/subhead that expands on the title
- article: The polished body in markdown format:
  - Use ## for section headings
  - Preserve the draft's section structure
  - Do NOT include the title as a heading (it's separate)
</Output Format>
"""

def polish_article(state: CopyEditorState, config: RunnableConfig = None):
    """Polish draft into final publication-ready article."""
    logger.info("-> polish_article")

    story_brief = state["story_brief"]
    draft_package = state["draft_package"]

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", default_model)

    messages = [
        SystemMessage(content=copy_editor_prompt),
        HumanMessage(content=f"Draft to polish:\n\n{draft_package.full_draft}")
    ]

    logger.info("  Polishing draft...")
    structured_model = model.with_structured_output(FinalArticle)
    final_article = structured_model.invoke(messages)

    logger.info(f"  Title: {final_article.title}")
    logger.info(f"  Polish complete: {len(final_article.article.split())} words")

    # Save final article to artifacts folder
    final_article.save(story_brief.slug)
    logger.info(f"  Saved to: artifacts/{story_brief.slug}/")

    return {"final_article": final_article}

def build_copy_editor_graph():
    builder = StateGraph(CopyEditorState)
    builder.add_node("polish_article", polish_article)
    builder.add_edge(START, "polish_article")
    builder.add_edge("polish_article", END)
    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.newsroom_logging import setup_logging

    setup_logging()

    def main():
        parser = argparse.ArgumentParser(description="Copy Editor Agent")
        parser.add_argument("slug", help="The article slug")
        args = parser.parse_args()

        slug = args.slug
        print(f"📝 Copy Editor: Polishing draft for '{slug}'...")

        try:
            story_brief = StoryBrief.load(slug)
            draft_package = DraftPackage.load(slug)
        except FileNotFoundError as e:
            print(f"❌ Missing input files: {e}")
            return

        graph = build_copy_editor_graph()

        result = graph.invoke({
            "story_brief": story_brief,
            "draft_package": draft_package
        })
        final_article = result.get("final_article")

        if final_article:
            print(f"✅ Article Polished!")
            print(f"   Title: {final_article.title}")
            print(f"   Words: {len(final_article.article.split())}")
            print(f"   Saved to: artifacts/{slug}/")
        else:
            print("❌ Failed to polish article.")

    main()
