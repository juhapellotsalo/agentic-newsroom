from dotenv import load_dotenv
load_dotenv()

import logging
import base64
from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas.states import GraphicDeskState
from agentic_newsroom.schemas.models import FinalArticle, StoryBrief
from agentic_newsroom.schemas.base import get_project_root
from agentic_newsroom.llm.openai import get_smart_model

logger = logging.getLogger(__name__)

# Initialize client
openai_client = OpenAI()

default_model = get_smart_model()

graphic_desk_prompt = """You are a photo editor. Your task is to generate a minimalist, realistic image prompt based on the article provided.

<Process>
1. **Identify the Subject:** Extract the main geographic location OR the primary physical object (e.g., a ship, a mountain, an island) discussed in the article.
2. **Remove Narrative:** Discard all specific events, actions, characters, emotions, and drama. Do not describe what happened there; only describe the place/object itself.
3. **Format:** Create a simple sentence: "A realistic photograph of [Subject] [Location/Context]."
</Process>

<Constraints>
- **NO Events:** Do not mention storms, battles, accidents, footprints, or specific plot points.
- **NO Adjectives:** Avoid dramatic adjectives like "looming," "treacherous," "lonely," or "mysterious." Keep it neutral.
- **Historical Rule:** If the article is set >200 years ago, specify "historical [Subject]" (e.g., "A realistic photograph of a historical ship...").
- **Style:** Append the following technical tags to the end: ", 4k, professional photography, cinematic lighting. --ar 16:9"
</Constraints>

<Output Format>
Return ONLY the prompt string.

Example 1 (Article about a crash in the Andes):
"A realistic photograph of the Andes mountains, 4k, professional photography, cinematic lighting. --ar 16:9"

Example 2 (Article about the mutiny on the Bounty):
"A realistic photograph of a historical ship on the Pacific Ocean, 4k, professional photography, cinematic lighting. --ar 16:9"

Example 3 (Article about Penguins):
"A realistic photograph of a colony of penguins in Antarctica, 4k, professional photography, cinematic lighting. --ar 16:9"
</Output Format>

Article:
{article_content}
"""


def generate_image_prompt(state: GraphicDeskState, config: RunnableConfig = None):
    """Generate an image prompt based on the final article."""
    logger.info("-> generate_image_prompt")

    story_brief = state["story_brief"]
    final_article = state["final_article"]

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", default_model)

    article_context = f"""Title: {final_article.title}
            Subtitle: {final_article.subtitle or 'None'}
            Topic: {story_brief.topic}
            Article:
            {final_article.article}
    """

    messages = [
        SystemMessage(content=graphic_desk_prompt),
        HumanMessage(content=article_context)
    ]

    logger.info("  Generating image prompt...")
    response = model.invoke(messages)
    image_prompt = response.content.strip()

    logger.info(f"  Prompt: {image_prompt[:100]}...")

    # Save to graphics subfolder
    output_dir = get_project_root() / "artifacts" / story_brief.slug / "graphics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "hero_prompt.txt", "w") as f:
        f.write(image_prompt)
    logger.info(f"  Saved to: artifacts/{story_brief.slug}/graphics/hero_prompt.txt")

    return {"image_prompt": image_prompt}


def _generate_image_openai(prompt: str, model: str, quality: str) -> bytes:
    """Generate image using OpenAI's image API."""
    result = openai_client.images.generate(
        model=model,
        prompt=prompt,
        size="1536x1024",  # Close to 16:9
        quality=quality,
        n=1
    )
    return base64.b64decode(result.data[0].b64_json)


def generate_hero_image(state: GraphicDeskState, config: RunnableConfig = None):
    """Generate hero image from prompt using OpenAI's image API.

    Config options:
        image_model: "gpt-image-1", "gpt-image-1-mini" (default: "gpt-image-1")
        image_quality: "low", "medium", "high" (default: "medium")
    """
    logger.info("-> generate_hero_image")

    story_brief = state["story_brief"]
    image_prompt = state["image_prompt"]

    # Get config options with defaults
    configuration = config.get("configurable", {}) if config else {}
    image_model = configuration.get("image_model", "gpt-image-1.5")
    image_quality = configuration.get("image_quality", "medium")

    logger.info(f"  Prompt: {image_prompt}")
    logger.info(f"  Model: {image_model}, Quality: {image_quality}")

    # Generate image
    logger.info("  Calling OpenAI image API...")
    image_bytes = _generate_image_openai(image_prompt, image_model, image_quality)

    # Save image
    output_dir = get_project_root() / "artifacts" / story_brief.slug / "graphics"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "hero_image.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    logger.info(f"  Saved to: artifacts/{story_brief.slug}/graphics/hero_image.png")

    return {"hero_image_path": str(image_path)}


def build_graphic_desk_graph():
    """
    Build graphic desk graph:
    START → generate_image_prompt → generate_hero_image → END
    """
    builder = StateGraph(GraphicDeskState)
    builder.add_node("generate_image_prompt", generate_image_prompt)
    builder.add_node("generate_hero_image", generate_hero_image)
    builder.add_edge(START, "generate_image_prompt")
    builder.add_edge("generate_image_prompt", "generate_hero_image")
    builder.add_edge("generate_hero_image", END)
    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.utils.newsroom_logging import setup_logging

    setup_logging()

    def main():
        parser = argparse.ArgumentParser(description="Graphic Desk Agent")
        parser.add_argument("slug", help="The article slug")
        args = parser.parse_args()

        slug = args.slug
        print(f"Graphic Desk: Creating image prompt for '{slug}'...")

        try:
            story_brief = StoryBrief.load(slug)
            final_article = FinalArticle.load(slug)
        except FileNotFoundError as e:
            print(f"Missing input files: {e}")
            return

        graph = build_graphic_desk_graph()

        result = graph.invoke({
            "story_brief": story_brief,
            "final_article": final_article
        })

        image_prompt = result.get("image_prompt")
        hero_image_path = result.get("hero_image_path")

        if hero_image_path:
            print(f"Hero image created!")
            print(f"   Prompt: {image_prompt}")
            print(f"   Image: artifacts/{slug}/graphics/hero_image.png")
        else:
            print("Failed to create hero image.")

    main()