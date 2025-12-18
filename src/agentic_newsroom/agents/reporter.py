from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

load_dotenv()

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

from agentic_newsroom.schemas.models import DraftPackage, FactReview, StyleReview, RevisedDraft
from agentic_newsroom.schemas.states import ReporterState
from agentic_newsroom.llm.openai import get_smart_model, get_mini_model
from agentic_newsroom.prompts.common import magazine_profile, magazine_guardrails
from agentic_newsroom.utils.content import count_words

logger = logging.getLogger(__name__)

# Default models per node type
# - Creative/revision tasks: smart model (quality critical)
# - Review/analytical tasks: mini model (structured output, cost-effective)
smart_model = get_smart_model()
mini_model = get_mini_model()

# =============================================================================
# PROMPTS
# =============================================================================

reporter_write_draft_prompt = f"""You are a science journalist writing for Agentic Newsroom.

{magazine_profile}

{magazine_guardrails}

<Task>
Write a complete article draft based on the Story Brief and Research Material.
</Task>

<Article Types>
- Web Daily: 400-700 words. Focus on a single study, discovery, or event.
- Full Feature: 1500-2000 words. Broader narrative, immersive, scene-setting.
</Article Types>

<Writing Rules>
1. Match length to article type. This is critical.
2. Break into sections with ## headings. Each section covers one aspect.
3. Open with a scene or vivid moment that pulls readers in. No dry summaries.
4. Ground abstract concepts in tangible details: places, numbers, sensory images.
5. Write in your own voice. NEVER copy-paste from sources.
6. No em dashes (—). Use commas, colons, or separate sentences.
7. No narrative quotes ("Dr. X said..."). You have no interviews.
8. Attribute non-obvious facts to their sources.
9. Stick to facts from Research Material only.
</Writing Rules>

<Voice>
- Short, punchy sentences. Vary rhythm.
- Use concrete numbers and specifics, not vague claims.
- Lead with what's surprising or at stake.
- No jargon. Explain technical terms naturally. 
- Build toward the discovery. Set up the mystery before the answer.
- When researchers appear, show them as people with motivations, not just names.
</Voice>

<Output>
- full_draft: The article text (no sources section in the draft)
- sources: List of full URLs used
- sources_section: Prose paragraph describing sources for end of article
</Output>
"""

# --- FACT REVIEW PROMPT ---
fact_review_prompt = """You are a fact-checker. Review this draft against the Research Material.

<Check>
1. **Accuracy**: Do claims match the research? Flag contradictions or unsupported statements.
2. **Attribution**: Are non-obvious facts sourced? Flag missing citations.
3. **Completeness**: Are all Key Questions answered? Flag gaps.
</Check>

<Output>
- issues: Specific problems. Example: "Paragraph 3 claims X but research says Y"
- rubric: Score accuracy, attribution, completeness 1-4. Default other dimensions to 3.
</Output>
"""

# --- STYLE REVIEW PROMPT ---
style_review_prompt = """You are a copy editor. Review this draft for style issues.

<Check>
1. **Compliance**: No em dashes (—), no narrative quotes ("Dr. X said...").
2. **Structure**: Strong lead, logical flow, good pacing, correct length for article type.
3. **Voice**: Engaging, clear, journalistic. No jargon.
</Check>

<Output>
- issues: Specific problems. Example: "Paragraph 2 uses em dash, rewrite as comma"
- rubric: Score compliance, structure, voice 1-4. Default other dimensions to 3.
</Output>
"""

# --- REVISION PROMPT ---
revision_prompt = """Fix the listed issues. Preserve everything else.

<Issues>
{issues}
</Issues>

<Rules>
- Fix ONLY listed issues
- Keep all ## section headings intact
- Do not merge or remove sections
- Keep same length
</Rules>
"""

# =============================================================================
# NODES
# =============================================================================

def _get_reporter_dir(slug: str):
    """Get the reporter artifacts directory."""
    from agentic_newsroom.schemas.base import get_project_root
    output_dir = get_project_root() / "artifacts" / slug / "reporter"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_draft(state: ReporterState, config: RunnableConfig = None):
    """Write the initial draft."""
    logger.info("-> write_draft")

    story_brief = state["story_brief"]
    research_package = state["research_package"]

    logger.debug(f"  Topic: {story_brief.topic}")
    logger.debug(f"  Article type: {story_brief.article_type}")
    logger.debug(f"  Research items: {len(research_package.results)}")

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", smart_model)  # Creative task: use smart model

    story_brief_md = story_brief.to_markdown()
    research_md = research_package.to_markdown()

    messages = [
        SystemMessage(content=reporter_write_draft_prompt),
        HumanMessage(content=f"This is the story brief:\n\n{story_brief_md}"),
        HumanMessage(content=f"Here is the research package:\n\n{research_md}")
    ]

    logger.info("  Generating draft...")
    structured_model = model.with_structured_output(DraftPackage)
    draft_package = structured_model.invoke(messages)

    word_count = count_words(draft_package.full_draft)
    logger.info(f"  Draft complete: {word_count} words")
    logger.info(f"  Sources used: {len(draft_package.sources)}")

    # Save initial draft
    output_dir = _get_reporter_dir(story_brief.slug)
    with open(output_dir / "1_initial_draft.md", "w") as f:
        f.write(draft_package.to_markdown())
    logger.info("  Saved: 1_initial_draft.md")

    return {"draft_package": draft_package}


def review_facts(state: ReporterState, config: RunnableConfig = None):
    """Review draft for factual accuracy, attribution, completeness."""
    logger.info("-> review_facts")

    story_brief = state["story_brief"]
    draft_package = state["draft_package"]
    research_package = state["research_package"]

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", mini_model)  # Analytical task: use mini model

    # Tight context: key questions + research + draft
    key_questions_md = "\n".join(f"- {q}" for q in story_brief.key_questions)

    messages = [
        SystemMessage(content=fact_review_prompt),
        HumanMessage(content=f"Key Questions to Answer:\n{key_questions_md}"),
        HumanMessage(content=f"Research Material:\n\n{research_package.to_markdown()}"),
        HumanMessage(content=f"Draft to Review:\n\n{draft_package.full_draft}")
    ]

    logger.info("  Reviewing facts...")
    structured_model = model.with_structured_output(FactReview)
    fact_review = structured_model.invoke(messages)

    logger.info(f"  Fact review complete: {len(fact_review.issues)} issues found")

    # Save fact review
    output_dir = _get_reporter_dir(story_brief.slug)
    with open(output_dir / "2_fact_review.md", "w") as f:
        f.write(fact_review.to_markdown())
    logger.info("  Saved: 2_fact_review.md")

    return {"fact_review": fact_review}


def revise_facts(state: ReporterState, config: RunnableConfig = None):
    """Revise draft to fix factual issues."""
    logger.info("-> revise_facts")

    story_brief = state["story_brief"]
    draft_package = state["draft_package"]
    fact_review = state["fact_review"]

    if not fact_review.issues:
        logger.info("  No factual issues to fix, skipping revision")
        return {}

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", smart_model)  # Revision task: use smart model

    issues_str = "\n".join(f"- {issue}" for issue in fact_review.issues)
    prompt = revision_prompt.format(issues=issues_str)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Current Draft:\n\n{draft_package.full_draft}")
    ]

    logger.info(f"  Revising {len(fact_review.issues)} factual issues...")
    structured_model = model.with_structured_output(RevisedDraft)
    revised = structured_model.invoke(messages)

    # Update only full_draft, preserve sources
    updated_package = DraftPackage(
        full_draft=revised.full_draft,
        sources=draft_package.sources,
        sources_section=draft_package.sources_section
    )

    word_count = count_words(updated_package.full_draft)
    logger.info(f"  Fact revision complete: {word_count} words")

    # Save revised draft
    output_dir = _get_reporter_dir(story_brief.slug)
    with open(output_dir / "3_after_fact_revision.md", "w") as f:
        f.write(updated_package.to_markdown())
    logger.info("  Saved: 3_after_fact_revision.md")

    return {"draft_package": updated_package}


def review_style(state: ReporterState, config: RunnableConfig = None):
    """Review draft for style compliance, structure, voice."""
    logger.info("-> review_style")

    story_brief = state["story_brief"]
    draft_package = state["draft_package"]

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", mini_model)  # Analytical task: use mini model

    # Tight context: just draft + article type (no research needed)
    messages = [
        SystemMessage(content=style_review_prompt),
        HumanMessage(content=f"Article Type: {story_brief.article_type}"),
        HumanMessage(content=f"Draft to Review:\n\n{draft_package.full_draft}")
    ]

    logger.info("  Reviewing style...")
    structured_model = model.with_structured_output(StyleReview)
    style_review = structured_model.invoke(messages)

    logger.info(f"  Style review complete: {len(style_review.issues)} issues found")

    # Save style review
    output_dir = _get_reporter_dir(story_brief.slug)
    with open(output_dir / "4_style_review.md", "w") as f:
        f.write(style_review.to_markdown())
    logger.info("  Saved: 4_style_review.md")

    return {"style_review": style_review}


def revise_style(state: ReporterState, config: RunnableConfig = None):
    """Revise draft to fix style issues."""
    logger.info("-> revise_style")

    story_brief = state["story_brief"]
    draft_package = state["draft_package"]
    style_review = state["style_review"]

    if not style_review.issues:
        logger.info("  No style issues to fix, skipping revision")
        return {}

    configuration = config.get("configurable", {}) if config else {}
    model = configuration.get("model", smart_model)  # Revision task: use smart model

    issues_str = "\n".join(f"- {issue}" for issue in style_review.issues)
    prompt = revision_prompt.format(issues=issues_str)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Current Draft:\n\n{draft_package.full_draft}")
    ]

    logger.info(f"  Revising {len(style_review.issues)} style issues...")
    structured_model = model.with_structured_output(RevisedDraft)
    revised = structured_model.invoke(messages)

    # Update only full_draft, preserve sources
    updated_package = DraftPackage(
        full_draft=revised.full_draft,
        sources=draft_package.sources,
        sources_section=draft_package.sources_section
    )

    word_count = count_words(updated_package.full_draft)
    logger.info(f"  Style revision complete: {word_count} words")

    # Save revised draft
    output_dir = _get_reporter_dir(story_brief.slug)
    with open(output_dir / "5_after_style_revision.md", "w") as f:
        f.write(updated_package.to_markdown())
    logger.info("  Saved: 5_after_style_revision.md")

    return {"draft_package": updated_package}


def finalize_draft(state: ReporterState, config: RunnableConfig = None):
    """Save the final draft package to slug-level artifacts folder."""
    logger.info("-> finalize_draft")

    story_brief = state["story_brief"]
    draft_package = state["draft_package"]

    word_count = count_words(draft_package.full_draft)
    logger.info(f"  Final word count: {word_count}")

    # Save final draft_package at slug level
    draft_package.save(story_brief.slug)
    logger.info(f"  Saved final draft to: artifacts/{story_brief.slug}/")

    return {"draft_package": draft_package}


# =============================================================================
# GRAPH
# =============================================================================

def build_reporter_graph():
    """
    Build the reporter graph with linear two-pass review:

    START → write_draft → review_facts → revise_facts → review_style → revise_style → finalize_draft → END
    """
    builder = StateGraph(ReporterState)

    builder.add_node("write_draft", write_draft)
    builder.add_node("review_facts", review_facts)
    builder.add_node("revise_facts", revise_facts)
    builder.add_node("review_style", review_style)
    builder.add_node("revise_style", revise_style)
    builder.add_node("finalize_draft", finalize_draft)

    # Linear flow - no conditionals
    builder.add_edge(START, "write_draft")
    builder.add_edge("write_draft", "review_facts")
    builder.add_edge("review_facts", "revise_facts")
    builder.add_edge("revise_facts", "review_style")
    builder.add_edge("review_style", "revise_style")
    builder.add_edge("revise_style", "finalize_draft")
    builder.add_edge("finalize_draft", END)

    return builder.compile()


if __name__ == "__main__":
    import argparse
    from agentic_newsroom.schemas.models import StoryBrief, ResearchPackage
    from agentic_newsroom.utils.newsroom_logging import setup_logging

    setup_logging()

    def main():

        parser = argparse.ArgumentParser(description="Reporter Agent")
        parser.add_argument("slug", help="The article slug")
        parser.add_argument("--mini", action="store_true", help="Use mini model instead of smart model")
        args = parser.parse_args()

        slug = args.slug
        model = get_mini_model() if args.mini else get_smart_model()
        model_name = "mini" if args.mini else "smart"

        print(f"Reporter: Writing draft for '{slug}' (model: {model_name})...")
        logger.info(f"Starting reporter agent for slug: {slug} with {model_name} model")

        try:
            story_brief = StoryBrief.load(slug)
            research_package = ResearchPackage.load(slug)
            logger.info(f"Loaded story brief: {story_brief.topic}")
            logger.info(f"Loaded research package with {len(research_package.results)} items")
        except FileNotFoundError as e:
            print(f"Missing input files: {e}")
            logger.error(f"Missing input files: {e}")
            return

        # Create graph
        logger.info("Building reporter graph")
        graph = build_reporter_graph()

        # Run graph with model config
        initial_state = {
            "story_brief": story_brief,
            "research_package": research_package,
        }
        config = {"configurable": {"model": model}}

        logger.info("Starting reporter workflow")
        result = graph.invoke(initial_state, config)

        draft_package = result.get("draft_package")

        if draft_package:
            print(f"Draft complete!")
            print(f"   Words: {count_words(draft_package.full_draft)}")
            print(f"   Saved to: artifacts/{slug}/")
            logger.info(f"Reporter complete. Draft saved to artifacts/{slug}/")
        else:
            print("Failed to write draft.")
            logger.error("Reporter workflow failed to produce draft")

    main()