#!/usr/bin/env python3
"""
Main entry point for running the Agentic Newsroom workflow.

This script runs the complete editorial pipeline from article idea to publication.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
from agentic_newsroom.workflows.newsroom_workflow import build_newsroom_workflow
from agentic_newsroom.utils.newsroom_logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Agentic Newsroom workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "The mysterious dragon blood trees of Socotra Island"
  python main.py "Deep sea hydrothermal vents and their unique ecosystems"
        """
    )
    parser.add_argument(
        "article_idea",
        help="The article idea to process through the full editorial pipeline"
    )

    args = parser.parse_args()

    setup_logging()

    article_idea = args.article_idea
    print(f"Agentic Newsroom: Processing article idea...")
    print(f"   Idea: {article_idea}\n")

    # Build the workflow
    workflow = build_newsroom_workflow()

    # Run the workflow
    initial_state = {
        "article_idea": article_idea,
    }

    print("Running workflow...")
    print("   -> Assignment Editor: Creating story brief...")
    print("   -> Research Assistant: Gathering research...")
    print("   -> Reporter: Writing draft...")
    print("   -> Copy Editor: Polishing article...")
    print("   -> Graphic Desk: Generating hero image...")
    print("   -> Editor-in-Chief: Reviewing and approving...\n")

    result = workflow.invoke(initial_state)

    # Extract results
    story_brief = result.get("story_brief")
    research_package = result.get("research_package")
    draft_package = result.get("draft_package")
    final_article = result.get("final_article")
    hero_image_path = result.get("hero_image_path")
    approval = result.get("approval")

    # Print summary
    print("=" * 80)
    print("WORKFLOW RESULTS")
    print("=" * 80)

    if story_brief:
        print(f"\nStory Brief:")
        print(f"   Topic: {story_brief.topic}")
        print(f"   Article Type: {story_brief.article_type}")
        print(f"   Slug: {story_brief.slug}")

    if research_package:
        print(f"\nResearch Package:")
        print(f"   Sources: {len(research_package.results)}")

    if draft_package:
        word_count = len(draft_package.full_draft.split())
        print(f"\nDraft Package:")
        print(f"   Word Count: {word_count}")
        print(f"   Sources: {len(draft_package.sources)}")

    if final_article:
        word_count = len(final_article.article.split())
        print(f"\nFinal Article:")
        print(f"   Title: {final_article.title}")
        print(f"   Word Count: {word_count}")

    if hero_image_path:
        print(f"\nHero Image:")
        print(f"   Path: {hero_image_path}")

    if approval:
        print(f"\nPublication Approval:")
        status = "APPROVED" if approval.approved else "NOT APPROVED"
        print(f"   Status: {status}")
        if approval.notes:
            print("   Notes:")
            for note in approval.notes:
                print(f"     - {note}")

    if story_brief:
        print(f"\nArtifacts saved to: artifacts/{story_brief.slug}/")

    print("\n" + "=" * 80)

    # Print the final article if approved
    if final_article and approval and approval.approved:
        print("\nFINAL ARTICLE:\n")
        print(final_article.to_markdown())
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
