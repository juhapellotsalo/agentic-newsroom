# Agentic Newsroom

A LangGraph-based simulation of a newspaper editorial workflow, demonstrating how AI agents can collaborate to produce high-quality articles through a structured multi-agent system.

## Overview

Agentic Newsroom simulates a magazine editorial workflow where specialized AI agents collaborate to transform story ideas into polished, publication-ready articles. Each agent has a specific role in the newsroom hierarchy, mimicking how real editorial teams work.

## Agents

The system includes six specialized agents:

1. **Assignment Editor** - Transforms raw story ideas into structured story briefs with clear angles and research questions
2. **Research Assistant** - Conducts iterative web and Wikipedia research to gather factual material
3. **Reporter** - Writes article drafts based on story briefs and research packages, with fact and style review passes
4. **Copy Editor** - Refines and polishes drafts into final articles with title and subtitle
5. **Graphic Desk** - Generates hero images for articles using AI image generation
6. **Editor-in-Chief** - Reviews articles against magazine guardrails and approves for publication

## Workflow

The full pipeline flows through all agents sequentially:

```
START → Assignment Editor → Research Assistant → Reporter → Copy Editor → Graphic Desk → Editor-in-Chief → END
```

## Architecture

Built with:
- **LangGraph** - Orchestrates multi-agent workflows with state management
- **LangChain** - Provides LLM integrations and structured outputs
- **Pydantic** - Enforces typed data schemas throughout the pipeline
- **OpenAI** - Powers both text generation and image generation

See [docs/architecture.md](docs/architecture.md) for detailed diagrams of each agent's graph.

## Project Structure

```
agentic-newsroom/
├── src/agentic_newsroom/
│   ├── agents/           # Individual agent implementations
│   ├── workflows/        # Multi-agent workflow orchestration
│   ├── schemas/          # Pydantic models for state and outputs
│   │   ├── models.py     # Data models (StoryBrief, DraftPackage, etc.)
│   │   └── states.py     # Agent state definitions
│   ├── prompts/          # System prompts and guidelines
│   ├── llm/              # LLM model configuration
│   └── utils/            # Helper utilities
├── notebooks/            # Jupyter notebooks for agent development
├── examples/             # Example article outputs
└── artifacts/            # Generated article artifacts (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)

### Installation

1. Install dependencies:
```bash
poetry install
```

2. Add your API keys to `.env`:
```bash
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key

# Optional: LangSmith for tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=agentic-newsroom
```

### Usage

#### Running the Full Workflow

From the command line:
```bash
python main.py "The mysterious dragon blood trees of Socotra Island"
```

**Note:** A full workflow run uses approximately 150,000 tokens and costs $0.30-0.50 USD.

Or programmatically:
```python
from agentic_newsroom.workflows.newsroom_workflow import build_newsroom_workflow

# Build and run the complete workflow
newsroom = build_newsroom_workflow()
result = newsroom.invoke({
    "article_idea": "The mysterious dragon blood trees of Socotra Island"
})

# Access outputs at each stage
story_brief = result["story_brief"]
research_package = result["research_package"]
draft_package = result["draft_package"]
final_article = result["final_article"]
hero_image_path = result["hero_image_path"]
approval = result["approval"]

print(final_article.to_markdown())
```

#### Running Individual Agents

Each agent can be run independently from the command line. Agents form a pipeline where each stage saves its output for the next:

```bash
# 1. Assignment Editor - Creates story brief from idea
python -m agentic_newsroom.agents.assignment_editor "Deep sea hydrothermal vents"

# 2. Research Assistant - Gathers material (uses story brief)
python -m agentic_newsroom.agents.research_assistant deep-sea-hydrothermal-vents

# 3. Reporter - Writes draft (uses story brief + research)
python -m agentic_newsroom.agents.reporter deep-sea-hydrothermal-vents

# 4. Copy Editor - Polishes draft into final article
python -m agentic_newsroom.agents.copy_editor deep-sea-hydrothermal-vents

# 5. Graphic Desk - Generates hero image
python -m agentic_newsroom.agents.graphic_desk deep-sea-hydrothermal-vents

# 6. Editor-in-Chief - Reviews and approves for publication
python -m agentic_newsroom.agents.editor_in_chief deep-sea-hydrothermal-vents
```

Most agents support a `--mini` flag to use the faster/cheaper mini model:
```bash
python -m agentic_newsroom.agents.reporter deep-sea-hydrothermal-vents --mini
```

Each agent:
- Takes either an article idea (Assignment Editor) or a slug (all others)
- Loads required inputs from `artifacts/[slug]/`
- Saves outputs to `artifacts/[slug]/`
- Can be imported and used programmatically via `build_*_graph()` functions

## Configuration

### LLM Models

The system uses OpenAI models configured in `src/agentic_newsroom/llm/openai.py`:
- **Smart model** - Used for creative/complex tasks (writing, editing)
- **Mini model** - Used for analytical tasks (research, review)

### Magazine Profile

The editorial voice and standards are defined in `src/agentic_newsroom/prompts/common.py`:
- `magazine_profile` - Publication's mission and voice
- `magazine_guardrails` - Editorial standards and policies
- `article_types` - Definitions for Web Daily (400-700 words) and Full Feature (1500-2000 words)

## Output

Generated articles and intermediate artifacts are saved to `artifacts/` (gitignored):

```
artifacts/
└── story-slug/
    ├── story_brief.json
    ├── research_package.json
    ├── draft_package.json
    ├── final_article.json
    ├── publication_approval.json
    ├── reporter/              # Reporter intermediate files
    │   ├── 1_initial_draft.md
    │   ├── 2_fact_review.md
    │   ├── 3_after_fact_revision.md
    │   ├── 4_style_review.md
    │   └── 5_after_style_revision.md
    └── graphics/              # Generated images
        ├── hero_prompt.txt
        └── hero_image.png
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with [LangGraph](https://github.com/langchain-ai/langgraph)