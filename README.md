# Agentic Newsroom

A LangGraph-based simulation of a newspaper editorial workflow, demonstrating how AI agents can collaborate to produce high-quality articles through a structured multi-agent system.

## Overview

Agentic Newsroom simulates a magazine editorial workflow where specialized AI agents collaborate to transform story ideas into polished, publication-ready articles. Each agent has a specific role in the newsroom hierarchy, mimicking how real editorial teams work.

## Agents

The system includes five specialized agents:

1. **Assignment Editor** - Transforms raw story ideas into structured story briefs with clear angles and research questions
2. **Research Assistant** - Conducts iterative web and Wikipedia research to gather factual material
3. **Reporter** - Writes article drafts based on story briefs and research packages
4. **Editor** - Refines and polishes drafts into final articles
5. **Editor-in-Chief** - Makes final publication decisions with approval or revision feedback

## Architecture

Built with:
- **LangGraph** - Orchestrates multi-agent workflows with state management
- **LangChain** - Provides LLM integrations and structured outputs
- **Pydantic** - Enforces typed data schemas throughout the pipeline

## Project Structure

```
agentic-newsroom/
├── src/agentic_newsroom/
│   ├── agents/          # Individual agent implementations
│   ├── workflows/       # Multi-agent workflow orchestration
│   ├── schemas.py       # Pydantic models for state and outputs
│   ├── prompts/         # System prompts and guidelines
│   └── tools/           # Research tools (Wikipedia, web search)
├── notebooks/           # Jupyter notebooks for agent development
│   └── [*_agent.ipynb]  # Frozen notebooks for each agent
└── examples/            # Example article outputs (gitignored)
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
   - OpenAI API key
   - Optional: LangSmith API key (for tracing)

### Usage

#### Running the Full Workflow

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
editor_decision = result["editor_decision"]

print(final_article.final_article)
```

#### Running Individual Agents

Each agent can be run independently from the command line. Agents form a pipeline where each stage saves its output for the next:

```bash
# 1. Assignment Editor - Creates story brief from idea
python -m agentic_newsroom.agents.assignment_editor "Deep sea hydrothermal vents"

# 2. Research Assistant - Gathers material (uses story brief)
python -m agentic_newsroom.agents.research_assistant deep_sea_hydrothermal_vents

# 3. Reporter - Writes draft (uses story brief + research)
python -m agentic_newsroom.agents.reporter deep_sea_hydrothermal_vents

# 4. Editor - Polishes draft (uses draft package)
python -m agentic_newsroom.agents.editor deep_sea_hydrothermal_vents

# 5. Editor-in-Chief - Makes publication decision (uses story brief + final article)
python -m agentic_newsroom.agents.editor_in_chief deep_sea_hydrothermal_vents
```

Each agent:
- Takes either an article idea (Assignment Editor) or a slug (all others)
- Loads required inputs from `tmp/[slug]/`
- Saves outputs to `tmp/[slug]/`
- Can be imported and used programmatically via `build_*_graph()` functions

## Notebooks

The `notebooks/` directory contains Jupyter notebooks used during initial development. These are **frozen snapshots** with inlined dependencies, kept as historical reference and quick testing tools. They are intentionally not kept in sync with the production codebase.

For actual use, import agents from `src/agentic_newsroom/agents/` as shown above.

## Configuration

### LLM Models

The system works with OpenAI model.

Default models:
- Assignment Editor: GPT-5.1 (reasoning: medium)
- Research Assistant: GPT-5-mini (reasoning: minimal)
- Reporter: GPT-5.1 (reasoning: medium)
- Editor: GPT-5.1 (reasoning: medium)
- Editor-in-Chief: GPT-5.1 (reasoning: medium)

### Magazine Profile

The editorial voice and standards are defined in `src/agentic_newsroom/prompts/common.py`. Customize these to match your publication's style.

## Development

### Code Formatting

```bash
poetry run black src/
poetry run ruff check src/
```

## Output

Generated articles and intermediate artifacts are saved to `tmp/` (gitignored):

```
tmp/
└── story-slug/
    ├── story_brief.json
    ├── research_package.json
    ├── draft_package.json
    ├── final_article.json
    └── editor_decision.json
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with [LangGraph](https://github.com/langchain-ai/langgraph)