# Architecture

Agentic Newsroom is a multi-agent system built with LangGraph. Each agent is a specialized LangGraph subgraph that handles one part of the editorial pipeline.

## Full Workflow

The complete pipeline runs six agents sequentially:

![Full Workflow](images/workflow.png)

Each agent receives state from the previous step, processes it, and passes results forward. All artifacts are saved to `artifacts/{slug}/` as the workflow progresses.

## Agents

### Assignment Editor

Transforms a raw article idea into a structured story brief with topic, angle, article type, and key research questions.

![Assignment Editor](images/assignment_editor.png)

### Research Assistant

Conducts iterative web research using Tavily search. Runs multiple turns, generating queries, searching, curating results, and extracting relevant information until research is complete.

![Research Assistant](images/research_assistant.png)

### Reporter

Writes the article draft through a two-pass review process: fact review followed by style review, with revisions after each.

![Reporter](images/reporter.png)

### Copy Editor

Polishes the draft into a final article with headline and subtitle.

![Copy Editor](images/copy_editor.png)

### Graphic Desk

Generates a hero image for the article using AI image generation.

![Graphic Desk](images/graphic_desk.png)

### Editor in Chief

Reviews the final article against magazine guardrails and approves for publication.

![Editor in Chief](images/editor_in_chief.png)

## Key Patterns

- **Subgraph composition**: Each agent is a compiled LangGraph that's invoked from the main workflow
- **Typed state**: Pydantic models ensure consistent data flow between agents
- **Artifact persistence**: Each agent saves its outputs to disk, enabling individual agent runs
- **Model flexibility**: Agents can use either smart or mini models via configuration