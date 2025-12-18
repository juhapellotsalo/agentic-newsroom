from langgraph.graph import START, END, StateGraph
from agentic_newsroom.schemas.states import NewsroomState
from agentic_newsroom.agents.assignment_editor import build_assignment_editor_graph
from agentic_newsroom.agents.research_assistant import build_research_assistant_graph, DEFAULT_MAX_TURNS
from agentic_newsroom.agents.reporter import build_reporter_graph
from agentic_newsroom.agents.copy_editor import build_copy_editor_graph
from agentic_newsroom.agents.graphic_desk import build_graphic_desk_graph
from agentic_newsroom.agents.editor_in_chief import build_editor_in_chief_graph

# Build subgraphs
assignment_editor_graph = build_assignment_editor_graph()
research_assistant_graph = build_research_assistant_graph()
reporter_graph = build_reporter_graph()
copy_editor_graph = build_copy_editor_graph()
graphic_desk_graph = build_graphic_desk_graph()
editor_in_chief_graph = build_editor_in_chief_graph()


def run_assignment_editor(state: NewsroomState):
    """Run assignment editor to create story brief from article idea."""
    result = assignment_editor_graph.invoke(state)
    return {"story_brief": result["story_brief"]}


def run_research_assistant(state: NewsroomState):
    """Run research assistant to gather material for the story."""
    initial_research_state = {
        "story_brief": state["story_brief"],
        "max_turns": DEFAULT_MAX_TURNS,
        "current_turn": 0,
        "context": [],
        "search_results": []
    }
    result = research_assistant_graph.invoke(initial_research_state)
    return {"research_package": result["research_package"]}


def run_reporter(state: NewsroomState):
    """Run reporter to write the article draft."""
    reporter_state = {
        "story_brief": state["story_brief"],
        "research_package": state["research_package"],
    }
    result = reporter_graph.invoke(reporter_state)
    return {"draft_package": result["draft_package"]}


def run_copy_editor(state: NewsroomState):
    """Run copy editor to polish the draft into final article."""
    copy_editor_state = {
        "story_brief": state["story_brief"],
        "draft_package": state["draft_package"]
    }
    result = copy_editor_graph.invoke(copy_editor_state)
    return {"final_article": result["final_article"]}


def run_graphic_desk(state: NewsroomState):
    """Run graphic desk to generate hero image."""
    graphic_state = {
        "story_brief": state["story_brief"],
        "final_article": state["final_article"]
    }
    result = graphic_desk_graph.invoke(graphic_state)
    return {"hero_image_path": result.get("hero_image_path")}


def run_editor_in_chief(state: NewsroomState):
    """Run editor in chief to review and approve the article."""
    eic_state = {
        "story_brief": state["story_brief"],
        "final_article": state["final_article"]
    }
    result = editor_in_chief_graph.invoke(eic_state)
    return {"approval": result["approval"]}


def build_newsroom_workflow():
    """
    Build the full newsroom workflow:

    START → assignment_editor → research_assistant → reporter → copy_editor → graphic_desk → editor_in_chief → END
    """
    builder = StateGraph(NewsroomState)

    builder.add_node("assignment_editor", run_assignment_editor)
    builder.add_node("research_assistant", run_research_assistant)
    builder.add_node("reporter", run_reporter)
    builder.add_node("copy_editor", run_copy_editor)
    builder.add_node("graphic_desk", run_graphic_desk)
    builder.add_node("editor_in_chief", run_editor_in_chief)

    builder.add_edge(START, "assignment_editor")
    builder.add_edge("assignment_editor", "research_assistant")
    builder.add_edge("research_assistant", "reporter")
    builder.add_edge("reporter", "copy_editor")
    builder.add_edge("copy_editor", "graphic_desk")
    builder.add_edge("graphic_desk", "editor_in_chief")
    builder.add_edge("editor_in_chief", END)

    return builder.compile()