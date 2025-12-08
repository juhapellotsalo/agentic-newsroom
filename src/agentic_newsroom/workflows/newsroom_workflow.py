from langgraph.graph import START, END, StateGraph
from agentic_newsroom.schemas import NewsroomState, ResearchPackage
from agentic_newsroom.agents.assignment_editor import build_assignment_editor_graph
from agentic_newsroom.agents.research_assistant import build_research_assistant_graph
from agentic_newsroom.agents.reporter import build_reporter_graph
from agentic_newsroom.agents.editor import build_editor_graph
from agentic_newsroom.agents.editor_in_chief import build_editor_in_chief_graph

# Build subgraphs
assignment_editor_graph = build_assignment_editor_graph()
research_assistant_graph = build_research_assistant_graph()
reporter_graph = build_reporter_graph()
editor_graph = build_editor_graph()
editor_in_chief_graph = build_editor_in_chief_graph()

def run_assignment_editor(state: NewsroomState):
    # Assignment editor graph uses NewsroomState, so we can pass it directly
    # But we need to be careful if it expects only specific keys.
    # The graph definition in assignment_editor.py uses NewsroomState.
    result = assignment_editor_graph.invoke(state)
    return {"story_brief": result["story_brief"]}

def run_research_assistant(state: NewsroomState):
    initial_research_state = {
        "story_brief": state["story_brief"],
        "max_num_turns": 3,
        "current_turn": 0,
        "context": [],
        "search_results": []
    }
    result = research_assistant_graph.invoke(initial_research_state)
    return {"research_package": ResearchPackage(results=result["search_results"])}

def run_reporter(state: NewsroomState):
    reporter_state = {
        "story_brief": state["story_brief"],
        "research_package": state["research_package"],
        "feedback": state.get("feedback")
    }
    result = reporter_graph.invoke(reporter_state)
    return {"draft_package": result["draft_package"]}

def run_editor(state: NewsroomState):
    editor_state = {
        "draft_package": state["draft_package"]
    }
    result = editor_graph.invoke(editor_state)
    return {"final_article": result["final_article"]}

def run_editor_in_chief(state: NewsroomState):
    eic_state = {
        "story_brief": state["story_brief"],
        "final_article": state["final_article"]
    }
    result = editor_in_chief_graph.invoke(eic_state)
    return {"editor_decision": result["editor_decision"]}

def process_feedback(state: NewsroomState):
    decision = state["editor_decision"]
    feedback = "The article was rejected. Please address the following issues:\n"
    for reason in decision.rejection_reasons:
        feedback += f"- {reason}\n"
    return {"feedback": feedback}

def check_decision(state: NewsroomState):
    decision = state["editor_decision"]
    if decision.decision == "approve":
        return END
    else:
        return "process_feedback"

def build_newsroom_workflow():
    builder = StateGraph(NewsroomState)
    
    builder.add_node("assignment_editor", run_assignment_editor)
    builder.add_node("research_assistant", run_research_assistant)
    builder.add_node("reporter", run_reporter)
    builder.add_node("editor", run_editor)
    builder.add_node("editor_in_chief", run_editor_in_chief)
    builder.add_node("process_feedback", process_feedback)
    
    builder.add_edge(START, "assignment_editor")
    builder.add_edge("assignment_editor", "research_assistant")
    builder.add_edge("research_assistant", "reporter")
    builder.add_edge("reporter", "editor")
    builder.add_edge("editor", "editor_in_chief")
    
    builder.add_conditional_edges(
        "editor_in_chief",
        check_decision,
        {END: END, "process_feedback": "process_feedback"}
    )
    
    builder.add_edge("process_feedback", "reporter")
    
    return builder.compile()
