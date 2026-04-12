"""
LangGraph Graph — Phase 3
==========================
Wires planner → executor → synthesiser into a StateGraph.

Entry point: run(question, store) → str

Usage:
    from agent.graph import run
    answer = run("Which candidates are not ready for closure?", store)
    print(answer)
"""

from __future__ import annotations

import logging
from pathlib import Path

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import planner_node, executor_node, synthesiser_node
from store.models import BPSSKnowledgeStore

logger = logging.getLogger(__name__)


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Graph structure:
        START → planner → executor → synthesiser → END

    All three nodes share the same AgentState dict.
    Each node returns a partial dict that gets merged into the state.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner",     planner_node)
    graph.add_node("executor",    executor_node)
    graph.add_node("synthesiser", synthesiser_node)

    # Wire edges
    graph.set_entry_point("planner")
    graph.add_edge("planner",     "executor")
    graph.add_edge("executor",    "synthesiser")
    graph.add_edge("synthesiser", END)

    return graph.compile()


# Cache the compiled graph — only built once per process
_compiled_graph = None

def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── Public entry point ────────────────────────────────────────────────────────

def run(question: str, store: BPSSKnowledgeStore) -> str:
    """
    Ask the BPSS agent a question and get a grounded cited answer.

    Args:
        question : natural language question about the BPSS dataset
        store    : populated BPSSKnowledgeStore from build_knowledge_store()

    Returns:
        A string answer with citations to source files and fields.
    """
    graph = _get_graph()

    initial_state: AgentState = {
        "question":     question,
        "store":        store,
        "plan":         [],
        "tool_results": [],
        "final_answer": "",
        "iterations":   0,
        "error":        "",
    }

    logger.info("Running agent for question: %s", question)
    final_state = graph.invoke(initial_state)

    if final_state.get("error"):
        logger.error("Agent error: %s", final_state["error"])
        return f"Error: {final_state['error']}"

    return final_state.get("final_answer", "No answer produced.")