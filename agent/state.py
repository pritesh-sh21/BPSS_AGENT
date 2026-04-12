"""
AgentState
==========
The single dict that flows through every node in the LangGraph graph.

Each node reads from it and writes back to it.
LangGraph merges the returned dict into the existing state automatically.
"""

from __future__ import annotations
from typing import TypedDict, Any


class AgentState(TypedDict, total=False):
    """
    Fields
    ------
    question        : the original user question (never changes)
    store           : the in-memory BPSS knowledge store (read-only)
    plan            : list of tool calls the planner wants to make
                      each item: {"tool": str, "cand_id": str | None, "reason": str}
    tool_results    : list of results returned by the executor
                      each item: {"tool": str, "cand_id": str | None, "result": dict}
    final_answer    : the grounded, cited answer string from the synthesiser
    iterations      : safety counter — graph aborts if this exceeds MAX_ITER
    error           : set if something goes wrong in any node
    """
    question:     str
    store:        Any                  # BPSSKnowledgeStore (Any avoids TypedDict issues)
    plan:         list[dict]
    tool_results: list[dict]
    final_answer: str
    iterations:   int
    error:        str