"""
Shared helpers for all BPSS business-logic tools.
"""

from __future__ import annotations
from typing import Any
from store.models import BPSSKnowledgeStore


def _cite(source: str, field: str, value: Any) -> dict:
    return {"source": source, "field": field, "value": str(value)}


def _ok(result: str, citations: list) -> dict:
    return {"result": result, "passed": True, "issues": [], "citations": citations}


def _fail(result: str, issues: list, citations: list) -> dict:
    return {"result": result, "passed": False, "issues": issues, "citations": citations}


def _get_record(store: BPSSKnowledgeStore, cand_id: str):
    """Return (record, error_dict). error_dict is set if candidate not found."""
    r = store.get_candidate(cand_id)
    if r is None:
        return None, _fail(
            f"Candidate {cand_id} not found.",
            [f"Unknown candidate ID: {cand_id}"],
            [],
        )
    return r, None