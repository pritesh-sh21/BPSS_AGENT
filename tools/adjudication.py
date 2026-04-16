"""
Tool: check_adjudication
=========================
Looks up whether a formally approved exception exists in the adjudication
register and whether it covers BPSS closure.

Policy: Screening_Operations_SOP.pdf
  "Where the tracker and adjudication register conflict, the adjudication
   register is the authority for exception approval."

Policy: BPSS_Screening_Policy_v3.pdf
  "Analyst notes, email statements, or verbal manager confirmation alone
   do not satisfy mandatory controls."

1. If no adjudication entry → return ok (no exception, nothing to check here)
2. Entry found → build citations
3. Check if "provisional start" in decision/scope but "closure" not mentioned → issue
4. Check if "no explicit waiver" in notes → issue
5. For provisional starts → check dual approval (Hiring Director + Screening Ops Lead)

"""

from __future__ import annotations
from store.models import BPSSKnowledgeStore
from tools.helpers import _cite, _ok, _fail, _get_record


def check_adjudication(store: BPSSKnowledgeStore, cand_id: str) -> dict:
    """
    Returns:
        passed=True  — no entry (no exception needed) OR entry fully authorises closure
        passed=False — entry exists but only covers provisional start, has no RTW waiver,
                       or lacks dual approval
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    citations = []
    issues = []

    # ── No entry = no approved exception ─────────────────────────────────────
    if r.adjudication is None:
        citations.append(_cite(
            "evidence/Adjudication_Register.pdf", "entry_for_candidate", "NONE"
        ))
        return _ok(
            f"{cand_id}: No adjudication entry. No approved exception exists. "
            f"All mandatory controls must be fully evidenced for closure.",
            citations,
        )

    adj = r.adjudication
    citations += [
        _cite("evidence/Adjudication_Register.pdf", "decision",  adj.decision),
        _cite("evidence/Adjudication_Register.pdf", "approvers", ", ".join(adj.approvers)),
        _cite("evidence/Adjudication_Register.pdf", "date",      adj.decision_date),
        _cite("evidence/Adjudication_Register.pdf", "scope",     adj.scope),
    ]

    # Check decision + scope + notes — "provisional start" can appear in any field
    full_text = (adj.decision + " " + adj.scope + " " + adj.notes).lower()
    scope_text = full_text  # kept for backward compat with other checks below

    # ── Provisional start ≠ BPSS closure ─────────────────────────────────────
    if "provisional start" in full_text and "closure" not in full_text:
        issues.append(
            f"Adjudication approves provisional start ONLY — NOT BPSS closure. "
            f"Case must remain open until all controls complete. "
            f"[evidence/Adjudication_Register.pdf | scope: {adj.scope}]"
        )

    # ── No explicit waiver of a mandatory control ─────────────────────────────
    if "no explicit waiver" in scope_text:
        issues.append(
            f"Register explicitly states no waiver of mandatory control requirement. "
            f"[evidence/Adjudication_Register.pdf]"
        )

    # ── Dual approval check for provisional starts ────────────────────────────
    if "provisional" in scope_text:
        has_hiring = any("hiring director" in a.lower() for a in adj.approvers)
        has_ops    = any("screening ops" in a.lower() or "operations lead" in a.lower()
                         for a in adj.approvers)
        if not (has_hiring and has_ops):
            issues.append(
                f"Provisional start requires BOTH Hiring Director AND Screening Ops Lead. "
                f"Found: {adj.approvers}. "
                f"[evidence/Adjudication_Register.pdf]"
            )

    if issues:
        return _fail(
            f"{cand_id}: Adjudication entry exists but does NOT authorise full closure.",
            issues,
            citations,
        )
    return _ok(
        f"{cand_id}: Adjudication entry — decision='{adj.decision}', "
        f"approvers={adj.approvers}, scope='{adj.scope}'.",
        citations,
    )