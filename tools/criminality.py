"""
Tool: check_criminality
========================
Verifies whether the criminality / basic disclosure check is complete.

Policy: BPSS_Screening_Policy_v3.pdf
  "Criminality/basic disclosure: required for all standard hires unless
   the role code is INTERN or CONTRACTOR-LTD."
"""

from __future__ import annotations
from store.models import BPSSKnowledgeStore
from tools.helpers import _cite, _ok, _fail, _get_record

EXEMPT_ROLES = {"INTERN", "CONTRACTOR-LTD"}
CRIM_KEYWORDS = ("disclosure", "dbs", "basic", "crim")


def check_criminality(store: BPSSKnowledgeStore, cand_id: str) -> dict:
    """
    Returns:
        passed=True  — DBS/disclosure doc present and clear, OR role is exempt
        passed=False — doc missing, pending, or tracker marks incomplete
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    t = r.tracker
    citations = [
        _cite("structured/bpps_tracker_export.csv", "role_code",              t.role_code),
        _cite("structured/bpps_tracker_export.csv", "criminality_complete",   t.criminality_complete),
        _cite("structured/bpps_tracker_export.csv", "criminality_na",         t.criminality_na),
    ]

    # ── Exempt roles ──────────────────────────────────────────────────────────
    if t.role_code.upper() in EXEMPT_ROLES or t.criminality_na:
        citations.append(_cite(
            "policies/BPSS_Screening_Policy_v3.pdf",
            "criminality_exemption",
            f"Role '{t.role_code}' is exempt from criminality check per policy."
        ))
        return _ok(
            f"{cand_id}: Criminality check NOT required — role '{t.role_code}' "
            f"is exempt per BPSS_Screening_Policy_v3.pdf.",
            citations,
        )

    # ── Look for DBS / disclosure document ───────────────────────────────────
    crim_docs = [
        d for d in r.documents
        if any(kw in d.doc_type.lower() for kw in CRIM_KEYWORDS)
    ]

    issues = []

    if not crim_docs:
        # Check if tracker also says incomplete — double confirmation
        if not t.criminality_complete:
            issues.append(
                f"No criminality/basic disclosure document found in evidence inventory "
                f"AND tracker marks criminality_complete=False. "
                f"[structured/document_inventory.csv | "
                f"structured/bpps_tracker_export.csv]"
            )
        else:
            issues.append(
                f"No criminality/basic disclosure document found in evidence inventory. "
                f"[structured/document_inventory.csv]"
            )
    else:
        for doc in crim_docs:
            citations.append(_cite(
                "structured/document_inventory.csv", "document_id", doc.document_id
            ))
            citations.append(_cite(
                "structured/document_inventory.csv", "remarks", doc.remarks
            ))
            if not doc.present_in_folder:
                issues.append(
                    f"{doc.document_id} ({doc.doc_type}) not present in evidence folder. "
                    f"[structured/document_inventory.csv | remarks: {doc.remarks}]"
                )

    # ── Tracker says incomplete even if doc exists ────────────────────────────
    if not t.criminality_complete and not issues:
        issues.append(
            f"Tracker marks criminality_complete=False — check may still be pending. "
            f"[structured/bpps_tracker_export.csv]"
        )

    # ── Check adjudication for provisional start scope ────────────────────────
    if r.adjudication:
        scope = (r.adjudication.scope + " " + r.adjudication.notes).lower()
        if "dbs" in scope or "before dbs" in scope:
            issues.append(
                f"Adjudication register confirms DBS/criminality check was NOT "
                f"complete at time of exception approval: '{r.adjudication.scope}'. "
                f"[evidence/Adjudication_Register.pdf]"
            )
            citations.append(_cite(
                "evidence/Adjudication_Register.pdf",
                "scope", r.adjudication.scope
            ))

    if issues:
        return _fail(
            f"{cand_id}: Criminality check INCOMPLETE ({len(issues)} issue(s)).",
            issues, citations,
        )
    return _ok(
        f"{cand_id}: Criminality/basic disclosure check complete.",
        citations,
    )