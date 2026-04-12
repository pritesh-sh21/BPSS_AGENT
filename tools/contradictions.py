"""
Tool: detect_contradictions
============================
Cross-references tracker claims against actual evidence to surface
contradictions.

Detects:
  1. Tracker='Clear' but live checks found open issues
  2. Individual control flags (identity/rtw/employment) vs live evidence
  3. ready_to_join=True with non-Clear status (Ready to Join ≠ BPSS closure)
  4. Verbal/email approval relied upon as evidence
"""

from __future__ import annotations
from store.models import BPSSKnowledgeStore
from tools.helpers import _cite, _ok, _fail, _get_record
from tools.freshness import check_freshness
from tools.employment import check_employment_coverage
from tools.rtw import check_rtw

VERBAL_SIGNALS = (
    "verbally confirmed",
    "verbal confirmation",
    "manager confirmed",
    "i spoke with",
    "am comfortable with",
)


def detect_contradictions(store: BPSSKnowledgeStore, cand_id: str) -> dict:
    """
    Returns:
        passed=True  — no contradictions found
        passed=False — one or more contradictions between tracker and evidence
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    t = r.tracker
    if t is None:
        return _fail(f"{cand_id}: No tracker row found.", ["Missing tracker data."], [])

    contradictions = []
    citations = [_cite("structured/bpps_tracker_export.csv", "status_tracker", t.status_tracker)]

    # Run live checks — same logic used by assess_closure_readiness
    live_checks = {
        "freshness":  check_freshness(store, cand_id),
        "employment": check_employment_coverage(store, cand_id),
        "rtw":        check_rtw(store, cand_id),
    }
    live_issues = {name: res["issues"] for name, res in live_checks.items()}

    # ── 1. Tracker='Clear' but live checks found issues ───────────────────────
    if t.status_tracker == "Clear":
        all_live = [i for issues in live_issues.values() for i in issues]
        if all_live:
            contradictions.append(
                f"CONTRADICTION: Tracker='Clear' but live checks found "
                f"{len(all_live)} unresolved issue(s). First: {all_live[0]} "
                f"[bpps_tracker_export.csv vs evidence folder]"
            )
        if all_live and r.adjudication is None:
            contradictions.append(
                f"CONTRADICTION: Status='Clear' with open issues but NO adjudication "
                f"register entry — no approved exception exists to justify closure. "
                f"[bpps_tracker_export.csv vs Adjudication_Register.pdf]"
            )

    # ── 2. Individual control flags vs live evidence ──────────────────────────
    flag_map = {
        "identity":   (t.identity_complete,   live_issues.get("freshness", [])),
        "employment": (t.employment_complete,  live_issues.get("employment", [])),
        "rtw":        (t.rtw_complete,         live_issues.get("rtw", [])),
    }
    for ctrl_name, (tracker_says, issues) in flag_map.items():
        if tracker_says and issues:
            citations.append(_cite(
                "structured/bpps_tracker_export.csv",
                f"{ctrl_name}_complete", tracker_says
            ))
            contradictions.append(
                f"CONTRADICTION: Tracker '{ctrl_name}_complete=True' but evidence shows: "
                f"{issues[0]} "
                f"[bpps_tracker_export.csv vs evidence folder]"
            )

    # ── 3. Ready to Join ≠ BPSS closure ──────────────────────────────────────
    # Flag ready_to_join=True whenever there are open issues — regardless of status
    all_live_issues = [i for issues in live_issues.values() for i in issues]
    if t.ready_to_join and all_live_issues:
        citations.append(_cite(
            "structured/bpps_tracker_export.csv", "ready_to_join", t.ready_to_join
        ))
        if t.status_tracker != "Clear":
            contradictions.append(
                f"CONTRADICTION: ready_to_join=True with status='{t.status_tracker}'. "
                f"SOP: 'Ready to Join indicates business onboarding readiness and is NOT "
                f"equivalent to BPSS closure.' "
                f"[bpps_tracker_export.csv vs Screening_Operations_SOP.pdf]"
            )
        else:
            contradictions.append(
                f"CONTRADICTION: ready_to_join=True and status='Clear' but "
                f"{len(all_live_issues)} mandatory control issue(s) remain unresolved. "
                f"Candidate should not be marked ready to join with open controls. "
                f"[bpps_tracker_export.csv vs evidence folder]"
            )

    # ── 4. Verbal/email approval treated as evidence ──────────────────────────
    sources = [
        (r.analyst_notes_text,  "evidence/Analyst_Working_Notes.docx"),
        (r.email_mentions_text, "evidence/Email_Approvals_and_Escalations.docx"),
        (r.candidate_pack_text, f"candidate_pack/{cand_id}_candidate_pack.docx"),
    ]
    for source_text, source_name in sources:
        for signal in VERBAL_SIGNALS:
            if signal in source_text.lower():
                contradictions.append(
                    f"CONTRADICTION: Informal approval detected in {source_name} "
                    f"('{signal}'). Policy: 'Verbal manager confirmation alone does "
                    f"not satisfy mandatory controls.' "
                    f"[{source_name} vs BPSS_Screening_Policy_v3.pdf]"
                )
                citations.append(_cite(source_name, "verbal_signal", signal))
                break  # one flag per source is enough

    if not contradictions:
        return _ok(
            f"{cand_id}: No contradictions detected between tracker and evidence.",
            citations,
        )
    return _fail(
        f"{cand_id}: {len(contradictions)} contradiction(s) detected.",
        contradictions,
        citations,
    )