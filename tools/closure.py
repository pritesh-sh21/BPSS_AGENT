"""
Tools: assess_closure_readiness + get_all_candidates_summary
=============================================================
Master tools that combine all five individual checks into a single verdict.

assess_closure_readiness(store, cand_id)
    Runs all checks for one candidate → unified verdict.

get_all_candidates_summary(store)
    Runs assess for every candidate → ranked summary (highest risk first).

1. Run all checks: freshness, employment, rtw, criminality, adjudication, contradictions
2. Collect all_issues and all_citations from every check
3. Determine verdict:
   - No issues → CLEAR
   - Has approved exception AND status=Risk Accepted → RISK ACCEPTED
   - Tracker=Clear BUT has issues → INCORRECTLY MARKED CLEAR
   - Anything else → NOT READY FOR CLOSURE
   
"""

from __future__ import annotations
from store.models import BPSSKnowledgeStore
from tools.helpers import _fail, _get_record
from tools.freshness import check_freshness
from tools.employment import check_employment_coverage
from tools.rtw import check_rtw
from tools.adjudication import check_adjudication
from tools.contradictions import detect_contradictions
from tools.criminality import check_criminality

PRIORITY = {
    "INCORRECTLY MARKED CLEAR": 0,
    "NOT READY FOR CLOSURE":    1,
    "RISK ACCEPTED":            2,
    "CLEAR":                    3,
}


def assess_closure_readiness(store: BPSSKnowledgeStore, cand_id: str) -> dict:
    """
    Runs all five checks and returns a unified closure verdict.

    Verdict options:
        CLEAR                    — all controls satisfied, no issues
        INCORRECTLY MARKED CLEAR — tracker says Clear but checks disagree
        NOT READY FOR CLOSURE    — open issues, no approved exception
        RISK ACCEPTED            — approved exception exists, operationally allowed
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    checks = {
        "freshness":      check_freshness(store, cand_id),
        "employment":     check_employment_coverage(store, cand_id),
        "rtw":            check_rtw(store, cand_id),
        "criminality":    check_criminality(store, cand_id),
        "adjudication":   check_adjudication(store, cand_id),
        "contradictions": detect_contradictions(store, cand_id),
    }

    all_issues    = [i for c in checks.values() for i in c.get("issues", [])]
    all_citations = [c for chk in checks.values() for c in chk.get("citations", [])]

    t = r.tracker
    tracker_status = t.status_tracker if t else "Unknown"
    has_approved_exception = (
        r.adjudication is not None and checks["adjudication"]["passed"]
    )

    # ── Determine verdict ─────────────────────────────────────────────────────
    if not all_issues:
        verdict, ready = "CLEAR", True
    elif has_approved_exception and tracker_status == "Risk Accepted":
        verdict, ready = "RISK ACCEPTED", True
    elif tracker_status == "Clear" and all_issues:
        verdict, ready = "INCORRECTLY MARKED CLEAR", False
    else:
        verdict, ready = "NOT READY FOR CLOSURE", False

    return {
        "candidate_id":      cand_id,
        "candidate_name":    t.candidate_name if t else "Unknown",
        "role":              t.role_code if t else "Unknown",
        "tracker_status":    tracker_status,
        "overall_verdict":   verdict,
        "ready_for_closure": ready,
        "risk_level":        t.risk_level if t else "Unknown",
        "issue_count":       len(all_issues),
        "checks":            checks,
        "all_issues":        all_issues,
        "all_citations":     all_citations,
    }


def get_all_candidates_summary(store: BPSSKnowledgeStore) -> dict:
    """
    Runs assess_closure_readiness for every candidate and returns
    a summary ranked highest risk first.
    """
    results = {
        cid: assess_closure_readiness(store, cid)
        for cid in store.all_candidate_ids()
    }

    ranked = sorted(
        results.values(),
        key=lambda x: (
            PRIORITY.get(x.get("overall_verdict", ""), 99),
            -x.get("issue_count", 0),
        )
    )

    return {
        "total_candidates":  len(results),
        "ready_for_closure": sum(1 for r in results.values() if r.get("ready_for_closure")),
        "not_ready":         sum(1 for r in results.values() if not r.get("ready_for_closure")),
        "ranked_results":    ranked,
    }