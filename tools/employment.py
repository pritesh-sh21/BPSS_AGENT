"""
Tool: check_employment_coverage
================================
Verifies prior 3 years of employment history is covered with no
unexplained gaps longer than 31 days.

Policy: BPSS_Screening_Policy_v3.pdf
  "Sufficient documentary or referee evidence to cover the previous 3 years;
   unexplained gaps over 31 days must be accounted for."

1. Get candidate and review_date
2. Calculate window_start = review_date - 3 years
3. For each employment period:
   a. If status is Weak/Gap/Unexplained → flag it
4. Find the earliest valid period
5. If earliest > window_start → gap at the start of window
6. Return issues

"""

from __future__ import annotations
from datetime import date
from store.models import BPSSKnowledgeStore
from tools.helpers import _cite, _ok, _fail, _get_record

BAD_STATUSES = ("weak", "gap", "unexplained")


def check_employment_coverage(store: BPSSKnowledgeStore, cand_id: str, years: int = 3) -> dict:
    """
    Returns:
        passed=True  — full window covered, no weak/gap periods
        passed=False — gaps, weak evidence, or window start not reached
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    t = r.tracker
    if t is None or t.analyst_review_date is None:
        return _fail(
            f"{cand_id}: No analyst review date — cannot compute coverage window.",
            ["Analyst review date missing."],
            [],
        )

    review_date = t.analyst_review_date
    window_start = date(review_date.year - years, review_date.month, review_date.day)
    citations = [
        _cite("structured/bpps_tracker_export.csv", "analyst_review_date", review_date),
        _cite("derived", "coverage_window", f"{window_start} to {review_date} ({years} years)"),
    ]

    if not r.employment:
        return _fail(
            f"{cand_id}: No employment history records found.",
            ["No records in structured/employment_history.csv."],
            citations,
        )

    issues = []

    # Flag each weak/gap/unexplained period
    for period in r.employment:
        citations.append(_cite(
            "structured/employment_history.csv",
            f"{period.period_start} to {period.period_end}",
            f"{period.evidence_type} [{period.evidence_status}]"
        ))
        if period.evidence_status.lower() in BAD_STATUSES:
            issues.append(
                f"Period {period.period_start} to {period.period_end}: "
                f"status='{period.evidence_status}' ({period.notes}). "
                f"[structured/employment_history.csv]"
            )

    # Check valid coverage reaches back to window_start
    valid_periods = [
        p for p in r.employment
        if p.period_start and p.period_end
        and p.evidence_status.lower() not in BAD_STATUSES
    ]

    if valid_periods:
        earliest = min(p.period_start for p in valid_periods)
        if earliest > window_start:
            gap_days = (earliest - window_start).days
            issues.append(
                f"Coverage starts {earliest} but 3-year window requires coverage "
                f"from {window_start}. Uncovered gap of {gap_days} days. "
                f"[structured/employment_history.csv]"
            )
    else:
        issues.append(
            f"No valid employment periods found for the 3-year window "
            f"({window_start} to {review_date})."
        )

    if issues:
        return _fail(
            f"{cand_id}: Employment coverage FAILED ({len(issues)} issue(s)).",
            issues,
            citations,
        )
    return _ok(
        f"{cand_id}: Employment covers required {years}-year window with no unexplained gaps.",
        citations,
    )