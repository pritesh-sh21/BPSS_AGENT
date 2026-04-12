"""
Tests for Phase 2 business-logic tools.
Run: python -m pytest tests/test_business_logic.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ingestion.ingest import build_knowledge_store
from tools import (
    check_freshness,
    check_employment_coverage,
    check_rtw,
    check_adjudication,
    detect_contradictions,
    assess_closure_readiness,
    get_all_candidates_summary,
)

DATA_DIR = Path(__file__).parent.parent.parent / "bpss_agentic_dataset"


@pytest.fixture(scope="module")
def store():
    return build_knowledge_store(DATA_DIR)


# ── check_freshness ───────────────────────────────────────────────────────────

class TestCheckFreshness:
    def test_cand101_passes(self, store):
        # Bank statement 2026-01-28, review 2026-02-11 → 14 days → pass
        r = check_freshness(store, "CAND-101")
        assert r["passed"] is True
        assert "101" in r["result"]

    def test_cand102_fails_stale_address(self, store):
        # Utility bill 2025-10-01, review 2026-02-04 → 126 days → fail
        r = check_freshness(store, "CAND-102")
        assert r["passed"] is False
        assert any("126" in issue for issue in r["issues"])

    def test_cand104_fails_missing_address(self, store):
        # Address proof not retained in folder
        r = check_freshness(store, "CAND-104")
        assert r["passed"] is False
        assert any("not retained" in i.lower() for i in r["issues"])

    def test_cand105_fails_no_address_doc(self, store):
        # No address proof document at all
        r = check_freshness(store, "CAND-105")
        assert r["passed"] is False

    def test_cand103_passes_fresh_address(self, store):
        # Address proof 2026-02-20, review 2026-02-25 → 5 days → pass
        r = check_freshness(store, "CAND-103")
        assert r["passed"] is True

    def test_citations_always_present(self, store):
        for cid in ["CAND-101", "CAND-102", "CAND-104"]:
            r = check_freshness(store, cid)
            assert len(r["citations"]) > 0

    def test_unknown_candidate(self, store):
        r = check_freshness(store, "CAND-999")
        assert r["passed"] is False
        assert "not found" in r["result"].lower()


# ── check_employment_coverage ─────────────────────────────────────────────────

class TestCheckEmploymentCoverage:
    def test_cand101_passes(self, store):
        # Full coverage Jan 2023 to Jan 2026, no gaps
        r = check_employment_coverage(store, "CAND-101")
        assert r["passed"] is True

    def test_cand102_fails_unexplained_gap(self, store):
        # Gap Apr-May 2024 marked Unexplained
        r = check_employment_coverage(store, "CAND-102")
        assert r["passed"] is False
        assert any("unexplained" in i.lower() or "gap" in i.lower() for i in r["issues"])

    def test_cand104_fails_weak_evidence(self, store):
        # 2023-01 to 2024-07 is CV only / Weak
        r = check_employment_coverage(store, "CAND-104")
        assert r["passed"] is False
        assert any("weak" in i.lower() for i in r["issues"])

    def test_cand105_fails_coverage_gap(self, store):
        # Only from 2024-06 onward — 18 months uncovered
        r = check_employment_coverage(store, "CAND-105")
        assert r["passed"] is False
        assert any("window" in i.lower() or "coverage" in i.lower() for i in r["issues"])

    def test_cand103_passes_education_accepted(self, store):
        # Student education declaration accepted
        r = check_employment_coverage(store, "CAND-103")
        assert r["passed"] is True

    def test_coverage_window_cited(self, store):
        r = check_employment_coverage(store, "CAND-101")
        window_citation = next(
            (c for c in r["citations"] if c["field"] == "coverage_window"), None
        )
        assert window_citation is not None
        assert "2023" in window_citation["value"]


# ── check_rtw ─────────────────────────────────────────────────────────────────

class TestCheckRTW:
    def test_cand101_passes_uk_passport(self, store):
        # Valid UK passport = valid RTW
        r = check_rtw(store, "CAND-101")
        assert r["passed"] is True

    def test_cand105_fails_expired_brp(self, store):
        # BRP expired 2024-12-31, review 2026-03-03
        r = check_rtw(store, "CAND-105")
        assert r["passed"] is False
        assert any("expired" in i.lower() and "brp" in i.lower() for i in r["issues"])

    def test_cand106_fails_contractor_rtw(self, store):
        # Recruiter assumption about contractor RTW exemption is invalid
        r = check_rtw(store, "CAND-106")
        assert r["passed"] is False
        assert any("contractor" in i.lower() for i in r["issues"])

    def test_cand102_passes_valid_evisa(self, store):
        # eVisa share code valid at check
        r = check_rtw(store, "CAND-102")
        assert r["passed"] is True

    def test_rtw_matrix_cited_for_contractor(self, store):
        r = check_rtw(store, "CAND-106")
        assert any("RTW_Evidence_Matrix" in c["source"] for c in r["citations"])


# ── check_adjudication ────────────────────────────────────────────────────────

class TestCheckAdjudication:
    def test_cand101_no_entry_ok(self, store):
        # No adjudication needed — all controls clean
        r = check_adjudication(store, "CAND-101")
        assert r["passed"] is True
        assert "no adjudication entry" in r["result"].lower()

    def test_cand102_provisional_not_closure(self, store):
        # Entry exists but approves provisional start only, not closure
        r = check_adjudication(store, "CAND-102")
        assert r["passed"] is False
        assert any("provisional" in i.lower() for i in r["issues"])

    def test_cand104_no_entry_is_problem(self, store):
        # No entry = no approved exception — yet tracker says Clear
        r = check_adjudication(store, "CAND-104")
        assert r["passed"] is True  # tool itself passes (no entry = nothing wrong here)
        # The contradiction is caught by detect_contradictions instead
        assert "no approved exception" in r["result"].lower()

    def test_cand106_risk_accepted_has_limits(self, store):
        # Entry exists but explicitly has no RTW waiver
        r = check_adjudication(store, "CAND-106")
        assert r["passed"] is False
        assert any("no explicit waiver" in i.lower() or "waiver" in i.lower()
                   for i in r["issues"])

    def test_adjudication_register_always_cited(self, store):
        for cid in ["CAND-101", "CAND-102", "CAND-106"]:
            r = check_adjudication(store, cid)
            assert any("Adjudication_Register" in c["source"] for c in r["citations"])


# ── detect_contradictions ─────────────────────────────────────────────────────

class TestDetectContradictions:
    def test_cand101_no_contradictions(self, store):
        r = detect_contradictions(store, "CAND-101")
        assert r["passed"] is True
        assert r["issues"] == []

    def test_cand104_tracker_vs_evidence(self, store):
        # Key trap: tracker='Clear' but evidence is incomplete
        r = detect_contradictions(store, "CAND-104")
        assert r["passed"] is False
        assert len(r["issues"]) >= 2
        # Must flag the Clear status contradiction
        assert any("clear" in i.lower() for i in r["issues"])
        # Must flag no adjudication register entry
        assert any("adjudication" in i.lower() for i in r["issues"])

    def test_cand104_verbal_confirmation_flagged(self, store):
        r = detect_contradictions(store, "CAND-104")
        assert any("verbal" in i.lower() for i in r["issues"])

    def test_cand102_ready_to_join_not_closure(self, store):
        r = detect_contradictions(store, "CAND-102")
        assert any("ready to join" in i.lower() or "ready_to_join" in i.lower()
                   for i in r["issues"])

    def test_sources_cited_in_contradictions(self, store):
        r = detect_contradictions(store, "CAND-104")
        sources = [c["source"] for c in r["citations"]]
        # Should reference the tracker as a source
        assert any("tracker" in s.lower() or "bpps_tracker" in s.lower() for s in sources)


# ── assess_closure_readiness ──────────────────────────────────────────────────

class TestAssessClosureReadiness:
    def test_cand101_verdict_clear(self, store):
        r = assess_closure_readiness(store, "CAND-101")
        assert r["overall_verdict"] == "CLEAR"
        assert r["ready_for_closure"] is True
        assert r["issue_count"] == 0

    def test_cand102_not_ready(self, store):
        r = assess_closure_readiness(store, "CAND-102")
        assert r["ready_for_closure"] is False
        assert r["issue_count"] > 0

    def test_cand104_incorrectly_marked_clear(self, store):
        # The most important test — catches the deliberate trap
        r = assess_closure_readiness(store, "CAND-104")
        assert r["overall_verdict"] == "INCORRECTLY MARKED CLEAR"
        assert r["ready_for_closure"] is False

    def test_cand105_not_ready(self, store):
        r = assess_closure_readiness(store, "CAND-105")
        assert r["ready_for_closure"] is False
        assert r["issue_count"] >= 3

    def test_cand103_clear(self, store):
        r = assess_closure_readiness(store, "CAND-103")
        assert r["overall_verdict"] == "CLEAR"
        assert r["ready_for_closure"] is True

    def test_result_has_all_fields(self, store):
        r = assess_closure_readiness(store, "CAND-102")
        for key in ("candidate_id", "candidate_name", "tracker_status",
                    "overall_verdict", "ready_for_closure", "issue_count",
                    "checks", "all_issues", "all_citations"):
            assert key in r, f"Missing key: {key}"

    def test_checks_contains_all_six(self, store):
        r = assess_closure_readiness(store, "CAND-101")
        assert set(r["checks"].keys()) == {
            "freshness", "employment", "rtw", "criminality",
            "adjudication", "contradictions"
        }


# ── get_all_candidates_summary ────────────────────────────────────────────────

class TestGetAllCandidatesSummary:
    def test_returns_all_six(self, store):
        r = get_all_candidates_summary(store)
        assert r["total_candidates"] == 6

    def test_cand104_ranked_first(self, store):
        r = get_all_candidates_summary(store)
        # INCORRECTLY MARKED CLEAR = highest priority (rank 0)
        top = r["ranked_results"][0]
        assert top["candidate_id"] == "CAND-104"
        assert top["overall_verdict"] == "INCORRECTLY MARKED CLEAR"

    def test_clear_candidates_ranked_last(self, store):
        r = get_all_candidates_summary(store)
        # Both CAND-101 and CAND-103 are CLEAR — they appear at the bottom
        bottom_two = {x["candidate_id"] for x in r["ranked_results"][-2:]}
        assert "CAND-101" in bottom_two
        assert "CAND-103" in bottom_two

    def test_ready_vs_not_ready_count(self, store):
        r = get_all_candidates_summary(store)
        assert r["ready_for_closure"] + r["not_ready"] == 6