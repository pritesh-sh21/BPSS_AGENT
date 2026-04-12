"""
Tests for the BPSS data ingestion layer.
Validates that all 14 files are parsed correctly and the knowledge store
reflects the known ground truth from the dataset.

Run with: python -m pytest tests/test_ingestion.py -v
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ingestion.ingest import build_knowledge_store
from store.models import BPSSKnowledgeStore

DATA_DIR = Path(__file__).parent.parent / "bpss_agentic_dataset"


@pytest.fixture(scope="module")
def store() -> BPSSKnowledgeStore:
    return build_knowledge_store(DATA_DIR)


# ── Store-level sanity ────────────────────────────────────────────────────────

class TestStoreStructure:
    def test_all_six_candidates_loaded(self, store):
        assert set(store.all_candidate_ids()) == {
            "CAND-101", "CAND-102", "CAND-103", "CAND-104", "CAND-105", "CAND-106"
        }

    def test_no_ingestion_errors(self, store):
        assert store.ingestion_errors == [], \
            f"Ingestion errors: {store.ingestion_errors}"

    def test_policy_texts_populated(self, store):
        assert "mandatory controls" in store.policy.bpss_policy_text.lower()
        assert "closure" in store.policy.sop_text.lower()
        assert "right-to-work" in store.policy.rtw_matrix_text.lower() or \
               "right to work" in store.policy.rtw_matrix_text.lower()
        assert "adjudication" in store.policy.adjudication_register_text.lower()


# ── CAND-101: All clear, no issues ───────────────────────────────────────────

class TestCAND101:
    def test_tracker_status(self, store):
        r = store.get_candidate("CAND-101")
        assert r.tracker.status_tracker == "Clear"
        assert r.tracker.risk_level == "Low"
        assert r.tracker.identity_complete is True
        assert r.tracker.rtw_complete is True
        assert r.tracker.employment_complete is True
        assert r.tracker.criminality_complete is True

    def test_candidate_pack_loaded(self, store):
        r = store.get_candidate("CAND-101")
        assert "CAND-101" in r.candidate_pack_text
        assert "Aarav Mehta" in r.candidate_pack_text
        assert "All mandatory controls complete" in r.candidate_pack_text

    def test_documents_present(self, store):
        r = store.get_candidate("CAND-101")
        ids = [d.document_id for d in r.documents]
        assert "ID-101-P" in ids     # Passport
        assert "ADDR-101-BS" in ids  # Bank statement
        assert "CRIM-101" in ids     # Disclosure

    def test_address_fresh(self, store):
        r = store.get_candidate("CAND-101")
        ctrl = r.get_control("identity")
        # Bank statement 2026-01-28, review 2026-02-11 → 14 days → fresh
        assert ctrl is not None
        assert not any("days old" in issue for issue in ctrl.issues), \
            f"Unexpected freshness issue: {ctrl.issues}"

    def test_employment_no_gaps(self, store):
        r = store.get_candidate("CAND-101")
        weak = [e for e in r.employment if e.evidence_status.lower() != "valid"]
        assert weak == [], f"Unexpected weak periods: {weak}"


# ── CAND-102: Multiple issues ─────────────────────────────────────────────────

class TestCAND102:
    def test_tracker_status(self, store):
        r = store.get_candidate("CAND-102")
        assert r.tracker.status_tracker == "Ready to Join"
        # Ready to Join is NOT BPSS closure
        assert r.tracker.criminality_complete is False

    def test_stale_address_detected(self, store):
        r = store.get_candidate("CAND-102")
        ctrl = r.get_control("identity")
        # Utility bill 2025-10-01, review 2026-02-04 → 126 days → stale
        assert any("days old" in issue for issue in ctrl.issues), \
            f"Expected stale address issue, got: {ctrl.issues}"

    def test_employment_gap_detected(self, store):
        r = store.get_candidate("CAND-102")
        ctrl = r.get_control("employment")
        assert any("gap" in issue.lower() or "unexplained" in issue.lower()
                   for issue in ctrl.issues), \
            f"Expected employment gap, got: {ctrl.issues}"

    def test_analyst_note_distributed(self, store):
        r = store.get_candidate("CAND-102")
        assert "CAND-102" in r.analyst_notes_text
        assert "closure not permitted" in r.analyst_notes_text.lower() or \
               "closure" in r.analyst_notes_text.lower()

    def test_adjudication_provisional_start(self, store):
        r = store.get_candidate("CAND-102")
        assert r.adjudication is not None
        assert "provisional" in r.adjudication.decision.lower() or \
               "provisional" in r.adjudication.notes.lower()


# ── CAND-103: Intern, expired passport OK ────────────────────────────────────

class TestCAND103:
    def test_criminality_exempt(self, store):
        r = store.get_candidate("CAND-103")
        assert r.tracker.role_code == "INTERN"
        assert r.tracker.criminality_na is True

    def test_tracker_status_clear(self, store):
        r = store.get_candidate("CAND-103")
        assert r.tracker.status_tracker == "Clear"

    def test_expired_passport_present(self, store):
        r = store.get_candidate("CAND-103")
        passport = r.get_document("passport")
        assert passport is not None
        # Passport expired 2025-11-01 — expired, but supported by student ID
        assert passport.valid_to == date(2025, 11, 1)

    def test_address_fresh(self, store):
        r = store.get_candidate("CAND-103")
        ctrl = r.get_control("identity")
        # Address proof 2026-02-20, review 2026-02-25 → 5 days → fresh
        assert not any("days old" in issue for issue in ctrl.issues), \
            f"Unexpected stale address: {ctrl.issues}"


# ── CAND-104: KEY TRAP — tracker says Clear, evidence is incomplete ───────────

class TestCAND104:
    def test_tracker_wrongly_says_clear(self, store):
        r = store.get_candidate("CAND-104")
        # Tracker says Clear — this is the contradiction we must catch
        assert r.tracker.status_tracker == "Clear"
        assert r.tracker.identity_complete is True   # tracker claims complete
        assert r.tracker.employment_complete is True  # tracker claims complete

    def test_address_proof_not_retained(self, store):
        r = store.get_candidate("CAND-104")
        # ADDR-104-CT: present_in_folder = No
        addr = r.get_document("address")
        assert addr is not None
        assert addr.present_in_folder is False, \
            "Address proof should NOT be in folder for CAND-104"

    def test_identity_control_has_issue(self, store):
        r = store.get_candidate("CAND-104")
        ctrl = r.get_control("identity")
        assert ctrl is not None
        assert len(ctrl.issues) > 0, \
            "Identity control should flag missing/unretained address proof"

    def test_employment_weak_period(self, store):
        r = store.get_candidate("CAND-104")
        ctrl = r.get_control("employment")
        # 2023-01 to 2024-07: CV only, no documentary corroboration
        assert any("weak" in issue.lower() or "cv" in issue.lower()
                   for issue in ctrl.issues), \
            f"Expected weak employment evidence, got: {ctrl.issues}"

    def test_no_adjudication_entry(self, store):
        r = store.get_candidate("CAND-104")
        # No entry in adjudication register → no approved exception
        assert r.adjudication is None, \
            "CAND-104 should have NO adjudication entry (it's an unapproved deviation)"

    def test_analyst_note_mentions_verbal_confirmation(self, store):
        r = store.get_candidate("CAND-104")
        assert "verbal" in r.analyst_notes_text.lower() or \
               "verbal" in r.candidate_pack_text.lower()


# ── CAND-105: High risk, multiple missing items ───────────────────────────────

class TestCAND105:
    def test_tracker_pending(self, store):
        r = store.get_candidate("CAND-105")
        assert r.tracker.status_tracker == "Pending"
        assert r.tracker.risk_level == "High"

    def test_expired_brp_detected(self, store):
        r = store.get_candidate("CAND-105")
        ctrl = r.get_control("rtw")
        # BRP expired 2024-12-31, review 2026-03-03
        assert any("expired" in issue.lower() for issue in ctrl.issues), \
            f"Expected expired RTW doc, got: {ctrl.issues}"

    def test_no_address_proof(self, store):
        r = store.get_candidate("CAND-105")
        ctrl = r.get_control("identity")
        assert len(ctrl.issues) > 0, "Should flag missing address proof"

    def test_employment_incomplete(self, store):
        r = store.get_candidate("CAND-105")
        # Only from 2024-06 onward — 18 months uncovered
        periods = r.employment
        assert len(periods) == 1
        assert periods[0].period_start.year == 2024


# ── CAND-106: Contractor, RTW dispute ────────────────────────────────────────

class TestCAND106:
    def test_criminality_exempt(self, store):
        r = store.get_candidate("CAND-106")
        assert r.tracker.role_code == "CONTRACTOR-LTD"
        assert r.tracker.criminality_na is True

    def test_adjudication_risk_accepted(self, store):
        r = store.get_candidate("CAND-106")
        assert r.adjudication is not None
        assert "risk accepted" in r.adjudication.notes.lower() or \
               "risk" in r.adjudication.decision.lower()

    def test_rtw_not_complete_in_tracker(self, store):
        r = store.get_candidate("CAND-106")
        assert r.tracker.rtw_complete is False

    def test_recruiter_email_captured(self, store):
        r = store.get_candidate("CAND-106")
        # Email from recruiter about RTW assumption should be in email_mentions
        assert "CAND-106" in r.email_mentions_text or \
               "rohan" in r.email_mentions_text.lower() or \
               "contractor" in r.email_mentions_text.lower()


# ── Cross-candidate checks ────────────────────────────────────────────────────

class TestCrossCandidate:
    def test_every_candidate_has_tracker(self, store):
        for cid in store.all_candidate_ids():
            assert store.get_candidate(cid).tracker is not None, \
                f"{cid} missing tracker row"

    def test_every_candidate_has_four_controls(self, store):
        for cid in store.all_candidate_ids():
            controls = store.get_candidate(cid).controls
            names = [c.control for c in controls]
            assert set(names) == {"identity", "rtw", "employment", "criminality"}, \
                f"{cid} controls: {names}"

    def test_every_candidate_has_pack_text(self, store):
        for cid in store.all_candidate_ids():
            r = store.get_candidate(cid)
            assert len(r.candidate_pack_text) > 100, \
                f"{cid} candidate pack text too short"

    def test_policy_docs_not_empty(self, store):
        assert len(store.policy.bpss_policy_text) > 200
        assert len(store.policy.sop_text) > 100
        assert len(store.policy.rtw_matrix_text) > 100