"""
Quick inspection script for the BPSS knowledge store.
Run: python inspect_store.py [CAND-ID]
  or: python inspect_store.py          (shows all candidates summary)
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

sys.path.insert(0, str(Path(__file__).parent))
from ingestion.ingest import build_knowledge_store

DATA_DIR = Path(__file__).parent / "bpss_agentic_dataset"


def print_sep(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def show_candidate(record):
    t = record.tracker
    print_sep(f"{record.candidate_id}  ·  {t.candidate_name if t else '?'}")

    # ── Tracker row ──
    if t:
        print(f"  Role              : {t.role_code}")
        print(f"  Tracker status    : {t.status_tracker}")
        print(f"  Ready to join     : {t.ready_to_join}")
        print(f"  Risk level        : {t.risk_level}")
        print(f"  Review date       : {t.analyst_review_date}")
        print(f"  Tracker notes     : {t.notes}")

    # ── Controls ──
    print_sep("Controls")
    for ctrl in record.controls:
        tick = "✓" if (ctrl.evidence_found and not ctrl.issues) else "✗"
        tracker_claim = "✓" if ctrl.tracker_says_complete else "✗"
        print(f"  [{tick}] {ctrl.control:<14}  tracker={tracker_claim}  docs={ctrl.supporting_docs}")
        for issue in ctrl.issues:
            print(f"        ⚠  {issue}")

    # ── Documents ──
    print_sep("Evidence documents")
    for d in record.documents:
        present = "✓" if d.present_in_folder else "✗ MISSING"
        print(f"  [{present}] {d.document_id:<18} {d.doc_type:<22} date={d.document_date}  valid_to={d.valid_to}")
        if d.remarks:
            print(f"             remarks: {d.remarks}")

    # ── Employment ──
    print_sep("Employment history")
    for e in record.employment:
        flag = "⚠" if e.evidence_status.lower() in ("weak", "gap", "unexplained") else " "
        print(f"  {flag} {str(e.period_start):<12} → {str(e.period_end):<12}  [{e.evidence_status}]  {e.evidence_type}  ({e.notes})")

    # ── Adjudication ──
    print_sep("Adjudication register")
    if record.adjudication:
        adj = record.adjudication
        print(f"  Decision   : {adj.decision}")
        print(f"  Approvers  : {', '.join(adj.approvers) if adj.approvers else 'none listed'}")
        print(f"  Date       : {adj.decision_date}")
        print(f"  Scope      : {adj.scope}")
    else:
        print("  No adjudication entry — no approved exception exists.")

    # ── Text excerpts ──
    print_sep("Analyst notes excerpt")
    txt = record.analyst_notes_text.strip()
    print(f"  {txt if txt else '(none)'}")

    print_sep("Email mentions excerpt")
    txt = record.email_mentions_text.strip()
    print(f"  {txt if txt else '(none)'}")

    # ── Candidate pack (truncated) ──
    print_sep("Candidate pack (first 400 chars)")
    print(f"  {record.candidate_pack_text[:400].replace(chr(10), ' | ')}")


def show_summary(store):
    print_sep("BPSS Knowledge Store — All Candidates")
    print(f"  {'ID':<12} {'Name':<22} {'Role':<18} {'Status':<16} {'Risk':<8} {'Issues'}")
    print_sep()
    for cid in store.all_candidate_ids():
        r = store.get_candidate(cid)
        t = r.tracker
        all_issues = [i for ctrl in r.controls for i in ctrl.issues]
        issue_count = len(all_issues)
        flag = "⚠ " * min(issue_count, 3)
        print(f"  {cid:<12} {t.candidate_name:<22} {t.role_code:<18} {t.status_tracker:<16} {t.risk_level:<8} {flag}{issue_count} issue(s)")

    print_sep("Policy documents loaded")
    p = store.policy
    print(f"  BPSS Policy        : {len(p.bpss_policy_text):>6} chars")
    print(f"  SOP                : {len(p.sop_text):>6} chars")
    print(f"  RTW Matrix         : {len(p.rtw_matrix_text):>6} chars")
    print(f"  Adjudication Reg.  : {len(p.adjudication_register_text):>6} chars")

    if store.ingestion_errors:
        print_sep("Ingestion errors")
        for e in store.ingestion_errors:
            print(f"  ✗ {e}")
    else:
        print_sep()
        print("  ✓ No ingestion errors.")


if __name__ == "__main__":
    print("\nBuilding knowledge store...")
    store = build_knowledge_store(DATA_DIR)

    if len(sys.argv) > 1:
        cand_id = sys.argv[1].upper()
        record = store.get_candidate(cand_id)
        if record:
            show_candidate(record)
        else:
            print(f"Unknown candidate: {cand_id}")
            print(f"Available: {store.all_candidate_ids()}")
    else:
        show_summary(store)