"""
Tool: check_freshness
=====================
Checks whether each address-proof document was dated within 90 days
of the analyst review date.

Policy: BPSS_Screening_Policy_v3.pdf
  "Proof of address must be dated within 90 days of the analyst review date."
"""

from __future__ import annotations
from store.models import BPSSKnowledgeStore
from tools.helpers import _cite, _ok, _fail, _get_record

ADDRESS_KEYWORDS = ("bank statement", "utility bill", "address proof", "council tax")


def check_freshness(store: BPSSKnowledgeStore, cand_id: str, max_days: int = 90) -> dict:
    """
    Returns:
        passed=True  — all address docs within max_days of review date
        passed=False — any doc is stale, missing, or not retained in folder
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    t = r.tracker
    if t is None or t.analyst_review_date is None:
        return _fail(
            f"{cand_id}: No analyst review date — cannot assess freshness.",
            ["Analyst review date missing from tracker."],
            [],
        )

    review_date = t.analyst_review_date
    citations = [_cite("structured/bpps_tracker_export.csv", "analyst_review_date", review_date)]

    address_docs = [
        d for d in r.documents
        if any(kw in d.doc_type.lower() for kw in ADDRESS_KEYWORDS)
    ]

    if not address_docs:
        return _fail(
            f"{cand_id}: No address-proof document found.",
            ["No address proof in structured/document_inventory.csv."],
            citations,
        )

    issues = []
    for doc in address_docs:
        citations.append(_cite("structured/document_inventory.csv", "document_id", doc.document_id))
        citations.append(_cite("structured/document_inventory.csv", "document_date", doc.document_date))

        if not doc.present_in_folder:
            issues.append(
                f"{doc.document_id} ({doc.doc_type}) referenced but NOT retained in "
                f"evidence folder. [{doc.source_file} | remarks: {doc.remarks}]"
            )
            continue

        if doc.document_date is None:
            issues.append(
                f"{doc.document_id} ({doc.doc_type}) has no document date — "
                f"cannot verify freshness. [{doc.source_file}]"
            )
            continue

        delta = (review_date - doc.document_date).days
        if delta > max_days:
            issues.append(
                f"{doc.document_id} ({doc.doc_type}) is {delta} days old at review "
                f"(limit: {max_days} days). Doc date: {doc.document_date}, "
                f"review date: {review_date}. [{doc.source_file}]"
            )

    if issues:
        return _fail(
            f"{cand_id}: Address proof freshness FAILED ({len(issues)} issue(s)).",
            issues,
            citations,
        )
    return _ok(
        f"{cand_id}: All address proof documents within {max_days}-day limit.",
        citations,
    )