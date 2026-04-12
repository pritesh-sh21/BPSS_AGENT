"""
Tool: check_document_staleness
================================
Returns all stale or expired documents for a candidate.

Staleness rules (BPSS_Screening_Policy_v3.pdf):
  - Address proof must be dated within 90 days of analyst review date
  - Expired BRP alone is not acceptable RTW evidence
  - Expired passport is only acceptable for identity corroboration
    if supported by a second current photo ID
"""

from __future__ import annotations
from store.models import BPSSKnowledgeStore
from tools.helpers import _cite, _ok, _fail, _get_record

ADDRESS_KEYWORDS  = ("bank statement", "utility bill", "address proof", "council tax")
RTW_KEYWORDS      = ("brp", "visa", "evisa", "share code", "rtw")
IDENTITY_KEYWORDS = ("passport", "licence", "driving", "id")


def check_document_staleness(store: BPSSKnowledgeStore, cand_id: str) -> dict:
    """
    Checks all documents for a candidate and flags:
      - Address proof older than 90 days at review date
      - Any document whose valid_to expired before the review date
      - Missing address proof entirely

    Returns:
        passed=True  — no stale or expired documents found
        passed=False — one or more stale/expired documents found
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    t = r.tracker
    if t is None or t.analyst_review_date is None:
        return _fail(
            f"{cand_id}: No analyst review date — cannot assess staleness.",
            ["Analyst review date missing from tracker."],
            [],
        )

    review_date = t.analyst_review_date
    citations = [_cite(
        "structured/bpps_tracker_export.csv",
        "analyst_review_date", review_date
    )]

    issues  = []
    flagged = []   # list of dicts for structured output

    for doc in r.documents:
        citations.append(_cite(
            "structured/document_inventory.csv",
            "document_id", doc.document_id
        ))

        # ── Expired before review date ────────────────────────────────────────
        if doc.valid_to and doc.valid_to < review_date:
            days_expired = (review_date - doc.valid_to).days
            is_address   = any(kw in doc.doc_type.lower() for kw in ADDRESS_KEYWORDS)
            is_rtw       = any(kw in doc.doc_type.lower() for kw in RTW_KEYWORDS)
            is_passport  = "passport" in doc.doc_type.lower()

            # Expired passport — check if a current backup photo ID exists
            if is_passport:
                backup_id = [
                    d for d in r.documents
                    if any(kw in d.doc_type.lower() for kw in ("student id", "student", "id card"))
                    and (d.valid_to is None or d.valid_to >= review_date)
                    and d.present_in_folder
                ]
                if backup_id:
                    flagged.append({
                        "document_id":  doc.document_id,
                        "doc_type":     doc.doc_type,
                        "valid_to":     str(doc.valid_to),
                        "review_date":  str(review_date),
                        "days_expired": days_expired,
                        "severity":     "LOW",
                        "reason":       "Expired passport acceptable for identity corroboration only "
                                        f"— backed by current photo ID: {backup_id[0].document_id}. "
                                        "Cannot be used for RTW evidence.",
                        "source":       doc.source_file,
                    })
                    # Not an issue — policy allows this with backup ID
                else:
                    issues.append(
                        f"{doc.document_id} ({doc.doc_type}) expired {doc.valid_to} "
                        f"— {days_expired} days before review {review_date}. "
                        f"No current backup photo ID found. "
                        f"[{doc.source_file}]"
                    )
                    flagged.append({
                        "document_id":  doc.document_id,
                        "doc_type":     doc.doc_type,
                        "valid_to":     str(doc.valid_to),
                        "review_date":  str(review_date),
                        "days_expired": days_expired,
                        "severity":     "HIGH",
                        "reason":       "Expired passport with no current backup photo ID.",
                        "source":       doc.source_file,
                    })

            # Expired BRP — never acceptable alone
            elif is_rtw and "brp" in doc.doc_type.lower():
                issues.append(
                    f"{doc.document_id} ({doc.doc_type}) expired {doc.valid_to} "
                    f"— {days_expired} days before review {review_date}. "
                    f"RTW matrix: 'Expired BRP alone not acceptable.' "
                    f"[reference/Permitted_RTW_Evidence_Matrix.pdf]"
                )
                flagged.append({
                    "document_id":  doc.document_id,
                    "doc_type":     doc.doc_type,
                    "valid_to":     str(doc.valid_to),
                    "review_date":  str(review_date),
                    "days_expired": days_expired,
                    "severity":     "HIGH",
                    "reason":       "Expired BRP is not acceptable RTW evidence. "
                                    "Current status evidence must be obtained.",
                    "source":       "reference/Permitted_RTW_Evidence_Matrix.pdf",
                })

        # ── Stale address proof (>90 days old) ───────────────────────────────
        if any(kw in doc.doc_type.lower() for kw in ADDRESS_KEYWORDS):
            if not doc.present_in_folder:
                issues.append(
                    f"{doc.document_id} ({doc.doc_type}) not retained in folder — "
                    f"cannot verify freshness. "
                    f"[{doc.source_file} | remarks: {doc.remarks}]"
                )
                flagged.append({
                    "document_id": doc.document_id,
                    "doc_type":    doc.doc_type,
                    "severity":    "HIGH",
                    "reason":      "Address proof referenced but not retained in evidence folder.",
                    "source":      doc.source_file,
                })
            elif doc.document_date:
                delta = (review_date - doc.document_date).days
                if delta > 90:
                    issues.append(
                        f"{doc.document_id} ({doc.doc_type}) is {delta} days old at review "
                        f"(limit: 90 days). Doc date: {doc.document_date}, "
                        f"review date: {review_date}. "
                        f"[{doc.source_file}]"
                    )
                    flagged.append({
                        "document_id":   doc.document_id,
                        "doc_type":      doc.doc_type,
                        "document_date": str(doc.document_date),
                        "review_date":   str(review_date),
                        "days_old":      delta,
                        "severity":      "HIGH",
                        "reason":        f"Address proof is {delta} days old — exceeds 90-day limit. "
                                         "Policy requires proof of address within 90 days of review.",
                        "source":        doc.source_file,
                    })

    # ── No address proof at all ───────────────────────────────────────────────
    has_address_doc = any(
        any(kw in d.doc_type.lower() for kw in ADDRESS_KEYWORDS)
        for d in r.documents
    )
    if not has_address_doc:
        issues.append(
            f"No address proof document found in evidence inventory. "
            f"[structured/document_inventory.csv]"
        )
        flagged.append({
            "document_id": "NONE",
            "doc_type":    "Address Proof",
            "severity":    "HIGH",
            "reason":      "No address proof document found at all.",
            "source":      "structured/document_inventory.csv",
        })

    result_summary = (
        f"{cand_id}: {len(flagged)} document(s) flagged "
        f"({len(issues)} requiring action)."
        if flagged else
        f"{cand_id}: No stale or expired documents found."
    )

    if issues:
        return _fail(result_summary, issues, citations) | {"flagged_documents": flagged}

    return _ok(result_summary, citations) | {"flagged_documents": flagged}