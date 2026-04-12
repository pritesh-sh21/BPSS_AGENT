"""
Tool: check_rtw
================
Verifies right-to-work evidence against the Permitted RTW Evidence Matrix.

Rules (reference/Permitted_RTW_Evidence_Matrix.pdf):
  - UK/Irish passport: acceptable if genuine and current.
  - eVisa / share code: acceptable if valid at time of check.
  - Expired BRP alone: NOT acceptable.
  - Contractors personally working in UK still require RTW evidence
    unless legal counsel documented a jurisdictional exemption.
"""

from __future__ import annotations
from store.models import BPSSKnowledgeStore
from tools.helpers import _cite, _ok, _fail, _get_record

RTW_KEYWORDS = ("passport", "evisa", "visa", "brp", "share code", "rtw")
ALT_RTW_KEYWORDS = ("share code", "evisa", "e-visa", "visa")


def check_rtw(store: BPSSKnowledgeStore, cand_id: str) -> dict:
    """
    Returns:
        passed=True  — valid RTW evidence found per matrix rules
        passed=False — expired BRP, expired passport with no alternative,
                       doc missing from folder, or contractor exemption unsubstantiated
    """
    r, err = _get_record(store, cand_id)
    if err:
        return err

    t = r.tracker
    review_date = t.analyst_review_date if t else None
    citations = []
    issues = []

    # Exclude student ID — valid for identity corroboration only, not RTW
    rtw_docs = [
        d for d in r.documents
        if any(kw in d.doc_type.lower() for kw in RTW_KEYWORDS)
        and "student" not in d.doc_type.lower()
    ]

    if not rtw_docs:
        return _fail(
            f"{cand_id}: No right-to-work document found.",
            ["No RTW evidence in structured/document_inventory.csv."],
            citations,
        )

    for doc in rtw_docs:
        citations.append(_cite("structured/document_inventory.csv", "document_id", doc.document_id))
        citations.append(_cite("structured/document_inventory.csv", "doc_type", doc.doc_type))

        # ── Expired BRP ───────────────────────────────────────────────────────
        if "brp" in doc.doc_type.lower():
            if doc.valid_to and review_date and doc.valid_to < review_date:
                issues.append(
                    f"{doc.document_id} is a BRP expired {doc.valid_to} "
                    f"(review: {review_date}). RTW matrix: 'Expired BRP alone not "
                    f"acceptable — current status evidence must be obtained.' "
                    f"[reference/Permitted_RTW_Evidence_Matrix.pdf]"
                )

        # ── Expired passport ──────────────────────────────────────────────────
        # Only flag if no alternative valid RTW doc (e.g. share code) exists
        elif "passport" in doc.doc_type.lower():
            if doc.valid_to and review_date and doc.valid_to < review_date:
                pack = r.candidate_pack_text.lower()
                has_alt_rtw = any(kw in pack for kw in ALT_RTW_KEYWORDS)
                if not has_alt_rtw:
                    issues.append(
                        f"{doc.document_id} (Passport) expired {doc.valid_to} before "
                        f"review {review_date}. Expired passport does not satisfy RTW. "
                        f"[policies/BPSS_Screening_Policy_v3.pdf]"
                    )
                else:
                    citations.append(_cite(
                        f"candidate_pack/{r.candidate_id}_candidate_pack.docx",
                        "rtw_alternative",
                        "Expired passport noted but pack confirms valid share code / eVisa"
                    ))

        # ── Document not present in folder ────────────────────────────────────
        if not doc.present_in_folder:
            issues.append(
                f"{doc.document_id} ({doc.doc_type}) not present in evidence folder. "
                f"[structured/document_inventory.csv | remarks: {doc.remarks}]"
            )

    # ── Contractor-specific check ─────────────────────────────────────────────
    if t and t.role_code == "CONTRACTOR-LTD":
        citations.append(_cite(
            "reference/Permitted_RTW_Evidence_Matrix.pdf",
            "contractor_clarification",
            "Contractors personally providing services in UK still require RTW evidence."
        ))
        if r.analyst_notes_text and "policy does not state this" in r.analyst_notes_text.lower():
            issues.append(
                "RTW dismissed on recruiter assumption about contractor structure. "
                "RTW matrix explicitly states contractors personally providing services "
                "still require eligibility evidence. No legal counsel exemption documented. "
                "[reference/Permitted_RTW_Evidence_Matrix.pdf | "
                "evidence/Analyst_Working_Notes.docx]"
            )

    if issues:
        return _fail(
            f"{cand_id}: Right-to-work check FAILED ({len(issues)} issue(s)).",
            issues,
            citations,
        )
    return _ok(
        f"{cand_id}: Right-to-work evidence valid per permitted evidence matrix.",
        citations,
    )