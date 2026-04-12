"""
Data models for the BPSS screening knowledge store.
All structured data is keyed by candidate_id.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import date


# ── Document evidence ────────────────────────────────────────────────────────

@dataclass
class DocumentRecord:
    """A single piece of evidence from the document_inventory CSV."""
    doc_type: str
    document_id: str
    document_date: Optional[date]
    valid_to: Optional[date]
    present_in_folder: bool
    remarks: str
    source_file: str = "structured/document_inventory.csv"


# ── Employment history ────────────────────────────────────────────────────────

@dataclass
class EmploymentPeriod:
    """A single employment/education period from the employment_history CSV."""
    period_start: Optional[date]
    period_end: Optional[date]
    evidence_type: str
    evidence_status: str   # Valid | Weak | Gap | Accepted
    notes: str
    source_file: str = "structured/employment_history.csv"


# ── Tracker row ───────────────────────────────────────────────────────────────

@dataclass
class TrackerRow:
    """One row from the BPSS case tracker (CSV export + XLSX)."""
    candidate_name: str
    role_code: str
    analyst_review_date: Optional[date]
    status_tracker: str           # Clear | Pending | Risk Accepted | Reject
    ready_to_join: bool
    identity_complete: bool
    rtw_complete: bool
    employment_complete: bool
    criminality_complete: bool    # May be N/A for INTERN/CONTRACTOR-LTD
    criminality_na: bool          # True when role exempts criminality check
    risk_level: str               # Low | Medium | High
    notes: str
    source_file: str = "structured/bpps_tracker_export.csv"


# ── Adjudication register entry ───────────────────────────────────────────────

@dataclass
class AdjudicationEntry:
    """An approved exception from the Adjudication_Register PDF."""
    decision: str
    approvers: list[str]
    decision_date: Optional[date]
    scope: str
    notes: str
    source_file: str = "evidence/Adjudication_Register.pdf"


# ── Control assessment ────────────────────────────────────────────────────────

@dataclass
class ControlStatus:
    """
    Assessed status of one mandatory control for a candidate.
    Populated by the ingestion layer from parsed evidence.
    """
    control: str                       # identity | rtw | employment | criminality
    tracker_says_complete: bool
    evidence_found: bool
    issues: list[str] = field(default_factory=list)
    supporting_docs: list[str] = field(default_factory=list)


# ── Full candidate record ─────────────────────────────────────────────────────

@dataclass
class CandidateRecord:
    """
    The complete knowledge record for one candidate.
    Populated from all 14 source files during ingestion.
    """
    candidate_id: str

    # ── Structured fields (from tracker CSV) ──
    tracker: Optional[TrackerRow] = None

    # ── Document evidence ──
    documents: list[DocumentRecord] = field(default_factory=list)

    # ── Employment history ──
    employment: list[EmploymentPeriod] = field(default_factory=list)

    # ── Adjudication register ──
    adjudication: Optional[AdjudicationEntry] = None

    # ── Unstructured raw text (for LLM retrieval) ──
    candidate_pack_text: str = ""        # from CAND-NNN_candidate_pack.docx
    analyst_notes_text: str = ""         # excerpts from Analyst_Working_Notes.docx
    email_mentions_text: str = ""        # excerpts from Email_Approvals_and_Escalations.docx

    # ── Pre-computed control status (populated by ingestion post-processing) ──
    controls: list[ControlStatus] = field(default_factory=list)

    def get_document(self, doc_type_fragment: str) -> Optional[DocumentRecord]:
        """Find the first document whose type contains the given fragment (case-insensitive)."""
        frag = doc_type_fragment.lower()
        for d in self.documents:
            if frag in d.doc_type.lower():
                return d
        return None

    def get_control(self, control_name: str) -> Optional[ControlStatus]:
        for c in self.controls:
            if c.control == control_name:
                return c
        return None


# ── Global policy store ───────────────────────────────────────────────────────

@dataclass
class PolicyStore:
    """
    Raw text from all policy / reference documents.
    Used by the search_policy tool.
    """
    bpss_policy_text: str = ""           # BPSS_Screening_Policy_v3.pdf
    sop_text: str = ""                   # Screening_Operations_SOP.pdf
    rtw_matrix_text: str = ""            # Permitted_RTW_Evidence_Matrix.pdf
    adjudication_register_text: str = "" # Adjudication_Register.pdf (full)


# ── Top-level knowledge store ─────────────────────────────────────────────────

@dataclass
class BPSSKnowledgeStore:
    """
    The complete in-memory knowledge store for the BPSS dataset.
    Keyed by candidate_id.
    """
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    policy: PolicyStore = field(default_factory=PolicyStore)
    ingestion_errors: list[str] = field(default_factory=list)

    def get_candidate(self, cand_id: str) -> Optional[CandidateRecord]:
        return self.candidates.get(cand_id.upper())

    def all_candidate_ids(self) -> list[str]:
        return sorted(self.candidates.keys())