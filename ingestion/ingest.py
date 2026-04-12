"""
BPSS Dataset Ingestion Layer
============================
Parses all 14 source files into a BPSSKnowledgeStore.

Entry point: build_knowledge_store(data_dir)

Parsing strategy per file type:
  PDF  → pypdf PdfReader
  DOCX → python-docx Document
  XLSX → openpyxl (sheet names + row iteration)
  CSV  → pandas read_csv
"""

import re
import logging
from pathlib import Path
from datetime import date
from typing import Optional

import pandas as pd
from pypdf import PdfReader
from docx import Document

from store.models import (
    BPSSKnowledgeStore,
    CandidateRecord,
    PolicyStore,
    TrackerRow,
    DocumentRecord,
    EmploymentPeriod,
    AdjudicationEntry,
    ControlStatus,
)

logger = logging.getLogger(__name__)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date(val) -> Optional[date]:
    """Safely parse various date representations into a date object."""
    if val is None or (isinstance(val, float) and __import__("math").isnan(val)):
        return None
    if isinstance(val, date):
        return val
    if hasattr(val, "date"):          # datetime → date
        return val.date()
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "n/a", ""):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return date.fromisoformat(s) if fmt == "%Y-%m-%d" else \
                   __import__("datetime").datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    logger.warning("Could not parse date: %r", val)
    return None


def _bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("yes", "true", "1", "y")


# ── PDF reader ────────────────────────────────────────────────────────────────

def _read_pdf(path: Path) -> str:
    """Extract all text from a PDF, joining pages with newlines."""
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        logger.error("Failed to read PDF %s: %s", path, exc)
        return f"[ERROR reading {path.name}: {exc}]"


# ── DOCX reader ───────────────────────────────────────────────────────────────

def _read_docx(path: Path) -> str:
    """Extract all paragraph text from a DOCX file."""
    try:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception as exc:
        logger.error("Failed to read DOCX %s: %s", path, exc)
        return f"[ERROR reading {path.name}: {exc}]"


# ── Candidate-specific text extraction ───────────────────────────────────────

def _extract_candidate_section(full_text: str, cand_id: str,
                               extra_terms: list | None = None) -> str:
    """
    Extract paragraphs mentioning a candidate ID or extra name terms from a
    shared document. Also captures the next 2 lines for context.
    """
    lines = full_text.splitlines()
    terms = [cand_id.upper()] + [t.upper() for t in (extra_terms or [])]
    result = []
    seen: set = set()
    for i, line in enumerate(lines):
        up = line.upper()
        if any(t in up for t in terms):
            for j in range(i, min(i + 3, len(lines))):
                if j not in seen:
                    result.append(lines[j].strip())
                    seen.add(j)
    return "\n".join(result)


# ── Policy + reference PDF ingestion ─────────────────────────────────────────

def _ingest_policy_docs(data_dir: Path, policy: PolicyStore) -> None:
    policy.bpss_policy_text = _read_pdf(
        data_dir / "policies" / "BPSS_Screening_Policy_v3.pdf"
    )
    policy.sop_text = _read_pdf(
        data_dir / "policies" / "Screening_Operations_SOP.pdf"
    )
    policy.rtw_matrix_text = _read_pdf(
        data_dir / "reference" / "Permitted_RTW_Evidence_Matrix.pdf"
    )
    policy.adjudication_register_text = _read_pdf(
        data_dir / "evidence" / "Adjudication_Register.pdf"
    )
    logger.info("Policy documents ingested.")


# ── Tracker CSV ingestion ─────────────────────────────────────────────────────

def _ingest_tracker(data_dir: Path, store: BPSSKnowledgeStore) -> None:
    path = data_dir / "structured" / "bpps_tracker_export.csv"
    try:
        df = pd.read_csv(str(path))
        for _, row in df.iterrows():
            cand_id = str(row["candidate_id"]).strip().upper()
            # criminality_complete may be blank/NaN for exempt roles (INTERN, CONTRACTOR-LTD)
            crim_val = row.get("criminality_complete")
            crim_na = (
                crim_val is None
                or (isinstance(crim_val, float) and __import__("math").isnan(crim_val))
                or str(crim_val).strip().upper() in ("N/A", "NA", "NAN", "")
            )
            crim_raw = "" if crim_na else str(crim_val).strip()
            crim_complete = False if crim_na else _bool(crim_raw)

            tracker = TrackerRow(
                candidate_name=str(row.get("candidate_name", "")).strip(),
                role_code=str(row.get("role_code", "")).strip().upper(),
                analyst_review_date=_parse_date(row.get("analyst_review_date")),
                status_tracker=str(row.get("status_tracker", "")).strip(),
                ready_to_join=_bool(row.get("ready_to_join", False)),
                identity_complete=_bool(row.get("identity_complete", False)),
                rtw_complete=_bool(row.get("rtw_complete", False)),
                employment_complete=_bool(row.get("employment_complete", False)),
                criminality_complete=crim_complete,
                criminality_na=crim_na,
                risk_level=str(row.get("risk_level", "")).strip(),
                notes=str(row.get("notes", "")).strip(),
            )
            if cand_id not in store.candidates:
                store.candidates[cand_id] = CandidateRecord(candidate_id=cand_id)
            store.candidates[cand_id].tracker = tracker
        logger.info("Tracker CSV ingested: %d candidates.", len(df))
    except Exception as exc:
        msg = f"Tracker CSV ingestion failed: {exc}"
        logger.error(msg)
        store.ingestion_errors.append(msg)


# ── Document inventory CSV ingestion ─────────────────────────────────────────

def _ingest_document_inventory(data_dir: Path, store: BPSSKnowledgeStore) -> None:
    path = data_dir / "structured" / "document_inventory.csv"
    try:
        df = pd.read_csv(str(path))
        for _, row in df.iterrows():
            cand_id = str(row["candidate_id"]).strip().upper()
            doc = DocumentRecord(
                doc_type=str(row.get("doc_type", "")).strip(),
                document_id=str(row.get("document_id", "")).strip(),
                document_date=_parse_date(row.get("document_date")),
                valid_to=_parse_date(row.get("valid_to")),
                present_in_folder=_bool(row.get("present_in_folder", False)),
                remarks=str(row.get("remarks", "")).strip(),
            )
            if cand_id not in store.candidates:
                store.candidates[cand_id] = CandidateRecord(candidate_id=cand_id)
            store.candidates[cand_id].documents.append(doc)
        logger.info("Document inventory ingested: %d records.", len(df))
    except Exception as exc:
        msg = f"Document inventory ingestion failed: {exc}"
        logger.error(msg)
        store.ingestion_errors.append(msg)


# ── Employment history CSV ingestion ──────────────────────────────────────────

def _ingest_employment_history(data_dir: Path, store: BPSSKnowledgeStore) -> None:
    path = data_dir / "structured" / "employment_history.csv"
    try:
        df = pd.read_csv(str(path))
        for _, row in df.iterrows():
            cand_id = str(row["candidate_id"]).strip().upper()
            period = EmploymentPeriod(
                period_start=_parse_date(row.get("period_start")),
                period_end=_parse_date(row.get("period_end")),
                evidence_type=str(row.get("evidence_type", "")).strip(),
                evidence_status=str(row.get("evidence_status", "")).strip(),
                notes=str(row.get("notes", "")).strip(),
            )
            if cand_id not in store.candidates:
                store.candidates[cand_id] = CandidateRecord(candidate_id=cand_id)
            store.candidates[cand_id].employment.append(period)
        logger.info("Employment history ingested: %d records.", len(df))
    except Exception as exc:
        msg = f"Employment history ingestion failed: {exc}"
        logger.error(msg)
        store.ingestion_errors.append(msg)


# ── Adjudication register PDF ingestion ──────────────────────────────────────

def _ingest_adjudication_register(data_dir: Path, store: BPSSKnowledgeStore) -> None:
    """
    Parse the Adjudication Register PDF.

    Why pypdf line-based parsing here?
    ------------------------------------
    The PDF has overlapping column geometry — adjacent cells share x0 ranges
    and pdfplumber's character/word extraction produces garbled text like
    "appHroirvinegd Director; Screening Ops Le2a0d26-02-06" because characters
    from Decision and Approver columns are interleaved in the PDF stream.

    pypdf's extract_text() processes each cell separately and returns them on
    individual lines in the correct reading order:
        CAND-102
        Provisional start approved       ← decision
        Hiring Director; Screening Ops   ← approvers
        2026-02-06                       ← date
        Start work before DBS only       ← scope
        Case remains Pending until...    ← notes

    This fixed structure (exactly 5 lines per CAND entry) is reliable and
    requires no hardcoded approver names.
    """
    path = data_dir / "evidence" / "Adjudication_Register.pdf"
    text = _read_pdf(path)
    entries = _parse_adjudication_lines(text)

    for cand_id, entry in entries.items():
        cand_id = cand_id.upper()
        if cand_id not in store.candidates:
            store.candidates[cand_id] = CandidateRecord(candidate_id=cand_id)
        store.candidates[cand_id].adjudication = entry

    logger.info("Adjudication register ingested: %d entries.", len(entries))


def _parse_adjudication_lines(text: str) -> dict[str, AdjudicationEntry]:
    """
    Parse adjudication entries from pypdf line output.

    pypdf extracts the table cells one per line in reading order.
    After each CAND-NNN line the next 5 lines are always:
        [0] decision
        [1] approvers  (split on ";" or ",")
        [2] date       (YYYY-MM-DD)
        [3] scope
        [4] notes

    This approach is robust to any approver name/title — no strings hardcoded.
    """
    CAND_PATTERN = re.compile(r"^(CAND-\d{3})$", re.IGNORECASE)
    FIELDS_PER_ENTRY = 5

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    entries: dict[str, AdjudicationEntry] = {}

    i = 0
    while i < len(lines):
        m = CAND_PATTERN.match(lines[i])
        if m:
            cand_id = m.group(1).upper()
            remaining = lines[i + 1:]

            # Collect the next FIELDS_PER_ENTRY non-empty lines as cell values
            cell_lines = []
            for line in remaining:
                if CAND_PATTERN.match(line):
                    break  # hit the next entry
                cell_lines.append(line)
                if len(cell_lines) == FIELDS_PER_ENTRY:
                    break

            # Pad with empty strings if fewer lines than expected
            while len(cell_lines) < FIELDS_PER_ENTRY:
                cell_lines.append("")

            decision      = cell_lines[0]
            approvers_raw = cell_lines[1]
            date_str      = cell_lines[2]
            scope         = cell_lines[3]
            notes         = cell_lines[4]

            # Parse date
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_str)
            entry_date = _parse_date(date_match.group(0)) if date_match else None

            # Split approvers on ";" or "," — generic, no hardcoded names
            approvers = [a.strip() for a in re.split(r"[;,]", approvers_raw) if a.strip()]

            entries[cand_id] = AdjudicationEntry(
                decision=decision,
                approvers=approvers,
                decision_date=entry_date,
                scope=scope,
                notes=notes,
            )
            i += 1 + len(cell_lines)
        else:
            i += 1

    return entries


def _parse_adjudication_lines(text: str) -> dict[str, AdjudicationEntry]:
    """
    Parse adjudication entries from pypdf line output.

    pypdf extracts the table cells one per line in reading order.
    After each CAND-NNN line the next 5 lines are always:
        [0] decision
        [1] approvers  (split on ";" or ",")
        [2] date       (YYYY-MM-DD)
        [3] scope
        [4] notes

    This approach is robust to any approver name/title — no strings hardcoded.
    """
    CAND_PATTERN = re.compile(r"^(CAND-\d{3})$", re.IGNORECASE)
    FIELDS_PER_ENTRY = 5

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    entries: dict[str, AdjudicationEntry] = {}

    i = 0
    while i < len(lines):
        m = CAND_PATTERN.match(lines[i])
        if m:
            cand_id = m.group(1).upper()
            remaining = lines[i + 1:]

            # Collect the next FIELDS_PER_ENTRY non-empty lines as cell values
            cell_lines = []
            for line in remaining:
                if CAND_PATTERN.match(line):
                    break  # hit the next entry
                cell_lines.append(line)
                if len(cell_lines) == FIELDS_PER_ENTRY:
                    break

            # Pad with empty strings if fewer lines than expected
            while len(cell_lines) < FIELDS_PER_ENTRY:
                cell_lines.append("")

            decision      = cell_lines[0]
            approvers_raw = cell_lines[1]
            date_str      = cell_lines[2]
            scope         = cell_lines[3]
            notes         = cell_lines[4]

            # Parse date
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_str)
            entry_date = _parse_date(date_match.group(0)) if date_match else None

            # Split approvers on ";" or "," — generic, no hardcoded names
            approvers = [a.strip() for a in re.split(r"[;,]", approvers_raw) if a.strip()]

            entries[cand_id] = AdjudicationEntry(
                decision=decision,
                approvers=approvers,
                decision_date=entry_date,
                scope=scope,
                notes=notes,
            )
            i += 1 + len(cell_lines)
        else:
            i += 1

    return entries


# ── Candidate pack DOCX ingestion ─────────────────────────────────────────────

def _ingest_candidate_packs(data_dir: Path, store: BPSSKnowledgeStore) -> None:
    pack_dir = data_dir / "candidate_pack"
    for docx_path in sorted(pack_dir.glob("CAND-*_candidate_pack.docx")):
        # Extract CAND-NNN from filename
        m = re.search(r"(CAND-\d{3})", docx_path.name, re.IGNORECASE)
        if not m:
            logger.warning("Could not determine cand_id from filename: %s", docx_path.name)
            continue
        cand_id = m.group(1).upper()
        text = _read_docx(docx_path)
        if cand_id not in store.candidates:
            store.candidates[cand_id] = CandidateRecord(candidate_id=cand_id)
        store.candidates[cand_id].candidate_pack_text = text
    logger.info("Candidate packs ingested: %d files.", len(list(pack_dir.glob("CAND-*_candidate_pack.docx"))))


# ── Evidence DOCX ingestion ───────────────────────────────────────────────────

def _ingest_evidence_docs(data_dir: Path, store: BPSSKnowledgeStore) -> None:
    analyst_notes_path = data_dir / "evidence" / "Analyst_Working_Notes.docx"
    email_path = data_dir / "evidence" / "Email_Approvals_and_Escalations.docx"

    analyst_full = _read_docx(analyst_notes_path)
    email_full = _read_docx(email_path)

    # Build a name lookup for richer extraction (emails reference names, not CAND-IDs)
    # Extract first name and full name from tracker data
    for cand_id, record in store.candidates.items():
        extra: list[str] = []
        if record.tracker:
            name = record.tracker.candidate_name
            extra.append(name)
            # also add first name alone for robustness
            parts = name.split()
            if parts:
                extra.append(parts[0])
        record.analyst_notes_text = _extract_candidate_section(analyst_full, cand_id, extra)
        record.email_mentions_text = _extract_candidate_section(email_full, cand_id, extra)

    logger.info("Evidence documents ingested and distributed to candidates.")


# ── Control status post-processing ───────────────────────────────────────────

def _build_control_statuses(store: BPSSKnowledgeStore) -> None:
    """
    After all files are parsed, compute ControlStatus for each candidate
    by cross-referencing tracker claims against actual evidence.
    """
    for cand_id, record in store.candidates.items():
        if record.tracker is None:
            continue
        t = record.tracker

        # ── Identity ──────────────────────────────────────────────────────────
        identity_issues = []
        identity_docs = []
        photo_id_docs = [d for d in record.documents
                         if any(kw in d.doc_type.lower()
                                for kw in ("passport", "licence", "driving", "id"))]
        address_docs = [d for d in record.documents
                        if any(kw in d.doc_type.lower()
                               for kw in ("bank statement", "utility bill", "address proof",
                                          "council tax"))]

        # Check address freshness (must be within 90 days of review date)
        if t.analyst_review_date:
            for addr in address_docs:
                if addr.document_date:
                    delta = (t.analyst_review_date - addr.document_date).days
                    if delta > 90:
                        identity_issues.append(
                            f"Address proof {addr.document_id} is {delta} days old at review "
                            f"(max 90 days). [{addr.source_file}]"
                        )

        # Check for missing address proof in folder
        for addr in address_docs:
            if not addr.present_in_folder:
                identity_issues.append(
                    f"Address proof {addr.document_id} mentioned but NOT retained in folder. "
                    f"[{addr.source_file} | remarks: {addr.remarks}]"
                )
        if not address_docs:
            identity_issues.append("No address proof document found in evidence inventory.")

        identity_docs = [d.document_id for d in photo_id_docs + address_docs]
        store.candidates[cand_id].controls.append(ControlStatus(
            control="identity",
            tracker_says_complete=t.identity_complete,
            evidence_found=bool(photo_id_docs) and bool(address_docs),
            issues=identity_issues,
            supporting_docs=identity_docs,
        ))

        # ── Right to work ────────────────────────────────────────────────────
        rtw_issues = []
        rtw_docs_found = [d for d in record.documents
                          if any(kw in d.doc_type.lower()
                                 for kw in ("passport", "evisa", "visa", "brp",
                                            "share code", "rtw"))]
        for rtw_doc in rtw_docs_found:
            if rtw_doc.valid_to and rtw_doc.valid_to < (t.analyst_review_date or date.today()):
                rtw_issues.append(
                    f"RTW document {rtw_doc.document_id} expired {rtw_doc.valid_to} "
                    f"(review date: {t.analyst_review_date}). [{rtw_doc.source_file}]"
                )
            if not rtw_doc.present_in_folder:
                rtw_issues.append(
                    f"RTW document {rtw_doc.document_id} not present in folder. "
                    f"[{rtw_doc.source_file}]"
                )

        store.candidates[cand_id].controls.append(ControlStatus(
            control="rtw",
            tracker_says_complete=t.rtw_complete,
            evidence_found=bool(rtw_docs_found),
            issues=rtw_issues,
            supporting_docs=[d.document_id for d in rtw_docs_found],
        ))

        # ── Employment history ────────────────────────────────────────────────
        emp_issues = []
        weak_periods = [e for e in record.employment
                        if e.evidence_status.lower() in ("weak", "gap", "unexplained")]
        for w in weak_periods:
            emp_issues.append(
                f"Employment period {w.period_start}–{w.period_end}: "
                f"status={w.evidence_status} ({w.notes}). [{w.source_file}]"
            )

        store.candidates[cand_id].controls.append(ControlStatus(
            control="employment",
            tracker_says_complete=t.employment_complete,
            evidence_found=bool(record.employment),
            issues=emp_issues,
            supporting_docs=[f"{e.period_start}–{e.period_end}: {e.evidence_type}"
                             for e in record.employment],
        ))

        # ── Criminality ───────────────────────────────────────────────────────
        crim_issues = []
        exempt = t.criminality_na or t.role_code in ("INTERN", "CONTRACTOR-LTD")
        crim_docs = [d for d in record.documents
                     if any(kw in d.doc_type.lower()
                            for kw in ("disclosure", "dbs", "basic", "crim"))]
        if not exempt and not crim_docs:
            crim_issues.append(
                "No criminality/basic disclosure document found in evidence inventory."
            )

        store.candidates[cand_id].controls.append(ControlStatus(
            control="criminality",
            tracker_says_complete=t.criminality_complete or exempt,
            evidence_found=exempt or bool(crim_docs),
            issues=crim_issues,
            supporting_docs=["EXEMPT (role: {})".format(t.role_code)] if exempt
                            else [d.document_id for d in crim_docs],
        ))

    logger.info("Control statuses computed for %d candidates.", len(store.candidates))


# ── Master entry point ────────────────────────────────────────────────────────

def build_knowledge_store(data_dir: str | Path) -> BPSSKnowledgeStore:
    """
    Parse all 14 BPSS dataset files and return a fully populated BPSSKnowledgeStore.

    Args:
        data_dir: Path to the bpss_agentic_dataset/ directory.

    Returns:
        BPSSKnowledgeStore keyed by candidate_id, with policy docs populated.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    store = BPSSKnowledgeStore()

    # 1. Policy PDFs (no candidate keying)
    logger.info("--- Ingesting policy documents ---")
    _ingest_policy_docs(data_dir, store.policy)

    # 2. Structured data (CSV / XLSX) — establishes candidate records
    logger.info("--- Ingesting structured data ---")
    _ingest_tracker(data_dir, store)
    _ingest_document_inventory(data_dir, store)
    _ingest_employment_history(data_dir, store)

    # 3. Adjudication register PDF
    logger.info("--- Ingesting adjudication register ---")
    _ingest_adjudication_register(data_dir, store)

    # 4. Candidate pack DOCX files
    logger.info("--- Ingesting candidate packs ---")
    _ingest_candidate_packs(data_dir, store)

    # 5. Shared evidence DOCX files (analyst notes, emails)
    logger.info("--- Ingesting evidence documents ---")
    _ingest_evidence_docs(data_dir, store)

    # 6. Post-process: compute control status from evidence vs tracker
    logger.info("--- Computing control statuses ---")
    _build_control_statuses(store)

    if store.ingestion_errors:
        logger.warning("Ingestion completed with %d error(s):", len(store.ingestion_errors))
        for err in store.ingestion_errors:
            logger.warning("  %s", err)
    else:
        logger.info("Ingestion complete. %d candidates loaded.", len(store.candidates))

    return store