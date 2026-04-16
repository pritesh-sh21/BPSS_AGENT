"""
BPSS Agent — FastAPI Backend
=============================
Endpoints:
  POST /ask          — ask the agent a question
  GET  /candidates   — list all candidates with summary
  GET  /candidate/{id} — get full details for one candidate
  GET  /health       — health check
  POST /admin/reload — reload the knowledge store

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.ingest import build_knowledge_store
from agent.graph import run
from tools import assess_closure_readiness, get_all_candidates_summary

# ── Global knowledge store ─────────────────────────────────────────────────

# Works both locally and on Railway/cloud
# Can override with DATA_DIR env variable
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "bpss_agentic_dataset")))
store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build knowledge store once at startup, reuse for all requests."""
    global store
    logger.info("Building knowledge store...")
    store = build_knowledge_store(DATA_DIR)
    logger.info("Knowledge store ready. %d candidates loaded.", len(store.candidates))
    yield
    logger.info("Shutting down.")


# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="BPSS Screening Agent",
    description="AI agent that answers questions over the BPSS screening dataset",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {"question": "Which candidates are not ready for closure?"}
        }


class AnswerResponse(BaseModel):
    question:    str
    answer:      str
    duration_s:  float


class CandidateSummary(BaseModel):
    candidate_id:      str
    candidate_name:    str
    role:              str
    tracker_status:    str
    overall_verdict:   str
    ready_for_closure: bool
    risk_level:        str
    issue_count:       int


class CandidateDetail(BaseModel):
    candidate_id:       str
    candidate_name:     str
    role:               str
    tracker_status:     str
    overall_verdict:    str
    ready_for_closure:  bool
    risk_level:         str
    issue_count:        int
    all_issues:         list[str]
    control_completion: dict
    has_adjudication:   bool
    adjudication_decision: str | None


class HealthResponse(BaseModel):
    status:        str
    store_loaded:  bool
    candidates:    int


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the UI from the root URL."""
    ui_path = Path(__file__).parent / "ui.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    return {"message": "UI not found. Place ui.html in the project root."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if the API and knowledge store are ready."""
    return HealthResponse(
        status       = "ok",
        store_loaded = store is not None,
        candidates   = len(store.candidates) if store else 0,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Agent"])
async def ask(request: QuestionRequest):
    """
    Ask the BPSS agent a natural language question.
    The agent plans tool calls, executes them, and returns a grounded cited answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if store is None:
        raise HTTPException(status_code=503, detail="Knowledge store not ready.")

    logger.info("Question received: %s", request.question)
    start = time.time()

    try:
        answer = run(request.question, store)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")

    duration = round(time.time() - start, 2)
    logger.info("Answer produced in %.2fs", duration)

    return AnswerResponse(
        question   = request.question,
        answer     = answer,
        duration_s = duration,
    )


@app.get("/candidates", response_model=list[CandidateSummary], tags=["Candidates"])
async def list_candidates():
    """
    Returns a summary of all candidates ranked by risk (highest first).
    Uses the business logic tools directly — no LLM call needed.
    """
    if store is None:
        raise HTTPException(status_code=503, detail="Knowledge store not ready.")

    summary = get_all_candidates_summary(store)
    results = []

    for r in summary["ranked_results"]:
        t = store.get_candidate(r["candidate_id"]).tracker
        results.append(CandidateSummary(
            candidate_id      = r["candidate_id"],
            candidate_name    = r["candidate_name"],
            role              = t.role_code if t else "Unknown",
            tracker_status    = r["tracker_status"],
            overall_verdict   = r["overall_verdict"],
            ready_for_closure = r["ready_for_closure"],
            risk_level        = t.risk_level if t else "Unknown",
            issue_count       = r["issue_count"],
        ))

    return results


@app.get("/candidate/{candidate_id}", response_model=CandidateDetail, tags=["Candidates"])
async def get_candidate(candidate_id: str):
    """
    Returns full details for a specific candidate including all issues
    and per-control completion status.
    Uses business logic tools directly — no LLM call.
    """
    if store is None:
        raise HTTPException(status_code=503, detail="Knowledge store not ready.")

    cid = candidate_id.upper()
    record = store.get_candidate(cid)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {cid} not found. Valid IDs: {store.all_candidate_ids()}"
        )

    result = assess_closure_readiness(store, cid)
    t      = record.tracker
    checks = result.get("checks", {})

    # build control completion from check results
    def crim_label():
        crim = checks.get("criminality", {})
        res  = crim.get("result", "")
        if "exempt" in res.lower() or "not required" in res.lower():
            return "EXEMPT"
        return "COMPLETE" if crim.get("passed") else "INCOMPLETE"

    control_completion = {
        "identity":    "COMPLETE" if checks.get("freshness",   {}).get("passed") else "INCOMPLETE",
        "employment":  "COMPLETE" if checks.get("employment",  {}).get("passed") else "INCOMPLETE",
        "rtw":         "COMPLETE" if checks.get("rtw",         {}).get("passed") else "INCOMPLETE",
        "criminality": crim_label(),
    }

    adj = record.adjudication

    return CandidateDetail(
        candidate_id          = cid,
        candidate_name        = t.candidate_name if t else "Unknown",
        role                  = t.role_code if t else "Unknown",
        tracker_status        = result["tracker_status"],
        overall_verdict       = result["overall_verdict"],
        ready_for_closure     = result["ready_for_closure"],
        risk_level            = t.risk_level if t else "Unknown",
        issue_count           = result["issue_count"],
        all_issues            = result["all_issues"],
        control_completion    = control_completion,
        has_adjudication      = adj is not None,
        adjudication_decision = adj.decision if adj else None,
    )


@app.post("/admin/reload", tags=["System"])
async def reload_store(x_admin_key: str = Header(default="")):
    """
    Reload the knowledge store from disk.
    Useful when dataset files are updated without restarting the server.
    Requires X-Admin-Key header matching ADMIN_KEY env variable.
    """
    expected_key = os.environ.get("ADMIN_KEY", "")
    if not expected_key or x_admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

    global store
    logger.info("Reloading knowledge store...")
    store = build_knowledge_store(DATA_DIR)
    logger.info("Knowledge store reloaded. %d candidates.", len(store.candidates))

    return {"status": "reloaded", "candidates": len(store.candidates)}