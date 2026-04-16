"""
LangGraph Nodes — Phase 3
==========================
Three nodes form the agent pipeline:

  planner_node     — LLM reads the question and decides which tools to call
  executor_node    — runs the chosen tools deterministically (no LLM)
  synthesiser_node — LLM turns tool results into a grounded cited answer

LLM: meta-llama/Llama-3.1-8B-Instruct via HuggingFaceEndpoint
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from tools import (
    check_document_staleness,
    check_freshness,
    check_employment_coverage,
    check_rtw,
    check_criminality,
    check_adjudication,
    detect_contradictions,
    assess_closure_readiness,
    get_all_candidates_summary,
)

logger = logging.getLogger(__name__)

MAX_ITER = 6
ALL_CANDS = ["CAND-101", "CAND-102", "CAND-103", "CAND-104", "CAND-105", "CAND-106"]

# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "check_document_staleness":     check_document_staleness,
    "check_freshness":            check_freshness,
    "check_employment_coverage":  check_employment_coverage,
    "check_rtw":                  check_rtw,
    "check_criminality":          check_criminality,
    "check_adjudication":         check_adjudication,
    "detect_contradictions":      detect_contradictions,
    "assess_closure_readiness":   assess_closure_readiness,
    "get_all_candidates_summary": get_all_candidates_summary,
}

CANDIDATE_TOOLS = {
    "check_document_staleness",
    "check_freshness",
    "check_employment_coverage",
    "check_rtw",
    "check_criminality",
    "check_adjudication",
    "detect_contradictions",
    "assess_closure_readiness",
}

SUMMARY_TOOLS = {
    "get_all_candidates_summary",
}


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm() -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.7,
        max_new_tokens=4096,
    )
    model = ChatHuggingFace(llm=llm)
    return model


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    """Invoke the LLM and return the response as a plain string."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ── Planner prompt ────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a BPSS screening analyst assistant.

You have access to a knowledge store of 6 candidates:
CAND-101 (Aarav Mehta, Analyst)
CAND-102 (Sofia Khan, Analyst)
CAND-103 (Nina Patel, Intern)
CAND-104 (Liam O'Connor, Ops)
CAND-105 (Emma Roy, Analyst)
CAND-106 (Rohan Sen, Contractor-LTD)

Available tools:
- check_document_staleness(cand_id)  : checks ALL documents for staleness or expiry before review
- check_freshness(cand_id)           : checks address proof is within 90 days of review
- check_employment_coverage(cand_id) : checks 3-year employment history coverage
- check_rtw(cand_id)                 : checks right-to-work evidence validity
- check_criminality(cand_id)         : checks DBS/basic disclosure is complete
- check_adjudication(cand_id)        : checks adjudication register for approved exceptions
- detect_contradictions(cand_id)     : finds contradictions between tracker and evidence
- assess_closure_readiness(cand_id)  : runs ALL checks for one candidate (master tool)
- get_all_candidates_summary()       : runs assess for ALL candidates at once

RULES:
1. If the question mentions "candidates" (plural), "all", "which", "any", or does not name a specific candidate → you MUST use get_all_candidates_summary. This is the ONLY correct tool for these questions.
2. If the question names ONE specific candidate (e.g. "CAND-104") → use assess_closure_readiness for that candidate only.
3. If the question asks about a SPECIFIC control (e.g. "right to work", "employment", "address proof") for one candidate → use the specific tool.
4. If the question asks about "stale", "expired", "documents", "evidence expiry" → use check_document_staleness for ALL candidates.
5. If the question asks about "policy exceptions", "approved exceptions", "unapproved deviations", "adjudication" → use BOTH check_adjudication AND detect_contradictions for ALL candidates individually. check_adjudication finds approved exceptions. detect_contradictions finds unapproved deviations (cases closed without proper evidence or register entry). Do NOT use get_all_candidates_summary for this question type.
4. NEVER call assess_closure_readiness for multiple candidates — use get_all_candidates_summary instead.
5. Always include a "reason" explaining why you chose each tool.

Respond ONLY with a valid JSON array. No explanation, no markdown, no extra text.

Example format:
[
  {"tool": "get_all_candidates_summary", "cand_id": null, "reason": "need all candidates"},
  {"tool": "assess_closure_readiness", "cand_id": "CAND-104", "reason": "specific candidate question"}
]"""


# ── Synthesiser prompt ────────────────────────────────────────────────────────

SYNTHESISER_SYSTEM = """You are a BPSS screening analyst.
You have just run screening tools and received their results.
Write a clear professional answer to the user's SPECIFIC question based ONLY on the tool results.

CRITICAL — ANSWER THE SPECIFIC QUESTION ASKED:
- Read the question carefully and answer ONLY what was asked
- Do NOT dump all tool data — filter to what is relevant to the question
- If asked about "stale documents" → only report stale/expired document issues, not employment or contradictions
- If asked about "closure readiness" → report overall verdicts and all issues
- If asked about a specific control → only report that control's results

IMPORTANT — HOW TO READ get_all_candidates_summary RESULTS:
The tool returns a "ranked_results" list. Each item in the list is one candidate.
You MUST read and report on EVERY candidate in ranked_results — do not skip any.
For each candidate read: candidate_id, candidate_name, overall_verdict, issue_count, all_issues, all_citations.

RULES:
1. Answer the specific question asked — stay focused
2. Report on ALL candidates — never say "no details available"
3. Every factual claim MUST cite its source like: [source: filename, field: fieldname]
4. Never invent facts not present in the tool results
5. Clearly distinguish what the tracker CLAIMS vs what EVIDENCE shows
6. Flag contradictions prominently
7. Use plain English
8. Structure: summary first, then one section per relevant candidate
9. When ranking by risk — ALWAYS use the order from ranked_results in tool output. Never reorder by your own judgement. The ranking is already computed correctly by issue count and verdict severity.
10. For each candidate always list the SPECIFIC issues found — never write "several issues remain unresolved". Name each issue explicitly with its source citation.
11. Never invent contradictions — only report ones explicitly listed in the issues list. If a candidate has no contradictions, do not mention any."""


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 · planner_node
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> AgentState:
    """
    LLM reads the question and returns a JSON plan of tool calls.
    Input  : question, iterations
    Output : plan, iterations (incremented)
    """
    iterations = state.get("iterations", 0)
    if iterations >= MAX_ITER:
        logger.warning("Max iterations reached.")
        return {"error": "Max iterations reached.", "final_answer": ""}

    question = state["question"]
    logger.info("Planner received question: %s", question)

    raw = _call_llm(
        system_prompt=PLANNER_SYSTEM,
        user_prompt=f"Question: {question}\n\nProduce the tool call plan as JSON.",
        max_tokens=512,
    )

    # Strip markdown fences if model added them
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break

    # Extract JSON array if buried in extra text
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        plan = json.loads(raw)
        if not isinstance(plan, list):
            raise ValueError("Plan must be a JSON array.")
        logger.info("Planner produced %d tool call(s).", len(plan))
    except Exception as exc:
        logger.error("Planner JSON parse failed: %s\nRaw: %s", exc, raw)
        # Safe fallback
        plan = [{"tool": "get_all_candidates_summary", "cand_id": None,
                 "reason": "fallback due to parse error"}]

    return {
        "plan":       plan,
        "iterations": iterations + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 · executor_node
# ─────────────────────────────────────────────────────────────────────────────

def executor_node(state: AgentState) -> AgentState:
    """
    Runs each tool in the plan deterministically. Zero LLM calls.
    Input  : plan, store
    Output : tool_results
    """
    plan  = state.get("plan", [])
    store = state["store"]

    tool_results = []
    seen = set()

    for call in plan:
        tool_name = call.get("tool", "")
        cand_id   = call.get("cand_id")
        reason    = call.get("reason", "")

        key = (tool_name, cand_id)
        if key in seen:
            continue
        seen.add(key)

        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            logger.warning("Unknown tool: %s", tool_name)
            tool_results.append({
                "tool": tool_name, "cand_id": cand_id,
                "result": {"passed": False,
                           "issues": [f"Unknown tool: {tool_name}"],
                           "citations": [], "result": "Tool not found."}
            })
            continue

        try:
            if tool_name in CANDIDATE_TOOLS:
                if not cand_id:
                    # No cand_id — run for all candidates
                    for cid in ALL_CANDS:
                        k2 = (tool_name, cid)
                        if k2 not in seen:
                            result = fn(store, cid)
                            tool_results.append({"tool": tool_name, "cand_id": cid,
                                                 "result": result, "reason": reason})
                            seen.add(k2)
                else:
                    result = fn(store, cand_id.upper())
                    tool_results.append({"tool": tool_name, "cand_id": cand_id,
                                         "result": result, "reason": reason})

            elif tool_name in SUMMARY_TOOLS:
                result = fn(store)
                tool_results.append({"tool": tool_name, "cand_id": None,
                                     "result": result, "reason": reason})

        except Exception as exc:
            logger.error("Tool %s failed for %s: %s", tool_name, cand_id, exc)
            tool_results.append({
                "tool": tool_name, "cand_id": cand_id,
                "result": {"passed": False,
                           "issues": [f"Tool error: {exc}"],
                           "citations": [], "result": str(exc)}
            })

    logger.info("Executor ran %d tool call(s), got %d result(s).",
                len(plan), len(tool_results))
    return {"tool_results": tool_results}


# ─────────────────────────────────────────────────────────────────────────────
# Helper · _flatten_tool_results
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_tool_results(tool_results: list) -> str:
    """
    Convert tool results into a compact readable text block for the LLM.
    Avoids sending 40KB of nested JSON — extracts only what matters.
    """
    lines = []

    for item in tool_results:
        tool    = item.get("tool", "")
        cand_id = item.get("cand_id")
        result  = item.get("result", {})

        # ── get_all_candidates_summary ────────────────────────────────────────
        if tool == "get_all_candidates_summary":
            lines.append(f"TOOL: get_all_candidates_summary")
            lines.append(f"Total candidates: {result.get('total_candidates')}")
            lines.append(f"Ready for closure: {result.get('ready_for_closure')}")
            lines.append(f"Not ready: {result.get('not_ready')}")
            lines.append("")
            lines.append("RANKING (by risk, highest first — do NOT change this order):")
            ranked = result.get("ranked_results", [])
            for idx, r in enumerate(ranked, 1):
                lines.append(f"  Rank {idx}: {r.get('candidate_id')} {r.get('candidate_name')} | verdict={r.get('overall_verdict')} | issue_count={r.get('issue_count')}")
            lines.append("")

            for r in ranked:
                cid      = r.get("candidate_id", "")
                name     = r.get("candidate_name", "")
                verdict  = r.get("overall_verdict", "")
                tracker  = r.get("tracker_status", "")
                n_issues = r.get("issue_count", 0)
                issues   = r.get("all_issues", [])

                # Detect ready_to_join from issues list
                rtj_issues = [i for i in issues if "ready_to_join" in i.lower() or "ready to join" in i.lower()]
                rtj_label  = " ← MARKED READY TO JOIN WITH OPEN ISSUES" if rtj_issues else ""

                lines.append(f"CANDIDATE: {cid} — {name}{rtj_label}")
                lines.append(f"  Tracker status : {tracker}")
                lines.append(f"  Verdict        : {verdict}")
                lines.append(f"  Risk level     : {r.get('risk_level', 'unknown')}")
                lines.append(f"  Issue count    : {n_issues}")
                if rtj_issues:
                    lines.append(f"  READY_TO_JOIN  : True — but has {n_issues} open issue(s)")

                # Per-control pass/fail — explicit so LLM does not have to infer
                checks = r.get("checks", {})
                lines.append("  Control completion (use these exact values, do not infer):")
                lines.append(f"    identity    = {'COMPLETE' if checks.get('freshness',{}).get('passed') else 'INCOMPLETE'}")
                lines.append(f"    employment  = {'COMPLETE' if checks.get('employment',{}).get('passed') else 'INCOMPLETE'}")
                lines.append(f"    rtw         = {'COMPLETE' if checks.get('rtw',{}).get('passed') else 'INCOMPLETE'}")
                crim_check = checks.get("criminality", {})
                crim_result = crim_check.get("result", "")
                if "exempt" in crim_result.lower() or "not required" in crim_result.lower():
                    crim_label = "EXEMPT"
                elif crim_check.get("passed"):
                    crim_label = "COMPLETE"
                else:
                    crim_label = "INCOMPLETE"
                lines.append(f"    criminality = {crim_label}")

                if issues:
                    lines.append("  Missing items / issues:")
                    for issue in issues:
                        lines.append(f"    - {issue}")

                # Key citations
                citations = r.get("all_citations", [])
                if citations:
                    lines.append("  Key citations:")
                    seen_sources = set()
                    for c in citations[:8]:
                        src = c.get("source", "")
                        if src not in seen_sources:
                            lines.append(f"    [{src}, field: {c.get('field')}, value: {c.get('value')}]")
                            seen_sources.add(src)
                lines.append("")

        # ── assess_closure_readiness ──────────────────────────────────────────
        elif tool == "assess_closure_readiness":
            cid     = result.get("candidate_id", cand_id)
            name    = result.get("candidate_name", "")
            verdict = result.get("overall_verdict", "")
            tracker = result.get("tracker_status", "")
            issues  = result.get("all_issues", [])

            lines.append(f"TOOL: assess_closure_readiness")
            lines.append(f"CANDIDATE: {cid} — {name}")
            lines.append(f"  Tracker status: {tracker}")
            lines.append(f"  Verdict: {verdict}")
            if issues:
                lines.append("  Issues:")
                for issue in issues:
                    lines.append(f"    - {issue}")
            lines.append("")

        # ── check_document_staleness ──────────────────────────────────────────
        elif tool == "check_document_staleness":
            res_txt = result.get("result", "")
            issues  = result.get("issues", [])
            flagged = result.get("flagged_documents", [])

            lines.append(f"TOOL: check_document_staleness | CANDIDATE: {cand_id}")
            lines.append(f"  Result: {res_txt}")
            if flagged:
                lines.append("  Flagged documents:")
                for f in flagged:
                    lines.append(
                        f"    - {f.get('document_id')} ({f.get('doc_type')}) "
                        f"[severity: {f.get('severity')}]: {f.get('reason')} "
                        f"[source: {f.get('source')}]"
                    )
            else:
                lines.append("  No stale or expired documents.")
            lines.append("")

        # ── detect_contradictions (policy deviation labelling) ───────────────
        elif tool == "detect_contradictions":
            passed  = result.get("passed", True)
            res_txt = result.get("result", "")
            issues  = result.get("issues", [])

            lines.append(f"TOOL: detect_contradictions | CANDIDATE: {cand_id}")
            lines.append(f"  Result: {res_txt}")

            if not passed and issues:
                # Check if this looks like an unapproved deviation
                # (Clear status with open issues and no adjudication entry)
                is_deviation = any(
                    "no adjudication register entry" in i.lower() or
                    "no approved exception" in i.lower()
                    for i in issues
                )
                if is_deviation:
                    lines.append(f"  DEVIATION TYPE: UNAPPROVED DEVIATION — case actioned without formal register entry")
                lines.append(f"  Contradictions found:")
                for issue in issues:
                    lines.append(f"    - {issue}")
            else:
                lines.append(f"  No contradictions — no unapproved deviation detected")
            lines.append("")

        # ── check_adjudication ───────────────────────────────────────────────
        elif tool == "check_adjudication":
            passed   = result.get("passed", True)
            res_txt  = result.get("result", "")
            issues   = result.get("issues", [])
            citations = result.get("citations", [])

            lines.append(f"TOOL: check_adjudication | CANDIDATE: {cand_id}")
            lines.append(f"  Result : {res_txt}")

            # Extract key adjudication fields from citations
            adj_fields = {c.get("field"): c.get("value") for c in citations
                         if c.get("source","").endswith("Adjudication_Register.pdf")}
            if adj_fields.get("entry_for_candidate") == "NONE":
                lines.append(f"  Adjudication entry : NONE — no approved exception exists")
                lines.append(f"  Exception type     : NO EXCEPTION")
            elif adj_fields:
                lines.append(f"  Decision   : {adj_fields.get('decision', 'unknown')}")
                lines.append(f"  Approvers  : {adj_fields.get('approvers', 'unknown')}")
                lines.append(f"  Scope      : {adj_fields.get('scope', 'unknown')}")
                if issues:
                    lines.append(f"  LIMITATIONS (does NOT constitute full closure):")
                    for issue in issues:
                        lines.append(f"    - {issue}")
                    lines.append(f"  Exception type : APPROVED BUT LIMITED")
                else:
                    lines.append(f"  Exception type : FULLY APPROVED")
            if not passed and not adj_fields.get("entry_for_candidate"):
                lines.append(f"  Issues:")
                for issue in issues:
                    lines.append(f"    - {issue}")
            lines.append("")

        # ── individual tools ──────────────────────────────────────────────────
        else:
            passed  = result.get("passed", False)
            res_txt = result.get("result", "")
            issues  = result.get("issues", [])

            lines.append(f"TOOL: {tool} | CANDIDATE: {cand_id}")
            lines.append(f"  Passed : {passed}")
            lines.append(f"  Result : {res_txt}")
            if issues:
                lines.append("  Issues:")
                for issue in issues:
                    lines.append(f"    - {issue}")
            lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 · synthesiser_node
# ─────────────────────────────────────────────────────────────────────────────

def synthesiser_node(state: AgentState) -> AgentState:
    """
    LLM reads tool results and writes a grounded cited answer.
    Input  : question, tool_results
    Output : final_answer
    """
    question     = state["question"]
    tool_results = state.get("tool_results", [])

    if not tool_results:
        return {"final_answer": "No tool results available to answer the question."}

    results_text = _flatten_tool_results(tool_results)

    final_answer = _call_llm(
        system_prompt=SYNTHESISER_SYSTEM,
        user_prompt=(
            f"Question: {question}\n\n"
            f"Tool results:\n{results_text}\n\n"
            f"Write the answer now."
        ),
        max_tokens=1500,
    )

    logger.info("Synthesiser produced answer (%d chars).", len(final_answer))
    return {"final_answer": final_answer}