"""
BPSS Screening Agent — Entry Point
====================================
Run from the project root:

    python run.py "Which candidates are not ready for closure?"
    python run.py "Compare CAND-104 adjudication against policy"
    python run.py --all        (answers all 10 evaluation questions)

Requires:
    HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) for Hugging Face Inference
"""

import sys
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root before anything else
load_dotenv(Path(__file__).parent / ".env")

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)

from ingestion.ingest import build_knowledge_store
from agent.graph import run

DATA_DIR = Path(__file__).parent / "bpss_agentic_dataset"

EVALUATION_QUESTIONS = [
    "Which candidate files are not yet ready for BPSS closure, and why?",
    "For each candidate, determine whether the identity, employment history, criminality, and right-to-work checks are complete. Cite evidence.",
    "Which cases appear to have policy exceptions? Distinguish approved exceptions from unapproved deviations.",
    "Compare the adjudication note for CAND-104 with the policy. Is closure justified?",
    "Which candidates have potentially stale documents or evidence that expired before review completion?",
    "Draft an escalation summary for the Screening Operations Lead covering the top 3 highest-risk cases.",
    "What information is missing to determine whether CAND-105 can be cleared?",
    "Reconcile any contradictions between the tracker and the analyst notes.",
    "Which candidates were marked ready to join even though one or more mandatory controls were incomplete?",
    "Create a structured JSON summary for each candidate with: status, control completion by category, missing items, risk level, supporting evidence.",
]


def main():
    # Check API key
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("ERROR: API KEY IS not set.")
        sys.exit(1)

    # Build knowledge store once
    print("\nBuilding knowledge store...")
    store = build_knowledge_store(DATA_DIR)
    print(f"Loaded {len(store.candidates)} candidates.\n")

    # Determine mode
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py \"your question here\"")
        print("  python run.py --all")
        sys.exit(0)

    if sys.argv[1] == "--all":
        # Answer all 10 evaluation questions
        print("=" * 70)
        print("Running all 10 evaluation questions")
        print("=" * 70)
        for i, question in enumerate(EVALUATION_QUESTIONS, 1):
            print(f"\nQ{i}: {question}")
            print("-" * 70)
            answer = run(question, store)
            print(answer)
            print()
    else:
        # Single question from command line
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}\n")
        print("-" * 70)
        answer = run(question, store)
        print(answer)


if __name__ == "__main__":
    main()