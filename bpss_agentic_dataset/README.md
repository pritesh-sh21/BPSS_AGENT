# BPSS Agentic AI Interview Exercise

## Scenario
Build a backend-only **agentic AI system** that answers questions over a mixed-format fictional BPSS screening dataset.

## Objective
Your system should:
1. Interpret the question.
2. Decide what tools to use.
3. Retrieve evidence from PDFs, DOCX, CSV, and XLSX files.
4. Combine structured and unstructured evidence.
5. Detect contradictions or missing information.
6. Produce a grounded answer with citations to source files and fields.

## Expected capabilities
- Multi-step planning / tool use
- Document retrieval over mixed file types
- Basic structured querying over tabular data
- Evidence-backed synthesis
- Inability handling: say when evidence is insufficient
- Simple session memory or state is a plus

## Deliverables
- Source code
- README with setup instructions
- One command to run the agent locally
- A short architecture note (1–2 pages max)
- Answers to the evaluation questions in `candidate_pack/sample_questions.md`

## Constraints
- No UI required
- You may use any framework
- Focus on correctness, traceability, and robustness rather than polish

## What the data represents
This is a fictional hiring and identity-screening dataset inspired by BPSS-style pre-employment controls. It includes:
- screening policy
- candidate packs
- adjudication notes
- identity/right-to-work evidence logs
- tabular trackers
- conflicting notes and exceptions

## Success criteria
A strong solution will:
- cite exact files/rows/sheets used
- distinguish policy from exceptions
- detect stale or conflicting evidence
- identify missing data explicitly
- avoid hallucinating absent facts
