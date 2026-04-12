"""
tools package — BPSS business-logic tools (Phase 2)

Import everything from here so the rest of the codebase
never needs to know which sub-module a function lives in.

Usage:
    from tools import check_freshness, assess_closure_readiness, ...
"""

from tools.freshness      import check_freshness
from tools.staleness     import check_document_staleness
from tools.criminality    import check_criminality
from tools.employment     import check_employment_coverage
from tools.rtw            import check_rtw
from tools.adjudication   import check_adjudication
from tools.contradictions import detect_contradictions
from tools.closure        import assess_closure_readiness, get_all_candidates_summary

__all__ = [
    "check_freshness",
    "check_document_staleness",
    "check_criminality",
    "check_employment_coverage",
    "check_rtw",
    "check_adjudication",
    "detect_contradictions",
    "assess_closure_readiness",
    "get_all_candidates_summary",
]