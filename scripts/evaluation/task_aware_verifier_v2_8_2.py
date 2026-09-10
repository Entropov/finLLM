"""v2.8.2 stock validator with a fail-closed missing-trend abstention path."""

from __future__ import annotations

import re
from typing import Any

from scripts.evaluation import task_aware_verifier_v2_7_core as v27
from scripts.rag.audit_schema import extract_citation_ids


VALIDATOR_VERSION = "evidence_anchor_task_validator.v2.8.2"


def _trend_available(evidence: dict[str, str]) -> bool:
    return any(v27._field_value(quote, "MA5") and v27._field_value(quote, "MA20") for quote in evidence.values())


def _cited_trend_abstention(lines: list[str], evidence: dict[str, str]) -> bool:
    for line in lines:
        citations = {item.upper() for item in extract_citation_ids(line)}
        if citations & set(evidence) and re.search(r"(?:MA5|MA20|均线|趋势).{0,24}(?:证据不足|无法确认|未提供|缺少)|(?:证据不足|无法确认|未提供|缺少).{0,24}(?:MA5|MA20|均线|趋势)", line):
            return True
    return False


def validate_task_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Preserve v2.7 behavior except for evidence-proven stock trend absence."""
    if str(case.get("task_type", "")) != "stock_analysis":
        return v27.validate_task_answer(case, answer)
    evidence = v27._evidence_by_id(case)
    lines = v27._content_lines(answer)
    failures = v27._invalid_citations(lines, evidence)
    if re.search(r"<think>.*?</think>", answer, re.DOTALL | re.IGNORECASE):
        failures.append("nonempty_thinking")
    if v27._has_prediction(answer):
        failures.append("unsupported_forward_prediction")
    if not evidence:
        requirements = {"evidence_present": False}
        failures.append("visible_evidence_missing")
    else:
        trend_present = _trend_available(evidence)
        requirements = {
            "market_price_or_return": v27._anchor_from_fields(lines, evidence, ("收盘价", "近20个交易日收益率")),
            "market_trend": (
                v27._anchor_from_fields(lines, evidence, ("MA5", "MA20"))
                or v27._keyword_supported(lines, evidence, ("趋势", "均线"), ("MA5", "MA20"))
                if trend_present
                else _cited_trend_abstention(lines, evidence)
            ),
            "market_risk": v27._anchor_from_fields(lines, evidence, ("年化波动率", "最大回撤")),
            "financial_anchor": v27._anchor_from_fields(lines, evidence, ("营业收入", "归母口径净利润", "经营活动现金流量净额")),
        }
    satisfied = [name for name, passed in requirements.items() if passed]
    missing = [name for name, passed in requirements.items() if not passed]
    return {
        "validator_version": VALIDATOR_VERSION,
        "task_type": "stock_analysis",
        "hard_gate_passed": bool(requirements) and not failures and not missing,
        "anchor_coverage": round(len(satisfied) / len(requirements), 4) if requirements else 0.0,
        "satisfied_anchors": satisfied,
        "missing_anchors": missing,
        "failures": sorted(set(failures)),
        "conditional_trend_abstention_allowed": bool(evidence) and not _trend_available(evidence),
    }
