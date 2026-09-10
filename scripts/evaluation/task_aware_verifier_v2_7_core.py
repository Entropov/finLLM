"""Evidence-only task validator for the SFT v2.7 core selector.

The validator deliberately has no access to evaluation ``scoring`` assertions.
It checks whether an answer covers the task-specific evidence anchors that were
visible in the prompt.  This makes it suitable for candidate selection and
hard-negative verification, unlike gold task scores.
"""

from __future__ import annotations

import re
from typing import Any

from scripts.evaluation.sft_v2_7_scope import CORE_TASKS, require_core_task
from scripts.rag.audit_schema import extract_citation_ids
from scripts.rag.quant_protocol import parse_quant_action


VALIDATOR_VERSION = "evidence_anchor_task_validator.v1"


def _content_lines(answer: str) -> list[str]:
    return [
        line.strip()
        for line in answer.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _evidence_by_id(case: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("evidence_id", "")).upper(): str(item.get("exact_quote", ""))
        for item in case.get("evidence", [])
        if item.get("evidence_id")
    }


def _field_value(quote: str, field: str) -> str | None:
    match = re.search(rf"{re.escape(field)}为([^，。；;\\s]+)", quote)
    return match.group(1).strip() if match else None


def _field_supported(lines: list[str], evidence_id: str, field: str, value: str) -> bool:
    return any(
        field in line
        and value in line
        and evidence_id in {item.upper() for item in extract_citation_ids(line)}
        for line in lines
    )


def _anchor_from_fields(
    lines: list[str],
    evidence: dict[str, str],
    fields: tuple[str, ...],
) -> bool:
    for evidence_id, quote in evidence.items():
        for field in fields:
            value = _field_value(quote, field)
            if value and _field_supported(lines, evidence_id, field, value):
                return True
    return False


def _keyword_supported(
    lines: list[str], evidence: dict[str, str], keywords: tuple[str, ...], evidence_terms: tuple[str, ...]
) -> bool:
    for evidence_id, quote in evidence.items():
        if not any(term in quote for term in evidence_terms):
            continue
        for line in lines:
            citations = {item.upper() for item in extract_citation_ids(line)}
            if evidence_id in citations and any(keyword in line for keyword in keywords):
                return True
    return False


def _invalid_citations(lines: list[str], evidence: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        citations = {item.upper() for item in extract_citation_ids(line)}
        if not citations:
            failures.append(f"line_{line_number}:missing_citation")
            continue
        if citations and not citations <= set(evidence):
            failures.append(f"line_{line_number}:invalid_evidence_id")
    return failures


def _has_prediction(answer: str) -> bool:
    return bool(
        re.search(
            r"(?:预计|预测|将会|必然|大概率).{0,24}(?:上涨|下跌|改善|恶化|目标价|买入|卖出|收益)",
            answer,
        )
    )


def _requirements(case: dict[str, Any], answer: str) -> tuple[dict[str, bool], list[str]]:
    task_type = str(case.get("task_type", ""))
    require_core_task(task_type)
    evidence = _evidence_by_id(case)
    lines = _content_lines(answer)
    failures = _invalid_citations(lines, evidence)
    if re.search(r"<think>.*?</think>", answer, re.DOTALL | re.IGNORECASE):
        failures.append("nonempty_thinking")
    if _has_prediction(answer):
        failures.append("unsupported_forward_prediction")

    if task_type == "quant_strategy":
        _, quant_errors = parse_quant_action(answer)
        return {"canonical_quant_action": not quant_errors}, quant_errors

    if not evidence:
        return {"evidence_present": False}, [*failures, "visible_evidence_missing"]

    if task_type == "financial_qa":
        return {
            "asset_liability_ratio": _anchor_from_fields(lines, evidence, ("资产负债率",)),
            "current_ratio": _anchor_from_fields(lines, evidence, ("流动比率",)),
        }, failures

    if task_type == "stock_analysis":
        return {
            "market_price_or_return": _anchor_from_fields(lines, evidence, ("收盘价", "近20个交易日收益率")),
            "market_trend": _anchor_from_fields(lines, evidence, ("MA5", "MA20"))
            or _keyword_supported(lines, evidence, ("趋势", "均线"), ("MA5", "MA20")),
            "market_risk": _anchor_from_fields(lines, evidence, ("年化波动率", "最大回撤")),
            "financial_anchor": _anchor_from_fields(
                lines, evidence, ("营业收入", "归母口径净利润", "经营活动现金流量净额")
            ),
        }, failures

    raise ValueError(f"unsupported core task: {task_type}")


def validate_task_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Validate a candidate using only its task type, visible evidence, and text."""
    task_type = str(case.get("task_type", ""))
    if task_type not in CORE_TASKS:
        raise ValueError(f"task outside the SFT v2.7 core scope: {task_type}")
    requirements, failures = _requirements(case, answer)
    satisfied = [name for name, passed in requirements.items() if passed]
    missing = [name for name, passed in requirements.items() if not passed]
    coverage = len(satisfied) / len(requirements) if requirements else 0.0
    hard_gate = bool(requirements) and not failures and not missing
    return {
        "validator_version": VALIDATOR_VERSION,
        "task_type": task_type,
        "hard_gate_passed": hard_gate,
        "anchor_coverage": round(coverage, 4),
        "satisfied_anchors": satisfied,
        "missing_anchors": missing,
        "failures": sorted(set(failures)),
    }
