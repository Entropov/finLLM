#!/usr/bin/env python3
"""SFT v2.5 quant action, renderer, and audit protocol tests."""

from __future__ import annotations

import json

import pytest

from scripts.data_processing.build_sft_v2_5_dataset import (
    TARGET_TASKS,
    build_target,
    decision_basis,
    rendered_answer,
    rendered_quant_errors,
    target_errors,
)
from scripts.evaluation.sft_v2_5_protocol import case_prompt, materialize_answer, score_answer
from scripts.rag.audit_schema import content_digest
from scripts.rag.quant_protocol import (
    QUANT_CODE,
    canonical_quant_action,
    materialize_quant_output,
    parse_quant_action,
    render_quant_artifact,
)


AS_OF = "2026-09-01T00:00:00+08:00"
FINANCIAL_QUOTE = (
    "示例公司（证券代码600000）报告期2025-09-30的营业收入为100.0000亿元，"
    "归母口径净利润为10.0000亿元，经营活动现金流量净额为8.0000亿元；"
    "同比可比期2024-09-30的营业收入为90.0000亿元，归母口径净利润为9.0000亿元。"
    "营业收入同比变化11.1111%，净利润同比变化11.1111%。"
    "同期资产总计200.0000亿元、负债合计80.0000亿元、流动资产合计60.0000亿元、"
    "流动负债合计30.0000亿元。按负债合计除以资产总计计算，资产负债率为40.0000%；"
    "按流动资产合计除以流动负债合计计算，流动比率为2.0000。"
)
MARKET_QUOTE = (
    "示例公司（证券代码600000）截至2026-03-06收盘价为10.0000元，近20个交易日收益率为5.0000%。"
    "最近60个交易日按日收益率标准差乘以根号252计算的年化波动率为20.0000%，"
    "按收盘价相对历史滚动高点计算的最大回撤为-8.0000%。MA5为10.2000元，"
    "MA20为9.8000元，RSI6为60.0000。"
)


def _evidence(evidence_id: str, quote: str, suffix: str) -> dict:
    source = f"https://example.test/{suffix}"
    return {
        "evidence_id": evidence_id,
        "exact_quote": quote,
        "source": source,
        "source_uri": source,
        "canonical_url": source,
        "content_hash": content_digest(quote),
        "publisher": "example",
        "source_type": "news",
        "reliability_tier": "verified_secondary",
        "published_at": "2026-03-06T15:00:00+08:00",
        "effective_at": "2026-03-06T15:00:00+08:00",
        "fetched_at": "2026-03-07T00:00:00+08:00",
        "document_version": suffix,
        "point_in_time_available": True,
    }


FINANCIAL = _evidence("E1111111111", FINANCIAL_QUOTE, "financial")
MARKET = _evidence("E2222222222", MARKET_QUOTE, "market")


def _prompt(task_type: str, evidence: list[dict] | None = None) -> dict:
    prompt = {
        "query": f"针对示例公司（600000）完成{task_type}任务。",
        "task_type": task_type,
        "request_as_of": AS_OF,
        "evidence": evidence if evidence is not None else [FINANCIAL, MARKET],
    }
    prompt["decision_basis"] = decision_basis(task_type, prompt)
    return prompt


def _case(task_type: str) -> dict:
    evidence = [{key: value for key, value in item.items() if key != "source"} for item in (FINANCIAL, MARKET)]
    return {
        "id": f"case-{task_type}",
        "task_type": task_type,
        "question": f"针对示例公司（600000）完成{task_type}任务。",
        "request_as_of": AS_OF,
        "source_group": f"source-{task_type}",
        "requires_audit": True,
        "evidence": evidence,
        "scoring": {"mode": "patterns", "required_patterns": [], "forbidden_patterns": [], "pass_threshold": 0.0},
    }


def test_quant_action_is_single_line_strict_json() -> None:
    target = canonical_quant_action()
    assert "\n" not in target
    assert parse_quant_action(target)[1] == []
    assert json.loads(target)["template_id"] == "ma5_ma20_long_only"
    assert parse_quant_action(target + "\n说明")[1]
    assert parse_quant_action(target.replace("ma5_ma20", "ma10_ma30"))[1]


def test_quant_renderer_is_deterministic_and_external_data_only() -> None:
    rendered = render_quant_artifact([FINANCIAL, MARKET])
    assert rendered == render_quant_artifact([FINANCIAL, MARKET])
    assert "10.0000元 [E2222222222]" in rendered
    assert "quant.v2.5" in rendered
    assert QUANT_CODE in rendered
    assert not any(term in QUANT_CODE for term in ("read_csv", "download", "yfinance", "np.random"))


@pytest.mark.parametrize("task_type", sorted(TARGET_TASKS))
def test_v2_5_targets_are_short_and_protocol_valid(task_type: str) -> None:
    prompt = _prompt(task_type)
    target = build_target(task_type, prompt)
    assert len(target) <= 900
    assert "<think>" not in target.lower()
    assert target_errors(task_type, prompt, target) == []
    if task_type == "quant_strategy":
        assert rendered_quant_errors(prompt, rendered_answer(task_type, prompt, target)) == []


def test_quant_protocol_materializes_before_scoring() -> None:
    case = _case("quant_strategy")
    raw = canonical_quant_action()
    rendered, errors = materialize_answer(case, raw)
    assert errors == []
    assert rendered.startswith("### 证据")
    score = score_answer(case, raw)
    assert score["materialized"] is True
    assert score["audit_hard_gate_passed"] is True
    rejected = score_answer(case, raw + "\nextra")
    assert rejected["materialized"] is False
    assert rejected["protocol_errors"]


def test_materializer_rejects_code_instead_of_repairing_it() -> None:
    rendered, errors = materialize_quant_output(QUANT_CODE, [MARKET])
    assert rendered is None
    assert "quant_action_extra_markup" in errors


def test_case_prompt_exposes_exact_quant_action_contract() -> None:
    prompt = case_prompt(_case("quant_strategy"))
    assert canonical_quant_action() in prompt["output_contract"]
    assert prompt["decision_basis"]["response_budget"] == "one_json_line"
