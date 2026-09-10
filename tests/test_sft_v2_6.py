"""SFT v2.6 compact target and scoring protocol tests."""

from __future__ import annotations

import json

import pytest

from scripts.data_processing.build_sft_v2_6_dataset import TARGET_TASKS, _compact_target, _target_errors
from scripts.evaluation.sft_v2_6_protocol import case_prompt, materialize_answer, prompt_contract, score_answer
from scripts.rag.audit_schema import content_digest
from scripts.rag.quant_protocol import canonical_quant_action

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
    "按收盘价相对历史滚动高点计算的最大回撤为-8.0000%。MA5为10.2000元，MA20为9.8000元。"
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
    return {
        "query": (
            f"针对示例公司完成{task_type}任务，核对营业收入、净利润、现金流、资产负债率、"
            "流动比率、收盘价、收益率、波动率、最大回撤与均线。"
        ),
        "task_type": task_type,
        "request_as_of": AS_OF,
        "evidence": evidence if evidence is not None else [FINANCIAL, MARKET],
    }


def _case(task_type: str) -> dict:
    evidence = [{key: value for key, value in item.items() if key != "source"} for item in (FINANCIAL, MARKET)]
    return {
        "id": f"case-{task_type}",
        "task_type": task_type,
        "question": (
            f"针对示例公司完成{task_type}任务，核对营业收入、净利润、现金流、资产负债率、"
            "流动比率、收盘价、收益率、波动率、最大回撤与均线。"
        ),
        "request_as_of": AS_OF,
        "source_group": f"source-{task_type}",
        "requires_audit": True,
        "evidence": evidence,
        "scoring": {"mode": "patterns", "required_patterns": [], "forbidden_patterns": [], "pass_threshold": 0.0},
    }


@pytest.mark.parametrize("task_type", sorted(TARGET_TASKS))
def test_v2_6_targets_are_compact_and_audit_valid(task_type: str) -> None:
    prompt = _prompt(task_type)
    target = _compact_target(task_type, prompt)
    assert len(target) <= 500
    assert "<think>" not in target.lower()
    assert _target_errors(task_type, prompt, target) == []
    if task_type != "quant_strategy":
        assert not target.startswith("### 证据")


@pytest.mark.parametrize("task_type", ["financial_qa", "financial_report", "sentiment_analysis"])
def test_v2_6_missing_financial_evidence_abstains(task_type: str) -> None:
    prompt = _prompt(task_type, [])
    target = _compact_target(task_type, prompt)
    assert target.startswith("- 证据不足，无法确认")
    assert len(target.splitlines()) == 1
    assert _target_errors(task_type, prompt, target) == []


def test_v2_6_quant_reuses_frozen_renderer_contract() -> None:
    target = _compact_target("quant_strategy", _prompt("quant_strategy"))
    assert target == canonical_quant_action()
    assert json.loads(target)["artifact_version"] == "quant.v2.5"
    contract = prompt_contract()
    assert contract["contract_version"] == "sft_v2.6"
    assert contract["quant_model_output"] == target


def test_v2_6_materializes_quant_before_scoring() -> None:
    case = _case("quant_strategy")
    rendered, errors = materialize_answer(case, canonical_quant_action())
    assert errors == []
    assert rendered.startswith("### 证据")
    result = score_answer(case, canonical_quant_action())
    assert result["materialized"] is True
    assert result["audit_hard_gate_passed"] is True


def test_v2_6_case_prompt_declares_compact_budget() -> None:
    assert case_prompt(_case("stock_analysis"))["decision_basis"]["response_budget"] == "compact_atomic_lines"
