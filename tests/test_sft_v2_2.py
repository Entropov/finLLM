#!/usr/bin/env python3
"""SFT v2.2 canonical answer and data-gate tests."""

from __future__ import annotations

import json

import pytest

from scripts.data_processing.build_sft_v2_2_dataset import (
    _has_required_evidence,
    audit_target,
    build_canonical_target,
)
from scripts.rag.audit_schema import strip_thinking


AS_OF = "2026-09-01T00:00:00+08:00"
FINANCIAL_ID = "E1111111111"
MARKET_ID = "E2222222222"
FINANCIAL = {
    "evidence_id": FINANCIAL_ID,
    "exact_quote": (
        "示例公司（证券代码600000）报告期2025-09-30的营业收入为100.0000亿元，"
        "归母口径净利润为10.0000亿元，经营活动现金流量净额为8.0000亿元；"
        "同比可比期2024-09-30的营业收入为90.0000亿元，归母口径净利润为9.0000亿元。"
        "营业收入同比变化11.1111%，净利润同比变化11.1111%。"
        "同期资产总计200.0000亿元、负债合计80.0000亿元、流动资产合计60.0000亿元、"
        "流动负债合计30.0000亿元。按负债合计除以资产总计计算，资产负债率为40.0000%；"
        "按流动资产合计除以流动负债合计计算，流动比率为2.0000。"
    ),
    "source": "https://example.test/financial",
    "publisher": "example",
    "reliability_tier": "verified_secondary",
    "published_at": "2025-10-01T00:00:00+08:00",
    "effective_at": "2025-09-30T00:00:00+08:00",
    "fetched_at": "2026-03-01T00:00:00+08:00",
}
MARKET = {
    "evidence_id": MARKET_ID,
    "exact_quote": (
        "示例公司（证券代码600000）截至2026-03-06收盘价为10.0000元，近20个交易日收益率为5.0000%。"
        "最近60个交易日按日收益率标准差乘以根号252计算的年化波动率为20.0000%，"
        "按收盘价相对历史滚动高点计算的最大回撤为-8.0000%。"
        "MA5为10.2000元，MA20为9.8000元，RSI6为60.0000。"
    ),
    "source": "https://example.test/market",
    "publisher": "example",
    "reliability_tier": "verified_secondary",
    "published_at": "2026-03-06T15:00:00+08:00",
    "effective_at": "2026-03-06T15:00:00+08:00",
    "fetched_at": "2026-03-07T00:00:00+08:00",
}


def _prompt(task_type: str) -> dict:
    return {
        "query": f"针对示例公司（600000）完成{task_type}任务并逐项引用证据。",
        "task_type": task_type,
        "request_as_of": AS_OF,
        "evidence": [FINANCIAL, MARKET],
    }


@pytest.mark.parametrize(
    "task_type",
    ["financial_qa", "financial_report", "quant_strategy", "risk_assessment", "sentiment_analysis", "stock_analysis"],
)
def test_canonical_targets_pass_production_audit(task_type: str):
    prompt = _prompt(task_type)
    target = build_canonical_target(task_type, prompt)
    audit = audit_target(task_type, prompt, target)
    assert audit["hard_gate_passed"] is True, audit["hard_failures"]
    expected_task_validity = 0.3333 if task_type == "sentiment_analysis" else 1.0
    assert audit["task_validity"] == expected_task_validity
    assert audit["citation_coverage"] == 1.0
    assert audit["citation_precision"] == 1.0
    assert audit["numeric_consistency"] == 1.0
    assert len(strip_thinking(target)) <= 6000


def test_complex_targets_use_short_structured_reasoning_but_sentiment_does_not():
    stock = build_canonical_target("stock_analysis", _prompt("stock_analysis"))
    sentiment = build_canonical_target("sentiment_analysis", _prompt("sentiment_analysis"))
    think_payload = stock.split("<think>\n", 1)[1].split("\n</think>", 1)[0]
    assert set(json.loads(think_payload)) == {"evidence_ids", "plan"}
    assert len(think_payload) < 300
    assert "<think>" not in sentiment


def test_task_evidence_completeness_gate_rejects_missing_financial_context():
    prompt = _prompt("stock_analysis")
    prompt["evidence"] = [MARKET]
    assert _has_required_evidence("stock_analysis", prompt) is False
    with pytest.raises(ValueError, match="missing required evidence"):
        build_canonical_target("stock_analysis", prompt)
