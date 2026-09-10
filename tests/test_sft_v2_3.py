#!/usr/bin/env python3
"""SFT v2.3 exact-number, atomic-citation, and quant-artifact tests."""

from __future__ import annotations

import json
import re

import pytest

from scripts.data_processing.build_sft_v2_3_dataset import (
    QUANT_CODE,
    STRICT_SYSTEM_PROMPT,
    TARGET_TASKS,
    _audit_target,
    _quant_artifact_errors,
    build_adversarial_cases,
    build_strict_target,
    strict_target_errors,
)
from scripts.rag.audit_schema import extract_citation_ids, strip_thinking


AS_OF = "2026-09-01T00:00:00+08:00"
FINANCIAL = {
    "evidence_id": "E1111111111",
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
    "evidence_id": "E2222222222",
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
        "query": f"针对示例公司（600000）执行{task_type}。",
        "task_type": task_type,
        "request_as_of": AS_OF,
        "evidence": [FINANCIAL, MARKET],
    }


@pytest.mark.parametrize("task_type", sorted(TARGET_TASKS))
def test_strict_targets_pass_production_and_v2_3_gates(task_type: str):
    prompt = _prompt(task_type)
    target = build_strict_target(task_type, prompt)
    reward = _audit_target(task_type, prompt, target)
    assert reward["hard_gate_passed"] is True, reward["hard_failures"]
    assert reward["numeric_consistency"] == 1.0
    assert reward["citation_coverage"] == 1.0
    assert reward["citation_precision"] == 1.0
    assert strict_target_errors(task_type, prompt, target) == []


def test_every_non_heading_prose_line_has_a_sentence_final_citation():
    target = strip_thinking(build_strict_target("stock_analysis", _prompt("stock_analysis")))
    for line in target.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert extract_citation_ids(line)
        assert re.search(r"(?:\[E[0-9A-F]{10,64}\])+。$", line)


def test_numeric_rounding_is_rejected_even_when_decimal_value_is_close():
    prompt = _prompt("financial_qa")
    target = build_strict_target("financial_qa", prompt).replace("40.0000%", "40.00%", 1)
    errors = strict_target_errors("financial_qa", prompt, target)
    assert any("numeric_lexeme_not_copied:40.00%" in error for error in errors)


def test_quant_artifact_consumes_external_data_without_synthesis_or_download():
    assert _quant_artifact_errors(QUANT_CODE) == []
    assert "np.random" not in QUANT_CODE
    assert "date_range" not in QUANT_CODE
    assert "read_csv" not in QUANT_CODE
    assert "download" not in QUANT_CODE


def test_quant_artifact_gate_rejects_random_data():
    bad = QUANT_CODE.replace("out = df.copy()", "out = df.copy()\n    out['x'] = np.random.rand(len(out))")
    assert any(error.startswith("quant_forbidden") for error in _quant_artifact_errors(bad))


def test_adversarial_builder_creates_balanced_answer_free_cases():
    rows = []
    for index, task_type in enumerate(sorted(TARGET_TASKS)):
        prompt = _prompt(task_type)
        rows.append(
            {
                "id": f"source-{task_type}",
                "group_id": f"g-{task_type}",
                "source_group": f"heldout-source-{index}",
                "task_type": task_type,
                "conversations": [
                    {"from": "human", "value": json.dumps(prompt, ensure_ascii=False)},
                    {"from": "gpt", "value": "THIS MUST NOT ENTER ADVERSARIAL GOLD"},
                ],
            }
        )
    cases, rejected = build_adversarial_cases(rows)
    assert rejected == {}
    assert len(cases) == 300
    assert all(sum(case["task_type"] == task for case in cases) == 50 for task in TARGET_TASKS)
    assert len({case["question"] for case in cases if case["task_type"] == "quant_strategy"}) == 50
    assert all("answer" not in case and "conversations" not in case for case in cases)
    assert all("THIS MUST NOT ENTER" not in json.dumps(case, ensure_ascii=False) for case in cases)


def test_system_contract_names_all_targeted_failure_modes():
    for phrase in ("one atomic claim per line", "numeric lexeme", "never recompute", "unsupported", "never synthesize"):
        assert phrase in STRICT_SYSTEM_PROMPT
