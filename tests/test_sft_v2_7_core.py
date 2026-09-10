"""Regression tests for the SFT v2.7 three-task optimization scope."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data_processing.build_sft_v2_7_core_dataset import CORE_TASKS, _filter_core
from scripts.evaluation.sft_v2_7_core_protocol import case_prompt, prompt_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v2_7_core_scope_is_exact() -> None:
    assert CORE_TASKS == {"financial_qa", "quant_strategy", "stock_analysis"}


def test_v2_7_core_filter_excludes_non_core_rows() -> None:
    rows = [
        {"task_type": "financial_qa"},
        {"task_type": "risk_assessment"},
        {"task_type": "quant_strategy"},
    ]
    assert [row["task_type"] for row in _filter_core(rows, label="unit")] == [
        "financial_qa",
        "quant_strategy",
    ]


def test_v2_7_core_build_report_has_three_balanced_tasks() -> None:
    report = json.loads((PROJECT_ROOT / "data/sft_v2_7_core/build_report.json").read_text(encoding="utf-8"))
    assert report["release_gate_passed"] is True
    assert report["core_tasks"] == sorted(CORE_TASKS)
    assert report["components"]["train"]["by_task"] == {
        "financial_qa": 270,
        "quant_strategy": 270,
        "stock_analysis": 270,
    }
    assert report["components"]["trusted_regression"]["by_task"] == {
        "financial_qa": 50,
        "quant_strategy": 50,
        "stock_analysis": 50,
    }


def test_v2_7_core_protocol_rejects_excluded_tasks() -> None:
    with pytest.raises(ValueError, match="outside the SFT v2.7 core scope"):
        case_prompt({"task_type": "sentiment_analysis"})


def test_v2_7_core_prompt_contract_records_scope() -> None:
    contract = prompt_contract()
    assert contract["contract_version"] == "sft_v2.7-core"
    assert contract["core_tasks"] == sorted(CORE_TASKS)
