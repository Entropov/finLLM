from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.task_aware_verifier_v2_7_core import validate_task_answer as validate_v27
from scripts.evaluation.task_aware_verifier_v2_8_2 import validate_task_answer as validate_v282


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _conditional_stock_row() -> tuple[dict, str]:
    rows = json.loads(
        (PROJECT_ROOT / "data/sft_v2_8_2_stock_contract/fin_agentic_sft_v2_8_2_core_answer_train.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(
        item
        for item in rows
        if item["task_type"] == "stock_analysis"
        and item["target_task_validator"]["conditional_trend_abstention_allowed"]
    )
    prompt = json.loads(next(turn["value"] for turn in row["conversations"] if turn["from"] == "human"))
    target = next(turn["value"] for turn in row["conversations"] if turn["from"] == "gpt")
    return {"task_type": "stock_analysis", "evidence": prompt["evidence"]}, target


def test_v2_8_2_conditional_trend_abstention_requires_missing_ma_and_citation() -> None:
    case, target = _conditional_stock_row()
    assert validate_v282(case, target)["hard_gate_passed"] is True
    assert validate_v27(case, target)["hard_gate_passed"] is False
    market_id = case["evidence"][0]["evidence_id"]
    assert validate_v282(case, target.replace(f"[{market_id}]", "", 1))["hard_gate_passed"] is False


def test_v2_8_2_repair_has_no_preflight_training_source_overlap() -> None:
    report = json.loads((PROJECT_ROOT / "data/sft_v2_8_2_stock_contract/build_report.json").read_text(encoding="utf-8"))
    assert report["stock_target_validator"] == {"validated": 279, "failed": 0}
    assert report["preflight_gold_source_overlap_with_train"] == 0
