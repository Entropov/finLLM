from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def test_v2_8_preference_is_validator_aligned() -> None:
    report = _load("data/rlhf/sft_v2_8_core_validator_preference_report.json")
    checked = _load("saves/eval_results/sft_v2_8_validator_preference_check.json")
    assert report["contract_gate_passed"] is True
    assert report["by_task"] == {"financial_qa": 201, "quant_strategy": 270, "stock_analysis": 166}
    assert checked["summary"]["chosen_accept_rate"] == 1.0
    assert checked["summary"]["rejected_false_accept_count"] == 0


def test_v2_8_diagnostic_retains_stock_recall_blocker() -> None:
    diagnostic = _load("saves/eval_results/sft_v2_8_validator_calibration/validator_confusion_matrix.json")
    stock = diagnostic["stock_analysis"]["adversarial"]
    assert stock["model_raw_correctness_at_8"] == 0.7
    assert stock["correct_rejected_candidate_rate"] == 0.9367
    assert stock["selector_e2e_at_8"] == 0.04
