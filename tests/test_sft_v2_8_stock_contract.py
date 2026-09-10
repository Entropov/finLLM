from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluation.build_sft_v2_8_stock_contract_gold import build_cases
from scripts.evaluation.diagnose_sft_v2_8_validator import _confidence_diagnostic
from scripts.evaluation.eval_sft_v2_4_e2e import _sequence_logprob_summary
from scripts.evaluation.sft_v2_8_stock_contract import score_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _value(quote: str, field: str) -> str:
    return re.search(rf"{re.escape(field)}为([^，。；;\s]+)", quote).group(1)


def _complete_answer(case: dict) -> str:
    market, financial = case["evidence"]
    return "\n".join(
        [
            f"收盘价为{_value(market['exact_quote'], '收盘价')} [{market['evidence_id']}]。",
            f"MA5为{_value(market['exact_quote'], 'MA5')}，MA20为{_value(market['exact_quote'], 'MA20')}，呈现均线关系 [{market['evidence_id']}]。",
            f"年化波动率为{_value(market['exact_quote'], '年化波动率')} [{market['evidence_id']}]。",
            f"营业收入为{_value(financial['exact_quote'], '营业收入')} [{financial['evidence_id']}]。",
        ]
    )


def test_stock_contract_gold_is_source_disjoint_and_balanced() -> None:
    manifest = json.loads((PROJECT_ROOT / "data/evaluation/v2_split_manifest.json").read_text(encoding="utf-8"))
    seen = json.loads((PROJECT_ROOT / "data/evaluation/sft_v2_7_core_trusted_regression.json").read_text(encoding="utf-8"))
    cases, report = build_cases(manifest, issuers=25, seen_cases=seen)
    assert len(cases) == 50
    assert report["training_source_disjoint"] is True
    assert report["train_heldout_issuer_overlap"] == []
    assert report["seen_regression_source_disjoint"] is False
    assert report["evaluation_identity"] == "seen_contract_preflight_not_final_holdout"
    assert report["by_challenge"] == {"trusted_style": 25, "adversarial": 25}


def test_stock_contract_requires_trend_anchor_but_accepts_complete_answer() -> None:
    case = json.loads(
        (PROJECT_ROOT / "data/evaluation/sft_v2_8_stock_contract_seen_preflight.json").read_text(encoding="utf-8")
    )[0]
    answer = _complete_answer(case)
    assert score_answer(case, answer)["passed"] is True
    missing = score_answer(case, answer.replace("MA5", "短期指标").replace("MA20", "长期指标").replace("均线", "关系"))
    assert missing["passed"] is False
    assert "market_trend" in missing["task_details"]["missing_anchors"]


def test_sequence_logprob_summary_is_normalized_and_fails_closed() -> None:
    logprob = SimpleNamespace(logprob=-0.25)
    output = SimpleNamespace(token_ids=[1, 2], logprobs=[{1: logprob}, {2: logprob}])
    assert _sequence_logprob_summary(output) == {
        "token_count": 2,
        "sequence_logprob": -0.5,
        "sequence_normalized_logprob": -0.25,
        "sequence_normalized_nll": 0.25,
    }
    assert _sequence_logprob_summary(SimpleNamespace(token_ids=[1], logprobs=[{}])) is None


def test_confidence_diagnostic_uses_nll_without_calling_it_entropy() -> None:
    e2e = {
        "evaluation": {
            "details": [
                {"id": "correct", "candidate_score_at_1": {"passed": True, "audit_hard_gate_passed": True, "protocol_errors": [], "audit_failures": []}, "candidate_sequence_logprob_at_1": {"sequence_normalized_nll": 0.2}},
                {"id": "wrong", "candidate_score_at_1": {"passed": False, "audit_hard_gate_passed": True, "protocol_errors": [], "audit_failures": []}, "candidate_sequence_logprob_at_1": {"sequence_normalized_nll": 0.1}},
            ]
        }
    }
    selector = {"details": [{"id": "correct", "candidate_task_validations": [{"hard_gate_passed": True}]}, {"id": "wrong", "candidate_task_validations": [{"hard_gate_passed": False}]}]}
    report = _confidence_diagnostic(e2e, selector)
    assert report["status"] == "sequence_nll_available_entropy_unavailable"
    assert report["mean_token_entropy"] is None
    assert report["correct_vs_incorrect_entropy"] == {"correct_mean_nll": 0.2, "incorrect_mean_nll": 0.1}
