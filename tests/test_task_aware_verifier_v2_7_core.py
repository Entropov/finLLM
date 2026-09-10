from __future__ import annotations

from scripts.evaluation.replay_sft_v2_7_core_task_selector import select_candidate_index
from scripts.evaluation.task_aware_verifier_v2_7_core import validate_task_answer


QA_CASE = {
    "task_type": "financial_qa",
    "evidence": [
        {
            "evidence_id": "E11111111111",
            "exact_quote": "资产负债率为74.4834%；流动比率为1.3356。",
        }
    ],
}

STOCK_CASE = {
    "task_type": "stock_analysis",
    "evidence": [
        {
            "evidence_id": "E22222222222",
            "exact_quote": "收盘价为80.4000元，近20个交易日收益率为-11.9483%，年化波动率为46.3658%，最大回撤为-15.6214%，MA5为81.6720元，MA20为84.0260元。",
        },
        {
            "evidence_id": "E33333333333",
            "exact_quote": "营业收入为24.0145亿元，归母口径净利润为27.5080亿元，经营活动现金流量净额为13.4043亿元。",
        },
    ],
}


def _score(audit: bool, audit_score: float) -> dict:
    return {
        "audit_hard_gate_passed": audit,
        "audit_score": audit_score,
        "audit_failures": [] if audit else ["unsupported_claim"],
        "protocol_errors": [],
    }


def test_financial_qa_requires_both_visible_ratio_anchors() -> None:
    answer = "\n".join(
        [
            "- 资产负债率为74.4834% [E11111111111]。",
            "- 流动比率为1.3356 [E11111111111]。",
            "- 证据不足，无法确认未来价格方向 [E11111111111]。",
        ]
    )
    assert validate_task_answer(QA_CASE, answer)["hard_gate_passed"] is True
    missing = validate_task_answer(QA_CASE, answer.replace("流动比率为1.3356", "流动比率为1.3000"))
    assert missing["hard_gate_passed"] is False
    assert "current_ratio" in missing["missing_anchors"]


def test_stock_validator_rejects_audit_looking_answer_without_trend_or_boundary() -> None:
    incomplete = "\n".join(
        [
            "- 收盘价为80.4000元 [E22222222222]。",
            "- 年化波动率为46.3658% [E22222222222]。",
            "- 营业收入为24.0145亿元 [E33333333333]。",
        ]
    )
    verdict = validate_task_answer(STOCK_CASE, incomplete)
    assert verdict["hard_gate_passed"] is False
    assert "market_trend" in verdict["missing_anchors"]


def test_validator_rejects_uncited_atomic_claims() -> None:
    answer = "\n".join(
        [
            "- 资产负债率为74.4834%。",
            "- 流动比率为1.3356 [E11111111111]。",
        ]
    )
    verdict = validate_task_answer(QA_CASE, answer)
    assert verdict["hard_gate_passed"] is False
    assert "line_1:missing_citation" in verdict["failures"]


def test_task_selector_prefers_task_complete_candidate_without_gold_scores() -> None:
    complete = "\n".join(
        [
            "- 收盘价为80.4000元 [E22222222222]。",
            "- MA5为81.6720元 [E22222222222]。",
            "- 最大回撤为-15.6214% [E22222222222]。",
            "- 归母口径净利润为27.5080亿元 [E33333333333]。",
            "- 证据不足，无法确认未来价格方向 [E22222222222]。",
        ]
    )
    incomplete = "- 收盘价为80.4000元 [E22222222222]。"
    answers = [incomplete, complete, *["" for _ in range(6)]]
    scores = [_score(True, 0.99), _score(True, 0.70), *[_score(False, 0.0) for _ in range(6)]]
    index, validations = select_candidate_index(STOCK_CASE, answers, ["stop"] * 8, scores)
    assert index == 1
    assert validations[0]["hard_gate_passed"] is False


def test_validator_rejects_nonempty_thinking_even_when_anchors_are_present() -> None:
    answer = "\n".join(
        [
            "<think>hidden chain</think>",
            "- 资产负债率为74.4834% [E11111111111]。",
            "- 流动比率为1.3356 [E11111111111]。",
        ]
    )
    verdict = validate_task_answer(QA_CASE, answer)
    assert verdict["hard_gate_passed"] is False
    assert "nonempty_thinking" in verdict["failures"]
