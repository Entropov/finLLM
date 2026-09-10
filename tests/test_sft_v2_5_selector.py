from __future__ import annotations

from scripts.evaluation.replay_sft_v2_5_selector import replay, select_candidate_index


def _score(*, audit: bool, audit_score: float, passed: bool, primary: float = 0.0) -> dict:
    return {
        "audit_hard_gate_passed": audit,
        "audit_score": audit_score,
        "audit_failures": [] if audit else ["unsupported_claim"],
        "protocol_errors": [],
        "passed": passed,
        "primary_score": primary,
        "task_score": 1.0 if passed else 0.0,
    }


def test_selector_does_not_optimize_hidden_task_outcome() -> None:
    answers = [f"answer-{index}" for index in range(8)]
    reasons = ["stop"] * 8
    scores = [_score(audit=False, audit_score=0.0, passed=False) for _ in range(8)]
    scores[0] = _score(audit=True, audit_score=0.9, passed=False, primary=0.3)
    scores[1] = _score(audit=True, audit_score=0.8, passed=True, primary=1.0)

    assert select_candidate_index(answers, reasons, scores) == 0


def test_selector_rejects_when_no_audit_valid_candidate_exists() -> None:
    answers = [f"answer-{index}" for index in range(8)]
    reasons = ["stop"] * 8
    scores = [_score(audit=False, audit_score=0.0, passed=False) for _ in range(8)]

    assert select_candidate_index(answers, reasons, scores) is None


def test_replay_reports_oracle_recovery_and_false_acceptance() -> None:
    scores = [_score(audit=False, audit_score=0.0, passed=False) for _ in range(8)]
    scores[0] = _score(audit=True, audit_score=0.9, passed=False, primary=0.4)
    scores[1] = _score(audit=True, audit_score=0.8, passed=True, primary=0.9)
    payload = {
        "completed_samples": 1,
        "evaluation": {
            "details": [
                {
                    "id": "case-1",
                    "task_type": "financial_qa",
                    "source_group": "group-1",
                    "candidate_answers_at_8": [f"answer-{index}" for index in range(8)],
                    "candidate_finish_reasons_at_8": ["stop"] * 8,
                    "candidate_scores_at_8": scores,
                    "candidate_score_at_1": _score(audit=False, audit_score=0.0, passed=False),
                }
            ]
        },
    }

    result = replay(payload)["summary"]
    assert result["oracle"]["e2e_at_8"] == 1.0
    assert result["selector"]["e2e_at_8"] == 0.0
    assert result["selector"]["oracle_e2e_recovery"] == 0.0
    assert result["selector"]["task_false_accept_rate_given_selection"] == 1.0
