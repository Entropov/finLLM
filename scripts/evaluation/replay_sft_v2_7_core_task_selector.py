#!/usr/bin/env python3
"""Replay an evidence-anchor task-aware best-of-8 selector for SFT v2.7 core."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.replay_sft_v2_5_selector import (  # noqa: E402
    EXPECTED_CANDIDATES,
    _round,
    _sha256_file,
    audit_selector_view,
)
from scripts.evaluation.sft_v2_7_scope import CORE_TASKS  # noqa: E402
from scripts.evaluation.task_aware_verifier_v2_7_core import VALIDATOR_VERSION, validate_task_answer  # noqa: E402


MANIFEST_VERSION = "sft_v2.7_core_task_selector_replay.v1"
POLICY_VERSION = "audit_plus_evidence_anchor_best_of_8.v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _visible_case(case: dict[str, Any]) -> dict[str, Any]:
    task_type = str(case.get("task_type", ""))
    if task_type not in CORE_TASKS:
        raise ValueError(f"non-core case in task selector: {task_type}")
    return {
        "id": str(case.get("id", "")),
        "task_type": task_type,
        "request_as_of": str(case.get("request_as_of", "")),
        "evidence": list(case.get("evidence", [])),
    }


def select_candidate_index(
    case: dict[str, Any],
    answers: list[str],
    finish_reasons: list[str],
    scores: list[dict[str, Any]],
) -> tuple[int | None, list[dict[str, Any]]]:
    """Select with audit fields plus evidence anchors, never gold task fields."""
    if not (len(answers) == len(finish_reasons) == len(scores) == EXPECTED_CANDIDATES):
        raise ValueError("selector replay requires exactly 8 aligned candidates")
    validations = [validate_task_answer(case, answer) for answer in answers]
    eligible = []
    for index, score in enumerate(scores):
        audit = audit_selector_view(score)
        if (
            audit["audit_hard_gate_passed"]
            and not audit["protocol_errors"]
            and not audit["audit_failures"]
            and validations[index]["hard_gate_passed"]
        ):
            eligible.append(index)
    if not eligible:
        return None, validations
    return (
        max(
            eligible,
            key=lambda index: (
                validations[index]["anchor_coverage"],
                audit_selector_view(scores[index])["audit_score"],
                finish_reasons[index] != "length",
                -len(answers[index]),
                -index,
            ),
        ),
        validations,
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["selected"]]
    selected_passes = sum(row["selected_passed"] for row in rows)
    oracle_passes = sum(row["oracle_pass_at_8"] for row in rows)
    false_accepts = sum(row["selected"] and not row["selected_passed"] for row in rows)
    return {
        "samples": len(rows),
        "greedy": {
            "e2e_at_1": _round(mean(float(row["greedy_passed"]) for row in rows)),
            "audit_at_1": _round(mean(float(row["greedy_audit_passed"]) for row in rows)),
        },
        "oracle": {"e2e_at_8": _round(mean(float(row["oracle_pass_at_8"]) for row in rows))},
        "selector": {
            "e2e_at_8": _round(mean(float(row["selected_passed"]) for row in rows)),
            "selection_rate": _round(mean(float(row["selected"]) for row in rows)),
            "task_false_accept_count": false_accepts,
            "task_false_accept_rate_given_selection": _round(false_accepts / len(selected) if selected else 0.0),
            "task_rejection_count": len(rows) - len(selected),
            "oracle_e2e_recovery": _round(selected_passes / oracle_passes if oracle_passes else 0.0),
            "e2e_delta_vs_greedy": _round(
                mean(float(row["selected_passed"]) for row in rows)
                - mean(float(row["greedy_passed"]) for row in rows)
            ),
        },
    }


def replay(payload: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    details = (payload.get("evaluation") or {}).get("details") or []
    if len(details) != int(payload.get("completed_samples", -1)):
        raise ValueError("selector input is incomplete")
    case_by_id = {str(case.get("id", "")): _visible_case(case) for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("gold case IDs must be unique")
    rows = []
    for source in details:
        identifier = str(source.get("id", ""))
        case = case_by_id.get(identifier)
        if case is None:
            raise ValueError(f"missing gold case for prediction: {identifier}")
        answers = [str(item) for item in source.get("candidate_answers_at_8", [])]
        finish_reasons = [str(item) for item in source.get("candidate_finish_reasons_at_8", [])]
        scores = source.get("candidate_scores_at_8", [])
        if not all(isinstance(item, dict) for item in scores):
            raise ValueError(f"invalid candidate scores: {identifier}")
        selected_index, validations = select_candidate_index(case, answers, finish_reasons, scores)
        selected_score = scores[selected_index] if selected_index is not None else {}
        greedy_score = source.get("candidate_score_at_1") or {}
        rows.append(
            {
                "id": identifier,
                "task_type": case["task_type"],
                "source_group": str(source.get("source_group", "")),
                "selected": selected_index is not None,
                "selected_index": selected_index,
                "selected_task_validator": validations[selected_index] if selected_index is not None else None,
                "candidate_task_validations": validations,
                "selected_passed": bool(selected_score.get("passed")),
                "oracle_pass_at_8": any(bool(item.get("passed")) for item in scores),
                "greedy_passed": bool(greedy_score.get("passed")),
                "greedy_audit_passed": bool(greedy_score.get("audit_hard_gate_passed")),
            }
        )
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
    return {"summary": _summarize(rows), "by_task": {key: _summarize(value) for key, value in sorted(by_task.items())}, "details": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a task-aware SFT v2.7 core selector")
    parser.add_argument("--input", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path, gold_path, output_path = Path(args.input), Path(args.gold), Path(args.output)
    payload = _load_json(input_path)
    raw_cases = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("gold must be a JSON list")
    replayed = replay(payload, raw_cases)
    output = {
        "manifest_version": MANIFEST_VERSION,
        "policy_version": POLICY_VERSION,
        "task_validator_version": VALIDATOR_VERSION,
        "selection_features": ["audit_verifier_fields", "visible_task_type", "visible_evidence", "candidate_text"],
        "forbidden_selection_features": ["gold.scoring", "task_score", "primary_score", "passed", "gold scoring assertions"],
        "input": {"path": str(input_path.resolve()), "sha256": _sha256_file(input_path)},
        "gold": {"path": str(gold_path.resolve()), "sha256": _sha256_file(gold_path)},
        **replayed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({"output": str(output_path), **replayed["summary"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
