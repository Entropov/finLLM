#!/usr/bin/env python3
"""Replay an audit-only selector over stored SFT v2.5 candidate generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MANIFEST_VERSION = "sft_v2.5_selector_replay.v1"
POLICY_VERSION = "audit_only_best_of_8.v1"
EXPECTED_CANDIDATES = 8


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(value, 4)


def audit_selector_view(score: dict[str, Any]) -> dict[str, Any]:
    """Expose only evidence-verifiable fields to the selection policy."""
    return {
        "audit_hard_gate_passed": bool(score.get("audit_hard_gate_passed")),
        "audit_score": float(score.get("audit_score", 0.0)),
        "protocol_errors": tuple(str(item) for item in score.get("protocol_errors", [])),
        "audit_failures": tuple(str(item) for item in score.get("audit_failures", [])),
    }


def select_candidate_index(
    answers: list[str],
    finish_reasons: list[str],
    scores: list[dict[str, Any]],
) -> int | None:
    """Choose an audit-valid candidate without task labels or gold scoring."""
    if not (len(answers) == len(finish_reasons) == len(scores) == EXPECTED_CANDIDATES):
        raise ValueError("selector replay requires exactly 8 aligned candidates")
    eligible = []
    for index, score in enumerate(scores):
        view = audit_selector_view(score)
        if view["audit_hard_gate_passed"] and not view["protocol_errors"] and not view["audit_failures"]:
            eligible.append(index)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda index: (
            audit_selector_view(scores[index])["audit_score"],
            finish_reasons[index] != "length",
            -len(answers[index]),
            -index,
        ),
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_rows = [row for row in rows if row["selected"]]
    oracle_passes = sum(row["oracle_pass_at_8"] for row in rows)
    selected_passes = sum(row["selected_passed"] for row in rows)
    selected_false_accepts = sum(row["selected"] and not row["selected_passed"] for row in rows)
    summary = {
        "samples": len(rows),
        "greedy": {
            "e2e_at_1": _round(mean(float(row["greedy_passed"]) for row in rows)),
            "audit_at_1": _round(mean(float(row["greedy_audit_passed"]) for row in rows)),
            "mean_primary_at_1": _round(mean(float(row["greedy_primary_score"]) for row in rows)),
        },
        "oracle": {
            "e2e_at_8": _round(mean(float(row["oracle_pass_at_8"]) for row in rows)),
            "audit_at_8": _round(mean(float(row["oracle_audit_at_8"]) for row in rows)),
            "mean_best_primary_at_8": _round(mean(float(row["oracle_best_primary_at_8"]) for row in rows)),
        },
        "selector": {
            "e2e_at_8": _round(mean(float(row["selected_passed"]) for row in rows)),
            "audit_safe_selection_rate": _round(mean(float(row["selected"]) for row in rows)),
            "mean_primary_with_reject_zero": _round(mean(float(row["selected_primary_score"]) for row in rows)),
            "no_safe_candidate_count": len(rows) - len(selected_rows),
            "selected_length_finish_count": sum(row["selected_finish_reason"] == "length" for row in rows),
            "task_false_accept_count": selected_false_accepts,
            "task_false_accept_rate_given_selection": _round(
                selected_false_accepts / len(selected_rows) if selected_rows else 0.0
            ),
            "mean_selected_index_1_based": _round(
                mean(float(row["selected_index"] + 1) for row in selected_rows) if selected_rows else 0.0
            ),
        },
    }
    summary["selector"]["oracle_e2e_recovery"] = _round(selected_passes / oracle_passes if oracle_passes else 0.0)
    summary["selector"]["e2e_gap_to_oracle"] = _round(
        summary["oracle"]["e2e_at_8"] - summary["selector"]["e2e_at_8"]
    )
    summary["selector"]["e2e_delta_vs_greedy"] = _round(
        summary["selector"]["e2e_at_8"] - summary["greedy"]["e2e_at_1"]
    )
    summary["selector"]["audit_delta_vs_greedy"] = _round(
        summary["selector"]["audit_safe_selection_rate"] - summary["greedy"]["audit_at_1"]
    )
    return summary


def replay(payload: dict[str, Any]) -> dict[str, Any]:
    evaluation = payload.get("evaluation") or {}
    source_rows = evaluation.get("details") or []
    if len(source_rows) != int(payload.get("completed_samples", -1)):
        raise ValueError("selector input is incomplete")
    rows = []
    for row in source_rows:
        answers = [str(item) for item in row.get("candidate_answers_at_8", [])]
        finish_reasons = [str(item) for item in row.get("candidate_finish_reasons_at_8", [])]
        scores = row.get("candidate_scores_at_8", [])
        if not all(isinstance(item, dict) for item in scores):
            raise ValueError(f"invalid candidate scores: {row.get('id')}")
        selected_index = select_candidate_index(answers, finish_reasons, scores)
        selected_score = scores[selected_index] if selected_index is not None else None
        greedy_score = row.get("candidate_score_at_1") or {}
        rows.append(
            {
                "id": str(row.get("id")),
                "task_type": str(row.get("task_type")),
                "source_group": str(row.get("source_group")),
                "selected": selected_index is not None,
                "selected_index": selected_index,
                "selected_finish_reason": finish_reasons[selected_index] if selected_index is not None else "rejected",
                "selected_answer_sha256": (
                    hashlib.sha256(answers[selected_index].encode()).hexdigest() if selected_index is not None else None
                ),
                "selected_verifier_view": (
                    audit_selector_view(scores[selected_index]) if selected_index is not None else None
                ),
                "selected_passed": bool(selected_score and selected_score.get("passed")),
                "selected_primary_score": float(selected_score.get("primary_score", 0.0)) if selected_score else 0.0,
                "oracle_pass_at_8": any(bool(item.get("passed")) for item in scores),
                "oracle_audit_at_8": any(bool(item.get("audit_hard_gate_passed")) for item in scores),
                "oracle_best_primary_at_8": max(float(item.get("primary_score", 0.0)) for item in scores),
                "greedy_passed": bool(greedy_score.get("passed")),
                "greedy_audit_passed": bool(greedy_score.get("audit_hard_gate_passed")),
                "greedy_primary_score": float(greedy_score.get("primary_score", 0.0)),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)
    return {
        "summary": _summarize_rows(rows),
        "by_task": {task: _summarize_rows(task_rows) for task, task_rows in sorted(grouped.items())},
        "details": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an audit-only best-of-8 selector")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = _load_json(input_path)
    replayed = replay(payload)
    output = {
        "manifest_version": MANIFEST_VERSION,
        "policy_version": POLICY_VERSION,
        "selection_features": [
            "audit_hard_gate_passed",
            "protocol_errors",
            "audit_failures",
            "audit_score",
            "finish_reason",
            "answer_length",
            "candidate_index",
        ],
        "forbidden_selection_features": ["task_score", "primary_score", "passed", "gold scoring assertions"],
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256_file(input_path),
            "manifest_version": payload.get("manifest_version"),
            "dataset_fingerprint": payload.get("dataset_fingerprint"),
            "adapter": payload.get("adapter"),
            "generation_settings": payload.get("generation_settings"),
        },
        "created_at_unix": int(time.time()),
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
