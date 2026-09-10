#!/usr/bin/env python3
"""Summarize SFT v2.1 collection runs and their released trajectory dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    by_task: dict[str, dict[str, int]] = {}
    task_names = sorted({str(row.get("task_type", "")) for row in rows})
    for task in task_names:
        task_rows = [row for row in rows if str(row.get("task_type", "")) == task]
        by_task[task] = {
            "responses": len(task_rows),
            "successful": sum(bool(row.get("ok")) for row in task_rows),
            "failed": sum(not bool(row.get("ok")) for row in task_rows),
            "answer_guard_applied": sum(
                bool((row.get("audit_summary") or {}).get("answer_guard_applied")) for row in task_rows
            ),
        }
    failures = Counter(
        str(reason)
        for row in rows
        if not row.get("ok")
        for reason in (
            (row.get("audit_summary") or {}).get("hard_failures")
            or (row.get("audit_summary") or {}).get("validation_errors")
            or [row.get("error", "unknown")]
        )
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path) if path.is_file() else "",
        "responses": len(rows),
        "unique_request_ids": len({str(row.get("id", "")) for row in rows if row.get("id")}),
        "successful": sum(bool(row.get("ok")) for row in rows),
        "failed": sum(not bool(row.get("ok")) for row in rows),
        "by_task": by_task,
        "attempts": dict(sorted(Counter(str(row.get("attempt", "")) for row in rows).items())),
        "failure_reasons": dict(failures.most_common()),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    request_path = Path(args.requests)
    requests = json.loads(request_path.read_text(encoding="utf-8"))
    question_to_task = {str(row["question"]): str(row["task_type"]) for row in requests}
    response_paths = [Path(item) for item in args.responses]
    runs = [_run_summary(path) for path in response_paths]

    best_outcomes: dict[str, dict[str, Any]] = {}
    for path in response_paths:
        for row in _read_jsonl(path):
            identifier = str(row.get("id", ""))
            if not identifier:
                continue
            if identifier not in best_outcomes or row.get("ok"):
                best_outcomes[identifier] = row
    unique_by_task = {}
    for task in sorted({str(row["task_type"]) for row in requests}):
        task_rows = [row for row in best_outcomes.values() if str(row.get("task_type")) == task]
        unique_by_task[task] = {
            "requests_with_outcome": len(task_rows),
            "successful": sum(bool(row.get("ok")) for row in task_rows),
            "failed": sum(not bool(row.get("ok")) for row in task_rows),
        }

    trajectory_rows = []
    for pattern in args.trajectories:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            trajectory_rows.extend(_read_jsonl(path))
    matched_trajectories = [row for row in trajectory_rows if str(row.get("query", "")) in question_to_task]
    trajectory_by_task = {}
    for task in sorted(set(question_to_task.values())):
        task_rows = [row for row in matched_trajectories if str(row.get("task_type", "")) == task]
        trajectory_by_task[task] = {
            "attempts": len(task_rows),
            "hard_gate_passes": sum(
                bool((row.get("reward_breakdown") or {}).get("hard_gate_passed")) for row in task_rows
            ),
            "answer_guard_applied": sum(bool(row.get("answer_guard_applied")) for row in task_rows),
        }

    dataset_report_path = Path(args.dataset_report)
    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    frozen_paths = [Path(item) for item in args.frozen_files]
    return {
        "schema_version": "agentic_collection_summary.v2.1",
        "requests": {
            "path": str(request_path.resolve()),
            "sha256": _sha256(request_path),
            "count": len(requests),
            "by_task": dict(sorted(Counter(str(row["task_type"]) for row in requests).items())),
        },
        "runs": runs,
        "unique_request_outcomes": {
            "requests_with_outcome": len(best_outcomes),
            "successful": sum(bool(row.get("ok")) for row in best_outcomes.values()),
            "failed": sum(not bool(row.get("ok")) for row in best_outcomes.values()),
            "by_task": unique_by_task,
        },
        "matched_trajectory_attempts": {
            "count": len(matched_trajectories),
            "by_task": trajectory_by_task,
        },
        "released_dataset": {
            "report_path": str(dataset_report_path.resolve()),
            "report_sha256": _sha256(dataset_report_path),
            "release_gate_passed": bool(dataset_report.get("release_gate_passed")),
            "release_gate": dataset_report.get("release_gate", {}),
            "accepted_trajectories": (dataset_report.get("joint_build") or {}).get("accepted_trajectories", 0),
            "accepted_by_task": (dataset_report.get("joint_build") or {}).get("accepted_by_task", {}),
            "components": dataset_report.get("components", {}),
        },
        "frozen_artifacts": {
            str(path): _sha256(path) for path in frozen_paths if path.is_file()
        },
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Agentic SFT v2.1 collection")
    parser.add_argument("--requests", default="data/rag/v2_1_collection_requests.json")
    parser.add_argument(
        "--responses",
        nargs="+",
        default=[
            "data/rag/v2_1_collection_responses.jsonl",
            "data/rag/v2_1_retry_responses.jsonl",
            "data/rag/v2_1_guard_responses.jsonl",
        ],
    )
    parser.add_argument("--trajectories", nargs="+", default=["data/rag/trajectories/*.jsonl"])
    parser.add_argument("--dataset-report", default="data/sft_v2_1/build_report.json")
    parser.add_argument(
        "--frozen-files",
        nargs="+",
        default=[
            "data/evaluation/trusted_finance_v2.json",
            "data/evaluation/v2_split_manifest.json",
            "data/rag/v2_collection_requests.json",
        ],
    )
    parser.add_argument("--output", default="saves/eval_results/agentic_v2_1_collection_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(args)
    _write_atomic(Path(args.output), report)
    print(
        json.dumps(
            {
                "unique_request_outcomes": report["unique_request_outcomes"],
                "matched_trajectory_attempts": report["matched_trajectory_attempts"]["count"],
                "release_gate_passed": report["released_dataset"]["release_gate_passed"],
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["released_dataset"]["release_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
