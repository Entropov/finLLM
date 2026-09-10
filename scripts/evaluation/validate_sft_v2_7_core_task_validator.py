#!/usr/bin/env python3
"""Validate the v2.7-core evidence-anchor verifier on preference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.sft_v2_7_scope import CORE_TASKS  # noqa: E402
from scripts.evaluation.task_aware_verifier_v2_7_core import VALIDATOR_VERSION, validate_task_answer  # noqa: E402


MANIFEST_VERSION = "sft_v2.7_core_task_validator_preference_check.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visible_case(row: dict[str, Any]) -> dict[str, Any]:
    conversations = row.get("conversations") or []
    prompt_text = next(
        (str(item.get("value", "")) for item in conversations if item.get("from") in {"human", "user"}),
        "",
    )
    prompt = json.loads(prompt_text)
    task_type = str(row.get("task_type", ""))
    if task_type not in CORE_TASKS or prompt.get("task_type") != task_type:
        raise ValueError(f"invalid core preference row: {row.get('id')}")
    return {"id": row.get("id"), "task_type": task_type, "evidence": list(prompt.get("evidence", []))}


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for row in rows:
        case = _visible_case(row)
        chosen = validate_task_answer(case, str((row.get("chosen") or {}).get("value", "")))
        rejected = validate_task_answer(case, str((row.get("rejected") or {}).get("value", "")))
        details.append(
            {
                "id": str(row.get("id", "")),
                "task_type": case["task_type"],
                "negative_type": str(row.get("negative_type", "")),
                "chosen_hard_gate_passed": chosen["hard_gate_passed"],
                "rejected_hard_gate_passed": rejected["hard_gate_passed"],
                "chosen_failures": chosen["failures"],
                "chosen_missing_anchors": chosen["missing_anchors"],
                "rejected_failures": rejected["failures"],
                "rejected_missing_anchors": rejected["missing_anchors"],
            }
        )
    rejected_by_type = Counter(item["negative_type"] for item in details if item["rejected_hard_gate_passed"])
    chosen_failures = [item for item in details if not item["chosen_hard_gate_passed"]]
    rejected_false_accepts = [item for item in details if item["rejected_hard_gate_passed"]]
    summary = {
        "samples": len(details),
        "chosen_accept_rate": round(sum(item["chosen_hard_gate_passed"] for item in details) / len(details), 4),
        "rejected_false_accept_count": len(rejected_false_accepts),
        "rejected_false_accept_rate": round(len(rejected_false_accepts) / len(details), 4),
        "chosen_false_reject_count": len(chosen_failures),
        "rejected_false_accept_by_negative_type": dict(sorted(rejected_by_type.items())),
        "release_gate_passed": not chosen_failures and not rejected_false_accepts,
    }
    return {"summary": summary, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SFT v2.7 task verifier against hard-negative pairs")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data/rlhf/sft_v2_7_core_hard_negative_preference.json"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_task_validator_preference_check.json"),
    )
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("preference input must be a JSON list")
    report = validate(rows)
    output = {
        "manifest_version": MANIFEST_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "input": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        **report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({"output": str(output_path), **report["summary"]}, ensure_ascii=False), flush=True)
    return 0 if report["summary"]["release_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
