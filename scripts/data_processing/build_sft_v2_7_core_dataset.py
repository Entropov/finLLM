#!/usr/bin/env python3
"""Build the task-scoped SFT v2.7 core dataset from audited v2.6 artifacts.

This iteration deliberately optimizes only financial_qa, quant_strategy, and
stock_analysis. Non-core tasks remain in prior artifacts for historical
analysis, but cannot enter training, Dev-Audit selection, or regression gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing.build_sft_v2_1_dataset import _assistant_text  # noqa: E402
from scripts.evaluation.eval_sft_v2_trusted import evidence_source_identities  # noqa: E402
from scripts.evaluation.sft_v2_7_scope import CORE_TASKS  # noqa: E402

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/sft_v2_6"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft_v2_7_core"
DEFAULT_DEV_INPUT = PROJECT_ROOT / "data/evaluation/sft_v2_6_dev_audit.json"
DEFAULT_TRUSTED_INPUT = PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"
DEFAULT_ADVERSARIAL_INPUT = PROJECT_ROOT / "data/evaluation/sft_v2_3_audit_adversarial.json"
DEFAULT_PREFERENCE_INPUT = PROJECT_ROOT / "data/rlhf/sft_v2_6_hard_negative_preference.json"
DEFAULT_DEV_OUTPUT = PROJECT_ROOT / "data/evaluation/sft_v2_7_core_dev_audit.json"
DEFAULT_TRUSTED_OUTPUT = PROJECT_ROOT / "data/evaluation/sft_v2_7_core_trusted_regression.json"
DEFAULT_ADVERSARIAL_OUTPUT = PROJECT_ROOT / "data/evaluation/sft_v2_7_core_adversarial_regression.json"
DEFAULT_PREFERENCE_OUTPUT = PROJECT_ROOT / "data/rlhf/sft_v2_7_core_hard_negative_preference.json"

INPUT_FILES = {
    "train": "fin_agentic_sft_v2_6_answer_train.json",
    "eval": "fin_agentic_sft_v2_6_answer_eval.json",
}
OUTPUT_FILES = {
    "train": "fin_agentic_sft_v2_7_core_answer_train.json",
    "eval": "fin_agentic_sft_v2_7_core_answer_eval.json",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected JSON object rows: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _filter_core(rows: list[dict[str, Any]], *, label: str) -> list[dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("task_type")) in CORE_TASKS]
    if not filtered:
        raise ValueError(f"{label} contains no core-task rows")
    non_core = {str(row.get("task_type")) for row in filtered} - CORE_TASKS
    if non_core:
        raise ValueError(f"{label} core filter failed: {sorted(non_core)}")
    return filtered


def _groups(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(group)
        for row in rows
        for group in [row.get("source_group"), *list(row.get("source_groups", []))]
        if group
    }


def _evidence_sources(rows: list[dict[str, Any]]) -> set[str]:
    return set().union(*(evidence_source_identities(row) for row in rows))


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["task_type"]) for row in rows).items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_train(rows: list[dict[str, Any]]) -> bool:
    return all(
        row.get("dataset_version") == "sft_v2.6"
        and bool(row.get("target_audit", {}).get("hard_gate_passed"))
        and not re.search(r"</?think>", _assistant_text(row), re.IGNORECASE)
        for row in rows
    )


def build(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    train = _filter_core(_load_rows(input_dir / INPUT_FILES["train"]), label="train")
    evaluation = _filter_core(_load_rows(input_dir / INPUT_FILES["eval"]), label="eval")
    dev = _filter_core(_load_rows(Path(args.dev_input)), label="Dev-Audit")
    trusted = _filter_core(_load_rows(Path(args.trusted_input)), label="trusted regression")
    adversarial = _filter_core(_load_rows(Path(args.adversarial_input)), label="adversarial regression")
    preference = _filter_core(_load_rows(Path(args.preference_input)), label="hard-negative preference")

    train_groups = _groups(train)
    regression_sources = _evidence_sources(trusted) | _evidence_sources(adversarial)
    dev_sources = _evidence_sources(dev)
    train_sources = _evidence_sources(train)
    train_counts = _counts(train)
    eval_counts = _counts(evaluation)
    dev_counts = _counts(dev)
    trusted_counts = _counts(trusted)
    adversarial_counts = _counts(adversarial)
    preference_counts = _counts(preference)
    gate = {
        "exact_core_task_set": set(train_counts) == CORE_TASKS,
        "task_balanced_train": len(set(train_counts.values())) == 1,
        "eval_contains_all_core_tasks": set(eval_counts) == CORE_TASKS,
        "dev_contains_all_core_tasks": set(dev_counts) == CORE_TASKS and min(dev_counts.values()) >= 8,
        "trusted_has_50_per_core_task": set(trusted_counts) == CORE_TASKS and min(trusted_counts.values()) >= 50,
        "adversarial_has_50_per_core_task": set(adversarial_counts) == CORE_TASKS and min(adversarial_counts.values()) >= 50,
        "preference_covers_train": len(preference) == len(train) and preference_counts == train_counts,
        "train_dev_source_disjoint": not (train_sources & dev_sources) and not (train_groups & _groups(dev)),
        "train_regression_source_disjoint": not (train_sources & regression_sources)
        and not (train_groups & (_groups(trusted) | _groups(adversarial))),
        "dev_regression_source_disjoint": not (dev_sources & regression_sources),
        "all_train_targets_audit_checked_and_no_think": _validate_train(train + evaluation),
    }
    report = {
        "schema_version": "sft_v2.7-core",
        "profile": "three_task_core_scope_from_audited_v2.6",
        "core_tasks": sorted(CORE_TASKS),
        "excluded_from_optimization": ["financial_report", "risk_assessment", "sentiment_analysis"],
        "source_artifacts": {
            "train": {"path": str((input_dir / INPUT_FILES["train"]).resolve()), "sha256": _sha256(input_dir / INPUT_FILES["train"])},
            "eval": {"path": str((input_dir / INPUT_FILES["eval"]).resolve()), "sha256": _sha256(input_dir / INPUT_FILES["eval"])},
            "dev": {"path": str(Path(args.dev_input).resolve()), "sha256": _sha256(Path(args.dev_input))},
            "trusted": {"path": str(Path(args.trusted_input).resolve()), "sha256": _sha256(Path(args.trusted_input))},
            "adversarial": {"path": str(Path(args.adversarial_input).resolve()), "sha256": _sha256(Path(args.adversarial_input))},
            "preference": {"path": str(Path(args.preference_input).resolve()), "sha256": _sha256(Path(args.preference_input))},
        },
        "components": {
            "train": {"samples": len(train), "by_task": train_counts, "max_target_chars": max(len(_assistant_text(row)) for row in train)},
            "eval": {"samples": len(evaluation), "by_task": eval_counts},
            "preference": {"samples": len(preference), "by_task": preference_counts},
            "dev_audit": {"samples": len(dev), "by_task": dev_counts},
            "trusted_regression": {"samples": len(trusted), "by_task": trusted_counts, "identity": "seen_regression_only"},
            "adversarial_regression": {"samples": len(adversarial), "by_task": adversarial_counts, "identity": "seen_regression_only"},
        },
        "release_gate": gate,
        "release_gate_passed": all(gate.values()),
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / OUTPUT_FILES["train"], train)
    _write_json(output_dir / OUTPUT_FILES["eval"], evaluation)
    _write_json(output_dir / "build_report.json", report)
    _write_json(Path(args.dev_output), dev)
    _write_json(Path(args.trusted_output), trusted)
    _write_json(Path(args.adversarial_output), adversarial)
    _write_json(Path(args.preference_output), preference)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["release_gate_passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SFT v2.7 three-task core dataset")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dev-input", default=str(DEFAULT_DEV_INPUT))
    parser.add_argument("--trusted-input", default=str(DEFAULT_TRUSTED_INPUT))
    parser.add_argument("--adversarial-input", default=str(DEFAULT_ADVERSARIAL_INPUT))
    parser.add_argument("--preference-input", default=str(DEFAULT_PREFERENCE_INPUT))
    parser.add_argument("--dev-output", default=str(DEFAULT_DEV_OUTPUT))
    parser.add_argument("--trusted-output", default=str(DEFAULT_TRUSTED_OUTPUT))
    parser.add_argument("--adversarial-output", default=str(DEFAULT_ADVERSARIAL_OUTPUT))
    parser.add_argument("--preference-output", default=str(DEFAULT_PREFERENCE_OUTPUT))
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
