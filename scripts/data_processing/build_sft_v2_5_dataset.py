#!/usr/bin/env python3
"""Build SFT v2.5 short-target data and source-disjoint Dev-Audit cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing import build_sft_v2_4_dataset as v24  # noqa: E402
from scripts.data_processing.build_sft_v2_1_dataset import _answer_target_errors, _assistant_text  # noqa: E402
from scripts.evaluation.eval_sft_v2_trusted import evidence_source_identities  # noqa: E402
from scripts.rag.quant_protocol import (  # noqa: E402
    QUANT_ARTIFACT_VERSION,
    canonical_quant_action,
    materialize_quant_output,
    parse_quant_action,
    render_quant_artifact,
)


TARGET_TASKS = v24.TARGET_TASKS
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/sft_v2_4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft_v2_5"
DEFAULT_DEV_AUDIT = PROJECT_ROOT / "data/evaluation/sft_v2_5_dev_audit.json"
DEFAULT_PREFERENCE = PROJECT_ROOT / "data/rlhf/sft_v2_5_hard_negative_preference.json"
DEFAULT_TRUSTED = PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"
DEFAULT_ADVERSARIAL = PROJECT_ROOT / "data/evaluation/sft_v2_3_audit_adversarial.json"
INPUT_FILES = {
    "train": "fin_agentic_sft_v2_4_answer_train.json",
    "eval": "fin_agentic_sft_v2_4_answer_eval.json",
}
OUTPUT_FILES = {
    "train": "fin_agentic_sft_v2_5_answer_train.json",
    "eval": "fin_agentic_sft_v2_5_answer_eval.json",
}
REPAIR_SAMPLES_PER_TASK = 54

SYSTEM_PROMPT = (
    "You are an auditable financial analyst. Use only supplied evidence and output no chain-of-thought or <think> "
    "tags. Follow decision_basis and stop immediately after the required response. For non-quant tasks, output only "
    "short atomic claims; every claim must end with its Evidence IDs, and every number must be copied exactly from "
    "exact_quote. Abstain instead of recomputing, forecasting, or filling missing facts. For quant_strategy, output "
    "only the single-line quant.action.v2.5 JSON object specified by output_contract, with no markdown, code, facts, "
    "or commentary. A deterministic audited node renders the Python artifact. Never promise returns."
)

TASK_CONTRACTS = dict(v24.TASK_CONTRACTS)
TASK_CONTRACTS["quant_strategy"] = (
    "Return exactly this JSON object and then EOS: " + canonical_quant_action()
)
STRICT_CONTRACT = dict(v24.STRICT_CONTRACT)
STRICT_CONTRACT.update(
    {
        "reasoning_policy": "no_think_output_use_supplied_decision_basis",
        "length_policy": "short_fixed_sections_then_eos",
        "quant_policy": "quant.action.v2.5_only_then_deterministic_renderer",
    }
)

REPAIR_CONSTRAINTS = {
    "financial_qa": ["copy_exact_ratio_lexemes", "cite_each_line", "no_recalculation", "stop_after_boundary"],
    "financial_report": ["copy_only_reported_facts", "cite_each_line", "no_outlook", "stop_after_boundary"],
    "quant_strategy": ["exact_action_json", "no_python", "no_markdown", "eos_after_json"],
    "risk_assessment": ["require_both_evidence_groups", "bounded_risk_label", "cite_each_line", "no_extra_risks"],
    "sentiment_analysis": ["one_supported_label", "cite_each_line", "no_extra_explanation", "stop_after_label"],
    "stock_analysis": ["copy_exact_market_and_financial_facts", "cite_each_line", "no_price_forecast"],
}


def decision_basis(task_type: str, prompt: dict[str, Any]) -> dict[str, Any]:
    basis = v24.decision_basis(task_type, prompt)
    if task_type == "quant_strategy":
        basis["decision"] = "emit_quant_action"
        basis["checks"] = ["exact_action_schema", "deterministic_renderer", "no_model_generated_code"]
    basis["response_budget"] = "one_json_line" if task_type == "quant_strategy" else "short_atomic_lines"
    return basis


def _answer_prompt(task_type: str, prompt: dict[str, Any]) -> dict[str, Any]:
    """Drop irrelevant market evidence from financial-only response targets."""
    output = copy.deepcopy(prompt)
    if task_type in {"financial_qa", "financial_report", "sentiment_analysis"}:
        grouped = v24._evidence_by_kind(output)
        output["evidence"] = grouped.get("financial", [])[:1]
    return output


def build_target(task_type: str, prompt: dict[str, Any]) -> str:
    if task_type == "quant_strategy":
        return canonical_quant_action()
    return v24.build_target(task_type, _answer_prompt(task_type, prompt))


def rendered_answer(task_type: str, prompt: dict[str, Any], target: str) -> str:
    if task_type != "quant_strategy":
        return target
    rendered, errors = materialize_quant_output(target, prompt.get("evidence", []))
    if errors or rendered is None:
        raise ValueError(f"invalid canonical quant action: {errors}")
    return rendered


def rendered_quant_errors(prompt: dict[str, Any], answer: str) -> list[str]:
    errors = []
    expected = render_quant_artifact(prompt.get("evidence", []))
    if answer != expected:
        errors.append("quant_render_not_canonical")
    reward = v24._audit_target("quant_strategy", prompt, answer)
    if not reward["hard_gate_passed"]:
        errors.extend(f"audit:{item}" for item in reward["hard_failures"])
    for component in ("numeric_consistency", "citation_coverage", "citation_precision"):
        if reward[component] != 1.0:
            errors.append(f"strict_{component}:{reward[component]}")
    return sorted(set(errors))


def target_errors(task_type: str, prompt: dict[str, Any], target: str) -> list[str]:
    if task_type == "quant_strategy":
        return parse_quant_action(target)[1]
    return v24.target_errors(task_type, prompt, target)


def _prompt(row: dict[str, Any]) -> dict[str, Any]:
    return v24._prompt(row)


def _convert(
    row: dict[str, Any],
    *,
    prompt_override: dict[str, Any] | None = None,
    suffix: str = "",
) -> dict[str, Any]:
    task_type = str(row["task_type"])
    prompt = copy.deepcopy(prompt_override if prompt_override is not None else _prompt(row))
    prompt["decision_basis"] = decision_basis(task_type, prompt)
    prompt.pop("claim_plan", None)
    prompt["strict_audit_contract"] = STRICT_CONTRACT
    prompt["output_contract"] = TASK_CONTRACTS[task_type]
    target = build_target(task_type, prompt)
    errors = target_errors(task_type, prompt, target)
    if errors:
        raise ValueError(f"{row.get('id')}:{','.join(errors)}")
    final_answer = rendered_answer(task_type, prompt, target)
    if task_type == "quant_strategy":
        render_errors = rendered_quant_errors(prompt, final_answer)
        if render_errors:
            raise ValueError(f"{row.get('id')}:render:{','.join(render_errors)}")

    output = copy.deepcopy(row)
    output["id"] = f"{row['id']}{suffix}"
    output["system"] = SYSTEM_PROMPT
    output["conversations"][0]["value"] = v24.canonical_json(prompt)
    output["conversations"][1]["value"] = target
    output["dataset_version"] = "sft_v2.5"
    output["repair_profile"] = "short_eos_claim_aligned_quant_action"
    output["canonical_target_sha256"] = hashlib.sha256(target.encode()).hexdigest()
    reward = v24._audit_target(task_type, prompt, final_answer)
    output["target_audit"] = {
        "evaluated_answer": "deterministic_render" if task_type == "quant_strategy" else "model_target",
        "hard_gate_passed": reward["hard_gate_passed"],
        "hard_failures": reward["hard_failures"],
        "claim_support": reward["claim_support"],
        "numeric_consistency": reward["numeric_consistency"],
        "citation_coverage": reward["citation_coverage"],
        "citation_precision": reward["citation_precision"],
        "task_validity": reward["task_validity"],
    }
    if task_type != "quant_strategy":
        schema_errors = _answer_target_errors(output)
        if schema_errors:
            raise ValueError(f"{row.get('id')}:{','.join(schema_errors)}")
    return output


def _repair_variant(row: dict[str, Any], index: int) -> dict[str, Any]:
    task_type = str(row["task_type"])
    prompt = copy.deepcopy(_prompt(row))
    prompt["repair_constraints"] = REPAIR_CONSTRAINTS[task_type]
    prompt["observed_failure_class"] = (
        "long_or_invalid_quant_generation"
        if task_type == "quant_strategy"
        else "unsupported_or_uncited_or_numeric_drift"
    )
    output = _convert(row, prompt_override=prompt, suffix=f":v2.5-repair-{index + 1}")
    output["challenge_profile"] = "observed_failure_repair_without_test_content"
    output["repair_source"] = "v2.4_aggregate_failure_taxonomy"
    return output


def _balance(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    target = max(len(grouped[task]) for task in TARGET_TASKS)
    output = list(rows)
    added = {}
    for task in sorted(TARGET_TASKS):
        items = sorted(grouped[task], key=lambda item: str(item["id"]))
        missing = target - len(items)
        added[task] = missing
        for index in range(missing):
            replica = copy.deepcopy(items[index % len(items)])
            replica["replica_of"] = str(items[index % len(items)]["id"])
            replica["id"] = f"{replica['id']}:v2.5-balance-{index + 1}"
            replica["oversampled"] = True
            output.append(replica)
    output.sort(key=lambda item: (str(item["task_type"]), str(item["id"])))
    return output, added


def make_hard_negative(task_type: str, target: str, seed: str) -> tuple[str, str]:
    if task_type != "quant_strategy":
        return v24.make_hard_negative(task_type, target, seed)
    variants = (
        ('{"action":"render_quant_artifact","artifact_version":"quant.v2.5"}', "missing_template"),
        (canonical_quant_action() + "\n附加说明", "trailing_prose"),
        ('{"action":"generate_python","artifact_version":"quant.v2.5","template_id":"ma5_ma20_long_only"}', "wrong_action"),
        ("```python\ndef quant_artifact(df):\n    return df\n```", "model_generated_code"),
    )
    index = hashlib.sha256(seed.encode()).digest()[0] % len(variants)
    return variants[index]


def build_preferences(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = []
    negative_types = Counter()
    errors_by_type = Counter()
    for row in rows:
        task_type = str(row["task_type"])
        prompt = _prompt(row)
        chosen = _assistant_text(row)
        rejected, negative_type = make_hard_negative(task_type, chosen, str(row["id"]))
        errors = target_errors(task_type, prompt, rejected)
        if not errors:
            raise ValueError(f"hard negative unexpectedly passed: {row['id']}:{negative_type}")
        negative_types[negative_type] += 1
        errors_by_type.update(errors)
        pairs.append(
            {
                "id": f"{row['id']}:hard-negative",
                "task_type": task_type,
                "system": row["system"],
                "conversations": [copy.deepcopy(row["conversations"][0])],
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": rejected},
                "negative_type": negative_type,
                "negative_contract_errors": errors,
                "source_group": row.get("source_group", ""),
            }
        )
    return pairs, {
        "samples": len(pairs),
        "by_task": dict(sorted(Counter(str(row["task_type"]) for row in pairs).items())),
        "negative_types": dict(sorted(negative_types.items())),
        "negative_contract_errors": dict(errors_by_type.most_common()),
    }


def _prompt_sources(rows: list[dict[str, Any]]) -> set[str]:
    return v24._prompt_sources(rows)


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = v24._stats(rows)
    stats["quant_action_targets"] = sum(
        row["task_type"] == "quant_strategy" and _assistant_text(row) == canonical_quant_action() for row in rows
    )
    stats["repair_samples"] = sum(
        row.get("challenge_profile") == "observed_failure_repair_without_test_content" for row in rows
    )
    return stats


def build_all(
    train_source: list[dict[str, Any]],
    eval_source: list[dict[str, Any]],
    *,
    trusted_path: Path,
    adversarial_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    converted_train = [_convert(row) for row in train_source]
    converted_eval = [_convert(row) for row in eval_source]
    for task in sorted(TARGET_TASKS):
        candidates = [row for row in train_source if row["task_type"] == task and not row.get("oversampled")]
        candidates.sort(key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())
        converted_train.extend(
            _repair_variant(row, index) for index, row in enumerate(candidates[:REPAIR_SAMPLES_PER_TASK])
        )
    train_rows, balance_added = _balance(converted_train)
    preferences, preference_stats = build_preferences(train_rows)

    dev_audit = v24.build_dev_audit(eval_source)
    for case in dev_audit:
        case["id"] = str(case["id"]).replace("dev-audit-v2.4-", "dev-audit-v2.5-")
        case["challenge_tags"] = [
            "dev_audit",
            "claim_evidence",
            "numeric_exact",
            "short_eos",
            "quant_action_renderer",
        ]

    train_sources = _prompt_sources(train_rows)
    dev_sources = set().union(*(evidence_source_identities(case) for case in dev_audit))
    trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
    adversarial = json.loads(adversarial_path.read_text(encoding="utf-8"))
    trusted_sources = set().union(*(evidence_source_identities(case) for case in trusted))
    adversarial_sources = set().union(*(evidence_source_identities(case) for case in adversarial))
    dev_counts = Counter(str(case["task_type"]) for case in dev_audit)
    train_stats, eval_stats = _stats(train_rows), _stats(converted_eval)
    frozen_paths = (
        trusted_path,
        adversarial_path,
        PROJECT_ROOT / "data/evaluation/v2_split_manifest.json",
        PROJECT_ROOT / "saves/eval_results/sft_v2_4_trusted_e2e.json",
        PROJECT_ROOT / "saves/eval_results/sft_v2_4_adversarial_e2e.json",
    )
    frozen = {
        str(path.relative_to(PROJECT_ROOT)): v24._sha256_file(path) if path.is_file() else ""
        for path in frozen_paths
    }
    gate = {
        "task_balanced_train": len(set(train_stats["by_task"].values())) == 1,
        "minimum_dev_per_task": all(dev_counts[task] >= 8 for task in TARGET_TASKS),
        "train_dev_source_disjoint": not (train_sources & dev_sources),
        "dev_regression_source_disjoint": not (dev_sources & trusted_sources) and not (dev_sources & adversarial_sources),
        "train_regression_source_disjoint": not (train_sources & trusted_sources) and not (train_sources & adversarial_sources),
        "all_targets_no_think": train_stats["thinking_targets"] == 0 and eval_stats["thinking_targets"] == 0,
        "all_rendered_targets_audit_pass": train_stats["audit_hard_pass_rate"] == 1.0 and eval_stats["audit_hard_pass_rate"] == 1.0,
        "all_targets_numeric_exact": train_stats["numeric_exact_rate"] == 1.0 and eval_stats["numeric_exact_rate"] == 1.0,
        "all_targets_citation_complete": train_stats["citation_coverage_rate"] == 1.0 and eval_stats["citation_coverage_rate"] == 1.0,
        "all_quant_targets_are_actions": train_stats["quant_action_targets"] == train_stats["by_task"]["quant_strategy"],
        "short_model_targets": train_stats["max_target_chars"] <= 900,
        "repair_taxonomy_covers_all_tasks": all(
            any(row["task_type"] == task and row.get("repair_source") for row in train_rows)
            for task in TARGET_TASKS
        ),
        "hard_negatives_cover_all_train_samples": len(preferences) == len(train_rows),
        "hard_negatives_are_rejected": all(row["negative_contract_errors"] for row in preferences),
        "frozen_artifacts_present": all(frozen.values()),
    }
    report = {
        "schema_version": "sft_v2.5",
        "profile": "short_eos_task_balanced_quant_action_renderer",
        "quant_artifact_version": QUANT_ARTIFACT_VERSION,
        "source_files": INPUT_FILES,
        "components": {"train": train_stats, "loss_eval": eval_stats},
        "balance_added": balance_added,
        "preference": preference_stats,
        "dev_audit": {
            "samples": len(dev_audit),
            "by_task": dict(sorted(dev_counts.items())),
            "source_groups": len({str(case["source_group"]) for case in dev_audit}),
            "train_source_overlap": len(train_sources & dev_sources),
            "trusted_source_overlap": len(trusted_sources & dev_sources),
            "adversarial_source_overlap": len(adversarial_sources & dev_sources),
        },
        "evaluation_status": {
            "dev_audit": "source_disjoint_checkpoint_selection",
            "trusted_finance_v2": "regression_only_seen_during_v2.5_design",
            "sft_v2_3_audit_adversarial": "regression_only_seen_during_v2.5_design",
            "untouched_final_holdout": "missing_requires_new_independent_gold",
        },
        "frozen_artifacts": frozen,
        "release_gate": gate,
        "release_gate_passed": all(gate.values()),
    }
    return {"train": train_rows, "eval": converted_eval}, preferences, dev_audit, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT v2.5 short-target and quant-action data")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preference-output", default=str(DEFAULT_PREFERENCE))
    parser.add_argument("--dev-audit-output", default=str(DEFAULT_DEV_AUDIT))
    parser.add_argument("--trusted", default=str(DEFAULT_TRUSTED))
    parser.add_argument("--adversarial", default=str(DEFAULT_ADVERSARIAL))
    parser.add_argument("--allow-failed-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    datasets, preferences, dev_audit, report = build_all(
        json.loads((input_dir / INPUT_FILES["train"]).read_text(encoding="utf-8")),
        json.loads((input_dir / INPUT_FILES["eval"]).read_text(encoding="utf-8")),
        trusted_path=Path(args.trusted),
        adversarial_path=Path(args.adversarial),
    )
    output_dir = Path(args.output_dir)
    for split, filename in OUTPUT_FILES.items():
        v24._write_json_atomic(output_dir / filename, datasets[split])
    v24._write_json_atomic(Path(args.preference_output), preferences)
    v24._write_json_atomic(Path(args.dev_audit_output), dev_audit)
    v24._write_json_atomic(output_dir / "build_report.json", report)
    print(
        json.dumps(
            {
                "release_gate_passed": report["release_gate_passed"],
                **report["components"],
                "preference": report["preference"],
                "dev_audit": report["dev_audit"],
            },
            ensure_ascii=False,
        )
    )
    if not report["release_gate_passed"] and not args.allow_failed_gate:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
