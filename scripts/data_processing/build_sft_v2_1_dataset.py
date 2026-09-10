#!/usr/bin/env python3
"""Build source-disjoint answer and policy datasets for SFT v2.1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing.build_sft_v2_dataset import (  # noqa: E402
    POLICY_NODES,
    TARGET_TASKS,
    _read_jsonl,
    build_dataset,
)
from scripts.evaluation.generate_sft_v2_trusted_predictions import (  # noqa: E402
    SYSTEM_PROMPT as ANSWER_SYSTEM_PROMPT,
    TASK_CONTRACTS,
    _claim_plan,
)
from scripts.rag.audit_schema import (  # noqa: E402
    build_claims_from_answer,
    canonical_json,
    strip_thinking,
)
from scripts.rag.reward_v2 import _numeric_tokens  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft_v2_1"
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
ABSTENTION_TERMS = ("无法确认", "资料不足", "证据不足", "未检索到", "需要补充")
COMPONENT_FILES = {
    "answer_train": "fin_agentic_sft_v2_1_answer_train.json",
    "answer_eval": "fin_agentic_sft_v2_1_answer_eval.json",
    "policy_train": "fin_agentic_sft_v2_1_policy_train.json",
    "policy_eval": "fin_agentic_sft_v2_1_policy_eval.json",
}


def _assistant_text(sample: dict[str, Any]) -> str:
    for turn in reversed(sample.get("conversations", [])):
        if turn.get("from") in {"gpt", "assistant"}:
            return str(turn.get("value", ""))
    return ""


def _normalize_answer_sample(sample: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(sample)
    task_type = str(normalized["task_type"])
    conversations = normalized.get("conversations", [])
    if len(conversations) != 2:
        raise ValueError(f"answer sample must have exactly two turns: {normalized.get('id')}")
    prompt = json.loads(str(conversations[0]["value"]))
    prompt["claim_plan"] = _claim_plan(task_type)
    prompt["output_contract"] = TASK_CONTRACTS[task_type]
    conversations[0]["value"] = canonical_json(prompt)
    if task_type == "sentiment_analysis":
        conversations[1]["value"] = strip_thinking(str(conversations[1]["value"]))
    normalized["system"] = ANSWER_SYSTEM_PROMPT
    normalized["training_component"] = "answer"
    normalized["dataset_version"] = "sft_v2.1"
    return normalized


def _normalize_policy_sample(sample: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(sample)
    normalized["training_component"] = "policy"
    normalized["dataset_version"] = "sft_v2.1"
    return normalized


def _answer_target_errors(sample: dict[str, Any]) -> list[str]:
    errors = []
    target = _assistant_text(sample)
    if CYRILLIC_RE.search(target):
        errors.append("cyrillic_target")
    if target.lower().count("<think>") != target.lower().count("</think>"):
        errors.append("unbalanced_thinking_tags")
    final_answer = strip_thinking(target)
    if not final_answer:
        errors.append("empty_final_answer")
        return errors
    try:
        prompt = json.loads(str(sample["conversations"][0]["value"]))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        errors.append("invalid_answer_prompt")
        return errors
    evidence_by_id = {
        str(item.get("evidence_id", "")).upper(): str(item.get("exact_quote", ""))
        for item in prompt.get("evidence", [])
        if item.get("evidence_id")
    }
    claims = build_claims_from_answer(final_answer, str(prompt.get("request_as_of", "")))
    if not claims and not any(term in final_answer for term in ABSTENTION_TERMS):
        errors.append("no_parseable_claims")
    for claim in claims:
        if any(term in claim.statement for term in ABSTENTION_TERMS):
            continue
        cited_ids = set(claim.supporting_evidence_ids)
        if not cited_ids:
            errors.append("claim_missing_citation")
            continue
        if not cited_ids <= set(evidence_by_id):
            errors.append("claim_unknown_citation")
            continue
        numbers = _numeric_tokens(claim.statement)
        cited_numbers = _numeric_tokens(" ".join(evidence_by_id[item] for item in cited_ids))
        if not numbers <= cited_numbers:
            errors.append("claim_numeric_mismatch")
    return sorted(set(errors))


def _sample_fingerprint(rows: list[dict[str, Any]]) -> str:
    stable = [
        {
            "id": row.get("id"),
            "group_id": row.get("group_id"),
            "task_type": row.get("task_type"),
            "sample_kind": row.get("sample_kind"),
            "policy_node": row.get("policy_node"),
            "prompt_sha256": hashlib.sha256(
                str((row.get("conversations") or [{}])[0].get("value", "")).encode("utf-8")
            ).hexdigest(),
            "target_sha256": hashlib.sha256(_assistant_text(row).encode("utf-8")).hexdigest(),
        }
        for row in sorted(rows, key=lambda item: str(item.get("id", "")))
    ]
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()


def _component_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "groups": len({str(row.get("group_id", "")) for row in rows}),
        "by_task": dict(sorted(Counter(str(row.get("task_type", "")) for row in rows).items())),
        "by_policy_node": dict(
            sorted(Counter(str(row.get("policy_node", "")) for row in rows if row.get("policy_node")).items())
        ),
        "cyrillic_targets": sum(bool(CYRILLIC_RE.search(_assistant_text(row))) for row in rows),
        "fingerprint": _sample_fingerprint(rows),
    }


def build_component_datasets(
    payloads: Iterable[dict[str, Any]],
    *,
    min_reward: float,
    eval_ratio: float,
    seed: int,
    min_per_task: int,
    min_answer_train_per_task: int,
    min_answer_eval_per_task: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    joint_train, joint_eval, joint_report = build_dataset(
        payloads,
        min_reward=min_reward,
        eval_ratio=eval_ratio,
        seed=seed,
        min_per_task=min_per_task,
    )
    unknown_kinds = Counter(
        str(row.get("sample_kind", ""))
        for row in joint_train + joint_eval
        if row.get("sample_kind") not in {"auditable_answer", "agent_policy"}
    )
    datasets: dict[str, list[dict[str, Any]]] = {}
    target_rejections: dict[str, dict[str, Any]] = {}
    for split, joint_rows in (("train", joint_train), ("eval", joint_eval)):
        answer_rows = [
            _normalize_answer_sample(row)
            for row in joint_rows
            if row.get("sample_kind") == "auditable_answer"
        ]
        rejection_reasons = Counter()
        valid_answers = []
        for row in answer_rows:
            errors = _answer_target_errors(row)
            if errors:
                rejection_reasons.update(errors)
            else:
                valid_answers.append(row)
        eligible_groups = {str(row["group_id"]) for row in valid_answers}
        policy_rows = [
            _normalize_policy_sample(row)
            for row in joint_rows
            if row.get("sample_kind") == "agent_policy" and str(row["group_id"]) in eligible_groups
        ]
        datasets[f"answer_{split}"] = valid_answers
        datasets[f"policy_{split}"] = policy_rows
        target_rejections[split] = {
            "rejected_trajectories": len(answer_rows) - len(valid_answers),
            "reasons": dict(rejection_reasons.most_common()),
        }
    stats = {name: _component_stats(rows) for name, rows in datasets.items()}
    answer_train_shortfalls = {
        task: max(0, min_answer_train_per_task - stats["answer_train"]["by_task"].get(task, 0))
        for task in sorted(TARGET_TASKS)
    }
    answer_eval_shortfalls = {
        task: max(0, min_answer_eval_per_task - stats["answer_eval"]["by_task"].get(task, 0))
        for task in sorted(TARGET_TASKS)
    }
    component_group_mismatches = {}
    for split in ("train", "eval"):
        answer_groups = {str(row["group_id"]) for row in datasets[f"answer_{split}"]}
        policy_groups = {str(row["group_id"]) for row in datasets[f"policy_{split}"]}
        component_group_mismatches[split] = {
            "answer_only": len(answer_groups - policy_groups),
            "policy_only": len(policy_groups - answer_groups),
        }
    missing_policy_nodes = {
        split: sorted(POLICY_NODES - set(stats[f"policy_{split}"]["by_policy_node"]))
        for split in ("train", "eval")
    }
    cyrillic_targets = sum(item["cyrillic_targets"] for item in stats.values())
    release_gate = {
        "joint_v2_gate_passed": bool(joint_report.get("release_gate_passed")),
        "known_sample_kinds_only": not unknown_kinds,
        "component_groups_match": not any(
            value for mismatch in component_group_mismatches.values() for value in mismatch.values()
        ),
        "answer_train_coverage": not any(answer_train_shortfalls.values()),
        "answer_eval_coverage": not any(answer_eval_shortfalls.values()),
        "policy_node_coverage": not any(missing_policy_nodes.values()),
        "no_cyrillic_targets": cyrillic_targets == 0,
        "query_source_disjoint": joint_report.get("query_overlap") == 0
        and joint_report.get("source_identity_overlap") == 0,
    }
    report = {
        "schema_version": "sft_v2.1",
        "profile": "separate_answer_policy_adapters",
        "seed": seed,
        "min_reward": min_reward,
        "joint_build": joint_report,
        "components": stats,
        "unknown_sample_kinds": dict(unknown_kinds),
        "component_group_mismatches": component_group_mismatches,
        "answer_train_shortfalls": answer_train_shortfalls,
        "answer_eval_shortfalls": answer_eval_shortfalls,
        "missing_policy_nodes": missing_policy_nodes,
        "answer_target_rejections": target_rejections,
        "release_gate": release_gate,
        "release_gate_passed": all(release_gate.values()),
    }
    return datasets, report


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build separate SFT v2.1 answer and policy datasets")
    parser.add_argument("--input", nargs="+", default=[str(PROJECT_ROOT / "data/rag/trajectories/*.jsonl")])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_DIR / "build_report.json"))
    parser.add_argument("--min-reward", type=float, default=0.65)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-task", type=int, default=100)
    parser.add_argument("--min-answer-train-per-task", type=int, default=80)
    parser.add_argument("--min-answer-eval-per-task", type=int, default=10)
    parser.add_argument("--allow-failed-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets, report = build_component_datasets(
        _read_jsonl(args.input),
        min_reward=args.min_reward,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_per_task=args.min_per_task,
        min_answer_train_per_task=args.min_answer_train_per_task,
        min_answer_eval_per_task=args.min_answer_eval_per_task,
    )
    output_dir = Path(args.output_dir)
    for name, filename in COMPONENT_FILES.items():
        _write_json_atomic(output_dir / filename, datasets[name])
    _write_json_atomic(Path(args.report), report)
    print(
        json.dumps(
            {
                "release_gate_passed": report["release_gate_passed"],
                "release_gate": report["release_gate"],
                "components": {
                    name: {"samples": item["samples"], "groups": item["groups"], "by_task": item["by_task"]}
                    for name, item in report["components"].items()
                },
            },
            ensure_ascii=False,
        )
    )
    if not report["release_gate_passed"] and not args.allow_failed_gate:
        print("SFT v2.1 data release gate failed; see build_report.json.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
