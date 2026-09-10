#!/usr/bin/env python3
"""Generate and score paired base/candidate E2E@1 and candidate E2E@8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.eval_sft_v2_trusted import (  # noqa: E402
    TARGET_TASKS,
    _load_json,
    dataset_fingerprint,
    evidence_source_identities,
    paired_cluster_bootstrap_interval,
    validate_cases,
)
from scripts.evaluation.generate_sft_v2_trusted_predictions import (  # noqa: E402
    DEFAULT_MODEL,
    _identity,
    _sha256_file,
)
from scripts.evaluation.sft_v2_4_protocol import prompt_contract, render_prompt, score_answer  # noqa: E402


DEFAULT_ADAPTER = PROJECT_ROOT / "saves/qwen3-8b/lora/sft-v2.4-hardneg-selected"
DEFAULT_GOLD = PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "saves/eval_results/sft_v2_4_trusted_e2e.json"
RL_THRESHOLDS = {
    "minimum_e2e_at_8": 0.80,
    "minimum_audit_at_8": 0.95,
    "minimum_task_e2e_at_8": 0.70,
    "minimum_e2e_gap": 0.15,
}
MANIFEST_VERSION = "sft_v2.4_e2e.v1"
CANDIDATE_REQUEST_NAME = "sft-v2.4-candidate"
EXPECTED_TASKS = TARGET_TASKS
MIN_SAMPLES_PER_TASK = 50


def _round(value: float) -> float:
    return round(value, 4)


def _sequence_logprob_summary(output: Any) -> dict[str, Any] | None:
    """Store selected-token sequence likelihoods, not bulky top-k distributions."""
    token_logprobs = getattr(output, "logprobs", None)
    token_ids = list(getattr(output, "token_ids", None) or [])
    if not token_logprobs or len(token_logprobs) != len(token_ids):
        return None
    values: list[float] = []
    for token_id, alternatives in zip(token_ids, token_logprobs):
        item = alternatives.get(token_id) if isinstance(alternatives, dict) else None
        value = getattr(item, "logprob", None) if item is not None else None
        if value is None:
            return None
        values.append(float(value))
    if not values:
        return None
    total = sum(values)
    return {
        "token_count": len(values),
        "sequence_logprob": round(total, 6),
        "sequence_normalized_logprob": round(total / len(values), 6),
        "sequence_normalized_nll": round(-total / len(values), 6),
    }


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    for task in sorted(TARGET_TASKS):
        task_rows = grouped[task]
        by_task[task] = {
            "samples": len(task_rows),
            "baseline_e2e_at_1": _round(mean(float(row["baseline_score"]["passed"]) for row in task_rows)),
            "candidate_e2e_at_1": _round(mean(float(row["candidate_score_at_1"]["passed"]) for row in task_rows)),
            "candidate_e2e_at_8": _round(mean(float(row["candidate_pass_at_8"]) for row in task_rows)),
            "candidate_audit_at_1": _round(
                mean(float(row["candidate_score_at_1"]["audit_hard_gate_passed"]) for row in task_rows)
            ),
            "candidate_audit_at_8": _round(mean(float(row["candidate_audit_at_8"]) for row in task_rows)),
            "baseline_mean_primary_at_1": _round(mean(row["baseline_score"]["primary_score"] for row in task_rows)),
            "candidate_mean_primary_at_1": _round(
                mean(row["candidate_score_at_1"]["primary_score"] for row in task_rows)
            ),
            "candidate_mean_best_primary_at_8": _round(mean(row["best_primary_at_8"] for row in task_rows)),
        }
        by_task[task]["primary_delta_at_1"] = _round(
            by_task[task]["candidate_mean_primary_at_1"] - by_task[task]["baseline_mean_primary_at_1"]
        )
    differences = [
        row["candidate_score_at_1"]["primary_score"] - row["baseline_score"]["primary_score"] for row in rows
    ]
    grouped_differences: dict[str, list[float]] = defaultdict(list)
    for row, difference in zip(rows, differences):
        grouped_differences[str(row["source_group"])].append(difference)
    e2e_at_1 = mean(float(row["candidate_score_at_1"]["passed"]) for row in rows)
    e2e_at_8 = mean(float(row["candidate_pass_at_8"]) for row in rows)
    audit_at_1 = mean(float(row["candidate_score_at_1"]["audit_hard_gate_passed"]) for row in rows)
    audit_at_8 = mean(float(row["candidate_audit_at_8"]) for row in rows)
    clustered_ci = paired_cluster_bootstrap_interval(grouped_differences)
    minimum_task_e2e_at_8 = min(item["candidate_e2e_at_8"] for item in by_task.values())
    e2e_gap = e2e_at_8 - e2e_at_1
    consider_rl = (
        e2e_at_8 >= RL_THRESHOLDS["minimum_e2e_at_8"]
        and audit_at_8 >= RL_THRESHOLDS["minimum_audit_at_8"]
        and minimum_task_e2e_at_8 >= RL_THRESHOLDS["minimum_task_e2e_at_8"]
        and e2e_gap >= RL_THRESHOLDS["minimum_e2e_gap"]
    )
    if consider_rl:
        recommendation = "consider_grpo_for_generation_stability"
    elif e2e_at_8 < RL_THRESHOLDS["minimum_e2e_at_8"] or minimum_task_e2e_at_8 < RL_THRESHOLDS["minimum_task_e2e_at_8"]:
        recommendation = "continue_sft_data_and_protocol_repair"
    else:
        recommendation = "no_rl_stability_case"
    baseline_e2e = mean(float(row["baseline_score"]["passed"]) for row in rows)
    candidate_primary = mean(row["candidate_score_at_1"]["primary_score"] for row in rows)
    baseline_primary = mean(row["baseline_score"]["primary_score"] for row in rows)
    return {
        "samples": len(rows),
        "baseline": {
            "e2e_at_1": _round(baseline_e2e),
            "audit_at_1": _round(mean(float(row["baseline_score"]["audit_hard_gate_passed"]) for row in rows)),
            "mean_primary_at_1": _round(baseline_primary),
        },
        "candidate": {
            "e2e_at_1": _round(e2e_at_1),
            "e2e_at_8": _round(e2e_at_8),
            "audit_at_1": _round(audit_at_1),
            "audit_at_8": _round(audit_at_8),
            "mean_primary_at_1": _round(candidate_primary),
            "mean_best_primary_at_8": _round(mean(row["best_primary_at_8"] for row in rows)),
            "e2e_gap_at_8_minus_at_1": _round(e2e_gap),
            "minimum_task_e2e_at_8": minimum_task_e2e_at_8,
        },
        "paired": {
            "mean_primary_delta_at_1": _round(candidate_primary - baseline_primary),
            "source_group_clustered_delta_95ci": clustered_ci,
        },
        "generation_quality": {
            "baseline_length_finishes": sum(row["baseline_finish_reason"] == "length" for row in rows),
            "candidate_length_finishes_at_1": sum(row["candidate_finish_reason_at_1"] == "length" for row in rows),
            "candidate_length_finishes_at_8": sum(
                finish == "length" for row in rows for finish in row["candidate_finish_reasons_at_8"]
            ),
            "candidate_nonempty_think_at_1": sum(
                "nonempty_thinking" in row["candidate_score_at_1"]["protocol_errors"] for row in rows
            ),
        },
        "by_task": by_task,
        "release_gate": {
            "audit_at_1_at_least_95pct": audit_at_1 >= 0.95,
            "paired_primary_noninferiority_95ci": clustered_ci[0] >= -0.02,
            "per_task_primary_noninferiority": all(item["primary_delta_at_1"] >= -0.02 for item in by_task.values()),
            "e2e_at_1_non_regression": e2e_at_1 >= baseline_e2e,
        },
        "rl_decision": {
            "thresholds": RL_THRESHOLDS,
            "consider_rl": consider_rl,
            "recommendation": recommendation,
        },
    }


def summarize_predictions(cases: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in cases:
        identifier = str(case["id"])
        prediction = predictions[identifier]
        baseline_score = score_answer(case, prediction["baseline_answer"])
        candidate_score_at_1 = score_answer(case, prediction["candidate_answer_at_1"])
        sample_scores = [score_answer(case, answer) for answer in prediction["candidate_answers_at_8"]]
        rows.append(
            {
                "id": identifier,
                "task_type": str(case["task_type"]),
                "source_group": str(case.get("source_group") or identifier),
                **prediction,
                "baseline_score": baseline_score,
                "candidate_score_at_1": candidate_score_at_1,
                "candidate_scores_at_8": sample_scores,
                "candidate_pass_at_8": any(item["passed"] for item in sample_scores),
                "candidate_audit_at_8": any(item["audit_hard_gate_passed"] for item in sample_scores),
                "best_primary_at_8": max(item["primary_score"] for item in sample_scores),
            }
        )
    return {"summary": _score_summary(rows), "details": rows}


def _write_output(
    path: Path,
    *,
    cases: list[dict[str, Any]],
    gold_path: Path,
    model_identity: dict[str, Any],
    adapter_identity: dict[str, Any],
    generation: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> None:
    scored = summarize_predictions(cases, predictions) if len(predictions) == len(cases) else None
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_fingerprint": dataset_fingerprint(cases),
        "gold": {"path": str(gold_path.resolve()), "sha256": _sha256_file(gold_path)},
        "model": model_identity,
        "adapter": adapter_identity,
        "prompt_contract": prompt_contract(str(Path(__file__).resolve())),
        "generation_settings": generation,
        "expected_samples": len(cases),
        "completed_samples": len(predictions),
        "predictions": [predictions[str(case["id"])] for case in cases if str(case["id"]) in predictions],
        "evaluation": scored,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_existing(
    path: Path,
    *,
    fingerprint: str,
    model_identity: dict[str, Any],
    adapter_identity: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    expected = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_fingerprint": fingerprint,
        "model": model_identity,
        "adapter": adapter_identity,
        "prompt_contract": prompt_contract(str(Path(__file__).resolve())),
        "generation_settings": generation,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"cannot resume {path}; incompatible fields: {', '.join(mismatches)}")
    rows = payload.get("predictions", [])
    existing = {str(row.get("id", "")): row for row in rows}
    if len(existing) != len(rows) or "" in existing:
        raise ValueError(f"invalid existing prediction IDs: {path}")
    return existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT v2.4 with paired E2E@1 and candidate E2E@8")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-file", default=str(PROJECT_ROOT / "data/sft_v2_4/fin_agentic_sft_v2_4_answer_train.json"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--collect-token-logprobs",
        action="store_true",
        help="Persist selected-token sequence logprob/NLL for confidence diagnostics.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples != 8:
        raise ValueError("the preregistered protocol requires exactly 8 sampled candidates")
    gold_path = Path(args.gold)
    cases = _load_json(gold_path)
    validate_cases(cases)
    counts = Counter(str(case["task_type"]) for case in cases)
    if set(counts) != EXPECTED_TASKS or any(counts[task] < MIN_SAMPLES_PER_TASK for task in EXPECTED_TASKS):
        raise ValueError(
            f"E2E gold requires every configured task with at least {MIN_SAMPLES_PER_TASK} cases: {dict(counts)}"
        )
    train_rows = _load_json(Path(args.train_file))
    train_sources = {
        str(item)
        for row in train_rows
        for item in ([row.get("source_group")] + list(row.get("source_groups", [])))
        if item
    }
    gold_sources = set().union(*(evidence_source_identities(case) for case in cases))
    direct_gold_groups = {str(case.get("source_group")) for case in cases if case.get("source_group")}
    overlaps = sorted(train_sources & (gold_sources | direct_gold_groups))
    if overlaps:
        raise ValueError(f"train/evaluation source leakage detected: {overlaps[:5]}")
    model_identity = _identity(args.model)
    adapter_identity = _identity(args.adapter, hash_weights=True)
    if not model_identity["exists"]:
        raise FileNotFoundError(f"base model does not exist: {args.model}")
    if not adapter_identity["exists"]:
        raise FileNotFoundError(f"candidate adapter does not exist: {args.adapter}")
    generation = {
        "backend": "vllm",
        "greedy": {"temperature": 0.0, "top_p": 1.0, "seed": args.seed},
        "sampled": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed, "n": args.samples},
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "enable_thinking": False,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "collect_token_logprobs": args.collect_token_logprobs,
    }
    output_path = Path(args.output)
    existing = {} if args.no_resume else _load_existing(
        output_path,
        fingerprint=dataset_fingerprint(cases),
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        generation=generation,
    )
    print(json.dumps({"samples": len(cases), "by_task": counts, "existing": len(existing), "source_overlap": 0}, ensure_ascii=False, default=dict), flush=True)
    if args.validate_only:
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
    )
    sampling_logprobs = 1 if args.collect_token_logprobs else None
    greedy = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        logprobs=sampling_logprobs,
    )
    sampled = SamplingParams(
        n=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        logprobs=sampling_logprobs,
    )
    request = LoRARequest(CANDIDATE_REQUEST_NAME, 1, lora_path=str(Path(args.adapter).resolve()))
    pending = [case for case in cases if str(case["id"]) not in existing]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        prompts = [render_prompt(tokenizer, case) for case in batch]
        started = time.time()
        baseline_outputs = llm.generate(prompts, greedy, use_tqdm=False)
        candidate_outputs = llm.generate(prompts, greedy, use_tqdm=False, lora_request=request)
        sampled_outputs = llm.generate(prompts, sampled, use_tqdm=False, lora_request=request)
        for case, prompt, base, candidate, candidates in zip(
            batch, prompts, baseline_outputs, candidate_outputs, sampled_outputs
        ):
            identifier = str(case["id"])
            existing[identifier] = {
                "id": identifier,
                "task_type": str(case["task_type"]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_tokens": len(base.prompt_token_ids or []),
                "baseline_answer": base.outputs[0].text.strip(),
                "baseline_finish_reason": base.outputs[0].finish_reason or "stop",
                "candidate_answer_at_1": candidate.outputs[0].text.strip(),
                "candidate_finish_reason_at_1": candidate.outputs[0].finish_reason or "stop",
                "candidate_answers_at_8": [item.text.strip() for item in candidates.outputs],
                "candidate_finish_reasons_at_8": [item.finish_reason or "stop" for item in candidates.outputs],
                "baseline_sequence_logprob_at_1": _sequence_logprob_summary(base.outputs[0]),
                "candidate_sequence_logprob_at_1": _sequence_logprob_summary(candidate.outputs[0]),
                "candidate_sequence_logprobs_at_8": [
                    _sequence_logprob_summary(item) for item in candidates.outputs
                ],
            }
        _write_output(
            output_path,
            cases=cases,
            gold_path=gold_path,
            model_identity=model_identity,
            adapter_identity=adapter_identity,
            generation=generation,
            predictions=existing,
        )
        print(f"completed={len(existing)}/{len(cases)} batch_seconds={time.time() - started:.1f}", flush=True)
    summary = summarize_predictions(cases, existing)["summary"]
    print(json.dumps({"output": str(output_path), **summary}, ensure_ascii=False), flush=True)
    return 0 if all(summary["release_gate"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
