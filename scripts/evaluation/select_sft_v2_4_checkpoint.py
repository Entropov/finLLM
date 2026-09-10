#!/usr/bin/env python3
"""Select an SFT or preference checkpoint using source-disjoint Dev-Audit generation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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

from scripts.evaluation.eval_sft_v2_trusted import TARGET_TASKS, _load_json, dataset_fingerprint, validate_cases  # noqa: E402
from scripts.evaluation.generate_sft_v2_trusted_predictions import DEFAULT_MODEL, _identity, _sha256_file  # noqa: E402
from scripts.evaluation.sft_v2_4_protocol import prompt_contract, render_prompt, score_answer  # noqa: E402


MANIFEST_VERSION = "dev_audit_predictions.v2.4"
SELECTION_VERSION = "dev_audit_checkpoint_selection.v2.4"


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        return int(match.group(1))
    state_path = path / "trainer_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("global_step", -1))
    return -1


def checkpoint_tag(path: Path) -> str:
    if re.fullmatch(r"checkpoint-\d+", path.name):
        return path.name
    return f"final-step-{checkpoint_step(path)}"


def discover_checkpoints(root: Path) -> list[Path]:
    checkpoints = [path for path in root.glob("checkpoint-*") if (path / "adapter_model.safetensors").is_file()]
    if (root / "adapter_model.safetensors").is_file():
        final_step = checkpoint_step(root)
        if all(checkpoint_step(path) != final_step for path in checkpoints):
            checkpoints.append(root)
    checkpoints.sort(key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"no adapter checkpoints under {root}")
    return checkpoints


def summarize(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row["score"])
    for task in sorted(TARGET_TASKS):
        scores = grouped[task]
        by_task[task] = {
            "samples": len(scores),
            "pass_rate": round(mean(float(item["passed"]) for item in scores), 4),
            "audit_pass_rate": round(mean(float(item["audit_hard_gate_passed"]) for item in scores), 4),
            "mean_primary_score": round(mean(float(item["primary_score"]) for item in scores), 4),
        }
    return {
        "samples": len(cases),
        "pass_rate": round(mean(float(row["score"]["passed"]) for row in rows), 4),
        "audit_pass_rate": round(mean(float(row["score"]["audit_hard_gate_passed"]) for row in rows), 4),
        "minimum_task_audit_pass_rate": min(item["audit_pass_rate"] for item in by_task.values()),
        "mean_primary_score": round(mean(float(row["score"]["primary_score"]) for row in rows), 4),
        "length_finish_count": sum(row["finish_reason"] == "length" for row in rows),
        "protocol_failure_count": sum(bool(row["score"]["protocol_errors"]) for row in rows),
        "by_task": by_task,
    }


def rank_key(report: dict[str, Any], step: int) -> tuple[float, float, float, float, int, int]:
    return (
        float(report["audit_pass_rate"]),
        float(report["minimum_task_audit_pass_rate"]),
        float(report["pass_rate"]),
        float(report["mean_primary_score"]),
        -int(report["length_finish_count"]),
        -step,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def replace_selected_directory(temporary: Path, selected_output: Path, *, overwrite: bool) -> None:
    """Promote a fully written directory and restore the old selection if promotion fails."""
    if not selected_output.exists():
        os.replace(temporary, selected_output)
        return
    if not overwrite:
        raise FileExistsError(f"selected output already exists: {selected_output}")
    backup = selected_output.with_name(f"{selected_output.name}.previous-{os.getpid()}")
    if backup.exists():
        raise FileExistsError(f"stale selection backup exists: {backup}")
    os.replace(selected_output, backup)
    try:
        os.replace(temporary, selected_output)
    except BaseException:
        os.replace(backup, selected_output)
        raise
    shutil.rmtree(backup)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select checkpoint by Dev-Audit generation")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--selected-output", required=True)
    parser.add_argument("--gold", default=str(PROJECT_ROOT / "data/evaluation/sft_v2_4_dev_audit.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_4_dev_checkpoints"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--overwrite-selected", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = discover_checkpoints(checkpoint_dir)
    gold_path = Path(args.gold)
    cases = _load_json(gold_path)
    validate_cases(cases)
    counts = Counter(str(case["task_type"]) for case in cases)
    if set(counts) != TARGET_TASKS or any(counts[task] < 8 for task in TARGET_TASKS):
        raise ValueError(f"Dev-Audit requires all tasks with at least 8 cases: {dict(counts)}")
    selected_output = Path(args.selected_output)
    if selected_output.exists() and not args.overwrite_selected:
        raise FileExistsError(f"selected output already exists: {selected_output}")
    print(json.dumps({"checkpoints": [str(path) for path in checkpoints], "dev_samples": len(cases), "by_task": counts}, ensure_ascii=False, default=dict), flush=True)
    if args.validate_only:
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    prompts = [render_prompt(tokenizer, case) for case in cases]
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, repetition_penalty=1.05, seed=46)
    summaries = []
    output_dir = Path(args.output_dir)
    protocol = prompt_contract(str(Path(__file__).resolve()))
    for adapter_index, checkpoint in enumerate(checkpoints, start=1):
        rows = []
        request = LoRARequest(f"dev-{adapter_index}", adapter_index, lora_path=str(checkpoint.resolve()))
        for offset in range(0, len(cases), args.batch_size):
            batch_cases = cases[offset : offset + args.batch_size]
            outputs = llm.generate(prompts[offset : offset + args.batch_size], sampling, use_tqdm=False, lora_request=request)
            for case, output in zip(batch_cases, outputs):
                candidate = output.outputs[0]
                answer = candidate.text.strip()
                rows.append(
                    {
                        "id": str(case["id"]),
                        "task_type": str(case["task_type"]),
                        "answer": answer,
                        "finish_reason": candidate.finish_reason or "stop",
                        "output_tokens": len(candidate.token_ids or []),
                        "score": score_answer(case, answer),
                    }
                )
        summary = summarize(cases, rows)
        step = checkpoint_step(checkpoint)
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "checkpoint": _identity(str(checkpoint), hash_weights=True),
            "checkpoint_step": step,
            "dataset_fingerprint": dataset_fingerprint(cases),
            "gold_sha256": _sha256_file(gold_path),
            "prompt_contract": protocol,
            "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens, "enable_thinking": False},
            "summary": summary,
            "predictions": rows,
        }
        write_json(output_dir / f"{checkpoint_tag(checkpoint)}.json", manifest)
        summaries.append({"checkpoint": str(checkpoint), "step": step, **summary})
        print(json.dumps({"checkpoint": checkpoint.name, **summary}, ensure_ascii=False), flush=True)

    winner = max(summaries, key=lambda item: rank_key(item, int(item["step"])))
    source = Path(str(winner["checkpoint"]))
    temporary = selected_output.with_name(selected_output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    selection = {
        "selection_version": SELECTION_VERSION,
        "selection_order": ["audit_pass_rate", "minimum_task_audit_pass_rate", "pass_rate", "mean_primary_score", "fewest_truncations", "earliest_step"],
        "dev_gold": {"path": str(gold_path.resolve()), "sha256": _sha256_file(gold_path), "fingerprint": dataset_fingerprint(cases)},
        "checkpoint_root": str(checkpoint_dir.resolve()),
        "selected_checkpoint": str(source.resolve()),
        "selected_step": winner["step"],
        "selected_adapter_sha256": _sha256_file(source / "adapter_model.safetensors"),
        "selected_summary": {key: value for key, value in winner.items() if key != "checkpoint"},
        "all_checkpoints": summaries,
        "created_at_unix": int(time.time()),
    }
    write_json(temporary / "dev_audit_selection.json", selection)
    replace_selected_directory(temporary, selected_output, overwrite=args.overwrite_selected)
    write_json(output_dir / "selection.json", selection)
    print(json.dumps({"selected": str(source), "step": winner["step"], "summary": selection["selected_summary"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
