#!/home/super/.conda/envs/finllm/bin/python
"""Generate paired predictions using the strict SFT v2.3 audit contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing.build_sft_v2_3_dataset import (  # noqa: E402
    STRICT_CONTRACT,
    STRICT_SYSTEM_PROMPT,
    STRICT_TASK_CONTRACTS,
    strict_claim_plan,
)
from scripts.evaluation.eval_sft_v2_trusted import (  # noqa: E402
    TARGET_TASKS,
    dataset_fingerprint,
    get_question,
    validate_cases,
)
from scripts.evaluation.generate_sft_v2_trusted_predictions import (  # noqa: E402
    DEFAULT_MODEL,
    _identity,
    _load_json,
    _sha256_file,
)
from scripts.inference.chat_template import apply_chat_template  # noqa: E402


DEFAULT_GOLD = PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"
DEFAULT_ADAPTER = PROJECT_ROOT / "saves/qwen3-8b/lora/sft-v2.3-answer"
DEFAULT_BASELINE_OUTPUT = PROJECT_ROOT / "saves/eval_results/qwen3_base_v2_3_contract_trusted_predictions.json"
DEFAULT_CANDIDATE_OUTPUT = PROJECT_ROOT / "saves/eval_results/qwen3_sft_v2_3_trusted_predictions.json"


def build_model_input(case: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for raw in case.get("evidence", []):
        evidence.append(
            {
                "evidence_id": raw["evidence_id"],
                "exact_quote": raw["exact_quote"],
                "source": raw.get("canonical_url") or raw.get("source_uri", ""),
                "publisher": raw.get("publisher", ""),
                "reliability_tier": raw.get("reliability_tier", ""),
                "published_at": raw.get("published_at", ""),
                "effective_at": raw.get("effective_at", ""),
                "fetched_at": raw.get("fetched_at", ""),
            }
        )
    task_type = str(case["task_type"])
    return {
        "query": get_question(case),
        "task_type": task_type,
        "request_as_of": case["request_as_of"],
        "claim_plan": strict_claim_plan(task_type),
        "strict_audit_contract": STRICT_CONTRACT,
        "evidence": evidence,
        "output_contract": STRICT_TASK_CONTRACTS[task_type],
    }


def build_prompt(tokenizer: Any, case: dict[str, Any], *, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(build_model_input(case), ensure_ascii=False, separators=(",", ":"))},
    ]
    return apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _prompt_contract() -> dict[str, Any]:
    return {
        "fixed_case_evidence_only": True,
        "gold_scoring_hidden": True,
        "strict_audit_contract": True,
        "contract_version": "sft_v2.3",
        "system_prompt_sha256": hashlib.sha256(STRICT_SYSTEM_PROMPT.encode()).hexdigest(),
        "generator_sha256": _sha256_file(Path(__file__).resolve()),
    }


def _load_existing(
    path: Path,
    fingerprint: str,
    arm: str,
    *,
    model_identity: dict[str, Any],
    adapter_identity: dict[str, Any] | None,
    generation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    if payload.get("dataset_fingerprint") != fingerprint or payload.get("arm") != arm:
        raise ValueError(f"cannot resume incompatible prediction file: {path}")
    expected = {
        "model": model_identity,
        "adapter": adapter_identity,
        "generation_settings": generation,
        "prompt_contract": _prompt_contract(),
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"cannot resume {path}; incompatible fields: {', '.join(mismatches)}")
    rows = payload.get("predictions", [])
    existing = {str(row.get("id", "")): row for row in rows}
    if len(existing) != len(rows) or "" in existing:
        raise ValueError(f"invalid existing prediction IDs: {path}")
    return existing


def _write_manifest(
    path: Path,
    *,
    arm: str,
    fingerprint: str,
    gold_path: Path,
    cases: list[dict[str, Any]],
    model_identity: dict[str, Any],
    adapter_identity: dict[str, Any] | None,
    generation: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> None:
    ordered = [predictions[str(case["id"])] for case in cases if str(case["id"]) in predictions]
    payload = {
        "manifest_version": "trusted_predictions.v1",
        "arm": arm,
        "dataset_fingerprint": fingerprint,
        "gold": {"path": str(gold_path.resolve()), "sha256": _sha256_file(gold_path)},
        "model": model_identity,
        "adapter": adapter_identity,
        "generation_settings": generation,
        "prompt_contract": _prompt_contract(),
        "expected_samples": len(cases),
        "completed_samples": len(ordered),
        "task_distribution": dict(sorted(Counter(row["task_type"] for row in ordered).items())),
        "predictions": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _generate_arm(
    *,
    llm: Any,
    sampling_params_cls: type,
    lora_request: Any | None,
    arm: str,
    output_path: Path,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    fingerprint: str,
    gold_path: Path,
    model_identity: dict[str, Any],
    adapter_identity: dict[str, Any] | None,
    generation: dict[str, Any],
    batch_size: int,
    resume: bool,
) -> None:
    existing = (
        _load_existing(
            output_path,
            fingerprint,
            arm,
            model_identity=model_identity,
            adapter_identity=adapter_identity,
            generation=generation,
        )
        if resume
        else {}
    )
    pending = [case for case in cases if str(case["id"]) not in existing]
    print(f"{arm}: existing={len(existing)} pending={len(pending)}", flush=True)
    sampling_params = sampling_params_cls(
        max_tokens=generation["max_tokens"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        repetition_penalty=generation["repetition_penalty"],
        seed=generation["seed"],
    )
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        prompts = [build_prompt(tokenizer, case, enable_thinking=generation["enable_thinking"]) for case in batch]
        started = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False, lora_request=lora_request)
        for case, prompt, output in zip(batch, prompts, outputs):
            candidate = output.outputs[0]
            identifier = str(case["id"])
            existing[identifier] = {
                "id": identifier,
                "task_type": case["task_type"],
                "answer": candidate.text.strip(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_tokens": len(output.prompt_token_ids or []),
                "output_tokens": len(candidate.token_ids or []),
                "finish_reason": candidate.finish_reason or "stop",
            }
        _write_manifest(
            output_path,
            arm=arm,
            fingerprint=fingerprint,
            gold_path=gold_path,
            cases=cases,
            model_identity=model_identity,
            adapter_identity=adapter_identity,
            generation=generation,
            predictions=existing,
        )
        print(f"{arm}: completed={len(existing)}/{len(cases)} batch_seconds={time.time() - started:.1f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired predictions for strict SFT v2.3")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--baseline-output", default=str(DEFAULT_BASELINE_OUTPUT))
    parser.add_argument("--candidate-output", default=str(DEFAULT_CANDIDATE_OUTPUT))
    parser.add_argument("--arm", choices=("both", "baseline", "candidate"), default="both")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_path = Path(args.gold)
    cases = _load_json(gold_path)
    validate_cases(cases)
    counts = Counter(str(case["task_type"]) for case in cases)
    if set(counts) != TARGET_TASKS or any(counts[task] < 50 for task in TARGET_TASKS):
        raise ValueError(f"evaluation set must contain every task with at least 50 cases: {dict(counts)}")
    fingerprint = dataset_fingerprint(cases)
    model_identity = _identity(args.model)
    adapter_identity = _identity(args.adapter, hash_weights=True)
    if not model_identity["exists"]:
        raise FileNotFoundError(f"base model does not exist: {args.model}")
    if args.arm in {"both", "candidate"} and not adapter_identity["exists"]:
        raise FileNotFoundError(f"candidate adapter does not exist: {args.adapter}")
    generation = {
        "backend": "vllm",
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 1.0 if args.temperature <= 0 else args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "enable_thinking": not args.disable_thinking,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
    }
    print(json.dumps({"samples": len(cases), "by_task": counts, "dataset_fingerprint": fingerprint, "generation": generation}, ensure_ascii=False, default=dict), flush=True)
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
        enable_lora=args.arm in {"both", "candidate"},
        max_lora_rank=64,
        max_loras=1,
    )
    common = {
        "llm": llm,
        "sampling_params_cls": SamplingParams,
        "tokenizer": tokenizer,
        "cases": cases,
        "fingerprint": fingerprint,
        "gold_path": gold_path,
        "model_identity": model_identity,
        "generation": generation,
        "batch_size": args.batch_size,
        "resume": not args.no_resume,
    }
    if args.arm in {"both", "baseline"}:
        _generate_arm(**common, arm="baseline", output_path=Path(args.baseline_output), adapter_identity=None, lora_request=None)
    if args.arm in {"both", "candidate"}:
        request = LoRARequest("sft-v2.3", 1, lora_path=str(Path(args.adapter).resolve()))
        _generate_arm(**common, arm="candidate", output_path=Path(args.candidate_output), adapter_identity=adapter_identity, lora_request=request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
