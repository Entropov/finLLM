#!/usr/bin/env python3
"""Build leakage-resistant Qwen3 trajectory SFT data from verified audit logs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag.audit_schema import canonical_json, load_audit_envelope  # noqa: E402
from scripts.rag.reward_v2 import compute_auditable_reward  # noqa: E402


POLICY_NODES = {
    "build_claim_plan",
    "select_policy",
    "plan_queries",
    "retrieve_local",
    "maybe_web_collect",
    "rerank_context",
    "normalize_evidence",
    "source_quality_gate",
    "resolve_contradictions",
    "compliance_gate",
}
TARGET_TASKS = {
    "stock_analysis",
    "quant_strategy",
    "financial_report",
    "sentiment_analysis",
    "financial_qa",
    "risk_assessment",
}
SYSTEM_POLICY = (
    "You are the policy component of an auditable financial agent. Return a concise structured decision. "
    "Use only observable state; never invent tool observations or citations."
)
SYSTEM_ANSWER = (
    "You are an auditable financial analyst. For complex tasks, use a short structured reasoning summary inside "
    "<think> tags, then provide the final answer. Every factual or calculated conclusion must cite a supplied "
    "Evidence ID. Treat publication, effective, and retrieval times as different fields. If evidence is insufficient, abstain."
)


def _uses_structured_thinking(task_type: str, claim_plan: dict[str, Any]) -> bool:
    if "complex_task" in claim_plan:
        return bool(claim_plan["complex_task"])
    return task_type in {"stock_analysis", "quant_strategy", "financial_report", "risk_assessment"}


def _supervision_target(reasoning: dict[str, Any], output: str, use_thinking: bool) -> str:
    if not use_thinking:
        return output
    return f"<think>\n{canonical_json(reasoning)}\n</think>\n{output}"


def _read_jsonl(patterns: Iterable[str]) -> Iterable[dict[str, Any]]:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for name in matches:
            with open(name, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        yield {"_load_error": f"{name}:{line_number}:{exc}"}
                        continue
                    payload["_source_file"] = name
                    payload["_source_line"] = line_number
                    yield payload


def _fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_identities(envelope) -> list[str]:
    return sorted(
        {
            f"{item.canonical_url or item.source_uri}|{item.document_version}"
            for item in envelope.evidence
            if item.canonical_url or item.source_uri
        }
    )


def _connected_source_groups(
    bundles: list[tuple[str, list[dict[str, Any]], list[str]]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep all trajectories sharing any source/version in the same split."""
    parent = {key: key for key, _, _ in bundles}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    source_owner: dict[str, str] = {}
    for key, _, identities in bundles:
        for identity in identities:
            owner = source_owner.setdefault(identity, key)
            union(key, owner)

    components: dict[str, list[tuple[list[dict[str, Any]], list[str]]]] = {}
    for key, samples, identities in bundles:
        components.setdefault(find(key), []).append((samples, identities))

    groups: dict[str, list[dict[str, Any]]] = {}
    for component in components.values():
        all_identities = sorted({identity for _, identities in component for identity in identities})
        fallback_ids = sorted(sample["group_id"] for samples, _ in component for sample in samples[:1])
        material = "||".join(all_identities or fallback_ids)
        group_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        grouped_samples = []
        for samples, identities in component:
            for sample in samples:
                sample["source_group"] = group_id
                sample["source_groups"] = identities
                grouped_samples.append(sample)
        groups[group_id] = grouped_samples
    return groups


def _trajectory_reward(envelope, payload: dict[str, Any]) -> dict[str, Any]:
    trajectory = [item.to_dict() for item in envelope.trajectory]
    market_feedback = payload.get("market_feedback")
    return compute_auditable_reward(
        query=envelope.query,
        answer=envelope.final_answer,
        evidence=envelope.evidence,
        claims=envelope.claims,
        calculations=envelope.calculations,
        task_type=envelope.task_type,
        request_as_of=envelope.request_as_of,
        trajectory=trajectory,
        market_feedback=market_feedback,
    )


def validate_trajectory(payload: dict[str, Any], min_reward: float) -> tuple[Any | None, dict[str, Any] | None, list[str]]:
    if payload.get("_load_error"):
        return None, None, [payload["_load_error"]]
    audit_payload = payload.get("audit") or (payload if payload.get("schema_version") == "2.0" else None)
    if not isinstance(audit_payload, dict):
        return None, None, ["missing_audit_v2"]
    try:
        envelope = load_audit_envelope(audit_payload)
    except (TypeError, ValueError) as exc:
        return None, None, [f"invalid_schema:{exc}"]
    errors = envelope.validation_errors()
    reward = _trajectory_reward(envelope, payload)
    if not reward["hard_gate_passed"]:
        errors.extend(f"reward:{item}" for item in reward["hard_failures"])
    if reward["total_reward"] < min_reward:
        errors.append("reward_below_threshold")
    if not envelope.query_id or not envelope.query or not envelope.final_answer:
        errors.append("missing_required_content")
    return envelope, reward, sorted(set(errors))


def _policy_samples(envelope, claim_plan: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    history: list[dict[str, Any]] = []
    use_thinking = _uses_structured_thinking(envelope.task_type, claim_plan)
    for event in envelope.trajectory:
        if event.node not in POLICY_NODES:
            history.append({"node": event.node, "action": event.action})
            continue
        state_summary = {
            "query": envelope.query,
            "task_type": envelope.task_type,
            "request_as_of": envelope.request_as_of,
            "claim_plan": claim_plan,
            "completed_steps": history[-6:],
            "current_node": event.node,
        }
        decision = {
            "node": event.node,
            "action": event.action,
            "action_args": event.action_args,
            "observation_ids": list(event.observation_ids),
        }
        reasoning = {
            "objective": event.node,
            "available_evidence_ids": list(event.observation_ids),
            "constraint": "use_observed_state_only",
        }
        answer = _supervision_target(reasoning, canonical_json(decision), use_thinking)
        samples.append(
            {
                "id": f"{envelope.query_id}:policy:{event.step_id}",
                "group_id": envelope.query_id,
                "task_type": envelope.task_type,
                "sample_kind": "agent_policy",
                "policy_node": event.node,
                "system": SYSTEM_POLICY,
                "conversations": [
                    {"from": "human", "value": canonical_json(state_summary)},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
        history.append({"node": event.node, "action": event.action})
    return samples


def _answer_sample(envelope, claim_plan: dict[str, Any]) -> dict[str, Any]:
    evidence_context = [
        {
            "evidence_id": item.evidence_id,
            "source": item.canonical_url or item.source_uri,
            "publisher": item.publisher,
            "published_at": item.published_at,
            "effective_at": item.effective_at,
            "fetched_at": item.fetched_at,
            "reliability_tier": item.reliability_tier,
            "exact_quote": item.exact_quote,
        }
        for item in envelope.evidence
    ]
    prompt = {
        "query": envelope.query,
        "task_type": envelope.task_type,
        "request_as_of": envelope.request_as_of,
        "claim_plan": claim_plan,
        "evidence": evidence_context,
    }
    reasoning = {
        "claim_plan": claim_plan.get("required_claim_groups", []),
        "claim_evidence_map": [
            {
                "claim_id": item.claim_id,
                "claim_type": item.claim_type,
                "supporting_evidence_ids": list(item.supporting_evidence_ids),
                "confidence": item.confidence,
            }
            for item in envelope.claims
        ],
        "uncertainties": sorted({assumption for item in envelope.claims for assumption in item.assumptions}),
    }
    answer = _supervision_target(
        reasoning,
        envelope.final_answer.strip(),
        _uses_structured_thinking(envelope.task_type, claim_plan),
    )
    return {
        "id": f"{envelope.query_id}:answer",
        "group_id": envelope.query_id,
        "task_type": envelope.task_type,
        "sample_kind": "auditable_answer",
        "system": SYSTEM_ANSWER,
        "conversations": [
            {"from": "human", "value": canonical_json(prompt)},
            {"from": "gpt", "value": answer},
        ],
    }


def build_dataset(
    payloads: Iterable[dict[str, Any]],
    *,
    min_reward: float,
    eval_ratio: float,
    seed: int,
    min_per_task: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bundles: list[tuple[str, list[dict[str, Any]], list[str]]] = []
    rejected = Counter()
    accepted_tasks = Counter()
    seen_queries = set()
    accepted_trajectories = 0
    inspected = 0
    for payload in payloads:
        inspected += 1
        envelope, reward, errors = validate_trajectory(payload, min_reward)
        if errors:
            rejected.update(errors)
            continue
        query_fingerprint = _fingerprint(envelope.query)
        if query_fingerprint in seen_queries:
            rejected["duplicate_query"] += 1
            continue
        seen_queries.add(query_fingerprint)
        claim_plan = payload.get("claim_plan") or {
            "required_claim_groups": sorted({item.claim_type for item in envelope.claims}),
            "requires_point_in_time": any(item.effective_at for item in envelope.evidence),
        }
        samples = _policy_samples(envelope, claim_plan)
        samples.append(_answer_sample(envelope, claim_plan))
        source_identities = _source_identities(envelope)
        for sample in samples:
            sample["audit_reward"] = reward["total_reward"]
            sample["query_fingerprint"] = query_fingerprint
        bundle_key = f"{envelope.query_id}:{query_fingerprint}"
        bundles.append((bundle_key, samples, source_identities))
        accepted_trajectories += 1
        accepted_tasks[envelope.task_type] += 1

    groups = _connected_source_groups(bundles)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    eval_count = int(round(len(group_ids) * eval_ratio))
    if len(group_ids) > 1 and eval_ratio > 0:
        eval_count = max(1, min(eval_count, len(group_ids) - 1))
    eval_ids = set(group_ids[:eval_count])
    train = [sample for group_id in group_ids if group_id not in eval_ids for sample in groups[group_id]]
    evaluation = [sample for group_id in group_ids if group_id in eval_ids for sample in groups[group_id]]
    task_shortfalls = {
        task: max(0, min_per_task - accepted_tasks.get(task, 0))
        for task in sorted(TARGET_TASKS)
    }
    query_overlap = len({row["query_fingerprint"] for row in train} & {row["query_fingerprint"] for row in evaluation})
    source_overlap = len(
        {identity for row in train for identity in row.get("source_groups", [])}
        & {identity for row in evaluation for identity in row.get("source_groups", [])}
    )
    report = {
        "schema_version": "sft_v2.0",
        "input_trajectories": inspected,
        "accepted_trajectories": accepted_trajectories,
        "rejected_trajectories": inspected - accepted_trajectories,
        "rejection_reasons": dict(rejected.most_common()),
        "accepted_by_task": dict(sorted(accepted_tasks.items())),
        "train_groups": len(groups) - len(eval_ids),
        "eval_groups": len(eval_ids),
        "train_samples": len(train),
        "eval_samples": len(evaluation),
        "query_overlap": query_overlap,
        "source_identity_overlap": source_overlap,
        "minimum_trajectories_per_task": min_per_task,
        "task_shortfalls": task_shortfalls,
        "release_gate_passed": bool(train and evaluation) and not query_overlap and not source_overlap and not any(task_shortfalls.values()),
        "min_reward": min_reward,
        "seed": seed,
    }
    return train, evaluation, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verified trajectory SFT v2 data")
    parser.add_argument("--input", nargs="+", default=[str(PROJECT_ROOT / "data/rag/trajectories/*.jsonl")])
    parser.add_argument("--train-output", default=str(PROJECT_ROOT / "data/sft_v2/fin_agentic_sft_v2_train.json"))
    parser.add_argument("--eval-output", default=str(PROJECT_ROOT / "data/sft_v2/fin_agentic_sft_v2_eval.json"))
    parser.add_argument("--report", default=str(PROJECT_ROOT / "data/sft_v2/build_report.json"))
    parser.add_argument("--min-reward", type=float, default=0.65)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-task", type=int, default=100)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train, evaluation, report = build_dataset(
        _read_jsonl(args.input),
        min_reward=args.min_reward,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_per_task=args.min_per_task,
    )
    outputs = (Path(args.train_output), Path(args.eval_output), Path(args.report))
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs[1].write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs[2].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["release_gate_passed"] and not args.allow_empty:
        print("SFT v2 data release gate failed; see build_report.json.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
