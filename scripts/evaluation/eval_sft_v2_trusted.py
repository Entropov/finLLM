#!/usr/bin/env python3
"""Paired, audit-aware evaluation for SFT v2 model predictions.

This evaluator intentionally does not use ROUGE or a single LLM judge as its
release criterion. Gold cases define exact labels or verifiable patterns and,
for evidence-grounded tasks, the point-in-time evidence visible to both models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag.audit_schema import (  # noqa: E402
    CalculationRecord,
    EvidenceRecord,
    build_claims_from_answer,
    canonical_json,
    content_digest,
    parse_timestamp,
)
from scripts.rag.reward_v2 import RewardV2Config, compute_auditable_reward  # noqa: E402

TARGET_TASKS = {
    "stock_analysis",
    "quant_strategy",
    "financial_report",
    "sentiment_analysis",
    "financial_qa",
    "risk_assessment",
}
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
PER_TASK_PRIMARY_NONINFERIORITY_MARGIN = 0.02


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    stable = [
        {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "question_hash": normalized_hash(get_question(row)),
            "source_group": row.get("source_group", ""),
            "request_as_of": row.get("request_as_of", ""),
            "evidence": [
                {"evidence_id": item.get("evidence_id"), "content_hash": item.get("content_hash")}
                for item in row.get("evidence", [])
            ],
            "scoring": row.get("scoring", {}),
        }
        for row in sorted(rows, key=lambda item: str(item.get("id", "")))
    ]
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()


def get_question(item: dict[str, Any]) -> str:
    if item.get("question"):
        return str(item["question"])
    for turn in item.get("conversations", []):
        if turn.get("from") in {"human", "user"}:
            return str(turn.get("value", ""))
    return ""


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload = _load_json(path)
    manifest = payload if isinstance(payload, dict) else {}
    rows = payload.get("predictions", []) if isinstance(payload, dict) else payload
    predictions = {}
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in predictions:
            raise ValueError(f"prediction IDs must be unique and non-empty: {identifier!r}")
        predictions[identifier] = str(row.get("answer") or row.get("prediction") or "")
    return predictions, manifest


def validate_prediction_protocol(
    cases: list[dict[str, Any]],
    baseline: dict[str, str],
    candidate: dict[str, str],
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify that paired predictions came from the declared fixed-evidence protocol."""
    errors: list[str] = []
    expected_ids = {str(case.get("id", "")) for case in cases}
    fingerprint = dataset_fingerprint(cases)
    for name, predictions, manifest, expected_arm in (
        ("baseline", baseline, baseline_manifest, "baseline"),
        ("candidate", candidate, candidate_manifest, "candidate"),
    ):
        if set(predictions) != expected_ids:
            errors.append(f"{name}:prediction_id_set_mismatch")
        if manifest.get("manifest_version") != "trusted_predictions.v1":
            errors.append(f"{name}:invalid_manifest_version")
        if manifest.get("arm") != expected_arm:
            errors.append(f"{name}:invalid_arm")
        if manifest.get("dataset_fingerprint") != fingerprint:
            errors.append(f"{name}:dataset_fingerprint_mismatch")
        if manifest.get("expected_samples") != len(cases):
            errors.append(f"{name}:expected_sample_count_mismatch")
        if manifest.get("completed_samples") != len(cases):
            errors.append(f"{name}:incomplete_predictions")
        prompt_contract = manifest.get("prompt_contract") or {}
        if prompt_contract.get("fixed_case_evidence_only") is not True:
            errors.append(f"{name}:fixed_evidence_not_declared")
        if prompt_contract.get("gold_scoring_hidden") is not True:
            errors.append(f"{name}:hidden_scoring_not_declared")
        if not prompt_contract.get("system_prompt_sha256"):
            errors.append(f"{name}:system_prompt_identity_missing")
        if not prompt_contract.get("generator_sha256"):
            errors.append(f"{name}:generator_identity_missing")
        model = manifest.get("model") or {}
        if model.get("exists") is not True or not model.get("resolved_path"):
            errors.append(f"{name}:model_identity_missing")
        if not manifest.get("generation_settings"):
            errors.append(f"{name}:generation_settings_missing")

    if baseline_manifest.get("adapter") is not None:
        errors.append("baseline:adapter_must_be_absent")
    candidate_adapter = candidate_manifest.get("adapter") or {}
    if candidate_adapter.get("exists") is not True or not candidate_adapter.get("resolved_path"):
        errors.append("candidate:adapter_identity_missing")
    adapter_weights = candidate_adapter.get("weight_files") or []
    if not adapter_weights or any(not item.get("sha256") for item in adapter_weights):
        errors.append("candidate:adapter_weight_hash_missing")
    if baseline_manifest.get("model") != candidate_manifest.get("model"):
        errors.append("paired:base_model_identity_mismatch")
    if baseline_manifest.get("generation_settings") != candidate_manifest.get("generation_settings"):
        errors.append("paired:generation_settings_mismatch")
    if baseline_manifest.get("prompt_contract") != candidate_manifest.get("prompt_contract"):
        errors.append("paired:prompt_contract_mismatch")
    baseline_prompt_hashes = {
        str(row.get("id", "")): row.get("prompt_sha256")
        for row in baseline_manifest.get("predictions", [])
    }
    candidate_prompt_hashes = {
        str(row.get("id", "")): row.get("prompt_sha256")
        for row in candidate_manifest.get("predictions", [])
    }
    if set(baseline_prompt_hashes) != expected_ids or any(not value for value in baseline_prompt_hashes.values()):
        errors.append("baseline:prompt_hashes_incomplete")
    if set(candidate_prompt_hashes) != expected_ids or any(not value for value in candidate_prompt_hashes.values()):
        errors.append("candidate:prompt_hashes_incomplete")
    if baseline_prompt_hashes != candidate_prompt_hashes:
        errors.append("paired:prompt_hash_mismatch")
    return {"passed": not errors, "errors": errors}


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _label(text: str) -> str:
    text = _strip_thinking(text).strip().lower()
    labels = {
        "积极": "positive", "正面": "positive", "positive": "positive", "yes": "yes",
        "消极": "negative", "负面": "negative", "negative": "negative", "no": "no",
        "中性": "neutral", "中立": "neutral", "neutral": "neutral",
    }
    for token, value in labels.items():
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", text):
            return value
    return "unknown"


def deterministic_task_score(case: dict[str, Any], answer: str) -> tuple[float, dict[str, Any]]:
    scoring = case.get("scoring") or {}
    mode = scoring.get("mode", "patterns")
    clean_answer = _strip_thinking(answer)
    if mode == "label":
        accepted = {str(item).lower() for item in scoring.get("accepted_labels", [])}
        predicted = _label(clean_answer)
        score = 1.0 if predicted in accepted else 0.0
        return score, {"mode": mode, "predicted_label": predicted, "accepted_labels": sorted(accepted)}
    if mode == "exact":
        accepted = {re.sub(r"\s+", "", str(item)).lower() for item in scoring.get("accepted_answers", [])}
        predicted = re.sub(r"\s+", "", clean_answer).lower()
        score = 1.0 if predicted in accepted else 0.0
        return score, {"mode": mode, "exact_match": bool(score)}

    required = [str(item) for item in scoring.get("required_patterns", [])]
    forbidden = [str(item) for item in scoring.get("forbidden_patterns", [])]
    required_hits = [bool(re.search(pattern, clean_answer, re.IGNORECASE)) for pattern in required]
    forbidden_hits = [pattern for pattern in forbidden if re.search(pattern, clean_answer, re.IGNORECASE)]
    coverage = sum(required_hits) / len(required_hits) if required_hits else 1.0
    score = coverage if not forbidden_hits else 0.0
    return round(score, 4), {
        "mode": mode,
        "required_coverage": round(coverage, 4),
        "forbidden_hits": forbidden_hits,
    }


def _evidence_from_case(case: dict[str, Any]) -> list[EvidenceRecord]:
    records = []
    for index, raw in enumerate(case.get("evidence", []), start=1):
        item = dict(raw)
        quote = str(item.get("exact_quote", ""))
        item.setdefault("evidence_id", f"E{index:010d}")
        item.setdefault("source_uri", "")
        item.setdefault("content_hash", content_digest(quote))
        item.setdefault("fetched_at", case.get("request_as_of", ""))
        records.append(EvidenceRecord(**item))
    return records


def _calculations_from_case(case: dict[str, Any]) -> list[CalculationRecord]:
    records = []
    for raw in case.get("calculations", []):
        item = dict(raw)
        item["evidence_ids"] = tuple(item.get("evidence_ids", ()))
        records.append(CalculationRecord(**item))
    return records


def validate_cases(cases: list[dict[str, Any]]) -> None:
    errors = []
    for index, case in enumerate(cases):
        prefix = str(case.get("id") or index)
        if not get_question(case):
            errors.append(f"{prefix}:missing_question")
        if case.get("task_type") not in TARGET_TASKS:
            errors.append(f"{prefix}:invalid_task_type")
        if not case.get("source_group"):
            errors.append(f"{prefix}:missing_source_group")
        if parse_timestamp(case.get("request_as_of")) is None:
            errors.append(f"{prefix}:invalid_request_as_of")
        mode = (case.get("scoring") or {}).get("mode")
        if mode not in {"exact", "label", "patterns"}:
            errors.append(f"{prefix}:invalid_scoring_mode")
        evidence = _evidence_from_case(case)
        if case.get("requires_audit", bool(evidence)) and not evidence:
            errors.append(f"{prefix}:audit_evidence_missing")
        for item in evidence:
            errors.extend(f"{prefix}:{item.evidence_id}:{error}" for error in item.validation_errors())
        for item in _calculations_from_case(case):
            errors.extend(f"{prefix}:{item.calculation_id}:{error}" for error in item.validation_errors({e.evidence_id for e in evidence}))
    if errors:
        raise ValueError("invalid trusted evaluation set: " + ", ".join(errors[:20]))


def score_case(case: dict[str, Any], answer: str) -> dict[str, Any]:
    task_score, task_details = deterministic_task_score(case, answer)
    evidence = _evidence_from_case(case)
    calculations = _calculations_from_case(case)
    final_answer = _strip_thinking(answer)
    claims = build_claims_from_answer(final_answer, request_as_of=str(case.get("request_as_of", "")))
    requires_audit = bool(case.get("requires_audit", bool(evidence)))
    if requires_audit:
        audit = compute_auditable_reward(
            query=get_question(case),
            answer=final_answer,
            evidence=evidence,
            claims=claims,
            calculations=calculations,
            task_type=str(case.get("task_type", "financial_qa")),
            request_as_of=str(case.get("request_as_of", "")),
            trajectory=[],
            config=RewardV2Config(require_complete_trajectory=False),
        )
        audit_pass = bool(audit["hard_gate_passed"])
        audit_score = float(audit["base_reward"] if audit_pass else 0.0)
    else:
        audit = {"hard_gate_passed": True, "hard_failures": [], "total_reward": 1.0}
        audit_pass = True
        audit_score = 1.0
    primary_score = 0.7 * task_score + 0.3 * audit_score
    passed = task_score >= float((case.get("scoring") or {}).get("pass_threshold", 1.0)) and audit_pass
    return {
        "id": case.get("id"),
        "task_type": case.get("task_type"),
        "task_score": round(task_score, 4),
        "audit_score": round(audit_score, 4),
        "primary_score": round(primary_score, 4),
        "passed": passed,
        "task_details": task_details,
        "audit_hard_gate_passed": audit_pass,
        "audit_failures": audit.get("hard_failures", []),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def paired_bootstrap_interval(differences: list[float], seed: int = 42, iterations: int = 5000) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        estimates.append(mean(rng.choice(differences) for _ in differences))
    estimates.sort()
    return [round(estimates[int(0.025 * iterations)], 4), round(estimates[min(iterations - 1, int(0.975 * iterations))], 4)]


def paired_cluster_bootstrap_interval(
    grouped_differences: dict[str, list[float]],
    seed: int = 42,
    iterations: int = 5000,
) -> list[float]:
    """Bootstrap source groups so repeated questions for an issuer stay correlated."""
    groups = sorted(grouped_differences)
    if not groups:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled = [rng.choice(groups) for _ in groups]
        values = [value for group in sampled for value in grouped_differences[group]]
        estimates.append(mean(values))
    estimates.sort()
    return [round(estimates[int(0.025 * iterations)], 4), round(estimates[min(iterations - 1, int(0.975 * iterations))], 4)]


def evidence_source_identities(case: dict[str, Any]) -> set[str]:
    return {
        f"{item.get('canonical_url') or item.get('source_uri', '')}|{item.get('document_version', '')}"
        for item in case.get("evidence", [])
        if item.get("canonical_url") or item.get("source_uri")
    }


def training_hashes(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    rows = _load_json(path)
    question_hashes: set[str] = set()
    source_groups: set[str] = set()
    for row in rows:
        if row.get("query_fingerprint"):
            question_hashes.add(str(row["query_fingerprint"]))
        question = get_question(row)
        if question:
            question_hashes.add(normalized_hash(question))
            try:
                structured = json.loads(question)
            except (TypeError, json.JSONDecodeError):
                structured = {}
            if isinstance(structured, dict) and structured.get("query"):
                question_hashes.add(normalized_hash(str(structured["query"])))
        if row.get("source_group"):
            source_groups.add(str(row["source_group"]))
        source_groups.update(str(item) for item in row.get("source_groups", []) if item)
    return question_hashes, source_groups


def evaluate(
    cases: list[dict[str, Any]],
    baseline: dict[str, str],
    candidate: dict[str, str],
    *,
    train_question_hashes: set[str],
    train_source_groups: set[str],
    min_samples_per_task: int = 0,
) -> dict[str, Any]:
    validate_cases(cases)
    ids = [str(case.get("id", "")) for case in cases]
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("gold case IDs must be unique and non-empty")
    missing_baseline = sorted(set(ids) - set(baseline))
    missing_candidate = sorted(set(ids) - set(candidate))
    if missing_baseline or missing_candidate:
        raise ValueError(f"missing predictions: baseline={missing_baseline[:5]}, candidate={missing_candidate[:5]}")

    leaks = []
    baseline_rows = []
    candidate_rows = []
    for case in cases:
        identifier = str(case["id"])
        question_leak = normalized_hash(get_question(case)) in train_question_hashes
        case_source_identities = evidence_source_identities(case)
        overlapping_source_identities = sorted(case_source_identities & train_source_groups)
        source_leak = bool(
            overlapping_source_identities
            or (case.get("source_group") and str(case["source_group"]) in train_source_groups)
        )
        if question_leak or source_leak:
            leaks.append(
                {
                    "id": identifier,
                    "question_overlap": question_leak,
                    "source_group_overlap": source_leak,
                    "overlapping_source_identities": overlapping_source_identities,
                }
            )
        baseline_rows.append(score_case(case, baseline[identifier]))
        candidate_rows.append(score_case(case, candidate[identifier]))

    differences = [candidate_row["primary_score"] - baseline_row["primary_score"] for baseline_row, candidate_row in zip(baseline_rows, candidate_rows)]
    baseline_passes = sum(row["passed"] for row in baseline_rows)
    candidate_passes = sum(row["passed"] for row in candidate_rows)
    regressions = sum(base["passed"] and not cand["passed"] for base, cand in zip(baseline_rows, candidate_rows))
    improvements = sum(not base["passed"] and cand["passed"] for base, cand in zip(baseline_rows, candidate_rows))
    by_task: dict[str, dict[str, Any]] = {}
    grouped = defaultdict(list)
    for base, cand in zip(baseline_rows, candidate_rows):
        grouped[str(cand["task_type"])].append((base, cand))
    for task, rows in sorted(grouped.items()):
        task_cases = [case for case in cases if str(case.get("task_type")) == task]
        baseline_primary_score = mean(base["primary_score"] for base, _ in rows)
        candidate_primary_score = mean(cand["primary_score"] for _, cand in rows)
        by_task[task] = {
            "samples": len(rows),
            "unique_questions": len({normalized_hash(get_question(case)) for case in task_cases}),
            "source_groups": len({str(case.get("source_group", "")) for case in task_cases}),
            "baseline_pass_rate": round(mean(float(base["passed"]) for base, _ in rows), 4),
            "candidate_pass_rate": round(mean(float(cand["passed"]) for _, cand in rows), 4),
            "candidate_audit_pass_rate": round(mean(float(cand["audit_hard_gate_passed"]) for _, cand in rows), 4),
            "baseline_mean_primary_score": round(baseline_primary_score, 4),
            "candidate_mean_primary_score": round(candidate_primary_score, 4),
            "primary_score_delta": round(candidate_primary_score - baseline_primary_score, 4),
        }
    delta_interval = paired_bootstrap_interval(differences)
    source_grouped_differences: dict[str, list[float]] = defaultdict(list)
    for case, difference in zip(cases, differences):
        source_grouped_differences[str(case.get("source_group") or case.get("id"))].append(difference)
    clustered_delta_interval = paired_cluster_bootstrap_interval(source_grouped_differences)
    task_sample_counts = Counter(str(case.get("task_type", "")) for case in cases)
    task_coverage_ok = all(task_sample_counts.get(task, 0) >= min_samples_per_task for task in TARGET_TASKS)
    unique_question_coverage_ok = all(
        row["unique_questions"] >= min_samples_per_task for row in by_task.values()
    )
    per_task_non_regression = all(row["candidate_pass_rate"] >= row["baseline_pass_rate"] for row in by_task.values())
    per_task_primary_noninferiority = all(
        row["primary_score_delta"] >= -PER_TASK_PRIMARY_NONINFERIORITY_MARGIN
        for row in by_task.values()
    )
    cyrillic_candidate_ids = sorted(
        identifier for identifier, answer in candidate.items() if CYRILLIC_RE.search(answer)
    )
    candidate_audit_pass_rate = mean(float(row["audit_hard_gate_passed"]) for row in candidate_rows)
    baseline_audit_pass_rate = mean(float(row["audit_hard_gate_passed"]) for row in baseline_rows)
    return {
        "evaluation_version": "trusted_sft_v2.0",
        "dataset_fingerprint": dataset_fingerprint(cases),
        "samples": len(cases),
        "release_gate_thresholds": {
            "per_task_primary_noninferiority_margin": PER_TASK_PRIMARY_NONINFERIORITY_MARGIN,
            "paired_primary_noninferiority_margin": 0.02,
            "minimum_audit_pass_rate": 0.95,
            "maximum_cyrillic_outputs": 0,
        },
        "data_leakage": {"count": len(leaks), "cases": leaks},
        "baseline": {
            "pass_rate": round(baseline_passes / len(cases), 4),
            "pass_rate_95ci": wilson_interval(baseline_passes, len(cases)),
            "audit_pass_rate": round(baseline_audit_pass_rate, 4),
            "mean_primary_score": round(mean(row["primary_score"] for row in baseline_rows), 4),
        },
        "candidate": {
            "pass_rate": round(candidate_passes / len(cases), 4),
            "pass_rate_95ci": wilson_interval(candidate_passes, len(cases)),
            "audit_pass_rate": round(candidate_audit_pass_rate, 4),
            "mean_primary_score": round(mean(row["primary_score"] for row in candidate_rows), 4),
            "cyrillic_output_count": len(cyrillic_candidate_ids),
            "cyrillic_output_ids": cyrillic_candidate_ids,
        },
        "paired_comparison": {
            "mean_primary_score_delta": round(mean(differences), 4),
            "delta_95ci": delta_interval,
            "source_group_clustered_delta_95ci": clustered_delta_interval,
            "source_group_clusters": len(source_grouped_differences),
            "improvements": improvements,
            "regressions": regressions,
        },
        "by_task": by_task,
        "release_gate": {
            "no_detected_leakage": not leaks,
            "minimum_task_coverage": task_coverage_ok,
            "minimum_unique_question_coverage": unique_question_coverage_ok,
            "no_pass_rate_regression": candidate_passes >= baseline_passes,
            "per_task_non_regression": per_task_non_regression,
            "per_task_primary_score_noninferiority": per_task_primary_noninferiority,
            "paired_noninferiority_95ci": clustered_delta_interval[0] >= -0.02,
            "audit_pass_at_least_95pct": candidate_audit_pass_rate >= 0.95,
            "no_audit_regression": candidate_audit_pass_rate >= baseline_audit_pass_rate,
            "no_more_regressions_than_improvements": regressions <= improvements,
            "no_cyrillic_candidate_outputs": not cyrillic_candidate_ids,
        },
        "details": {"baseline": baseline_rows, "candidate": candidate_rows},
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    base = report["baseline"]
    candidate = report["candidate"]
    paired = report["paired_comparison"]
    lines = [
        "# Trusted SFT V2 Evaluation",
        "",
        f"- Dataset fingerprint: `{report['dataset_fingerprint']}`",
        f"- Samples: {report['samples']}",
        f"- Leakage cases: {report['data_leakage']['count']}",
        f"- Baseline pass rate: {base['pass_rate']} (95% CI {base['pass_rate_95ci']})",
        f"- Candidate pass rate: {candidate['pass_rate']} (95% CI {candidate['pass_rate_95ci']})",
        f"- Paired score delta: {paired['mean_primary_score_delta']} (95% CI {paired['delta_95ci']})",
        f"- Source-group clustered 95% CI: {paired['source_group_clustered_delta_95ci']} ({paired['source_group_clusters']} groups)",
        f"- Paired improvements/regressions: {paired['improvements']}/{paired['regressions']}",
        f"- Candidate Cyrillic outputs: {candidate['cyrillic_output_count']}",
        f"- Prediction protocol valid: {report.get('prediction_protocol', {}).get('passed', False)}",
        "",
        "| Task | N | Unique queries | Source groups | Baseline pass | Candidate pass | Candidate audit | Baseline score | Candidate score | Score delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, row in report["by_task"].items():
        lines.append(
            f"| {task} | {row['samples']} | {row['unique_questions']} | {row['source_groups']} | "
            f"{row['baseline_pass_rate']} | {row['candidate_pass_rate']} | "
            f"{row['candidate_audit_pass_rate']} | {row['baseline_mean_primary_score']} | "
            f"{row['candidate_mean_primary_score']} | {row['primary_score_delta']} |"
        )
    gates = report["release_gate"]
    lines.extend(("", "## Release Gates", ""))
    lines.extend(f"- {name}: {passed}" for name, passed in gates.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired trusted evaluation for SFT v2")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--train-file", "--train-data", dest="train_file")
    parser.add_argument("--min-samples-per-task", type=int, default=50)
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_trusted.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_trusted.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = _load_json(Path(args.gold))
    baseline, baseline_manifest = load_predictions(Path(args.baseline_predictions))
    candidate, candidate_manifest = load_predictions(Path(args.candidate_predictions))
    protocol = validate_prediction_protocol(
        cases,
        baseline,
        candidate,
        baseline_manifest,
        candidate_manifest,
    )
    train_questions, train_sources = training_hashes(Path(args.train_file) if args.train_file else None)
    report = evaluate(
        cases,
        baseline,
        candidate,
        train_question_hashes=train_questions,
        train_source_groups=train_sources,
        min_samples_per_task=args.min_samples_per_task,
    )
    report["prediction_protocol"] = protocol
    report["release_gate"]["paired_protocol_valid"] = protocol["passed"]
    output_json = Path(args.output_json)
    output_md = Path(args.output)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output_md)
    print(json.dumps({"samples": report["samples"], **report["release_gate"]}, ensure_ascii=False))
    return 0 if all(report["release_gate"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
