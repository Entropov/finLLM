#!/usr/bin/env python3
"""Produce v2.8 validator calibration diagnostics from immutable v2.7 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.sft_v2_7_scope import CORE_TASKS  # noqa: E402


DIAGNOSTIC_VERSION = "sft_v2.8_validator_calibration_diagnostic.v1"
CORE_BUCKETS = ("financial_qa", "stock_analysis", "quant_strategy")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _bucket(task_type: str) -> str:
    return task_type if task_type in CORE_BUCKETS else "other_existing_tasks"


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _confusion(rows: list[tuple[bool, bool]]) -> dict[str, Any]:
    """Rows are (gold correctness, validator/policy acceptance)."""
    tp = sum(correct and accepted for correct, accepted in rows)
    fp = sum(not correct and accepted for correct, accepted in rows)
    tn = sum(not correct and not accepted for correct, accepted in rows)
    fn = sum(correct and not accepted for correct, accepted in rows)
    return {
        "samples": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "false_acceptance_rate": _ratio(fp, fp + tn),
        "false_rejection_rate": _ratio(fn, fn + tp),
        "acceptance_rate": _ratio(tp + fp, len(rows)),
    }


def _audit_eligible(score: dict[str, Any]) -> bool:
    return bool(score.get("audit_hard_gate_passed")) and not score.get("protocol_errors") and not score.get("audit_failures")


def _group_confusion(rows: list[tuple[str, bool, bool]]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for task_type, correct, accepted in rows:
        grouped[_bucket(task_type)].append((correct, accepted))
    output = {bucket: _confusion(grouped[bucket]) for bucket in (*CORE_BUCKETS, "other_existing_tasks") if bucket in grouped}
    output["overall"] = _confusion([(correct, accepted) for _, correct, accepted in rows])
    return output


def _taxonomy(detail: dict[str, Any]) -> tuple[str, str, str]:
    failures = set(detail.get("chosen_failures") or [])
    missing = set(detail.get("chosen_missing_anchors") or [])
    task_type = str(detail.get("task_type", ""))
    if "visible_evidence_missing" in failures:
        return "data_annotation_issue", "data", "chosen_prompt_has_no_visible_evidence"
    if any(reason.endswith("missing_citation") for reason in failures):
        return "citation_incomplete", "data_or_model", "atomic_claim_without_evidence_id"
    if any("invalid_evidence" in reason for reason in failures):
        return "claim_evidence_mismatch", "model_or_data", "citation_does_not_resolve_to_visible_evidence"
    if "nonempty_thinking" in failures:
        return "format_or_thinking", "model", "nonempty_thinking_tag"
    if "unsupported_forward_prediction" in failures:
        return "true_model_error", "model", "unsupported_forward_prediction"
    if any("quant_action" in reason for reason in failures) or "canonical_quant_action" in missing:
        return "json_or_artifact_format", "model", "canonical_quant_action_missing_or_invalid"
    if task_type == "stock_analysis" and "market_trend" in missing:
        return "stock_specific_rule_overstrict", "validator_or_data_contract", "chosen_target_omits_trend_anchor"
    if missing:
        return "evidence_anchor_not_satisfied", "validator_or_model", ";".join(sorted(missing))
    return "validator_implementation_bug", "validator", "rejected_without_recorded_failure_or_missing_anchor"


def _write_taxonomy_csv(path: Path, preference_report: dict[str, Any]) -> dict[str, Any]:
    rows = [item for item in preference_report.get("details", []) if not item.get("chosen_hard_gate_passed")]
    fieldnames = ["id", "task_type", "negative_type", "taxonomy", "ownership", "reason_detail", "chosen_failures", "chosen_missing_anchors"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            taxonomy, ownership, reason = _taxonomy(item)
            writer.writerow(
                {
                    "id": item.get("id"),
                    "task_type": item.get("task_type"),
                    "negative_type": item.get("negative_type"),
                    "taxonomy": taxonomy,
                    "ownership": ownership,
                    "reason_detail": reason,
                    "chosen_failures": ";".join(item.get("chosen_failures") or []),
                    "chosen_missing_anchors": ";".join(item.get("chosen_missing_anchors") or []),
                }
            )
    counts = Counter(_taxonomy(item)[0] for item in rows)
    return {
        "rejected_chosen_count": len(rows),
        "by_taxonomy": {
            key: {"count": value, "share": _ratio(value, len(rows))} for key, value in sorted(counts.items())
        },
    }


def _candidate_pool(e2e: dict[str, Any], selector: dict[str, Any]) -> tuple[list[tuple[str, bool, bool]], dict[str, Any]]:
    selector_by_id = {str(item["id"]): item for item in selector.get("details", [])}
    rows: list[tuple[str, bool, bool]] = []
    stock_rows = []
    for source in (e2e.get("evaluation") or {}).get("details", []):
        identifier = str(source.get("id", ""))
        selected = selector_by_id.get(identifier)
        if selected is None:
            raise ValueError(f"selector result missing case: {identifier}")
        task_type = str(source.get("task_type", ""))
        scores = source.get("candidate_scores_at_8") or []
        validations = selected.get("candidate_task_validations") or []
        if len(scores) != len(validations):
            raise ValueError(f"unaligned candidates: {identifier}")
        correct_accepts = 0
        correct_rejects = 0
        wrong_accepts = 0
        wrong_rejects = 0
        correct_rejection_reasons: list[str] = []
        wrong_accept_reasons: list[str] = []
        for score, validation in zip(scores, validations):
            correct = bool(score.get("passed"))
            audit_accepted = _audit_eligible(score)
            task_accepted = bool(validation.get("hard_gate_passed"))
            accepted = audit_accepted and task_accepted
            rows.append((task_type, correct, accepted))
            if correct and accepted:
                correct_accepts += 1
            elif correct:
                correct_rejects += 1
                if not task_accepted:
                    correct_rejection_reasons.extend(validation.get("missing_anchors") or [])
                    correct_rejection_reasons.extend(validation.get("failures") or [])
                elif not audit_accepted:
                    correct_rejection_reasons.append("audit_gate")
            elif accepted:
                wrong_accepts += 1
                wrong_accept_reasons.extend(validation.get("missing_anchors") or [])
            else:
                wrong_rejects += 1
        if task_type == "stock_analysis":
            greedy = source.get("candidate_score_at_1") or {}
            stock_rows.append(
                {
                    "raw_correct_at_1": bool(greedy.get("passed")),
                    "raw_correct_at_8": any(bool(score.get("passed")) for score in scores),
                    "correct_accept_candidates": correct_accepts,
                    "correct_reject_candidates": correct_rejects,
                    "wrong_accept_candidates": wrong_accepts,
                    "wrong_reject_candidates": wrong_rejects,
                    "correct_rejection_reasons": correct_rejection_reasons,
                    "wrong_accept_reasons": wrong_accept_reasons,
                    "selected_passed": bool(selected.get("selected_passed")),
                    "selected": bool(selected.get("selected")),
                }
            )
    return rows, {"stock_candidate_rows": stock_rows}


def _stock_report(stock_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(stock_rows)
    total_correct_candidates = sum(row["correct_accept_candidates"] + row["correct_reject_candidates"] for row in stock_rows)
    total_wrong_candidates = sum(row["wrong_accept_candidates"] + row["wrong_reject_candidates"] for row in stock_rows)
    correct_rejected = sum(row["correct_reject_candidates"] for row in stock_rows)
    wrong_accepted = sum(row["wrong_accept_candidates"] for row in stock_rows)
    correct_rejection_reasons = Counter(
        reason for row in stock_rows for reason in row["correct_rejection_reasons"]
    )
    return {
        "samples": total,
        "model_raw_correctness_at_1": _ratio(sum(row["raw_correct_at_1"] for row in stock_rows), total),
        "model_raw_correctness_at_8": _ratio(sum(row["raw_correct_at_8"] for row in stock_rows), total),
        "validator_accept_rate_candidate_pool": _ratio(
            sum(row["correct_accept_candidates"] + row["wrong_accept_candidates"] for row in stock_rows),
            total_correct_candidates + total_wrong_candidates,
        ),
        "correct_rejected_candidate_count": correct_rejected,
        "correct_rejected_candidate_rate": _ratio(correct_rejected, total_correct_candidates),
        "wrong_accepted_candidate_count": wrong_accepted,
        "wrong_accepted_candidate_rate": _ratio(wrong_accepted, total_wrong_candidates),
        "false_acceptance_rate": _ratio(wrong_accepted, total_wrong_candidates),
        "false_rejection_rate": _ratio(correct_rejected, total_correct_candidates),
        "selector_e2e_at_8": _ratio(sum(row["selected_passed"] for row in stock_rows), total),
        "selector_selection_rate": _ratio(sum(row["selected"] for row in stock_rows), total),
        "correct_rejected_by_reason": dict(sorted(correct_rejection_reasons.items())),
        "diagnosis": "validator_calibration_or_protocol_blocker"
        if total_correct_candidates and _ratio(correct_rejected, total_correct_candidates) >= 0.5
        else "model_generation_or_task_construction_blocker",
    }


def _selector_confusion(selector: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item in selector.get("details", []):
        accepted = bool(item.get("selected"))
        correct = bool(item.get("selected_passed")) if accepted else not bool(item.get("oracle_pass_at_8"))
        rows.append((str(item.get("task_type", "")), correct, accepted))
    return _group_confusion(rows)


def _primary_delta(e2e: dict[str, Any]) -> dict[str, float]:
    details = (e2e.get("evaluation") or {}).get("details") or []
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in details:
        candidate = item.get("candidate_score_at_1") or {}
        baseline = item.get("baseline_score") or {}
        grouped[str(item.get("task_type", ""))].append(
            float(candidate.get("primary_score", 0.0)) - float(baseline.get("primary_score", 0.0))
        )
    return {task: round(sum(values) / len(values), 4) for task, values in sorted(grouped.items()) if values}


def _mean_nll(rows: list[float]) -> float | None:
    return round(mean(rows), 6) if rows else None


def _confidence_diagnostic(e2e: dict[str, Any], selector: dict[str, Any]) -> dict[str, Any]:
    details = (e2e.get("evaluation") or {}).get("details") or []
    selector_by_id = {str(item.get("id", "")): item for item in selector.get("details", [])}
    records: list[tuple[bool, bool, float]] = []
    for item in details:
        likelihood = item.get("candidate_sequence_logprob_at_1") or {}
        nll = likelihood.get("sequence_normalized_nll")
        candidate_score = item.get("candidate_score_at_1") or {}
        selection = selector_by_id.get(str(item.get("id", "")))
        if nll is None or selection is None:
            continue
        validations = selection.get("candidate_task_validations") or []
        task_accepted = bool(validations and validations[0].get("hard_gate_passed"))
        accepted = _audit_eligible(candidate_score) and task_accepted
        records.append((bool(candidate_score.get("passed")), accepted, float(nll)))
    if not records:
        return {
            "status": "unavailable_from_existing_generation_artifacts",
            "coverage": 0.0,
            "available_probability_fields": [],
            "mean_token_entropy": None,
            "sequence_normalized_logprob_or_nll": None,
            "correct_vs_incorrect_entropy": None,
            "correct_accept_vs_correct_reject_entropy": None,
            "wrong_accept_vs_wrong_reject_entropy": None,
            "pass_at_1_at_8_relation": "E2E@1/@8 is available, but no confidence relationship can be inferred without sequence likelihoods.",
            "next_collection_requirement": "Rerun fixed evaluation with --collect-token-logprobs; preserve candidate IDs, sampling seeds, and selector replay.",
        }
    correct = [nll for passed, _, nll in records if passed]
    incorrect = [nll for passed, _, nll in records if not passed]
    correct_accept = [nll for passed, accepted, nll in records if passed and accepted]
    correct_reject = [nll for passed, accepted, nll in records if passed and not accepted]
    wrong_accept = [nll for passed, accepted, nll in records if not passed and accepted]
    wrong_reject = [nll for passed, accepted, nll in records if not passed and not accepted]
    return {
        "status": "sequence_nll_available_entropy_unavailable",
        "coverage": _ratio(len(records), len(details)),
        "available_probability_fields": ["sequence_normalized_nll"],
        "mean_token_entropy": None,
        "sequence_normalized_logprob_or_nll": {"mean_nll": _mean_nll([nll for _, _, nll in records])},
        "correct_vs_incorrect_entropy": {"correct_mean_nll": _mean_nll(correct), "incorrect_mean_nll": _mean_nll(incorrect)},
        "correct_accept_vs_correct_reject_entropy": {"correct_accept_mean_nll": _mean_nll(correct_accept), "correct_reject_mean_nll": _mean_nll(correct_reject)},
        "wrong_accept_vs_wrong_reject_entropy": {"wrong_accept_mean_nll": _mean_nll(wrong_accept), "wrong_reject_mean_nll": _mean_nll(wrong_reject)},
        "pass_at_1_at_8_relation": "NLL is descriptive only. Compare it with E2E@1/@8 and false acceptance; low NLL is never treated as correctness.",
        "next_collection_requirement": "Collect full token distributions only if true entropy is required; selected-token NLL is sufficient for this diagnostic.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose v2.8 validator calibration")
    parser.add_argument("--trusted-e2e", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_trusted_regression_e2e.json"))
    parser.add_argument("--adversarial-e2e", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_adversarial_regression_e2e.json"))
    parser.add_argument("--trusted-selector", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_trusted_task_selector_replay.json"))
    parser.add_argument("--adversarial-selector", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_adversarial_task_selector_replay.json"))
    parser.add_argument("--preference-check", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_task_validator_preference_check.json"))
    parser.add_argument("--audit-trusted-selector", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_trusted_selector_replay.json"))
    parser.add_argument("--audit-adversarial-selector", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_7_core_adversarial_selector_replay.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "saves/eval_results/sft_v2_8_validator_calibration"))
    args = parser.parse_args()

    trusted_e2e = _load(Path(args.trusted_e2e))
    adversarial_e2e = _load(Path(args.adversarial_e2e))
    trusted_selector = _load(Path(args.trusted_selector))
    adversarial_selector = _load(Path(args.adversarial_selector))
    preference = _load(Path(args.preference_check))
    audit_trusted = _load(Path(args.audit_trusted_selector))
    audit_adversarial = _load(Path(args.audit_adversarial_selector))
    output_dir = Path(args.output_dir)

    trusted_pool, trusted_extra = _candidate_pool(trusted_e2e, trusted_selector)
    adversarial_pool, adversarial_extra = _candidate_pool(adversarial_e2e, adversarial_selector)
    preference_rows = []
    for item in preference.get("details", []):
        task_type = str(item.get("task_type", ""))
        preference_rows.extend(
            [
                (task_type, True, bool(item.get("chosen_hard_gate_passed"))),
                (task_type, False, bool(item.get("rejected_hard_gate_passed"))),
            ]
        )
    taxonomy = _write_taxonomy_csv(output_dir / "rejected_chosen_by_reason.csv", preference)
    report = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "scope": {"core_tasks": sorted(CORE_TASKS), "other_existing_tasks": "out_of_scope_for_v2.7_validator; no acceptance claim"},
        "confusion_matrix": {
            "trusted_candidate_pool": _group_confusion(trusted_pool),
            "adversarial_candidate_pool": _group_confusion(adversarial_pool),
            "trusted_task_selector": _selector_confusion(trusted_selector),
            "adversarial_task_selector": _selector_confusion(adversarial_selector),
            "preference_chosen_rejected": _group_confusion(preference_rows),
        },
        "taxonomy": taxonomy,
        "stock_analysis": {
            "trusted": _stock_report(trusted_extra["stock_candidate_rows"]),
            "adversarial": _stock_report(adversarial_extra["stock_candidate_rows"]),
        },
        "before_after_selector": {
            "trusted": {"audit_only": audit_trusted.get("summary", {}).get("selector", {}), "task_aware": trusted_selector.get("summary", {}).get("selector", {})},
            "adversarial": {"audit_only": audit_adversarial.get("summary", {}).get("selector", {}), "task_aware": adversarial_selector.get("summary", {}).get("selector", {})},
        },
        "hard_negative_regression": preference.get("summary", {}),
        "e2e_primary": {
            "trusted_primary_delta_by_task": _primary_delta(trusted_e2e),
            "adversarial_primary_delta_by_task": _primary_delta(adversarial_e2e),
            "trusted_e2e": trusted_e2e.get("evaluation", {}).get("summary", {}),
            "adversarial_e2e": adversarial_e2e.get("evaluation", {}).get("summary", {}),
        },
        "confidence_entropy": {
            "trusted": _confidence_diagnostic(trusted_e2e, trusted_selector),
            "adversarial": _confidence_diagnostic(adversarial_e2e, adversarial_selector),
        },
        "decision": "FAIL: task-aware false acceptance is zero on seen regression and current hard negatives, but preference chosen acceptance and adversarial stock recall fail calibration gates.",
    }
    _write_json(output_dir / "validator_confusion_matrix.json", report)
    _write_json(output_dir / "stock_analysis_diagnostic.json", report["stock_analysis"])
    _write_json(output_dir / "entropy_confidence_diagnostic.json", report["confidence_entropy"])
    print(json.dumps({"output_dir": str(output_dir), "taxonomy": taxonomy, "decision": report["decision"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
