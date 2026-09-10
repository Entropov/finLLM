#!/usr/bin/env python3
"""Diagnose the v2.8 stock-contract seen preflight without release claims."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.diagnose_sft_v2_8_validator import _confidence_diagnostic, _confusion  # noqa: E402
from scripts.evaluation.replay_sft_v2_5_selector import audit_selector_view  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "saves/eval_results/sft_v2_8_stock_contract_seen_preflight_e2e.json"
DEFAULT_SELECTOR = PROJECT_ROOT / "saves/eval_results/sft_v2_8_stock_contract_seen_preflight_task_selector.json"
DEFAULT_GOLD = PROJECT_ROOT / "data/evaluation/sft_v2_8_stock_contract_seen_preflight.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "saves/eval_results/sft_v2_8_stock_contract_seen_preflight_diagnostic.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _eligible(score: dict[str, Any], validation: dict[str, Any]) -> bool:
    audit = audit_selector_view(score)
    return bool(
        audit["audit_hard_gate_passed"]
        and not audit["protocol_errors"]
        and not audit["audit_failures"]
        and validation.get("hard_gate_passed")
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _by_challenge(
    details: list[dict[str, Any]], selector: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    selection_by_id = {str(item["id"]): item for item in selector["details"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        tags = cases[str(item["id"])].get("challenge_tags", [])
        grouped["adversarial" if "instruction_conflict" in tags else "trusted_style"].append(item)
    report = {}
    for name, rows in sorted(grouped.items()):
        selection_rows = [selection_by_id[str(row["id"])] for row in rows]
        report[name] = {
            "samples": len(rows),
            "raw_e2e_at_1": _ratio(sum(bool(row["candidate_score_at_1"].get("passed")) for row in rows), len(rows)),
            "raw_e2e_at_8": _ratio(sum(bool(row.get("candidate_pass_at_8")) for row in rows), len(rows)),
            "selector_e2e_at_8": _ratio(sum(bool(row.get("selected_passed")) for row in selection_rows), len(rows)),
            "selector_false_accept_count": sum(bool(row.get("selected")) and not bool(row.get("selected_passed")) for row in selection_rows),
            "selector_rejection_count": sum(not bool(row.get("selected")) for row in selection_rows),
        }
    return report


def diagnose(e2e: dict[str, Any], selector: dict[str, Any], gold: list[dict[str, Any]]) -> dict[str, Any]:
    details = (e2e.get("evaluation") or {}).get("details") or []
    selection_by_id = {str(item["id"]): item for item in selector.get("details", [])}
    cases = {str(case["id"]): case for case in gold}
    if len(details) != len(selection_by_id) or set(selection_by_id) != set(cases):
        raise ValueError("E2E, selector, and gold case IDs must match")

    pool_rows: list[tuple[bool, bool]] = []
    correct_reject_reasons: Counter[str] = Counter()
    wrong_accept_reasons: Counter[str] = Counter()
    wrong_reject_validator_reasons: Counter[str] = Counter()
    wrong_reject_audit_reasons: Counter[str] = Counter()
    wrong_reject_candidate_taxonomy: Counter[str] = Counter()
    for item in details:
        selected = selection_by_id[str(item["id"])]
        scores = item.get("candidate_scores_at_8") or []
        validations = selected.get("candidate_task_validations") or []
        if len(scores) != 8 or len(validations) != 8:
            raise ValueError(f"expected eight aligned candidates for {item['id']}")
        for score, validation in zip(scores, validations):
            correct = bool(score.get("passed"))
            accepted = _eligible(score, validation)
            pool_rows.append((correct, accepted))
            if correct and not accepted:
                correct_reject_reasons.update(validation.get("missing_anchors") or [])
                correct_reject_reasons.update(validation.get("failures") or [])
                if not validation.get("hard_gate_passed") and not validation.get("missing_anchors") and not validation.get("failures"):
                    correct_reject_reasons["audit_gate"] += 1
            elif not correct and accepted:
                wrong_accept_reasons.update(validation.get("missing_anchors") or [])
            elif not correct:
                wrong_reject_validator_reasons.update(validation.get("missing_anchors") or [])
                wrong_reject_validator_reasons.update(validation.get("failures") or [])
                audit = audit_selector_view(score)
                categories = set(validation.get("missing_anchors") or [])
                failures = list(validation.get("failures") or [])
                if any(reason.endswith("missing_citation") for reason in failures):
                    categories.add("citation_incomplete")
                if "unsupported_forward_prediction" in failures:
                    categories.add("unsupported_forward_prediction")
                if not audit["audit_hard_gate_passed"]:
                    wrong_reject_audit_reasons.update(score.get("audit_failures") or ["audit_gate"])
                wrong_reject_audit_reasons.update(score.get("protocol_errors") or [])
                audit_failures = set(score.get("audit_failures") or [])
                if "unsupported_claim" in audit_failures:
                    categories.add("unsupported_claim")
                if "numeric_mismatch" in audit_failures:
                    categories.add("numeric_mismatch")
                if "insufficient_citation_coverage" in audit_failures:
                    categories.add("insufficient_citation_coverage")
                wrong_reject_candidate_taxonomy.update(categories or {"audit_or_contract_failure"})
    summary = (e2e.get("evaluation") or {}).get("summary") or {}
    pool_confusion = _confusion(pool_rows)
    wrong_rejected_total = int(pool_confusion["true_negative"])
    overlap = {
        "seen_contract_preflight": True,
        "not_final_holdout": True,
        "reason": "The gold report records 50/50 document overlap with the trusted regression.",
    }
    return {
        "diagnostic_version": "sft_v2.8_stock_contract_seen_preflight.v1",
        "evaluation_identity": overlap,
        "input_summary": summary,
        "candidate_pool_confusion": pool_confusion,
        "selector": selector.get("summary", {}).get("selector", {}),
        "by_prompt_bucket": _by_challenge(details, selector, cases),
        "correct_rejected_by_reason": dict(sorted(correct_reject_reasons.items())),
        "wrong_accepted_by_reason": dict(sorted(wrong_accept_reasons.items())),
        "wrong_rejected_by_validator_reason": dict(sorted(wrong_reject_validator_reasons.items())),
        "wrong_rejected_by_audit_reason": dict(sorted(wrong_reject_audit_reasons.items())),
        "wrong_rejected_candidate_taxonomy": {
            key: {"candidate_count": count, "candidate_share": _ratio(count, wrong_rejected_total)}
            for key, count in sorted(wrong_reject_candidate_taxonomy.items())
        },
        "confidence_nll": _confidence_diagnostic(e2e, selector),
        "decision": (
            "FAIL: selector has no observed false acceptance and recovers all contract-correct candidates, "
            "but raw candidate E2E@8/audit@8 are below preregistered thresholds, @1 primary is inferior, "
            "and the evidence is seen regression data. Do not start RL."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose v2.8 stock contract preflight")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--selector", default=str(DEFAULT_SELECTOR))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = diagnose(_load(Path(args.input)), _load(Path(args.selector)), _load(Path(args.gold)))
    _write(Path(args.output), report)
    print(json.dumps({"output": args.output, "candidate_pool": report["candidate_pool_confusion"], "selector": report["selector"], "decision": report["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
