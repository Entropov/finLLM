#!/usr/bin/env python3
"""Build evidence-complete, validator-aligned v2.8 core preference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.sft_v2_7_scope import CORE_TASKS  # noqa: E402
from scripts.evaluation.task_aware_verifier_v2_7_core import validate_task_answer  # noqa: E402
from scripts.rag.quant_protocol import canonical_quant_action  # noqa: E402


SCHEMA_VERSION = "sft_v2.8_validator_aligned_preference.v1"
DEFAULT_INPUT = PROJECT_ROOT / "data/sft_v2_7_core/fin_agentic_sft_v2_7_core_answer_train.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/rlhf/sft_v2_8_core_validator_preference.json"
DEFAULT_REPORT = PROJECT_ROOT / "data/rlhf/sft_v2_8_core_validator_preference_report.json"
_CITATION_RE = re.compile(r"\s*\[\s*E[0-9A-Fa-f]{10,64}\s*\]")


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return payload


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _prompt(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("conversations") or []
    text = next((str(item.get("value", "")) for item in messages if item.get("from") in {"human", "user"}), "")
    prompt = json.loads(text)
    if str(prompt.get("task_type")) != str(row.get("task_type")):
        raise ValueError(f"task mismatch: {row.get('id')}")
    return prompt


def _assistant(row: dict[str, Any]) -> str:
    messages = row.get("conversations") or []
    return next((str(item.get("value", "")) for item in messages if item.get("from") in {"gpt", "assistant"}), "")


def _stock_trend_line(prompt: dict[str, Any]) -> str:
    for evidence in prompt.get("evidence", []):
        quote = str(evidence.get("exact_quote", ""))
        ma5 = re.search(r"MA5为([^，。；;\s]+)", quote)
        ma20 = re.search(r"MA20为([^，。；;\s]+)", quote)
        evidence_id = str(evidence.get("evidence_id", "")).upper()
        if ma5 and ma20 and evidence_id:
            return f"- MA5为{ma5.group(1)}，MA20为{ma20.group(1)} [{evidence_id}]。"
    raise ValueError("stock prompt lacks MA5/MA20 evidence")


def _chosen(task_type: str, prompt: dict[str, Any], answer: str) -> str:
    if task_type == "quant_strategy":
        return canonical_quant_action()
    if task_type == "stock_analysis":
        trend = _stock_trend_line(prompt)
        if "MA5" not in answer and "MA20" not in answer:
            return f"{answer.rstrip()}\n{trend}"
    return answer.strip()


def _remove_citation_from_field(text: str, field: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if field in line and _CITATION_RE.search(line):
            lines[index] = _CITATION_RE.sub("", line, count=1)
            return "\n".join(lines)
    raise ValueError(f"cannot build citation-drop negative for {field}")


def _negative(task_type: str, chosen: str, seed: str) -> tuple[str, str, list[str]]:
    if task_type == "quant_strategy":
        variants = (
            ('{"action":"render_quant_artifact","artifact_version":"quant.v2.5"}', "json_or_artifact_format", ["quant_action_schema"]),
            (chosen + "\n说明", "json_or_artifact_format", ["quant_action_invalid_json"]),
            ('{"action":"generate_python","artifact_version":"quant.v2.5","template_id":"ma5_ma20_long_only"}', "json_or_artifact_format", ["quant_action_value:action"]),
        )
    else:
        citation_field = "资产负债率" if task_type == "financial_qa" else "MA5"
        variants = (
            (_remove_citation_from_field(chosen, citation_field), "citation_drop", ["missing_citation"]),
            (chosen + "\n- 预计下一期业绩改善且股价上涨。", "unsupported_forecast", ["unsupported_forward_prediction"]),
            ("<think>hidden reasoning</think>\n" + chosen, "nonempty_thinking", ["nonempty_thinking"]),
        )
    return variants[hashlib.sha256(seed.encode()).digest()[0] % len(variants)]


def build(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preference = []
    skipped_by_task: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    for row in rows:
        task_type = str(row.get("task_type", ""))
        if task_type not in CORE_TASKS:
            raise ValueError(f"non-core training row: {row.get('id')}")
        prompt = _prompt(row)
        visible_case = {"id": row.get("id"), "task_type": task_type, "evidence": list(prompt.get("evidence", []))}
        try:
            chosen = _chosen(task_type, prompt, _assistant(row))
        except ValueError as error:
            skipped_by_task[task_type] += 1
            skipped_reasons[str(error)] += 1
            continue
        chosen_verdict = validate_task_answer(visible_case, chosen)
        if not chosen_verdict["hard_gate_passed"]:
            skipped_by_task[task_type] += 1
            skipped_reasons.update(chosen_verdict["failures"] or chosen_verdict["missing_anchors"] or ["unknown"])
            continue
        rejected, negative_type, expected_reasons = _negative(task_type, chosen, str(row.get("id")))
        rejected_verdict = validate_task_answer(visible_case, rejected)
        if rejected_verdict["hard_gate_passed"]:
            raise ValueError(f"rejected passes v2.8 validator: {row.get('id')}:{negative_type}")
        preference.append(
            {
                "id": f"{row.get('id')}:v2.8-validator-preference",
                "task_type": task_type,
                "system": row.get("system", ""),
                "conversations": [dict((row.get("conversations") or [])[0])],
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": rejected},
                "negative_type": negative_type,
                "expected_rejection_reasons": expected_reasons,
                "source_group": row.get("source_group", ""),
                "validator_contract": "evidence_anchor_task_validator.v1",
            }
        )
    counts = Counter(str(item["task_type"]) for item in preference)
    negative_counts = Counter(str(item["negative_type"]) for item in preference)
    report = {
        "schema_version": SCHEMA_VERSION,
        "samples": len(preference),
        "by_task": dict(sorted(counts.items())),
        "by_negative_type": dict(sorted(negative_counts.items())),
        "excluded_source_rows": {"by_task": dict(sorted(skipped_by_task.items())), "by_reason": dict(sorted(skipped_reasons.items()))},
        "chosen_accept_rate": 1.0,
        "rejected_false_accept_rate": 0.0,
        "contract_gate_passed": len(preference) > 0 and set(counts) == CORE_TASKS and min(counts.values()) > 0,
    }
    return preference, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v2.8 validator-aligned core preference data")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    preference, report = build(_load(Path(args.input)))
    _write(Path(args.output), preference)
    _write(Path(args.report), report)
    print(json.dumps({"output": args.output, "report": args.report, **report}, ensure_ascii=False), flush=True)
    return 0 if report["contract_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
