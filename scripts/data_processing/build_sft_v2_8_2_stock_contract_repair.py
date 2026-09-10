#!/usr/bin/env python3
"""Build v2.8.2 core SFT data with deterministic four-anchor stock targets."""

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

from scripts.evaluation.eval_sft_v2_trusted import evidence_source_identities  # noqa: E402
from scripts.evaluation.task_aware_verifier_v2_8_2 import validate_task_answer  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/sft_v2_7_core"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft_v2_8_2_stock_contract"
PREFLIGHT_GOLD = PROJECT_ROOT / "data/evaluation/sft_v2_8_stock_contract_seen_preflight.json"
FILES = {
    "train": "fin_agentic_sft_v2_7_core_answer_train.json",
    "eval": "fin_agentic_sft_v2_7_core_answer_eval.json",
}
OUTPUT_FILES = {
    "train": "fin_agentic_sft_v2_8_2_core_answer_train.json",
    "eval": "fin_agentic_sft_v2_8_2_core_answer_eval.json",
}


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected JSON object rows: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _assistant_index(row: dict[str, Any]) -> int:
    for index, turn in enumerate(row.get("conversations", [])):
        if turn.get("from") in {"gpt", "assistant"}:
            return index
    raise ValueError(f"{row.get('id')}: assistant target missing")


def _prompt(row: dict[str, Any]) -> dict[str, Any]:
    for turn in row.get("conversations", []):
        if turn.get("from") in {"human", "user"}:
            value = json.loads(str(turn.get("value", "")))
            if not isinstance(value, dict):
                break
            return value
    raise ValueError(f"{row.get('id')}: structured prompt missing")


def _field(quote: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}为([^，。；;\s]+)", quote)
    if match is None:
        raise ValueError(f"missing {name}")
    return match.group(1)


def _stock_case(prompt: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": "stock_analysis",
        "evidence": [
            {
                "evidence_id": evidence["evidence_id"],
                "exact_quote": evidence["exact_quote"],
            }
            for evidence in prompt.get("evidence", [])
        ],
    }


def _stock_target(prompt: dict[str, Any]) -> str:
    evidence = list(prompt.get("evidence", []))
    market = next(item for item in evidence if "收盘价" in str(item.get("exact_quote", "")))
    financial = next(item for item in evidence if "营业收入" in str(item.get("exact_quote", "")))
    market_quote, financial_quote = str(market["exact_quote"]), str(financial["exact_quote"])
    market_id, financial_id = market["evidence_id"], financial["evidence_id"]
    trend = (
        f"- MA5为{_field(market_quote, 'MA5')}，MA20为{_field(market_quote, 'MA20')}，仅据此描述当前均线趋势关系 [{market_id}]。"
        if "MA5为" in market_quote and "MA20为" in market_quote
        else f"- 证据未提供MA5或MA20，无法确认均线趋势 [{market_id}]。"
    )
    return "\n".join([
        f"- 收盘价为{_field(market_quote, '收盘价')} [{market_id}]。",
        trend,
        f"- 年化波动率为{_field(market_quote, '年化波动率')}，最大回撤为{_field(market_quote, '最大回撤')}，该时点存在波动与回撤风险 [{market_id}]。",
        f"- 营业收入为{_field(financial_quote, '营业收入')} [{financial_id}]。",
        f"- 证据不足，无法确认未来价格方向、目标价或上涨概率 [{market_id}][{financial_id}]。",
    ])


def _repair(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output, changed = [], Counter()
    for source in rows:
        row = json.loads(json.dumps(source, ensure_ascii=False))
        if row.get("task_type") == "stock_analysis":
            prompt = _prompt(row)
            target = _stock_target(prompt)
            verdict = validate_task_answer(_stock_case(prompt), target)
            if not verdict["hard_gate_passed"]:
                raise ValueError(f"{row.get('id')}: generated stock target failed validator: {verdict}")
            row["conversations"][_assistant_index(row)]["value"] = target
            row["repair_profile"] = "v2.8.2_stock_four_anchor_atomic_contract"
            row["target_task_validator"] = verdict
            changed["stock_analysis"] += 1
        row["dataset_version"] = "sft_v2.8.2"
        output.append(row)
    return output, dict(changed)


def _source_ids(rows: list[dict[str, Any]]) -> set[str]:
    cases = []
    for row in rows:
        prompt = _prompt(row)
        cases.append({"evidence": prompt.get("evidence", [])})
    return set().union(*(evidence_source_identities(case) for case in cases))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v2.8.2 stock contract repair SFT data")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preflight-gold", default=str(PREFLIGHT_GOLD))
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    repaired = {}
    changed = {}
    for split, filename in FILES.items():
        repaired[split], changed[split] = _repair(_load(input_dir / filename))
        _write(output_dir / OUTPUT_FILES[split], repaired[split])
    # Avoid accidental evaluation leakage even though the preflight is explicitly seen regression data.
    gold = _load(Path(args.preflight_gold))
    gold_sources = set().union(*(evidence_source_identities(case) for case in gold))
    train_sources = _source_ids(repaired["train"])
    report = {
        "schema_version": "sft_v2.8.2_stock_contract_repair.v1",
        "components": {
            split: {"samples": len(rows), "by_task": dict(sorted(Counter(row["task_type"] for row in rows).items())), "stock_repaired": changed[split]}
            for split, rows in repaired.items()
        },
        "stock_target_validator": {
            "validated": changed["train"].get("stock_analysis", 0) + changed["eval"].get("stock_analysis", 0),
            "failed": 0,
        },
        "preflight_gold_source_overlap_with_train": len(train_sources & gold_sources),
        "preflight_gold_source_overlap_with_eval": len(_source_ids(repaired["eval"]) & gold_sources),
        "source_inputs": {split: {"path": str((input_dir / filename).resolve()), "sha256": _sha256(input_dir / filename)} for split, filename in FILES.items()},
        "release_note": "Training data is repair-only. This report does not turn the seen preflight into a release holdout.",
    }
    if report["preflight_gold_source_overlap_with_train"]:
        raise ValueError("preflight gold leaked into v2.8.2 train data")
    _write(output_dir / "build_report.json", report)
    print(json.dumps({"output_dir": str(output_dir), **report["components"], "preflight_overlap": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
