#!/usr/bin/env python3
"""Build SFT v2.6 compact claim-aligned answer data.

The v2.6 contract removes provenance prose from the model target.  Provenance
is still present in the input evidence records; the model emits only compact
atomic claims (or an explicit abstention) with sentence-final Evidence IDs.
Quant tasks continue to use a fixed action rendered by the audited node.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing import build_sft_v2_4_dataset as v24  # noqa: E402
from scripts.data_processing.build_sft_v2_1_dataset import _answer_target_errors, _assistant_text  # noqa: E402
from scripts.evaluation.eval_sft_v2_trusted import evidence_source_identities  # noqa: E402
from scripts.rag.audit_schema import extract_citation_ids  # noqa: E402
from scripts.rag.quant_protocol import canonical_quant_action, materialize_quant_output, parse_quant_action  # noqa: E402

TARGET_TASKS = v24.TARGET_TASKS
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/sft_v2_5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft_v2_6"
DEFAULT_DEV_AUDIT = PROJECT_ROOT / "data/evaluation/sft_v2_6_dev_audit.json"
DEFAULT_PREFERENCE = PROJECT_ROOT / "data/rlhf/sft_v2_6_hard_negative_preference.json"
DEFAULT_TRUSTED = PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"
DEFAULT_ADVERSARIAL = PROJECT_ROOT / "data/evaluation/sft_v2_3_audit_adversarial.json"
INPUT_FILES = {
    "train": "fin_agentic_sft_v2_5_answer_train.json",
    "eval": "fin_agentic_sft_v2_5_answer_eval.json",
}
OUTPUT_FILES = {
    "train": "fin_agentic_sft_v2_6_answer_train.json",
    "eval": "fin_agentic_sft_v2_6_answer_eval.json",
}

SYSTEM_PROMPT = (
    "You are an auditable financial analyst. Do not output <think> or free-form reasoning. "
    "Use only the supplied evidence and the decision_basis. Emit only compact section headings and "
    "atomic claim lines. Every factual, calculation, inference, or recommendation line must end with "
    "the supporting Evidence IDs in square brackets. Copy numeric lexemes, signs, trailing zeros, "
    "precision, and units exactly from exact_quote; never recalculate, round, forecast, or fill gaps. "
    "When required evidence is absent, emit one concise evidence-insufficient abstention and stop. "
    "For quant_strategy emit exactly the one-line quant.action.v2.5 JSON object; an audited deterministic "
    "node renders the artifact. Never promise returns."
)

STRICT_CONTRACT = {
    "reasoning_policy": "no_think_output_use_supplied_decision_basis",
    "claim_policy": "one_atomic_claim_per_line",
    "citation_policy": "each_claim_sentence_final_evidence_ids",
    "numeric_policy": "exact_lexeme_sign_trailing_zero_precision_and_unit_copy_only",
    "unsupported_policy": "single_explicit_abstention_then_eos",
    "quant_policy": "quant.action.v2.5_only_then_deterministic_renderer",
}

TASK_CONTRACTS = {
    "financial_qa": "Output only the reported asset-liability and current ratios, formula meaning when present, and a cited boundary; refuse recalculation.",
    "financial_report": "Output only reported identity, revenue, net profit, operating cash flow, revenue change, leverage, and liquidity; refuse outlook.",
    "quant_strategy": "Return exactly the canonical quant.action.v2.5 JSON object and EOS; never output Python or measured performance.",
    "risk_assessment": "Output available financial and market risk facts, a bounded risk level only when both groups exist, and one cited mitigation; refuse extra risks.",
    "sentiment_analysis": "Output supported financial facts and exactly one evidence-bound sentiment label; refuse news or price claims.",
    "stock_analysis": "Output available market and financial facts, a bounded current trend/risk statement, and refuse price forecasts.",
}

ABSTENTION_TERMS = ("证据不足", "无法确认", "无法计算", "不外推")


def _prompt(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["conversations"][0]["value"])


def _line(text: str, items: list[dict[str, Any]]) -> str:
    ids = "".join(f"[{str(item['evidence_id']).upper()}]" for item in items)
    return f"- {text} {ids}。" if ids else f"- {text}。"


def _abstain(text: str, items: list[dict[str, Any]] | None = None) -> str:
    return _line(f"证据不足，无法确认{text}", items or [])


def _facts(item: dict[str, Any], labels: tuple[str, ...]) -> list[str]:
    return v24.v23._available_facts(item, labels)


def _compact_target(task_type: str, prompt: dict[str, Any]) -> str:
    grouped = v24._evidence_by_kind(prompt)
    financial = grouped.get("financial", [])[:1]
    market = grouped.get("market", [])[:1]
    used = financial + market

    if task_type == "quant_strategy":
        return canonical_quant_action()

    if task_type == "financial_qa":
        if not financial:
            return _abstain("资产负债率、流动比率及其输入值")
        lines = ["### 财务比率"]
        lines.extend(_facts(financial[0], ("资产负债率", "流动比率")))
        lines.append(_line("观点：只复述证据已报告的比率，不重新计算或改变精度", financial))
        lines.append(_abstain("由比率单独推出未来偿债表现", financial))
        return "\n".join(lines)

    if task_type == "financial_report":
        if not financial:
            return _abstain("财务报告结论")
        lines = ["### 财务事实", v24.v23._entity_fact(financial[0])]
        lines.extend(_facts(financial[0], ("营业收入", "归母口径净利润", "经营活动现金流量净额", "营业收入同比变化", "资产负债率", "流动比率")))
        lines.append(_abstain("下一报告期业绩或行业变化", financial))
        return "\n".join(lines)

    if task_type == "risk_assessment":
        if not (financial and market):
            return _abstain("综合风险等级，因为所需财务与市场证据不完整", used)
        lines = ["### 风险事实"]
        lines.extend(_facts(financial[0], ("归母口径净利润", "经营活动现金流量净额", "资产负债率", "流动比率")))
        lines.extend(_facts(market[0], ("年化波动率", "最大回撤")))
        lines.append(_line(f"观点：综合风险等级为{v24._risk_level(str(financial[0]['exact_quote']), str(market[0]['exact_quote']))}", used))
        lines.append(_line("观点：缓释措施为控制仓位并在财务或市场证据更新后复核", used))
        lines.append(_abstain("证据之外的行业政策风险或未来风险等级", used))
        return "\n".join(lines)

    if task_type == "sentiment_analysis":
        if not financial:
            return _abstain("基本面情绪标签")
        lines = ["### 基本面事实"]
        lines.extend(_facts(financial[0], ("营业收入", "归母口径净利润", "经营活动现金流量净额", "营业收入同比变化")))
        lines.append(_line(f"观点：基本面情绪标签为{v24._sentiment_label(str(financial[0]['exact_quote']))}", financial))
        return "\n".join(lines)

    if task_type == "stock_analysis":
        if not (financial and market):
            return _abstain("完整股票分析；所需财务与市场证据不完整", used)
        lines = ["### 行情事实"]
        lines.extend(_facts(market[0], ("收盘价", "近20个交易日收益率", "年化波动率", "最大回撤")))
        lines.append("### 财务事实")
        lines.extend(_facts(financial[0], ("营业收入", "归母口径净利润", "经营活动现金流量净额")))
        lines.append(_line("观点：现有指标只描述证据时点的趋势与风险", used))
        lines.append(_abstain("未来价格方向、目标价或上涨概率", used))
        return "\n".join(lines)

    raise ValueError(f"unsupported task: {task_type}")


def _target_errors(task_type: str, prompt: dict[str, Any], target: str) -> list[str]:
    errors: list[str] = []
    if re.search(r"</?think>", target, re.IGNORECASE):
        errors.append("thinking_output_forbidden")
    if task_type == "quant_strategy":
        action, action_errors = parse_quant_action(target)
        errors.extend(action_errors)
        if action and action.get("artifact_version") != "quant.v2.5":
            errors.append("wrong_quant_version")
        rendered, render_errors = materialize_quant_output(target, prompt.get("evidence", []))
        if rendered is None or render_errors:
            errors.extend(f"quant:{item}" for item in render_errors)
        return sorted(set(errors))
    reward = v24._audit_target(task_type, prompt, target)
    if not reward["hard_gate_passed"]:
        errors.extend(f"audit:{item}" for item in reward["hard_failures"])
    for component in ("numeric_consistency", "citation_coverage", "citation_precision"):
        if reward[component] != 1.0:
            errors.append(f"strict_{component}:{reward[component]}")
    evidence_by_id = {str(item["evidence_id"]).upper(): str(item.get("exact_quote", "")) for item in prompt.get("evidence", [])}
    for line_number, raw_line in enumerate(target.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        citations = extract_citation_ids(line)
        abstention = any(term in line for term in ABSTENTION_TERMS)
        if not citations and not abstention:
            errors.append(f"line_{line_number}:missing_citation")
        if citations and not re.search(r"(?:\[\s*E[0-9A-Fa-f]{10,64}\s*\])+[。.!?]?$", line):
            errors.append(f"line_{line_number}:citation_not_sentence_final")
        statement = re.sub(r"\[\s*E[0-9A-Fa-f]{10,64}\s*\]", "", line)
        cited_text = " ".join(evidence_by_id.get(item, "") for item in citations)
        for token in v24.v23._raw_numeric_tokens(statement):
            if token not in cited_text:
                errors.append(f"line_{line_number}:numeric_lexeme_not_copied:{token}")
    return sorted(set(errors))


def _convert(row: dict[str, Any], suffix: str = "") -> dict[str, Any]:
    task_type = str(row["task_type"])
    prompt = copy.deepcopy(_prompt(row))
    prompt["strict_audit_contract"] = STRICT_CONTRACT
    prompt["output_contract"] = TASK_CONTRACTS[task_type]
    prompt["decision_basis"] = v24.decision_basis(task_type, prompt)
    prompt["decision_basis"]["response_budget"] = "one_json_line" if task_type == "quant_strategy" else "compact_atomic_lines"
    target = _compact_target(task_type, prompt)
    errors = _target_errors(task_type, prompt, target)
    if errors:
        raise ValueError(f"{row.get('id')}:{','.join(errors)}")
    output = copy.deepcopy(row)
    output["id"] = f"{row['id']}{suffix}"
    output["system"] = SYSTEM_PROMPT
    output["conversations"][0]["value"] = v24.canonical_json(prompt)
    output["conversations"][1]["value"] = target
    output["dataset_version"] = "sft_v2.6"
    output["repair_profile"] = "compact_claim_aligned_abstention_quant_action"
    output["canonical_target_sha256"] = hashlib.sha256(target.encode()).hexdigest()
    output["target_audit"] = {"hard_gate_passed": True, "protocol_version": "sft_v2.6"}
    if task_type != "quant_strategy":
        schema_errors = _answer_target_errors(output)
        if schema_errors:
            raise ValueError(f"{row.get('id')}:{','.join(schema_errors)}")
    return output


def _make_hard_negative(task_type: str, target: str, seed: str) -> tuple[str, str]:
    if task_type == "quant_strategy":
        variants = (
            ('{"action":"render_quant_artifact","artifact_version":"quant.v2.5"}', "missing_template"),
            (target + "\n说明", "trailing_prose"),
            ('{"action":"generate_python","artifact_version":"quant.v2.5","template_id":"ma5_ma20_long_only"}', "wrong_action"),
        )
        return variants[hashlib.sha256(seed.encode()).digest()[0] % len(variants)]
    variants = (
        (re.sub(r"\s*(?:\[\s*E[0-9A-Fa-f]{10,64}\s*\])+[。.!?]?$", "。", target, count=1, flags=re.MULTILINE), "citation_drop"),
        (target + "\n- 预计下一期业绩改善且股价上涨。", "unsupported_forecast"),
        ("<think>详细推理" * 80 + "</think>\n" + target, "long_think"),
    )
    return variants[hashlib.sha256(seed.encode()).digest()[0] % len(variants)]


def _build_preferences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        target = _assistant_text(row)
        rejected, negative_type = _make_hard_negative(str(row["task_type"]), target, str(row["id"]))
        output.append({
            "id": f"{row['id']}:hard-negative",
            "task_type": row["task_type"],
            "system": row["system"],
            "conversations": [copy.deepcopy(row["conversations"][0])],
            "chosen": {"from": "gpt", "value": target},
            "rejected": {"from": "gpt", "value": rejected},
            "negative_type": negative_type,
            "source_group": row.get("source_group", ""),
        })
    return output


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    train_source = json.loads((input_dir / INPUT_FILES["train"]).read_text(encoding="utf-8"))
    eval_source = json.loads((input_dir / INPUT_FILES["eval"]).read_text(encoding="utf-8"))
    train = [_convert(row) for row in train_source]
    evaluation = [_convert(row) for row in eval_source]
    preferences = _build_preferences(train)
    dev = v24.build_dev_audit(eval_source)
    for case in dev:
        case["id"] = str(case["id"]).replace("dev-audit-v2.4-", "dev-audit-v2.6-")
        case["challenge_tags"] = ["dev_audit", "compact_claims", "abstention_boundary", "quant_action_v2.5"]
    trusted = json.loads(Path(args.trusted).read_text(encoding="utf-8"))
    adversarial = json.loads(Path(args.adversarial).read_text(encoding="utf-8"))
    train_sources = v24._prompt_sources(train)
    dev_sources = set().union(*(evidence_source_identities(case) for case in dev))
    trusted_sources = set().union(*(evidence_source_identities(case) for case in trusted))
    adversarial_sources = set().union(*(evidence_source_identities(case) for case in adversarial))
    by_task = Counter(str(row["task_type"]) for row in train)
    gate = {
        "task_balanced_train": len(set(by_task.values())) == 1,
        "train_dev_source_disjoint": not (train_sources & dev_sources),
        "dev_regression_source_disjoint": not (dev_sources & trusted_sources) and not (dev_sources & adversarial_sources),
        "train_regression_source_disjoint": not (train_sources & trusted_sources) and not (train_sources & adversarial_sources),
        "all_targets_no_think": all(not re.search(r"</?think>", _assistant_text(row), re.I) for row in train + evaluation),
        "all_targets_short": max(len(_assistant_text(row)) for row in train + evaluation) <= 1000,
        "all_targets_audit_checked": all(bool(row.get("target_audit", {}).get("hard_gate_passed")) for row in train + evaluation),
        "hard_negatives_cover_all_train": len(preferences) == len(train),
    }
    report = {
        "schema_version": "sft_v2.6",
        "profile": "compact_claim_aligned_abstention_quant_action",
        "source_files": INPUT_FILES,
        "components": {
            "train": {"samples": len(train), "by_task": dict(sorted(by_task.items())), "max_target_chars": max(map(lambda r: len(_assistant_text(r)), train))},
            "eval": {"samples": len(evaluation), "by_task": dict(sorted(Counter(str(r["task_type"]) for r in evaluation).items()))},
            "preference": {"samples": len(preferences)},
        },
        "dev_audit": {"samples": len(dev), "by_task": dict(sorted(Counter(str(c["task_type"]) for c in dev).items()))},
        "evaluation_status": {"trusted": "seen_regression_only", "adversarial": "seen_regression_only", "untouched_final_holdout": "pending_new_independent_gold"},
        "release_gate": gate,
        "release_gate_passed": all(gate.values()),
    }
    output_dir = Path(args.output_dir)
    _write(output_dir / OUTPUT_FILES["train"], train)
    _write(output_dir / OUTPUT_FILES["eval"], evaluation)
    _write(output_dir / "build_report.json", report)
    _write(Path(args.preference_output), preferences)
    _write(Path(args.dev_audit_output), dev)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["release_gate_passed"] or args.allow_failed_gate else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preference-output", default=str(DEFAULT_PREFERENCE))
    parser.add_argument("--dev-audit-output", default=str(DEFAULT_DEV_AUDIT))
    parser.add_argument("--trusted", default=str(DEFAULT_TRUSTED))
    parser.add_argument("--adversarial", default=str(DEFAULT_ADVERSARIAL))
    parser.add_argument("--allow-failed-gate", action="store_true")
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
