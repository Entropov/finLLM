#!/usr/bin/env python3
"""Build SFT v2.4 data, hard negatives, and a source-disjoint Dev-Audit set."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing import build_sft_v2_3_dataset as v23  # noqa: E402
from scripts.data_processing.build_sft_v2_1_dataset import CYRILLIC_RE, _answer_target_errors, _assistant_text  # noqa: E402
from scripts.data_processing.build_sft_v2_2_dataset import (  # noqa: E402
    TARGET_TASKS,
    _evidence_by_kind,
    _prompt,
    _provenance_heading,
    _record_from_prompt,
    _risk_level,
    _sentiment_label,
    _sha256_file,
    _write_json_atomic,
)
from scripts.evaluation.eval_sft_v2_trusted import (  # noqa: E402
    evidence_source_identities,
    validate_cases,
)
from scripts.rag.audit_schema import build_claims_from_answer, canonical_json, extract_citation_ids  # noqa: E402
from scripts.rag.reward_v2 import RewardV2Config, compute_auditable_reward  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/sft_v2_3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft_v2_4"
DEFAULT_DEV_AUDIT = PROJECT_ROOT / "data/evaluation/sft_v2_4_dev_audit.json"
DEFAULT_PREFERENCE = PROJECT_ROOT / "data/rlhf/sft_v2_4_hard_negative_preference.json"
DEFAULT_TRUSTED = PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"
DEFAULT_ADVERSARIAL = PROJECT_ROOT / "data/evaluation/sft_v2_3_audit_adversarial.json"
INPUT_FILES = {
    "train": "fin_agentic_sft_v2_3_answer_train.json",
    "eval": "fin_agentic_sft_v2_3_answer_eval.json",
}
OUTPUT_FILES = {
    "train": "fin_agentic_sft_v2_4_answer_train.json",
    "eval": "fin_agentic_sft_v2_4_answer_eval.json",
}
WEAK_TASKS = {"financial_qa", "quant_strategy", "risk_assessment"}
ABSTENTION_TERMS = ("证据不足", "无法确认", "无法计算", "不外推")

SYSTEM_PROMPT = (
    "You are an auditable financial analyst. Use only supplied evidence. Do not output free-form chain-of-thought "
    "or <think> tags; the supplied decision_basis is the complete reasoning plan. Output only the final answer. "
    "Every non-heading prose line must contain exactly one atomic claim and end with all supporting Evidence IDs. "
    "Copy numeric lexemes, signs, trailing zeros, precision, and units exactly from exact_quote. Never recompute, "
    "round, fill missing values, or add forecasts. If evidence cannot support a requested claim, emit a concise "
    "explicit abstention instead. For quant_strategy, emit only the fixed quant.v2.4 artifact consuming an external "
    "DataFrame; never generate or download data and never claim unmeasured performance. Never promise returns."
)

TASK_CONTRACTS = {
    "financial_qa": (
        "Copy only the supplied ratio inputs and reported ratio values. Use separate cited lines for formula meaning "
        "and limitations. Refuse recalculation, rounding, or any answer whose required evidence is absent."
    ),
    "risk_assessment": (
        "Copy available financial and market risk facts one per cited line. Give a risk level only when both evidence "
        "groups are present; otherwise explicitly abstain. Never introduce industry, policy, or forecast claims."
    ),
    "quant_strategy": (
        "Use artifact_version quant.v2.4 and the exact input/output schema. The code must consume caller-provided "
        "DataFrame data and expose signal, entry, exit, position, stop_loss, sharpe, and drawdown. Refuse measured "
        "performance claims and never add alternate strategies."
    ),
    "financial_report": "Copy reported facts one per cited line and refuse any unsupported outlook.",
    "sentiment_analysis": "Copy reported facts one per cited line and emit exactly one cited sentiment label.",
    "stock_analysis": "Copy market and financial facts one per cited line and refuse future price or industry claims.",
}

STRICT_CONTRACT = {
    "reasoning_policy": "no_think_output_use_supplied_decision_basis",
    "claim_policy": "one_atomic_claim_per_non_heading_line",
    "citation_policy": "all_supporting_evidence_ids_at_sentence_end",
    "numeric_policy": "exact_lexeme_sign_trailing_zero_precision_and_unit_copy_only",
    "unsupported_policy": "explicit_abstention_without_guessing",
    "quant_policy": "quant.v2.4_fixed_schema_external_dataframe_only",
}

DECISION_CHECKS = {
    "financial_qa": ["required_ratio_evidence_present", "copy_reported_ratio", "refuse_recalculation"],
    "risk_assessment": ["financial_evidence_present", "market_evidence_present", "bound_risk_conclusion"],
    "quant_strategy": ["external_dataframe_only", "fixed_output_fields", "no_performance_claim"],
    "financial_report": ["copy_reported_values", "cite_each_claim", "refuse_outlook"],
    "sentiment_analysis": ["copy_reported_values", "single_supported_label", "cite_each_claim"],
    "stock_analysis": ["copy_market_and_financial_values", "cite_each_claim", "refuse_forecast"],
}

QUANT_CODE = """```python
import numpy as np
import pandas as pd

def quant_artifact(df):
    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        raise ValueError("external DataFrame with close is required")
    out = df.loc[:, ["close"]].copy()
    close = pd.to_numeric(out["close"], errors="raise")
    fast, slow = close.rolling(5).mean(), close.rolling(20).mean()
    up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    out["signal"] = np.select([up, down], [1, -1], default=0)
    out["entry"], out["exit"] = up, down
    out["position"] = out["signal"].replace(0, np.nan).ffill().fillna(0).clip(lower=0).shift(1).fillna(0)
    out["stop_loss"] = close.cummax() * 0.95
    out.loc[close <= out["stop_loss"], "position"] = 0
    strategy_return = out["position"] * close.pct_change()
    volatility = strategy_return.std()
    sharpe = np.nan if pd.isna(volatility) or volatility == 0 else np.sqrt(252) * strategy_return.mean() / volatility
    equity = (1 + strategy_return.fillna(0)).cumprod()
    out["drawdown"] = equity / equity.cummax() - 1
    return {"artifact": out, "sharpe": sharpe, "drawdown": out["drawdown"]}
```"""
_QUANT_CODE_BODY = re.fullmatch(r"```python\n(.*)\n```", QUANT_CODE, re.DOTALL).group(1)
_QUANT_AST = ast.dump(ast.parse(_QUANT_CODE_BODY), include_attributes=False)


def decision_basis(task_type: str, prompt: dict[str, Any]) -> dict[str, Any]:
    grouped = _evidence_by_kind(prompt)
    evidence_ids = [str(item["evidence_id"]).upper() for item in prompt.get("evidence", [])]
    required = {
        "financial": bool(grouped.get("financial")),
        "market": bool(grouped.get("market")),
    }
    if task_type in {"financial_qa", "financial_report", "sentiment_analysis"}:
        sufficient = required["financial"]
    elif task_type in {"stock_analysis", "risk_assessment"}:
        sufficient = required["financial"] and required["market"]
    else:
        sufficient = True
    return {
        "task_type": task_type,
        "decision": "answer_supported_only" if sufficient else "partial_or_abstain",
        "checks": DECISION_CHECKS[task_type],
        "available_evidence_ids": evidence_ids,
        "required_evidence_present": required,
    }


def _provenance(items: list[dict[str, Any]]) -> list[str]:
    return [_provenance_heading(item) for item in items]


def _opinion(text: str, items: list[dict[str, Any]]) -> str:
    citations = "".join(f"[{str(item['evidence_id']).upper()}]" for item in items)
    return f"- 观点：{text} {citations}。"


def _abstain(text: str, items: list[dict[str, Any]] | None = None) -> str:
    citations = "".join(f"[{str(item['evidence_id']).upper()}]" for item in (items or []))
    suffix = f" {citations}" if citations else ""
    return f"- 证据不足，无法确认{text}{suffix}。"


def build_target(task_type: str, prompt: dict[str, Any]) -> str:
    grouped = _evidence_by_kind(prompt)
    financial = grouped.get("financial", [])[:1]
    market = grouped.get("market", [])[:1]
    used = financial + market

    if task_type == "financial_qa":
        if not financial:
            body = ["### 审计结论", _abstain("资产负债率、流动比率及其输入值")]
        else:
            body = [
                "### 报告值",
                v23._fact(financial[0], "资产总计"),
                v23._fact(financial[0], "负债合计"),
                v23._fact(financial[0], "流动资产合计"),
                v23._fact(financial[0], "流动负债合计"),
                v23._fact(financial[0], "资产负债率"),
                v23._fact(financial[0], "流动比率"),
                "### 边界",
                _opinion("只复述证据已报告的比率，不重新计算或改变精度", financial),
                _abstain("由这两个比率单独推出未来偿债表现", financial),
            ]
    elif task_type == "risk_assessment":
        body = ["### 可验证风险事实"]
        if financial:
            body.extend(
                [
                    v23._fact(financial[0], "归母口径净利润"),
                    v23._fact(financial[0], "经营活动现金流量净额"),
                    v23._fact(financial[0], "资产负债率"),
                    v23._fact(financial[0], "流动比率"),
                ]
            )
        if market:
            body.extend([v23._fact(market[0], "年化波动率"), v23._fact(market[0], "最大回撤")])
        body.append("### 边界")
        if financial and market:
            body.extend(
                [
                    _opinion(
                        f"综合风险等级为{_risk_level(str(financial[0]['exact_quote']), str(market[0]['exact_quote']))}",
                        used,
                    ),
                    _opinion("缓释措施为控制仓位并在财务或市场证据更新后复核", used),
                    _abstain("证据之外的行业政策风险或未来风险等级", used),
                ]
            )
        else:
            body.append(_abstain("综合风险等级，因为所需财务与市场证据不完整", used))
    elif task_type == "quant_strategy":
        body = [
            "### artifact_version: quant.v2.4",
            "### input_schema: pandas.DataFrame[close]",
            "### output_fields: signal, entry, exit, position, stop_loss, sharpe, drawdown",
        ]
        if market:
            body.extend(
                [
                    "### 可审计行情输入",
                    *v23._available_facts(market[0], ("收盘价", "近20个交易日收益率", "年化波动率", "最大回撤")),
                ]
            )
        body.extend(["### 固定代码", QUANT_CODE, "### 边界"])
        body.append(_abstain("该 artifact 的实测 Sharpe、drawdown 或未来收益", market))
    elif task_type == "financial_report":
        if not financial:
            body = ["### 审计结论", _abstain("财务报告结论")]
        else:
            body = [
                "### 财务事实",
                v23._entity_fact(financial[0]),
                v23._fact(financial[0], "营业收入"),
                v23._fact(financial[0], "归母口径净利润"),
                v23._fact(financial[0], "经营活动现金流量净额"),
                v23._fact(financial[0], "营业收入同比变化"),
                v23._fact(financial[0], "资产负债率"),
                v23._fact(financial[0], "流动比率"),
                "### 边界",
                _abstain("下一报告期业绩或行业变化", financial),
            ]
    elif task_type == "sentiment_analysis":
        if not financial:
            body = ["### 审计结论", _abstain("基本面情绪标签")]
        else:
            body = [
                "### 财务事实",
                v23._fact(financial[0], "营业收入"),
                v23._fact(financial[0], "归母口径净利润"),
                v23._fact(financial[0], "经营活动现金流量净额"),
                v23._fact(financial[0], "营业收入同比变化"),
                _opinion(f"基本面情绪标签为{_sentiment_label(str(financial[0]['exact_quote']))}", financial),
            ]
    elif task_type == "stock_analysis":
        if not (financial and market):
            body = ["### 审计结论", _abstain("完整股票分析；所需财务与市场证据不完整", used)]
        else:
            body = [
                "### 行情事实",
                *v23._available_facts(market[0], ("收盘价", "近20个交易日收益率", "年化波动率", "最大回撤")),
                "### 财务事实",
                v23._fact(financial[0], "营业收入"),
                v23._fact(financial[0], "归母口径净利润"),
                v23._fact(financial[0], "经营活动现金流量净额"),
                "### 边界",
                _opinion("现有指标只描述证据时点的趋势与风险", used),
                _abstain("未来价格方向、目标价或上涨概率", used),
            ]
    else:
        raise ValueError(f"unsupported task: {task_type}")

    return "\n".join([*_provenance(used), *body]).strip()


def _audit_target(task_type: str, prompt: dict[str, Any], target: str) -> dict[str, Any]:
    evidence = [_record_from_prompt(item) for item in prompt.get("evidence", [])]
    return compute_auditable_reward(
        query=str(prompt.get("query", "")),
        answer=target,
        evidence=evidence,
        claims=build_claims_from_answer(target, str(prompt.get("request_as_of", ""))),
        task_type=task_type,
        request_as_of=str(prompt.get("request_as_of", "")),
        trajectory=[],
        config=RewardV2Config(require_complete_trajectory=False),
    )


def _quant_errors(target: str) -> list[str]:
    matches = re.findall(r"```python\s*\n(.*?)```", target, re.DOTALL)
    if not matches:
        return ["quant_code_missing"]
    if len(matches) != 1:
        return ["quant_code_count"]
    code = matches[0]
    errors = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["quant_code_syntax"]
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "quant_artifact":
        errors.append("quant_function_schema")
    elif [arg.arg for arg in functions[0].args.args] != ["df"]:
        errors.append("quant_input_schema")
    required = ("signal", "entry", "exit", "position", "stop_loss", "sharpe", "drawdown")
    errors.extend(f"quant_missing:{field}" for field in required if field not in code)
    forbidden = (r"np\.random", r"\brandom\.", r"read_csv", r"read_parquet", r"requests\.", r"yfinance", r"akshare", r"tushare", r"download", r"mock", r"synthetic")
    errors.extend(f"quant_forbidden:{pattern}" for pattern in forbidden if re.search(pattern, code, re.IGNORECASE))
    metadata = (
        "### artifact_version: quant.v2.4",
        "### input_schema: pandas.DataFrame[close]",
        "### output_fields: signal, entry, exit, position, stop_loss, sharpe, drawdown",
    )
    if any(target.splitlines().count(line) != 1 for line in metadata):
        errors.append("quant_metadata_schema")
    if ast.dump(tree, include_attributes=False) != _QUANT_AST:
        errors.append("quant_artifact_schema")
    return errors


def target_errors(task_type: str, prompt: dict[str, Any], target: str) -> list[str]:
    errors: list[str] = []
    if re.search(r"</?think>", target, re.IGNORECASE):
        errors.append("thinking_output_forbidden")
    reward = _audit_target(task_type, prompt, target)
    if not reward["hard_gate_passed"]:
        errors.extend(f"audit:{item}" for item in reward["hard_failures"])
    for component in ("numeric_consistency", "citation_coverage", "citation_precision"):
        if reward[component] != 1.0:
            errors.append(f"strict_{component}:{reward[component]}")

    evidence_by_id = {str(item["evidence_id"]).upper(): str(item.get("exact_quote", "")) for item in prompt.get("evidence", [])}
    in_code = False
    for line_number, raw_line in enumerate(target.splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not line or in_code or line.startswith("#"):
            continue
        citations = extract_citation_ids(line)
        abstention = any(term in line for term in ABSTENTION_TERMS)
        if not citations and not abstention:
            errors.append(f"line_{line_number}:missing_citation")
            continue
        if citations and not re.search(r"(?:\[\s*E[0-9A-Fa-f]{10,64}\s*\])+[\u3002.!?]?$", line):
            errors.append(f"line_{line_number}:citation_not_sentence_final")
        statement = re.sub(r"\[\s*E[0-9A-Fa-f]{10,64}\s*\]", "", line)
        cited_text = " ".join(evidence_by_id.get(item, "") for item in citations)
        for token in v23._raw_numeric_tokens(statement):
            if token not in cited_text:
                errors.append(f"line_{line_number}:numeric_lexeme_not_copied:{token}")
    if task_type == "quant_strategy":
        errors.extend(_quant_errors(target))
    if CYRILLIC_RE.search(target):
        errors.append("cyrillic_target")
    limit = 3200 if task_type == "quant_strategy" else 2200
    if len(target) > limit:
        errors.append("target_too_long")
    return sorted(set(errors))


def _convert(row: dict[str, Any], *, prompt_override: dict[str, Any] | None = None, suffix: str = "") -> dict[str, Any]:
    task_type = str(row["task_type"])
    prompt = copy.deepcopy(prompt_override if prompt_override is not None else _prompt(row))
    prompt["decision_basis"] = decision_basis(task_type, prompt)
    prompt.pop("claim_plan", None)
    prompt["strict_audit_contract"] = STRICT_CONTRACT
    prompt["output_contract"] = TASK_CONTRACTS[task_type]
    target = build_target(task_type, prompt)
    errors = target_errors(task_type, prompt, target)
    if errors:
        raise ValueError(f"{row.get('id')}:{','.join(errors)}")

    output = copy.deepcopy(row)
    output["id"] = f"{row['id']}{suffix}"
    output["system"] = SYSTEM_PROMPT
    output["conversations"][0]["value"] = canonical_json(prompt)
    output["conversations"][1]["value"] = target
    output["dataset_version"] = "sft_v2.4"
    output["repair_profile"] = "nothink_claim_aligned_quant_fixed"
    output["canonical_target_sha256"] = hashlib.sha256(target.encode()).hexdigest()
    reward = _audit_target(task_type, prompt, target)
    output["target_audit"] = {
        "hard_gate_passed": reward["hard_gate_passed"],
        "hard_failures": reward["hard_failures"],
        "claim_support": reward["claim_support"],
        "numeric_consistency": reward["numeric_consistency"],
        "citation_coverage": reward["citation_coverage"],
        "citation_precision": reward["citation_precision"],
        "task_validity": reward["task_validity"],
    }
    schema_errors = _answer_target_errors(output)
    if schema_errors:
        raise ValueError(f"{row.get('id')}:{','.join(schema_errors)}")
    return output


def _insufficient_variant(row: dict[str, Any], index: int) -> dict[str, Any]:
    task_type = str(row["task_type"])
    prompt = copy.deepcopy(_prompt(row))
    if task_type in {"financial_qa", "quant_strategy"}:
        prompt["evidence"] = []
    elif task_type == "risk_assessment":
        grouped = _evidence_by_kind(prompt)
        retained = grouped.get("financial" if index % 2 == 0 else "market", [])[:1]
        prompt["evidence"] = retained
    else:
        raise ValueError(f"insufficient variant not supported for {task_type}")
    prompt["query"] = f"{prompt.get('query', '')}\n缺证据测试：不得用常识或默认值补齐缺失字段。"
    output = _convert(row, prompt_override=prompt, suffix=":v2.4-insufficient")
    output["challenge_profile"] = "missing_evidence_boundary"
    output["source_groups"] = [
        f"{item.canonical_url or item.source_uri}|{item.document_version}"
        for item in (_record_from_prompt(raw) for raw in prompt.get("evidence", []))
    ]
    return output


def _balance(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    target = max(len(grouped[task]) for task in TARGET_TASKS)
    output = list(rows)
    added = {}
    for task in sorted(TARGET_TASKS):
        items = sorted(grouped[task], key=lambda item: str(item["id"]))
        missing = target - len(items)
        added[task] = missing
        for index in range(missing):
            replica = copy.deepcopy(items[index % len(items)])
            replica["replica_of"] = str(items[index % len(items)]["id"])
            replica["id"] = f"{replica['id']}:v2.4-balance-{index + 1}"
            replica["oversampled"] = True
            output.append(replica)
    output.sort(key=lambda item: (str(item["task_type"]), str(item["id"])))
    return output, added


def _drop_citation(target: str) -> str:
    lines = target.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("- ") and extract_citation_ids(line) and not any(term in line for term in ABSTENTION_TERMS):
            lines[index] = re.sub(r"\s*(?:\[\s*E[0-9A-Fa-f]{10,64}\s*\])+", "", line)
            return "\n".join(lines)
    return target + "\n- 该结论缺少引用。"


def _drift_number(target: str) -> str:
    lines = target.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("- ") or line.startswith(("- artifact_", "- input_schema", "- output_fields")):
            continue
        match = re.search(r"(?<![A-Za-z0-9_])([-+]?\d+(?:\.\d+)?%?)", line)
        if not match:
            continue
        token = match.group(1)
        replacement = token[:-1] + ("1" if token[-1] != "1" else "2")
        lines[index] = line[: match.start(1)] + replacement + line[match.end(1) :]
        return "\n".join(lines)
    return target + "\n- 未验证数值为1.00%。"


def _quant_violation(target: str) -> str:
    bad_code = """```python
import numpy as np
import pandas as pd

def quant_artifact(df):
    mock = pd.DataFrame({"close": np.random.randn(252)})
    return {"artifact": mock, "sharpe": 2.10, "drawdown": -0.08}
```"""
    return re.sub(r"```python\s*\n.*?```", bad_code, target, count=1, flags=re.DOTALL)


def make_hard_negative(task_type: str, target: str, seed: str) -> tuple[str, str]:
    pools = {
        "financial_qa": ("numeric_drift", "citation_drop", "unsupported_forecast", "long_think"),
        "risk_assessment": ("unsupported_forecast", "citation_drop", "numeric_drift", "long_think"),
        "quant_strategy": ("quant_violation", "unsupported_forecast", "citation_drop", "long_think"),
        "financial_report": ("numeric_drift", "citation_drop", "unsupported_forecast"),
        "sentiment_analysis": ("citation_drop", "numeric_drift", "unsupported_forecast"),
        "stock_analysis": ("unsupported_forecast", "citation_drop", "numeric_drift"),
    }
    digest = hashlib.sha256(seed.encode()).digest()
    kind = pools[task_type][digest[0] % len(pools[task_type])]
    if kind == "numeric_drift":
        return _drift_number(target), kind
    if kind == "citation_drop":
        return _drop_citation(target), kind
    if kind == "quant_violation":
        return _quant_violation(target), kind
    if kind == "long_think":
        reasoning = "先进行详细推演。" * 80
        return f"<think>{reasoning}</think>\n{target}", kind
    return target + "\n- 预计下一报告期业绩改善且股价上涨概率较高。", kind


def build_preferences(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = []
    negative_types = Counter()
    errors_by_type = Counter()
    for row in rows:
        task_type = str(row["task_type"])
        prompt = _prompt(row)
        chosen = _assistant_text(row)
        rejected, negative_type = make_hard_negative(task_type, chosen, str(row["id"]))
        errors = target_errors(task_type, prompt, rejected)
        if not errors:
            raise ValueError(f"hard negative unexpectedly passed: {row['id']}:{negative_type}")
        negative_types[negative_type] += 1
        errors_by_type.update(errors)
        pairs.append(
            {
                "id": f"{row['id']}:hard-negative",
                "task_type": task_type,
                "system": row["system"],
                "conversations": [copy.deepcopy(row["conversations"][0])],
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": rejected},
                "negative_type": negative_type,
                "negative_contract_errors": errors,
                "source_group": row.get("source_group", ""),
            }
        )
    return pairs, {
        "samples": len(pairs),
        "by_task": dict(sorted(Counter(str(row["task_type"]) for row in pairs).items())),
        "negative_types": dict(sorted(negative_types.items())),
        "negative_contract_errors": dict(errors_by_type.most_common()),
    }


def build_dev_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters = Counter()
    cases = []
    for row in sorted(rows, key=lambda item: (str(item["task_type"]), str(item["id"]))):
        task_type = str(row["task_type"])
        prompt = _prompt(row)
        counters[task_type] += 1
        cases.append(
            {
                "id": f"dev-audit-v2.4-{task_type}-{counters[task_type]:04d}",
                "task_type": task_type,
                "question": str(prompt.get("query", "")),
                "request_as_of": str(prompt.get("request_as_of", "")),
                "source_group": str(row.get("source_group", "")),
                "requires_audit": True,
                "evidence": [_record_from_prompt(item).to_dict() for item in prompt.get("evidence", [])],
                "scoring": v23._scoring_for(task_type, prompt),
                "challenge_tags": ["dev_audit", "claim_evidence", "numeric_exact", "bounded_output"],
            }
        )
    validate_cases(cases)
    return cases


def _prompt_sources(rows: list[dict[str, Any]]) -> set[str]:
    sources = set()
    for row in rows:
        for item in _prompt(row).get("evidence", []):
            record = _record_from_prompt(item)
            sources.add(f"{record.canonical_url or record.source_uri}|{record.document_version}")
    return sources


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(_assistant_text(row)) for row in rows]
    return {
        "samples": len(rows),
        "by_task": dict(sorted(Counter(str(row["task_type"]) for row in rows).items())),
        "source_groups": len({str(row.get("source_group", "")) for row in rows}),
        "mean_target_chars": round(sum(lengths) / len(lengths), 1),
        "max_target_chars": max(lengths),
        "thinking_targets": sum(bool(re.search(r"</?think>", _assistant_text(row), re.IGNORECASE)) for row in rows),
        "audit_hard_pass_rate": round(sum(bool(row["target_audit"]["hard_gate_passed"]) for row in rows) / len(rows), 4),
        "numeric_exact_rate": round(sum(row["target_audit"]["numeric_consistency"] == 1.0 for row in rows) / len(rows), 4),
        "citation_coverage_rate": round(sum(row["target_audit"]["citation_coverage"] == 1.0 for row in rows) / len(rows), 4),
        "missing_evidence_samples": sum(row.get("challenge_profile") == "missing_evidence_boundary" for row in rows),
    }


def build_all(
    train_source: list[dict[str, Any]],
    eval_source: list[dict[str, Any]],
    *,
    trusted_path: Path,
    adversarial_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    converted_train = [_convert(row) for row in train_source]
    converted_eval = [_convert(row) for row in eval_source]

    for task in sorted(WEAK_TASKS):
        candidates = [row for row in train_source if row["task_type"] == task and not row.get("oversampled")]
        candidates.sort(key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())
        for index, row in enumerate(candidates[:54]):
            converted_train.append(_insufficient_variant(row, index))
    train_rows, balance_added = _balance(converted_train)
    preferences, preference_stats = build_preferences(train_rows)
    dev_audit = build_dev_audit(converted_eval)

    train_sources = _prompt_sources(train_rows)
    dev_sources = set().union(*(evidence_source_identities(case) for case in dev_audit))
    trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
    adversarial = json.loads(adversarial_path.read_text(encoding="utf-8"))
    trusted_sources = set().union(*(evidence_source_identities(case) for case in trusted))
    adversarial_sources = set().union(*(evidence_source_identities(case) for case in adversarial))
    dev_counts = Counter(str(case["task_type"]) for case in dev_audit)
    train_stats, eval_stats = _stats(train_rows), _stats(converted_eval)
    frozen = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in (trusted_path, adversarial_path, PROJECT_ROOT / "data/evaluation/v2_split_manifest.json")
    }
    gate = {
        "task_balanced_train": len(set(train_stats["by_task"].values())) == 1,
        "minimum_dev_per_task": all(dev_counts[task] >= 8 for task in TARGET_TASKS),
        "train_dev_source_disjoint": not (train_sources & dev_sources),
        "dev_test_source_disjoint": not (dev_sources & trusted_sources) and not (dev_sources & adversarial_sources),
        "train_test_source_disjoint": not (train_sources & trusted_sources) and not (train_sources & adversarial_sources),
        "all_targets_no_think": train_stats["thinking_targets"] == 0 and eval_stats["thinking_targets"] == 0,
        "all_targets_audit_pass": train_stats["audit_hard_pass_rate"] == 1.0 and eval_stats["audit_hard_pass_rate"] == 1.0,
        "all_targets_numeric_exact": train_stats["numeric_exact_rate"] == 1.0 and eval_stats["numeric_exact_rate"] == 1.0,
        "all_targets_citation_complete": train_stats["citation_coverage_rate"] == 1.0 and eval_stats["citation_coverage_rate"] == 1.0,
        "weak_tasks_have_missing_evidence": all(
            any(row["task_type"] == task and row.get("challenge_profile") == "missing_evidence_boundary" for row in train_rows)
            for task in WEAK_TASKS
        ),
        "hard_negatives_cover_all_train_samples": len(preferences) == len(train_rows),
        "hard_negatives_are_rejected": all(row["negative_contract_errors"] for row in preferences),
        "frozen_test_artifacts_present": all(frozen.values()),
    }
    report = {
        "schema_version": "sft_v2.4",
        "profile": "nothink_task_balanced_audit_weighted_hard_negative",
        "source_files": INPUT_FILES,
        "components": {"train": train_stats, "loss_eval": eval_stats},
        "balance_added": balance_added,
        "preference": preference_stats,
        "dev_audit": {
            "samples": len(dev_audit),
            "by_task": dict(sorted(dev_counts.items())),
            "source_groups": len({str(case["source_group"]) for case in dev_audit}),
            "train_source_overlap": len(train_sources & dev_sources),
            "trusted_source_overlap": len(trusted_sources & dev_sources),
            "adversarial_source_overlap": len(adversarial_sources & dev_sources),
        },
        "frozen_artifacts": frozen,
        "release_gate": gate,
        "release_gate_passed": all(gate.values()),
    }
    return {"train": train_rows, "eval": converted_eval}, preferences, dev_audit, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT v2.4 data and audit artifacts")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preference-output", default=str(DEFAULT_PREFERENCE))
    parser.add_argument("--dev-audit-output", default=str(DEFAULT_DEV_AUDIT))
    parser.add_argument("--trusted", default=str(DEFAULT_TRUSTED))
    parser.add_argument("--adversarial", default=str(DEFAULT_ADVERSARIAL))
    parser.add_argument("--allow-failed-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    datasets, preferences, dev_audit, report = build_all(
        json.loads((input_dir / INPUT_FILES["train"]).read_text(encoding="utf-8")),
        json.loads((input_dir / INPUT_FILES["eval"]).read_text(encoding="utf-8")),
        trusted_path=Path(args.trusted),
        adversarial_path=Path(args.adversarial),
    )
    output_dir = Path(args.output_dir)
    for split, filename in OUTPUT_FILES.items():
        _write_json_atomic(output_dir / filename, datasets[split])
    _write_json_atomic(Path(args.preference_output), preferences)
    _write_json_atomic(Path(args.dev_audit_output), dev_audit)
    _write_json_atomic(output_dir / "build_report.json", report)
    print(json.dumps({"release_gate_passed": report["release_gate_passed"], **report["components"], "preference": report["preference"], "dev_audit": report["dev_audit"]}, ensure_ascii=False))
    if not report["release_gate_passed"] and not args.allow_failed_gate:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
