#!/usr/bin/env python3
"""Build concise, audit-gated SFT v2.2 answer data from frozen v2.1 splits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing.build_sft_v2_1_dataset import (  # noqa: E402
    CYRILLIC_RE,
    _answer_target_errors,
    _assistant_text,
)
from scripts.evaluation.eval_sft_v2_trusted import (  # noqa: E402
    evidence_source_identities,
    get_question,
    normalized_hash,
)
from scripts.evaluation.generate_sft_v2_trusted_predictions import (  # noqa: E402
    SYSTEM_PROMPT,
    TASK_CONTRACTS,
    _claim_plan,
)
from scripts.rag.audit_schema import (  # noqa: E402
    EvidenceRecord,
    build_claims_from_answer,
    canonical_json,
    canonicalize_url,
    content_digest,
    infer_source_quality,
    parse_timestamp,
    strip_thinking,
)
from scripts.rag.reward_v2 import RewardV2Config, compute_auditable_reward  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/sft_v2_1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/sft_v2_2"
DEFAULT_GOLD = PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"
DEFAULT_FROZEN_MANIFEST = PROJECT_ROOT / "saves/eval_results/agentic_v2_1_collection_summary.json"
OUTPUT_FILES = {
    "train": "fin_agentic_sft_v2_2_answer_train.json",
    "eval": "fin_agentic_sft_v2_2_answer_eval.json",
}
INPUT_FILES = {
    "train": "fin_agentic_sft_v2_1_answer_train.json",
    "eval": "fin_agentic_sft_v2_1_answer_eval.json",
}
TARGET_TASKS = set(TASK_CONTRACTS)
FINANCIAL_MARKERS = ("营业收入", "归母口径净利润", "经营活动现金流量净额", "资产负债率", "流动比率")
MARKET_MARKERS = ("收盘价", "近20个交易日收益率", "年化波动率", "最大回撤", "MA5", "MA20")
REQUIRED_EVIDENCE_KINDS = {
    "financial_qa": {"financial"},
    "financial_report": {"financial"},
    "quant_strategy": {"market"},
    "risk_assessment": {"financial", "market"},
    "sentiment_analysis": {"financial"},
    "stock_analysis": {"financial", "market"},
}
THINK_PLANS = {
    "financial_qa": ["核对公式与报告期数值", "区分含义与局限", "逐项引用"],
    "financial_report": ["核对报告期与同比口径", "覆盖现金流杠杆流动性", "分离事实与观点"],
    "quant_strategy": ["核对行情输入", "给出可运行研究代码", "拒绝虚构回测结果"],
    "risk_assessment": ["核对财务与市场风险", "形成风险等级", "给出可验证缓释措施"],
    "stock_analysis": ["核对行情与财务事实", "判断趋势信号", "明确主要风险"],
}


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _prompt(sample: dict[str, Any]) -> dict[str, Any]:
    conversations = sample.get("conversations") or []
    if len(conversations) != 2:
        raise ValueError("answer sample must contain exactly two turns")
    payload = json.loads(str(conversations[0].get("value", "")))
    if not isinstance(payload, dict):
        raise ValueError("answer prompt must be a JSON object")
    return payload


def _evidence_kind(item: dict[str, Any]) -> str:
    quote = str(item.get("exact_quote", ""))
    if any(marker in quote for marker in FINANCIAL_MARKERS):
        return "financial"
    if any(marker in quote for marker in MARKET_MARKERS):
        return "market"
    return "other"


def _evidence_by_kind(prompt: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prompt.get("evidence", []):
        if isinstance(item, dict) and item.get("evidence_id") and item.get("exact_quote"):
            grouped[_evidence_kind(item)].append(item)
    return grouped


def _has_required_evidence(task_type: str, prompt: dict[str, Any]) -> bool:
    present = {kind for kind, rows in _evidence_by_kind(prompt).items() if rows}
    return REQUIRED_EVIDENCE_KINDS[task_type] <= present


def _record_from_prompt(item: dict[str, Any]) -> EvidenceRecord:
    quote = str(item.get("exact_quote", "")).strip()
    source_uri = str(item.get("source") or item.get("source_uri") or "").strip()
    metadata = {
        "source_type": item.get("source_type", ""),
        "reliability_tier": item.get("reliability_tier", ""),
        "publisher": item.get("publisher", ""),
    }
    source_type, tier, publisher = infer_source_quality(source_uri, metadata)
    digest = content_digest(quote)
    return EvidenceRecord(
        evidence_id=str(item["evidence_id"]).upper(),
        source_uri=source_uri,
        canonical_url=canonicalize_url(str(item.get("canonical_url") or source_uri)),
        publisher=publisher,
        source_type=source_type,
        reliability_tier=tier,
        exact_quote=quote,
        content_hash=digest,
        document_version=str(item.get("document_version") or digest[:16]),
        published_at=str(item.get("published_at") or ""),
        effective_at=str(item.get("effective_at") or ""),
        fetched_at=str(item.get("fetched_at") or ""),
        point_in_time_available=True,
    )


def _provenance_heading(item: dict[str, Any]) -> str:
    evidence_id = str(item["evidence_id"]).upper()
    publisher = str(item.get("publisher") or "unknown")
    tier = str(item.get("reliability_tier") or "unknown")
    published = str(item.get("published_at") or "未提供")
    effective = str(item.get("effective_at") or "未提供")
    fetched = str(item.get("fetched_at") or "未提供")
    return (
        f"### 证据 [{evidence_id}] | 来源：{publisher}（{tier}） | "
        f"发布时间：{published} | 生效时间：{effective} | 获取时间：{fetched}"
    )


def _quote_fragments(item: dict[str, Any], keywords: tuple[str, ...] = ()) -> list[str]:
    text = re.sub(r"^#+\s*可引用事实\s*", "", str(item.get("exact_quote", "")).strip())
    fragments = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*", text) if part.strip()]
    if keywords:
        fragments = [part for part in fragments if any(keyword in part for keyword in keywords)]
    evidence_id = str(item["evidence_id"]).upper()
    cited = []
    for fragment in fragments:
        if fragment[-1:] in "。！？!?；;":
            cited.append(f"{fragment[:-1].rstrip()} [{evidence_id}]{fragment[-1]}")
        else:
            cited.append(f"{fragment} [{evidence_id}]。")
    return cited


def _metric(text: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}(?:为|变化|达到)?\s*([-+]?\d+(?:\.\d+)?)%?", text)
    return float(match.group(1)) if match else None


def _citations(items: list[dict[str, Any]]) -> str:
    return "".join(f"[{str(item['evidence_id']).upper()}]" for item in items)


def _opinion(text: str, items: list[dict[str, Any]]) -> str:
    return f"观点：{text} {_citations(items)}。"


def _structured_thinking(task_type: str, items: list[dict[str, Any]]) -> str:
    if task_type == "sentiment_analysis":
        return ""
    payload = {
        "evidence_ids": [str(item["evidence_id"]).upper() for item in items],
        "plan": THINK_PLANS[task_type],
    }
    return f"<think>\n{canonical_json(payload)}\n</think>\n\n"


def _financial_direction(text: str) -> str:
    revenue = _metric(text, "营业收入同比变化")
    profit = _metric(text, "净利润同比变化")
    if revenue is not None and profit is not None:
        if revenue > 0 and profit > 0:
            return "收入与利润同比方向均为增长，基本面增长信号一致"
        if revenue < 0 and profit < 0:
            return "收入与利润同比方向均为下降，基本面承压信号一致"
    return "收入与利润同比方向并不一致，基本面信号分化"


def _sentiment_label(text: str) -> str:
    revenue = _metric(text, "营业收入同比变化")
    profit = _metric(text, "净利润同比变化")
    cash_flow = _metric(text, "经营活动现金流量净额")
    profit_loss = "净利润由盈利转为亏损" in text
    if profit_loss or (revenue is not None and profit is not None and revenue < 0 and profit < 0):
        return "negative"
    if revenue is not None and profit is not None and cash_flow is not None:
        if revenue > 0 and profit > 0 and cash_flow > 0:
            return "positive"
        if profit < 0 and cash_flow < 0:
            return "negative"
    return "neutral"


def _risk_level(financial_text: str, market_text: str) -> str:
    leverage = _metric(financial_text, "资产负债率")
    liquidity = _metric(financial_text, "流动比率")
    volatility = _metric(market_text, "年化波动率")
    drawdown = _metric(market_text, "最大回撤")
    if (
        (leverage is not None and leverage >= 70)
        or (liquidity is not None and liquidity < 1)
        or (volatility is not None and volatility >= 50)
        or (drawdown is not None and drawdown <= -20)
    ):
        return "高"
    if (
        (leverage is not None and leverage >= 50)
        or (liquidity is not None and liquidity < 1.5)
        or (volatility is not None and volatility >= 30)
        or (drawdown is not None and drawdown <= -10)
    ):
        return "中"
    return "低"


def _trend_view(market_text: str) -> str:
    short = _metric(market_text, "MA5")
    long = _metric(market_text, "MA20")
    recent_return = _metric(market_text, "近20个交易日收益率")
    if short is not None and long is not None and recent_return is not None:
        if short > long and recent_return > 0:
            return "短期均线与近期收益方向共同指向偏强趋势"
        if short < long and recent_return < 0:
            return "短期均线与近期收益方向共同指向偏弱趋势"
    return "短期均线与近期收益方向出现分化，趋势信号需谨慎解释"


QUANT_CODE = """```python
import numpy as np
import pandas as pd

def research_strategy(df, price_col="close", fast=5, slow=20,
                      risk_fraction=0.01, stop_loss_pct=0.05):
    if price_col not in df.columns:
        raise ValueError(f"missing column: {price_col}")
    out = df.copy()
    close = out[price_col].astype(float)
    out["return"] = close.pct_change()
    out["ma_fast"] = close.rolling(fast).mean()
    out["ma_slow"] = close.rolling(slow).mean()
    cross_up = (out["ma_fast"] > out["ma_slow"]) & (out["ma_fast"].shift(1) <= out["ma_slow"].shift(1))
    cross_down = (out["ma_fast"] < out["ma_slow"]) & (out["ma_fast"].shift(1) >= out["ma_slow"].shift(1))
    out["signal"] = np.select([cross_up, cross_down], [1, -1], default=0)
    state = out["signal"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)
    annual_vol = out["return"].rolling(slow).std() * np.sqrt(252)
    size = (risk_fraction / annual_vol.replace(0, np.nan)).clip(0, 1).fillna(0)
    out["position"] = (state * size).shift(1).fillna(0)
    entry_price = close.where(cross_up).ffill()
    out["stop_price"] = entry_price * (1 - stop_loss_pct)
    out.loc[close <= out["stop_price"], "position"] = 0
    out["strategy_return"] = out["position"] * out["return"]
    equity = (1 + out["strategy_return"].fillna(0)).cumprod()
    out["drawdown"] = equity / equity.cummax() - 1
    volatility = out["strategy_return"].std()
    sharpe = np.nan if not volatility or np.isnan(volatility) else np.sqrt(252) * out["strategy_return"].mean() / volatility
    return out, {"sharpe": sharpe, "max_drawdown": out["drawdown"].min()}
```"""


def build_canonical_target(task_type: str, prompt: dict[str, Any]) -> str:
    grouped = _evidence_by_kind(prompt)
    financial = grouped.get("financial", [])[:1]
    market = grouped.get("market", [])[:1]
    if not _has_required_evidence(task_type, prompt):
        raise ValueError(f"missing required evidence for {task_type}")

    if task_type == "financial_qa":
        used = financial
        facts = _quote_fragments(
            financial[0],
            ("资产总计", "负债合计", "流动资产", "流动负债", "资产负债率", "流动比率"),
        )
        body = [
            "### 公式核对",
            *facts,
            "### 含义与局限",
            _opinion("资产负债率公式使用负债合计与资产总计，指标用于观察杠杆，但不能单独解释偿债压力", used),
            _opinion("流动比率公式使用流动资产与流动负债，指标用于观察短期偿债能力，但未反映资产变现质量", used),
            _opinion("上述定义和结果仅适用于证据所列报告期，跨期比较前应统一口径", used),
        ]
    elif task_type == "financial_report":
        used = financial
        financial_text = str(financial[0]["exact_quote"])
        body = [
            "### 财务事实",
            *_quote_fragments(financial[0]),
            "### 解读与局限",
            _opinion(_financial_direction(financial_text), used),
            _opinion("经营现金流、杠杆和流动性应结合后续报告期持续复核，当前结论不外推到未来", used),
        ]
    elif task_type == "sentiment_analysis":
        used = financial
        financial_text = str(financial[0]["exact_quote"])
        label = _sentiment_label(financial_text)
        body = [
            "### 财务事实",
            *_quote_fragments(financial[0], ("营业收入", "净利润", "经营活动现金流量净额", "同比变化")),
            _opinion(f"基本面情绪标签为{label}", used),
            _opinion(f"该标签依据收入、利润及经营现金流方向综合判定，当前结果为{label}", used),
        ]
    elif task_type == "stock_analysis":
        used = market + financial
        market_text = str(market[0]["exact_quote"])
        financial_text = str(financial[0]["exact_quote"])
        body = [
            "### 行情事实",
            *_quote_fragments(market[0]),
            "### 财务事实",
            *_quote_fragments(financial[0], ("营业收入", "净利润", "经营活动现金流量净额", "同比变化")),
            "### 趋势与风险",
            _opinion(_trend_view(market_text), market),
            _opinion(_financial_direction(financial_text), financial),
            _opinion("波动与回撤是主要市场风险，财务结论还需结合后续报告期持续复核", used),
        ]
    elif task_type == "risk_assessment":
        used = financial + market
        financial_text = str(financial[0]["exact_quote"])
        market_text = str(market[0]["exact_quote"])
        level = _risk_level(financial_text, market_text)
        body = [
            "### 财务风险事实",
            *_quote_fragments(
                financial[0],
                ("净利润", "经营活动现金流量净额", "资产负债率", "流动比率", "流动资产", "流动负债"),
            ),
            "### 市场风险事实",
            *_quote_fragments(market[0], ("年化波动率", "最大回撤")),
            "### 风险结论与缓释",
            _opinion(f"综合风险等级为{level}，排序时优先检查触发该等级的市场、杠杆或流动性指标", used),
            _opinion("缓释措施包括控制仓位、设置止损、监控现金流与流动性，并在数据更新后复核风险等级", used),
        ]
    elif task_type == "quant_strategy":
        used = market
        body = [
            "### 可审计行情输入",
            *_quote_fragments(market[0]),
            "### 研究代码",
            QUANT_CODE,
            "### 规则与评估",
            _opinion("signal 由均线交叉产生，入场和出场使用相反方向的交叉条件", used),
            _opinion("仓位采用波动率约束，止损参数作为可配置研究假设，不代表收益承诺", used),
            "证据不足，无法确认该策略的Sharpe和回撤回测结果，需要使用独立历史序列并计入交易成本后评估。",
        ]
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    provenance = [_provenance_heading(item) for item in used]
    return _structured_thinking(task_type, used) + "\n".join([*provenance, *body]).strip()


def audit_target(task_type: str, prompt: dict[str, Any], target: str) -> dict[str, Any]:
    evidence = [_record_from_prompt(item) for item in prompt.get("evidence", [])]
    return compute_auditable_reward(
        query=str(prompt.get("query", "")),
        answer=strip_thinking(target),
        evidence=evidence,
        claims=build_claims_from_answer(target, str(prompt.get("request_as_of", ""))),
        task_type=task_type,
        request_as_of=str(prompt.get("request_as_of", "")),
        trajectory=[],
        config=RewardV2Config(require_complete_trajectory=False),
    )


def repair_sample(sample: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    task_type = str(sample.get("task_type", ""))
    if task_type not in TARGET_TASKS:
        return None, ["invalid_task_type"]
    try:
        prompt = _prompt(sample)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None, ["invalid_prompt"]
    if not _has_required_evidence(task_type, prompt):
        return None, ["incomplete_task_evidence"]

    prompt["claim_plan"] = _claim_plan(task_type)
    prompt["output_contract"] = TASK_CONTRACTS[task_type]
    try:
        target = build_canonical_target(task_type, prompt)
        reward = audit_target(task_type, prompt, target)
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"canonicalization_error:{type(exc).__name__}"]

    repaired = copy.deepcopy(sample)
    original_target = _assistant_text(sample)
    repaired["system"] = SYSTEM_PROMPT
    repaired["conversations"][0]["value"] = canonical_json(prompt)
    repaired["conversations"][1]["value"] = target
    repaired["dataset_version"] = "sft_v2.2"
    repaired["training_component"] = "answer"
    repaired["repair_profile"] = "concise_audit_first"
    repaired["source_dataset_version"] = str(sample.get("dataset_version", "sft_v2.1"))
    repaired["original_target_sha256"] = hashlib.sha256(original_target.encode("utf-8")).hexdigest()
    repaired["canonical_target_sha256"] = hashlib.sha256(target.encode("utf-8")).hexdigest()
    repaired["target_audit"] = {
        "hard_gate_passed": reward["hard_gate_passed"],
        "hard_failures": reward["hard_failures"],
        "base_reward": reward["base_reward"],
        "task_validity": reward["task_validity"],
        "claim_support": reward["claim_support"],
        "numeric_consistency": reward["numeric_consistency"],
        "citation_coverage": reward["citation_coverage"],
        "citation_precision": reward["citation_precision"],
    }

    errors.extend(_answer_target_errors(repaired))
    if not reward["hard_gate_passed"]:
        errors.extend(f"audit:{item}" for item in reward["hard_failures"])
    minimum_task_validity = 0.3333 if task_type == "sentiment_analysis" else 1.0
    if reward["task_validity"] < minimum_task_validity:
        errors.append("task_validity_below_minimum")
    if CYRILLIC_RE.search(target):
        errors.append("cyrillic_target")
    if target.count("<think>") != target.count("</think>"):
        errors.append("unbalanced_thinking_tags")
    if len(strip_thinking(target)) > 6000:
        errors.append("target_too_long")
    return (None, sorted(set(errors))) if errors else (repaired, [])


def oversample_train(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    missing_tasks = TARGET_TASKS - set(grouped)
    if missing_tasks:
        raise ValueError(f"cannot oversample tasks with zero accepted samples: {sorted(missing_tasks)}")
    target = max(len(items) for items in grouped.values())
    output = list(rows)
    added: dict[str, int] = {}
    for task_type in sorted(TARGET_TASKS):
        items = sorted(grouped[task_type], key=lambda item: str(item.get("id", "")))
        missing = target - len(items)
        added[task_type] = missing
        for index in range(missing):
            source = items[index % len(items)]
            replica = copy.deepcopy(source)
            replica["replica_of"] = str(source["id"])
            replica["id"] = f"{source['id']}:v2.2-repeat-{index // len(items) + 1}"
            replica["oversampled"] = True
            output.append(replica)
    output.sort(key=lambda item: (str(item.get("task_type", "")), str(item.get("id", ""))))
    return output, added


def _dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_lengths = [len(_assistant_text(row)) for row in rows]
    audit_passes = sum(bool((row.get("target_audit") or {}).get("hard_gate_passed")) for row in rows)
    return {
        "samples": len(rows),
        "unique_groups": len({str(row.get("group_id", "")) for row in rows}),
        "by_task": dict(sorted(Counter(str(row.get("task_type", "")) for row in rows).items())),
        "unique_groups_by_task": dict(
            sorted(
                (task, len({str(row.get("group_id", "")) for row in rows if row.get("task_type") == task}))
                for task in TARGET_TASKS
            )
        ),
        "audit_hard_pass_rate": round(audit_passes / len(rows), 4) if rows else 0.0,
        "cyrillic_targets": sum(bool(CYRILLIC_RE.search(_assistant_text(row))) for row in rows),
        "thinking_targets": sum("<think>" in _assistant_text(row) for row in rows),
        "max_target_chars": max(target_lengths, default=0),
        "mean_target_chars": round(sum(target_lengths) / len(target_lengths), 1) if rows else 0.0,
        "fingerprint": hashlib.sha256(
            canonical_json(
                [
                    {
                        "id": row.get("id"),
                        "group_id": row.get("group_id"),
                        "task_type": row.get("task_type"),
                        "prompt": hashlib.sha256(str(row["conversations"][0]["value"]).encode()).hexdigest(),
                        "target": hashlib.sha256(_assistant_text(row).encode()).hexdigest(),
                    }
                    for row in rows
                ]
            ).encode()
        ).hexdigest(),
    }


def _heldout_overlap(train_rows: list[dict[str, Any]], gold_path: Path) -> dict[str, int]:
    gold = _load_json(gold_path)
    train_questions = set()
    train_sources = set()
    for row in train_rows:
        prompt = _prompt(row)
        train_questions.add(normalized_hash(str(prompt.get("query", ""))))
        train_sources.update(str(item) for item in row.get("source_groups", []) if item)
        if row.get("source_group"):
            train_sources.add(str(row["source_group"]))
    gold_questions = {normalized_hash(get_question(case)) for case in gold}
    gold_sources = set().union(*(evidence_source_identities(case) for case in gold))
    gold_sources.update(str(case.get("source_group", "")) for case in gold if case.get("source_group"))
    return {
        "question_overlap": len(train_questions & gold_questions),
        "source_identity_overlap": len(train_sources & gold_sources),
    }


def build_datasets(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    gold_path: Path,
    frozen_manifest_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    repaired: dict[str, list[dict[str, Any]]] = {}
    rejections: dict[str, Any] = {}
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        accepted = []
        reasons = Counter()
        for row in rows:
            result, errors = repair_sample(row)
            if result is None:
                reasons.update(errors)
            else:
                accepted.append(result)
        repaired[split] = accepted
        rejections[split] = {"input": len(rows), "accepted": len(accepted), "rejected": len(rows) - len(accepted), "reasons": dict(reasons.most_common())}

    unique_train = repaired["train"]
    repaired["train"], oversampling = oversample_train(unique_train)
    stats = {split: _dataset_stats(rows) for split, rows in repaired.items()}
    train_groups = {str(row.get("group_id", "")) for row in repaired["train"]}
    eval_groups = {str(row.get("group_id", "")) for row in repaired["eval"]}
    overlap = _heldout_overlap(unique_train, gold_path)

    frozen_manifest = _load_json(frozen_manifest_path)
    expected_hashes = dict(frozen_manifest.get("frozen_artifacts") or {})
    frozen = {}
    for relative in ("data/evaluation/trusted_finance_v2.json", "data/evaluation/v2_split_manifest.json"):
        path = PROJECT_ROOT / relative
        current = _sha256_file(path)
        expected = str(expected_hashes.get(relative, ""))
        frozen[relative] = {"expected_sha256": expected, "current_sha256": current, "matches": bool(expected) and current == expected}

    release_gate = {
        "derived_only_from_v2_1_splits": True,
        "train_eval_group_disjoint": not (train_groups & eval_groups),
        "minimum_train_unique_groups_per_task": all(value >= 50 for value in stats["train"]["unique_groups_by_task"].values()),
        "minimum_eval_unique_groups_per_task": all(value >= 8 for value in stats["eval"]["unique_groups_by_task"].values()),
        "balanced_training_samples": len(set(stats["train"]["by_task"].values())) == 1,
        "all_targets_audit_hard_pass": stats["train"]["audit_hard_pass_rate"] == 1.0 and stats["eval"]["audit_hard_pass_rate"] == 1.0,
        "no_cyrillic_targets": stats["train"]["cyrillic_targets"] == 0 and stats["eval"]["cyrillic_targets"] == 0,
        "bounded_target_length": stats["train"]["max_target_chars"] <= 6000 and stats["eval"]["max_target_chars"] <= 6000,
        "heldout_question_disjoint": overlap["question_overlap"] == 0,
        "heldout_source_disjoint": overlap["source_identity_overlap"] == 0,
        "frozen_heldout_artifacts_unchanged": all(item["matches"] for item in frozen.values()),
    }
    report = {
        "schema_version": "sft_v2.2",
        "profile": "concise_audit_first_answer_repair",
        "source_files": {split: INPUT_FILES[split] for split in INPUT_FILES},
        "rejections": rejections,
        "oversampling": {"strategy": "balance_to_largest_task", "added_by_task": oversampling},
        "components": stats,
        "heldout_overlap": overlap,
        "frozen_artifacts": frozen,
        "release_gate": release_gate,
        "release_gate_passed": all(release_gate.values()),
    }
    return repaired, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audit-first SFT v2.2 answer data")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--frozen-manifest", default=str(DEFAULT_FROZEN_MANIFEST))
    parser.add_argument("--allow-failed-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    inputs = {split: _load_json(input_dir / filename) for split, filename in INPUT_FILES.items()}
    datasets, report = build_datasets(
        inputs["train"],
        inputs["eval"],
        gold_path=Path(args.gold),
        frozen_manifest_path=Path(args.frozen_manifest),
    )
    for split, filename in OUTPUT_FILES.items():
        _write_json_atomic(output_dir / filename, datasets[split])
    _write_json_atomic(output_dir / "build_report.json", report)
    print(
        json.dumps(
            {
                "release_gate_passed": report["release_gate_passed"],
                "release_gate": report["release_gate"],
                "components": report["components"],
                "rejections": report["rejections"],
            },
            ensure_ascii=False,
        )
    )
    if not report["release_gate_passed"] and not args.allow_failed_gate:
        print("SFT v2.2 data release gate failed; see build_report.json.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
