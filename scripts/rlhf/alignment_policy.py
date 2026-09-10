#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-aware RLVF alignment policy and deterministic rewards."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ALL_TASKS = [
    "stock_analysis",
    "quant_strategy",
    "financial_report",
    "sentiment_analysis",
    "financial_qa",
    "risk_assessment",
]


@dataclass(frozen=True)
class AlignmentPolicy:
    task_type: str
    primary_method: str
    secondary_methods: tuple[str, ...]
    dpo_target: int
    grpo_target: int
    reward_name: str
    metrics: tuple[str, ...]
    notes: str


ALIGNMENT_POLICIES: dict[str, AlignmentPolicy] = {
    "stock_analysis": AlignmentPolicy(
        task_type="stock_analysis",
        primary_method="dpo",
        secondary_methods=("grpo",),
        dpo_target=5000,
        grpo_target=1000,
        reward_name="technical_structure_reward",
        metrics=("technical_coverage", "rouge_l", "risk_disclaimer_rate"),
        notes="DPO optimizes complete analysis; GRPO only rewards verifiable technical structure.",
    ),
    "quant_strategy": AlignmentPolicy(
        task_type="quant_strategy",
        primary_method="grpo",
        secondary_methods=("dpo",),
        dpo_target=2500,
        grpo_target=5000,
        reward_name="quant_code_reward",
        metrics=("syntax_correct_rate", "strategy_element_rate", "backtest_metric_rate"),
        notes="GRPO is preferred because code syntax and strategy elements are rule-verifiable.",
    ),
    "financial_report": AlignmentPolicy(
        task_type="financial_report",
        primary_method="dpo",
        secondary_methods=(),
        dpo_target=5000,
        grpo_target=0,
        reward_name="financial_report_reward",
        metrics=("financial_metric_coverage", "numeric_reference_rate", "risk_highlight_coverage"),
        notes="Open-ended report interpretation is best aligned with preference pairs.",
    ),
    "sentiment_analysis": AlignmentPolicy(
        task_type="sentiment_analysis",
        primary_method="grpo",
        secondary_methods=("dpo",),
        dpo_target=1000,
        grpo_target=5000,
        reward_name="sentiment_label_reward",
        metrics=("accuracy", "macro_f1", "format_compliance_rate"),
        notes="Label accuracy and concise output are directly rewardable.",
    ),
    "financial_qa": AlignmentPolicy(
        task_type="financial_qa",
        primary_method="dpo",
        secondary_methods=("grpo",),
        dpo_target=5000,
        grpo_target=1500,
        reward_name="qa_answer_reward",
        metrics=("rouge_l", "keyword_coverage", "answer_key_accuracy"),
        notes="Open QA uses DPO; extractive or exam-style QA can use GRPO rewards.",
    ),
    "risk_assessment": AlignmentPolicy(
        task_type="risk_assessment",
        primary_method="dpo",
        secondary_methods=("grpo",),
        dpo_target=4000,
        grpo_target=3000,
        reward_name="risk_assessment_reward",
        metrics=("risk_level_extraction_rate", "risk_factor_coverage", "mitigation_coverage"),
        notes="Preference learning handles reasoning quality; GRPO rewards structured risk fields.",
    ),
}


LABELS = ("positive", "negative", "neutral", "yes", "no")
SENTIMENT_KEYWORDS = {
    "positive": ("积极", "正面", "利好", "看多", "上涨", "增长", "positive", "bullish"),
    "negative": ("消极", "负面", "利空", "看空", "下跌", "亏损", "negative", "bearish"),
    "neutral": ("中性", "中立", "观望", "neutral"),
    "yes": ("yes", "是", "有", "包含"),
    "no": ("no", "否", "没有", "不包含"),
}


def get_policy(task_type: str) -> AlignmentPolicy:
    return ALIGNMENT_POLICIES.get(task_type, ALIGNMENT_POLICIES["financial_qa"])


def policy_as_dict() -> dict[str, dict[str, Any]]:
    return {task: asdict(policy) for task, policy in ALIGNMENT_POLICIES.items()}


def write_policy_report(path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(policy_as_dict(), f, ensure_ascii=False, indent=2)


def get_question(item: dict) -> str:
    for turn in item.get("conversations", []):
        if turn.get("from") == "human":
            return turn.get("value", "")
    return item.get("input") or item.get("question") or ""


def get_reference(item: dict) -> str:
    if isinstance(item.get("chosen"), dict):
        return item["chosen"].get("value", "")
    if isinstance(item.get("chosen"), str):
        return item["chosen"]
    if item.get("reference"):
        return item["reference"]
    if item.get("answer"):
        return item["answer"]
    for turn in item.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return item.get("output") or ""


def extract_sentiment_label(text: str) -> str:
    stripped = text.strip()
    if re.fullmatch(r"yes[.!。]?", stripped, re.IGNORECASE):
        return "yes"
    if re.fullmatch(r"no[.!。]?", stripped, re.IGNORECASE):
        return "no"
    lowered = stripped.lower()
    for label, keywords in SENTIMENT_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return label
    return "unknown"


def extract_risk_level(text: str) -> str:
    patterns = [
        ("very_high", ("极高风险", "极高")),
        ("high", ("高风险", "较高", "高")),
        ("medium", ("中等风险", "中风险", "中等", "中")),
        ("low", ("低风险", "较低", "低")),
    ]
    for label, keywords in patterns:
        if any(keyword in text for keyword in keywords):
            return label
    return "unknown"


def extract_answer_key(text: str) -> str:
    match = re.search(r"(?:答案|选项|正确答案)\s*[:：]?\s*([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    stripped = text.strip()
    if re.fullmatch(r"[A-D]", stripped, re.IGNORECASE):
        return stripped.upper()
    return ""


def extract_python_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def _presence_rate(text: str, patterns: list[str]) -> float:
    if not patterns:
        return 1.0
    hits = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
    return hits / len(patterns)


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d+(\.\d+)?%|\d+(\.\d+)?(亿|万|元|倍)", text))


def _format_compliance(text: str, task_type: str) -> float:
    if not text.strip():
        return 0.0
    if task_type == "sentiment_analysis":
        return 1.0 if extract_sentiment_label(text[:80]) != "unknown" and len(text) <= 160 else 0.5
    if task_type == "quant_strategy":
        return 1.0 if "```" in text and "def " in text else 0.5
    return 1.0 if ("\n" in text or re.search(r"^\s*(?:\d+[.、]|[-*])", text, re.MULTILINE)) else 0.5


def _risk_disclaimer(text: str) -> float:
    return 1.0 if any(term in text for term in ["风险", "仅供参考", "不构成投资建议", "谨慎"]) else 0.0


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def reward_sentiment(response: str, reference: str = "", answer_key: str = "") -> dict[str, float]:
    pred = extract_sentiment_label(response)
    gold = answer_key or extract_sentiment_label(reference)
    accuracy = 1.0 if gold != "unknown" and pred == gold else 0.0
    concise = 1.0 if 1 <= len(response.strip()) <= 160 else 0.4
    known = 1.0 if pred != "unknown" else 0.0
    return {
        "label_accuracy": accuracy,
        "format_compliance": known,
        "conciseness": concise,
        "reward": _bounded(0.7 * accuracy + 0.2 * known + 0.1 * concise),
    }


def reward_quant_strategy(response: str, reference: str = "", answer_key: str = "") -> dict[str, float]:
    code = extract_python_code(response)
    try:
        ast.parse(code)
        syntax = 1.0
    except SyntaxError:
        syntax = 0.0
    has_import = 1.0 if re.search(r"^import\s|^from\s", code, re.MULTILINE) else 0.0
    has_function = 1.0 if re.search(r"^def\s", code, re.MULTILINE) else 0.0
    strategy = _presence_rate(response, [r"入场|买入|开仓|信号", r"出场|卖出|平仓", r"止损|风控", r"仓位"])
    metrics = _presence_rate(response, [r"夏普|sharpe", r"回撤|drawdown", r"年化|annual"])
    return {
        "syntax_correct": syntax,
        "has_import": has_import,
        "has_function": has_function,
        "strategy_element_coverage": round(strategy, 4),
        "backtest_metric_coverage": round(metrics, 4),
        "reward": _bounded(0.25 * syntax + 0.15 * has_import + 0.15 * has_function + 0.25 * strategy + 0.2 * metrics),
    }


def reward_stock_analysis(response: str, reference: str = "", answer_key: str = "") -> dict[str, float]:
    technical = _presence_rate(response, [r"趋势", r"支撑", r"压力|阻力", r"成交量|量能", r"MACD", r"RSI"])
    structure = _format_compliance(response, "stock_analysis")
    risk = _risk_disclaimer(response)
    return {
        "technical_coverage": round(technical, 4),
        "format_compliance": structure,
        "risk_disclaimer": risk,
        "reward": _bounded(0.65 * technical + 0.2 * structure + 0.15 * risk),
    }


def reward_financial_report(response: str, reference: str = "", answer_key: str = "") -> dict[str, float]:
    metrics = _presence_rate(response, [r"营收|营业收入", r"净利润", r"ROE|净资产收益率", r"现金流", r"毛利率", r"资产负债率"])
    balance = _presence_rate(response, [r"亮点|优势|改善|增长", r"风险|压力|下滑|异常"])
    numeric = 1.0 if _has_number(response) else 0.0
    return {
        "financial_metric_coverage": round(metrics, 4),
        "risk_highlight_coverage": round(balance, 4),
        "numeric_reference": numeric,
        "reward": _bounded(0.5 * metrics + 0.25 * balance + 0.25 * numeric),
    }


def reward_financial_qa(response: str, reference: str = "", answer_key: str = "") -> dict[str, float]:
    pred_key = extract_answer_key(response)
    gold_key = answer_key or extract_answer_key(reference)
    key_accuracy = 1.0 if gold_key and pred_key == gold_key else 0.0
    reference_terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", reference))
    response_terms = set(re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", response))
    coverage = len(reference_terms & response_terms) / len(reference_terms) if reference_terms else 0.0
    quality = _presence_rate(response, [r"定义|含义|指", r"公式|计算|等于", r"风险|适用|区别|例"])
    if gold_key:
        reward = 0.7 * key_accuracy + 0.2 * min(coverage, 1.0) + 0.1 * quality
    else:
        reward = 0.55 * min(coverage, 1.0) + 0.45 * quality
    return {
        "answer_key_accuracy": key_accuracy if gold_key else 0.0,
        "keyword_coverage": round(min(coverage, 1.0), 4),
        "qa_quality": round(quality, 4),
        "reward": _bounded(reward),
    }


def reward_risk_assessment(response: str, reference: str = "", answer_key: str = "") -> dict[str, float]:
    level = 1.0 if extract_risk_level(response) != "unknown" else 0.0
    factors = _presence_rate(response, [r"信用风险", r"市场风险", r"流动性风险|流动性", r"波动|VaR|回撤", r"杠杆|负债"])
    mitigation = _presence_rate(response, [r"缓释|对冲", r"止损|限额", r"分散|监控|预警"])
    quantitative = 1.0 if re.search(r"\d+(\.\d+)?%|VaR|Beta|回撤|波动率", response, re.IGNORECASE) else 0.0
    return {
        "risk_level_present": level,
        "risk_factor_coverage": round(factors, 4),
        "mitigation_coverage": round(mitigation, 4),
        "quantitative_indicator": quantitative,
        "reward": _bounded(0.25 * level + 0.35 * factors + 0.25 * mitigation + 0.15 * quantitative),
    }


REWARD_FUNCTIONS = {
    "sentiment_analysis": reward_sentiment,
    "quant_strategy": reward_quant_strategy,
    "stock_analysis": reward_stock_analysis,
    "financial_report": reward_financial_report,
    "financial_qa": reward_financial_qa,
    "risk_assessment": reward_risk_assessment,
}


def compute_reward(task_type: str, response: str, reference: str = "", answer_key: str = "") -> dict[str, float]:
    fn = REWARD_FUNCTIONS.get(task_type, reward_financial_qa)
    return fn(response or "", reference or "", answer_key or "")


def summarize_rewards(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row.get("reward", 0.0)) for row in rows]
    if not rewards:
        return {"count": 0, "avg_reward": 0.0, "p10_reward": 0.0, "p50_reward": 0.0, "p90_reward": 0.0}
    sorted_rewards = sorted(rewards)

    def pct(p: float) -> float:
        idx = min(len(sorted_rewards) - 1, max(0, int(round((len(sorted_rewards) - 1) * p))))
        return round(sorted_rewards[idx], 4)

    return {
        "count": len(rewards),
        "avg_reward": round(mean(rewards), 4),
        "p10_reward": pct(0.1),
        "p50_reward": pct(0.5),
        "p90_reward": pct(0.9),
    }
