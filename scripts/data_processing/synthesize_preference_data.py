#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好数据合成脚本
================
从已有 SFT 数据生成 DPO 训练所需的偏好数据对 (chosen / rejected)。

支持三种模式:
1. rules  — 基于规则自动标注（快速，适合情感分析等有明确标准的任务）
2. llm    — 使用外部 LLM (如 GPT-4o-mini) 对两个候选回答评分，选出 chosen/rejected
3. both   — 两种模式结合（推荐）

输出格式符合 LLaMA-Factory DPO 数据规范:
  [{"conversations": [...], "chosen": {"from":"gpt","value":"..."}, "rejected": {"from":"gpt","value":"..."}}]

用法:
  # 规则模式（快速生成，无需 API）
  python scripts/data_processing/synthesize_preference_data.py \\
      --input data/sft/fin_instruct_train.json \\
      --output data/rlhf/fin_preference_train.json \\
      --mode rules --max-samples 2000

  # LLM Judge 模式（高质量，需配置 API Key）
  export SYNTH_API_KEY="your-key"
  export SYNTH_API_BASE="https://api.openai.com/v1"
  export SYNTH_API_MODEL="gpt-4o-mini"
  python scripts/data_processing/synthesize_preference_data.py \\
      --input data/sft/fin_instruct_train.json \\
      --output data/rlhf/fin_preference_train.json \\
      --mode llm --max-samples 5000

  # 两种模式结合（推荐）
  python scripts/data_processing/synthesize_preference_data.py \\
      --input data/sft/fin_instruct_train.json \\
      --output data/rlhf/fin_preference_train.json \\
      --mode both --max-samples 5000
"""

import argparse
import hashlib
import json
import os
import re
import random
import time
import logging
import sys
from collections import Counter, defaultdict
from typing import Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.rlhf.alignment_policy import (
    ALIGNMENT_POLICIES,
    ALL_TASKS,
    compute_reward,
    extract_risk_level,
    extract_sentiment_label,
    get_policy,
    write_policy_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────────────────────
SENTIMENT_KEYWORDS = {
    "positive": [
        "积极", "正面", "利好", "看涨", "买入", "增持", "强烈推荐",
        "大幅增长", "超预期", "突破", "创新高", "爆发", "优秀",
    ],
    "negative": [
        "消极", "负面", "利空", "看跌", "卖出", "减持", "回避",
        "大幅下跌", "不及预期", "跌破", "创新低", "暴跌", "亏损",
    ],
    "neutral": ["中性", "中立", "观望", "持平", "不变"],
}

# 金融分析高质量要素
QUALITY_INDICATORS = {
    "high": [
        r"\d+\.?\d*%",       # 包含具体百分比数字
        r"(?:ROE|ROA|PE|PB|EPS|MACD|RSI|KDJ|VaR|夏普比率)",  # 专业指标
        r"(?:支撑位|压力位|阻力位|均线|趋势线)",  # 技术分析术语
        r"(?:现金流|利润率|净资产|市值|营收)",    # 财务术语
        r"(?:①|②|③|\d+\.\s)",  # 结构化输出
    ],
    "low": [
        r"^.{0,50}$",        # 过短回答（少于50字）
        r"(?:无法回答|不清楚|不确定|建议咨询|无法判断)",  # 拒绝回答
        r"(?:如有需要|如需更多|希望对您有帮助)$",  # 泛化结尾无实质内容
    ],
}

DEGRADED_RESPONSES = [
    "这需要进一步分析，请咨询专业人士。",
    "我无法对此给出具体建议，投资需谨慎。",
    "这是一个复杂的问题，答案取决于多种因素。",
    "建议您查阅更详细的资料或咨询专业的金融顾问。",
    "针对您的问题，我建议您保持谨慎态度，毕竟市场充满变数。",
    "这个问题没有简单的答案。市场情况复杂，需要综合考量。",
    "投资有风险，我无法给出具体的买卖建议。",
    "从表面来看，这家公司可能存在一些机会，但也有风险。",
]

TASK_DEGRADED_RESPONSES = {
    "stock_analysis": [
        "该股票走势需要继续观察，短期难以判断，建议谨慎。",
        "整体看可能有机会，但也存在风险，具体买卖点需要再分析。",
    ],
    "financial_report": [
        "公司财务表现一般，具体指标还需要查看报表后再判断。",
        "财报有亮点也有风险，建议继续关注后续公告。",
    ],
    "financial_qa": [
        "这是一个金融基础概念，具体含义需要结合实际情况理解。",
        "该问题较复杂，建议查阅教材或咨询专业人士。",
    ],
    "risk_assessment": [
        "综合来看存在一定风险，建议投资者保持谨慎。",
        "风险水平需要结合更多资料判断，暂不做明确评级。",
    ],
    "quant_strategy": [
        "可以根据均线变化买入卖出，并注意控制风险。",
        "策略需要回测后才能判断是否有效。",
    ],
}


# ──────────────────────────────────────────────────────────────
# 质量评分
# ──────────────────────────────────────────────────────────────
def score_response(response: str, task_type: str = "financial_qa") -> float:
    """基于规则对一个回答进行质量评分，返回 0~1."""
    score = 0.5  # 基础分

    # 长度奖励（金融分析需要一定篇幅）
    length = len(response)
    if length > 500:
        score += 0.15
    elif length > 200:
        score += 0.08
    elif length < 80:
        score -= 0.25  # 严重惩罚过短回答

    # 高质量标志
    for pattern in QUALITY_INDICATORS["high"]:
        if re.search(pattern, response):
            score += 0.05

    # 低质量标志
    for pattern in QUALITY_INDICATORS["low"]:
        if re.search(pattern, response):
            score -= 0.15

    # 结构化输出奖励（包含 markdown 标题/列表）
    if re.search(r"^#{1,3}\s", response, re.MULTILINE):
        score += 0.08
    if re.search(r"^\s*[-*•]\s", response, re.MULTILINE):
        score += 0.05
    if re.search(r"\|.*\|.*\|", response):  # 表格
        score += 0.08

    # 免责声明存在但不是主要内容（加分）
    if "免责声明" in response or "仅供参考" in response:
        if length > 300:
            score += 0.03

    return max(0.0, min(1.0, score))


def _wrong_sentiment_response(original: str) -> str:
    current = extract_sentiment_label(original)
    labels = ["positive", "negative", "neutral"]
    if current in {"yes", "no"}:
        labels = ["yes", "no"]
    alternatives = [label for label in labels if label != current]
    wrong = random.choice(alternatives or labels)
    zh = {"positive": "积极", "negative": "消极", "neutral": "中性", "yes": "Yes", "no": "No"}[wrong]
    if wrong in {"yes", "no"}:
        return zh
    return f"情感倾向：{zh}。理由：该新闻对市场影响较为明确。"


def _weaken_quant_code(original: str) -> str:
    return (
        "策略思路：使用简单均线判断买卖点，但未提供完整回测。\n"
        "```python\n"
        "def strategy(data):\n"
        "    data['signal'] = 0\n"
        "    return data\n"
        "```"
    )


def _remove_numbers_and_terms(text: str) -> str:
    text = re.sub(r"\d+(\.\d+)?%?", "若干", text)
    text = re.sub(r"(ROE|ROA|PE|PB|EPS|MACD|RSI|KDJ|VaR|Beta)", "指标", text, flags=re.IGNORECASE)
    return text


def create_degraded_response(original: str, task_type: str) -> str:
    """基于原始回答生成一个"更差"的版本，用作 rejected。"""
    if task_type == "sentiment_analysis":
        return _wrong_sentiment_response(original)
    if task_type == "quant_strategy" and random.random() < 0.55:
        return _weaken_quant_code(original)

    # 随机选择劣化策略
    strategy = random.choice(["truncate", "template", "flatten", "drop_terms"])

    if strategy == "truncate" and len(original) > 200:
        # 截断，只保留前 1/3
        cut = len(original) // 3
        return original[:cut] + "……（内容省略）"

    elif strategy == "template":
        # 使用泛化模板回答
        return random.choice(TASK_DEGRADED_RESPONSES.get(task_type, DEGRADED_RESPONSES))

    elif strategy == "drop_terms":
        degraded = _remove_numbers_and_terms(original)
        if len(degraded) > 350:
            degraded = degraded[:350] + "..."
        return degraded.strip() or random.choice(TASK_DEGRADED_RESPONSES.get(task_type, DEGRADED_RESPONSES))

    else:
        # 去除结构化格式，保留纯文本
        flat = re.sub(r"#{1,6}\s", "", original)
        flat = re.sub(r"^\s*[-*•]\s", "，", flat, flags=re.MULTILINE)
        flat = re.sub(r"\|[^\n]+\|", "", flat)
        flat = re.sub(r"\n{2,}", "\n", flat)
        # 截断到较短
        if len(flat) > 300:
            flat = flat[:300] + "..."
        return flat.strip() or random.choice(DEGRADED_RESPONSES)


# ──────────────────────────────────────────────────────────────
# 规则模式
# ──────────────────────────────────────────────────────────────
def _task_quality_score(response: str, task_type: str, reference: str = "") -> float:
    reward = compute_reward(task_type, response, reference)
    base = score_response(response, task_type)
    return round(0.55 * base + 0.45 * reward.get("reward", 0.0), 4)


def _include_for_dpo(task_type: str, current_count: int, max_per_task: int | None) -> bool:
    policy = get_policy(task_type)
    if task_type == "sentiment_analysis":
        target = min(policy.dpo_target, max_per_task or policy.dpo_target)
        return current_count < target
    if max_per_task is None:
        return True
    return current_count < max_per_task


def process_rules_mode(samples: list[dict], max_per_task: int | None = None) -> list[dict]:
    """
    基于规则为每个样本生成 chosen/rejected 对。

    - 原始高质量回答 → chosen
    - 合成的劣化版本 → rejected
    """
    results = []
    skipped = 0

    task_counts = Counter()

    for item in samples:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            skipped += 1
            continue

        human_turn = next((c for c in convs if c["from"] == "human"), None)
        gpt_turn = next((c for c in convs if c["from"] == "gpt"), None)

        if not human_turn or not gpt_turn:
            skipped += 1
            continue

        chosen_text = gpt_turn["value"]
        task_type = item.get("task_type", "financial_qa")
        if task_type not in ALIGNMENT_POLICIES:
            task_type = "financial_qa"

        if not _include_for_dpo(task_type, task_counts[task_type], max_per_task):
            continue

        # 质量检查：原始回答不能太短
        min_len = 2 if task_type == "sentiment_analysis" else 60
        if len(chosen_text) < min_len:
            skipped += 1
            continue

        chosen_score = _task_quality_score(chosen_text, task_type, chosen_text)
        min_score = 0.1 if task_type == "sentiment_analysis" else 0.35
        if chosen_score < min_score:
            skipped += 1
            continue

        rejected_text = create_degraded_response(chosen_text, task_type)
        rejected_score = _task_quality_score(rejected_text, task_type, chosen_text)

        results.append({
            "conversations": [{"from": "human", "value": human_turn["value"]}],
            "chosen": {"from": "gpt", "value": chosen_text},
            "rejected": {"from": "gpt", "value": rejected_text},
            "system": item.get("system", ""),
            "task_type": task_type,
            "alignment_method": "dpo",
            "alignment_policy": get_policy(task_type).primary_method,
            "_source": "rules",
            "_chosen_score": round(chosen_score, 3),
            "_rejected_score": round(rejected_score, 3),
        })
        task_counts[task_type] += 1

    logger.info(f"Rules mode: {len(results)} pairs generated, {skipped} skipped")
    return results


# ──────────────────────────────────────────────────────────────
# LLM Judge 模式
# ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """你是一位资深金融分析专家，负责评估两个金融AI助手的回答质量。

评估维度：
1. **专业性**：是否使用了正确的金融术语和分析框架
2. **准确性**：内容是否符合金融常识和事实
3. **实用性**：回答是否有实质性帮助，而非空洞泛泛
4. **结构性**：是否有清晰的逻辑结构和层次
5. **完整性**：是否覆盖了问题的关键方面

请根据以上维度，判断哪个回答更优质，回复格式必须是:
WINNER: A 或 WINNER: B
REASON: 简短理由（一句话）"""

JUDGE_USER_TEMPLATE = """用户问题：
{question}

---
回答A：
{response_a}

---
回答B：
{response_b}

---
请判断哪个回答更优质？"""

JUDGE_RETRY_SYSTEM_PROMPT = """你只需要判断 A 或 B 哪个回答更好。
必须只输出一个字母：A 或 B。不要输出解释。"""

JUDGE_RETRY_USER_TEMPLATE = """问题：{question}

A：
{response_a}

B：
{response_b}

更好的回答是 A 还是 B？只输出 A 或 B。"""


def _extract_judge_text(resp) -> tuple[str, str]:
    """Extract text from OpenAI-compatible chat responses."""
    if not getattr(resp, "choices", None):
        return "", "no_choices"

    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", "") or ""
    message = getattr(choice, "message", None)
    if message is None:
        return "", finish_reason

    parts = []
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))

    # Some reasoning-model compatible APIs put text here and leave content empty.
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        parts.append(reasoning_content)

    return "\n".join(parts).strip(), finish_reason


def _parse_judge_winner(text: str) -> Optional[str]:
    """Parse A/B winner from several common judge output formats."""
    if not text:
        return None

    match = re.search(r"WINNER\s*[:：]\s*([AB])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'"winner"\s*:\s*"?(A|B)"?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    stripped = text.strip()
    if re.fullmatch(r"[AB]", stripped, re.IGNORECASE):
        return stripped.upper()

    match = re.search(r"(?:选择|答案|更好的是|更优的是)\s*[:：]?\s*([AB])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    if "回答A" in text and any(word in text for word in ["更好", "更优", "胜出"]):
        return "A"
    if "回答B" in text and any(word in text for word in ["更好", "更优", "胜出"]):
        return "B"

    return None


def call_llm_judge(
    question: str,
    response_a: str,
    response_b: str,
    api_key: str,
    api_base: str,
    model: str,
    max_retries: int = 3,
) -> Optional[str]:
    """
    调用外部 LLM 评判两个回答，返回 'A' 或 'B' 或 None（失败时）。
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("请安装 openai: pip install openai")
        return None

    client = OpenAI(api_key=api_key, base_url=api_base)

    for attempt in range(max_retries):
        try:
            if attempt == 0:
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(
                            question=question[:500],
                            response_a=response_a[:800],
                            response_b=response_b[:800],
                        ),
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": JUDGE_RETRY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": JUDGE_RETRY_USER_TEMPLATE.format(
                            question=question[:300],
                            response_a=response_a[:500],
                            response_b=response_b[:500],
                        ),
                    },
                ]

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
            text, finish_reason = _extract_judge_text(resp)
            winner = _parse_judge_winner(text)
            if winner:
                return winner

            if not text:
                logger.warning(
                    "LLM judge returned empty content "
                    f"(attempt={attempt+1}, finish_reason={finish_reason or 'unknown'}, model={model})"
                )
            else:
                logger.warning(f"LLM judge output not parseable: {text[:200]!r}")

        except Exception as e:
            logger.warning(f"LLM judge attempt {attempt+1} failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return None


def process_llm_mode(
    samples: list[dict],
    api_key: str,
    api_base: str,
    model: str,
    request_delay: float = 0.5,
    max_per_task: int | None = None,
) -> list[dict]:
    """
    使用 LLM Judge 为每个样本生成 chosen/rejected 对。

    策略：
    1. 原始高质量回答 = 候选 A
    2. 规则劣化版本 = 候选 B
    3. LLM 判断哪个更好 → 确保标注质量
    """
    results = []
    skipped = 0
    judge_wrong = 0

    task_counts = Counter()

    for i, item in enumerate(samples):
        convs = item.get("conversations", [])
        if len(convs) < 2:
            skipped += 1
            continue

        human_turn = next((c for c in convs if c["from"] == "human"), None)
        gpt_turn = next((c for c in convs if c["from"] == "gpt"), None)

        if not human_turn or not gpt_turn:
            skipped += 1
            continue

        quality_response = gpt_turn["value"]
        task_type = item.get("task_type", "financial_qa")
        if task_type not in ALIGNMENT_POLICIES:
            task_type = "financial_qa"
        if not _include_for_dpo(task_type, task_counts[task_type], max_per_task):
            continue

        min_len = 2 if task_type == "sentiment_analysis" else 60
        if len(quality_response) < min_len:
            skipped += 1
            continue

        degraded_response = create_degraded_response(quality_response, task_type)

        # 随机化 A/B 顺序，避免 position bias
        if random.random() < 0.5:
            resp_a, resp_b = quality_response, degraded_response
            a_is_quality = True
        else:
            resp_a, resp_b = degraded_response, quality_response
            a_is_quality = False

        winner = call_llm_judge(
            question=human_turn["value"],
            response_a=resp_a,
            response_b=resp_b,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )

        if winner is None:
            # LLM 判断失败，回退到规则模式
            chosen_text = quality_response
            rejected_text = degraded_response
            source = "rules_fallback"
        elif (winner == "A") == a_is_quality:
            # LLM 判断与预期一致 → 高质量标注
            chosen_text = quality_response
            rejected_text = degraded_response
            source = "llm_verified"
        else:
            # LLM 认为劣化版更好？大概率是劣化度不够
            # 换一个更严重的劣化版本
            judge_wrong += 1
            rejected_text = random.choice(DEGRADED_RESPONSES)
            chosen_text = quality_response
            source = "llm_corrected"

        results.append({
            "conversations": [{"from": "human", "value": human_turn["value"]}],
            "chosen": {"from": "gpt", "value": chosen_text},
            "rejected": {"from": "gpt", "value": rejected_text},
            "system": item.get("system", ""),
            "task_type": task_type,
            "alignment_method": "dpo",
            "alignment_policy": get_policy(task_type).primary_method,
            "_source": source,
        })
        task_counts[task_type] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(samples)}, pairs: {len(results)}")

        time.sleep(request_delay)

    logger.info(
        f"LLM mode: {len(results)} pairs, {skipped} skipped, {judge_wrong} LLM corrections"
    )
    return results


def stratified_split(items: list[dict], eval_ratio: float = 0.05, seed: int = 42) -> tuple[list[dict], list[dict]]:
    random.seed(seed)
    grouped = defaultdict(list)
    for item in items:
        grouped[item.get("task_type", "unknown")].append(item)

    train_data = []
    eval_data = []
    for _, group_items in sorted(grouped.items()):
        group_items = list(group_items)
        random.shuffle(group_items)
        if len(group_items) <= 1:
            train_part, eval_part = group_items, []
        else:
            eval_count = max(1, int(round(len(group_items) * eval_ratio)))
            eval_count = min(eval_count, len(group_items) - 1)
            eval_part = group_items[:eval_count]
            train_part = group_items[eval_count:]
        train_data.extend(train_part)
        eval_data.extend(eval_part)

    random.shuffle(train_data)
    random.shuffle(eval_data)
    return train_data, eval_data


def write_json(path: Path, data: list[dict] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_split_outputs(output: Path, data: list[dict], eval_ratio: float, seed: int, write_legacy: bool) -> dict:
    train_data, eval_data = stratified_split(data, eval_ratio=eval_ratio, seed=seed)
    if output.name.endswith("_train.json"):
        train_path = output
        eval_path = output.with_name(output.name.replace("_train.json", "_eval.json"))
    else:
        train_path = output
        eval_path = output.with_name(output.stem + "_eval.json")

    write_json(train_path, train_data)
    write_json(eval_path, eval_data)

    if write_legacy:
        write_json(output.parent / "fin_preference_train.json", train_data)
        write_json(output.parent / "fin_preference_eval.json", eval_data)

    report = {
        "total": len(data),
        "train": len(train_data),
        "eval": len(eval_data),
        "task_distribution": dict(sorted(Counter(item.get("task_type", "unknown") for item in data).items())),
        "train_distribution": dict(sorted(Counter(item.get("task_type", "unknown") for item in train_data).items())),
        "eval_distribution": dict(sorted(Counter(item.get("task_type", "unknown") for item in eval_data).items())),
        "policy": {task: get_policy(task).primary_method for task in ALL_TASKS},
    }
    report_path = train_path.with_name(train_path.stem.replace("_train", "") + "_report.json")
    write_json(report_path, report)
    write_policy_report(output.parent / "alignment_policy_report.json")
    return {"train_path": str(train_path), "eval_path": str(eval_path), "report_path": str(report_path), **report}


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="生成 DPO 偏好训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/sft/fin_instruct_train.json",
        help="输入 SFT 数据路径",
    )
    parser.add_argument(
        "--output",
        default="data/rlhf/fin_dpo_preference_train.json",
        help="输出偏好训练数据路径",
    )
    parser.add_argument(
        "--mode",
        choices=["rules", "llm", "both"],
        default="both",
        help="生成模式: rules(规则), llm(LLM评判), both(两者结合)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="最大处理样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        help="LLM API 请求间隔（秒），避免速率限制",
    )
    parser.add_argument(
        "--task-filter",
        nargs="+",
        default=None,
        help="只处理指定 task_type，如: --task-filter sentiment_analysis financial_qa",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="每个任务最多生成多少 DPO 样本；sentiment_analysis 默认受任务策略上限约束",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="按任务分层切分验证集比例",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="只写 --output 指定文件，不额外生成 *_eval.json",
    )
    parser.add_argument(
        "--no-legacy-alias",
        action="store_true",
        help="不写 fin_preference_train/eval.json 兼容别名",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # 读取原始数据
    logger.info(f"读取 SFT 数据: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"原始样本数: {len(raw_data)}")

    # 任务过滤
    if args.task_filter:
        raw_data = [d for d in raw_data if d.get("task_type") in args.task_filter]
        logger.info(f"过滤后样本数: {len(raw_data)} (task_filter={args.task_filter})")
    else:
        raw_data = [d for d in raw_data if d.get("task_type") in ALIGNMENT_POLICIES]
        logger.info(f"保留 RLVF 任务样本: {len(raw_data)}")

    # 随机采样
    if len(raw_data) > args.max_samples:
        raw_data = random.sample(raw_data, args.max_samples)
        logger.info(f"随机采样后: {len(raw_data)} 条")

    # API 配置（LLM 模式需要）
    api_key = os.environ.get("SYNTH_API_KEY", "")
    api_base = os.environ.get("SYNTH_API_BASE", "https://www.micuapi.ai/v1")
    model = os.environ.get("SYNTH_API_MODEL", "deepseek-v4-pro")

    results = []

    if args.mode == "rules":
        results = process_rules_mode(raw_data, max_per_task=args.max_per_task)

    elif args.mode == "llm":
        if not api_key:
            raise ValueError("LLM 模式需要设置 SYNTH_API_KEY 环境变量")
        results = process_llm_mode(raw_data, api_key, api_base, model, args.request_delay, args.max_per_task)

    elif args.mode == "both":
        # 2/3 规则模式，1/3 LLM 模式（节省 API 费用）
        random.shuffle(raw_data)
        split = len(raw_data) * 2 // 3
        rules_samples = raw_data[:split]
        llm_samples = raw_data[split:]

        logger.info(f"Both 模式: {len(rules_samples)} rules + {len(llm_samples)} llm")
        results.extend(process_rules_mode(rules_samples, max_per_task=args.max_per_task))

        if api_key and llm_samples:
            results.extend(
                process_llm_mode(llm_samples, api_key, api_base, model, args.request_delay, args.max_per_task)
            )
        elif llm_samples:
            logger.warning("未设置 SYNTH_API_KEY，LLM 部分回退到规则模式")
            results.extend(process_rules_mode(llm_samples, max_per_task=args.max_per_task))

    # 去重（按 task + human 问题全文 hash 去重）
    seen_questions = set()
    deduped = []
    for item in results:
        q_full = item["conversations"][0]["value"]
        q_key = hashlib.md5(f"{item.get('task_type')}::{q_full}".encode("utf-8")).hexdigest()
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            deduped.append(item)
    logger.info(f"去重后: {len(deduped)} 对（原 {len(results)} 对）")

    # 保存
    out_path = Path(args.output)
    if args.no_split:
        write_json(out_path, deduped)
        output_info = {"train_path": str(out_path), "eval_path": None, "report_path": None}
    else:
        output_info = write_split_outputs(
            out_path,
            deduped,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            write_legacy=not args.no_legacy_alias,
        )

    # 统计
    source_counts = {}
    task_counts = {}
    for item in deduped:
        src = item.get("_source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        tt = item.get("task_type", "unknown")
        task_counts[tt] = task_counts.get(tt, 0) + 1

    logger.info("=" * 50)
    logger.info(f"✅ 偏好数据生成完成！")
    logger.info(f"   输出路径: {out_path}")
    if output_info.get("eval_path"):
        logger.info(f"   验证集路径: {output_info['eval_path']}")
    if output_info.get("report_path"):
        logger.info(f"   分布报告: {output_info['report_path']}")
    logger.info(f"   总样本数: {len(deduped)}")
    logger.info(f"   来源分布: {source_counts}")
    logger.info(f"   任务分布: {task_counts}")
    logger.info("=" * 50)

    # 打印前 3 条样本供检查
    logger.info("\n前 3 条样本预览:")
    for i, item in enumerate(deduped[:3]):
        q = item["conversations"][0]["value"][:60]
        c = item["chosen"]["value"][:80]
        r = item["rejected"]["value"][:60]
        logger.info(f"  [{i+1}] Q: {q}...")
        logger.info(f"       Chosen: {c}...")
        logger.info(f"       Rejected: {r}...")


if __name__ == "__main__":
    main()
