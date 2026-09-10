#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare task-aware GRPO prompt data and offline reward reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from scripts.rlhf.alignment_policy import (
    ALIGNMENT_POLICIES,
    ALL_TASKS,
    compute_reward,
    extract_answer_key,
    extract_sentiment_label,
    get_policy,
    get_question,
    get_reference,
    summarize_rewards,
    write_policy_report,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_GRPO_TASKS = [
    "sentiment_analysis",
    "quant_strategy",
    "risk_assessment",
    "financial_qa",
    "stock_analysis",
]


def _stable_id(task_type: str, question: str) -> str:
    digest = hashlib.md5(f"{task_type}::{question}".encode("utf-8")).hexdigest()
    return f"{task_type}_{digest[:12]}"


def _answer_key_for_item(item: dict, task_type: str, reference: str) -> str:
    if item.get("answer_key"):
        return str(item["answer_key"])
    if task_type == "sentiment_analysis":
        return extract_sentiment_label(reference)
    return extract_answer_key(reference)


def _is_grpo_eligible(item: dict, task_type: str, reference: str, answer_key: str) -> bool:
    if task_type == "sentiment_analysis":
        return answer_key in {"positive", "negative", "neutral", "yes", "no"}
    if task_type == "financial_qa":
        question = get_question(item)
        return bool(answer_key or re.search(r"\b[A-D][.、)]", question) or re.search(r"正确答案|答案", reference))
    if task_type in {"quant_strategy", "risk_assessment", "stock_analysis"}:
        return bool(reference.strip())
    return False


def build_grpo_record(item: dict, task_type: str) -> dict[str, Any] | None:
    question = get_question(item)
    reference = get_reference(item)
    if not question or not reference:
        return None

    answer_key = _answer_key_for_item(item, task_type, reference)
    if not _is_grpo_eligible(item, task_type, reference, answer_key):
        return None

    policy = get_policy(task_type)
    reward = compute_reward(task_type, reference, reference, answer_key)
    return {
        "id": _stable_id(task_type, question),
        "task_type": task_type,
        "alignment_method": "grpo",
        "conversations": [{"from": "human", "value": question}],
        "system": item.get("system", ""),
        "reference": reference,
        "answer_key": answer_key,
        "reward_spec": {
            "name": policy.reward_name,
            "primary_method": policy.primary_method,
            "metrics": list(policy.metrics),
            "max_reward": 1.0,
        },
        "offline_reference_reward": reward,
    }


def load_input_data(path: Path, tasks: set[str]) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"输入文件必须是 JSON list: {path}")
    return [item for item in raw if item.get("task_type") in tasks]


def dedupe_records(records: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for record in records:
        key = record["id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def cap_by_task(records: list[dict], max_per_task: int | None, seed: int) -> list[dict]:
    if max_per_task is None:
        return records
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task_type"]].append(record)
    capped = []
    for task_type, items in grouped.items():
        rng.shuffle(items)
        policy_limit = get_policy(task_type).grpo_target or max_per_task
        limit = min(max_per_task, policy_limit)
        capped.extend(items[:limit])
    rng.shuffle(capped)
    return capped


def stratified_split(records: list[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task_type"]].append(record)
    train_data = []
    eval_data = []
    for _, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) <= 1:
            train_part, eval_part = items, []
        else:
            eval_count = max(1, int(round(len(items) * eval_ratio)))
            eval_count = min(eval_count, len(items) - 1)
            eval_part = items[:eval_count]
            train_part = items[eval_count:]
        train_data.extend(train_part)
        eval_data.extend(eval_part)
    rng.shuffle(train_data)
    rng.shuffle(eval_data)
    return train_data, eval_data


def write_json(path: Path, data: list[dict] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_report(records: list[dict], train_data: list[dict], eval_data: list[dict], skipped: Counter) -> dict:
    reward_rows_by_task = defaultdict(list)
    for record in records:
        reward_rows_by_task[record["task_type"]].append(record.get("offline_reference_reward", {}))
    return {
        "total": len(records),
        "train": len(train_data),
        "eval": len(eval_data),
        "task_distribution": dict(sorted(Counter(record["task_type"] for record in records).items())),
        "train_distribution": dict(sorted(Counter(record["task_type"] for record in train_data).items())),
        "eval_distribution": dict(sorted(Counter(record["task_type"] for record in eval_data).items())),
        "skipped_by_task": dict(sorted(skipped.items())),
        "reward_summary_by_task": {
            task: summarize_rewards(rows) for task, rows in sorted(reward_rows_by_task.items())
        },
        "grpo_backend": {
            "required": False,
            "supported_backends": ["EasyR1", "TRL-GRPO"],
            "note": "This project prepares GRPO data and reward specs. Training is skipped unless a compatible backend is detected.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GRPO prompt data and reward metadata")
    parser.add_argument("--input", default="data/sft/fin_instruct_train.json", help="输入 SFT 数据")
    parser.add_argument("--output", default="data/rlhf/fin_grpo_prompts_train.json", help="输出 GRPO 训练 prompt 数据")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_GRPO_TASKS, choices=ALL_TASKS, help="启用 GRPO 的任务")
    parser.add_argument("--max-samples", type=int, default=None, help="全局最大输入样本数")
    parser.add_argument("--max-per-task", type=int, default=5000, help="每任务最大 GRPO 样本数")
    parser.add_argument("--eval-ratio", type=float, default=0.05, help="分层验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    tasks = set(args.tasks)
    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"读取 SFT 数据: {input_path}")
    raw = load_input_data(input_path, tasks)
    if args.max_samples and len(raw) > args.max_samples:
        raw = random.sample(raw, args.max_samples)
    logger.info(f"候选样本: {len(raw)}")

    records = []
    skipped = Counter()
    for item in raw:
        task_type = item.get("task_type", "financial_qa")
        record = build_grpo_record(item, task_type)
        if record is None:
            skipped[task_type] += 1
            continue
        records.append(record)

    records = dedupe_records(records)
    records = cap_by_task(records, args.max_per_task, args.seed)
    train_data, eval_data = stratified_split(records, args.eval_ratio, args.seed)

    if output_path.name.endswith("_train.json"):
        train_path = output_path
        eval_path = output_path.with_name(output_path.name.replace("_train.json", "_eval.json"))
    else:
        train_path = output_path
        eval_path = output_path.with_name(output_path.stem + "_eval.json")

    write_json(train_path, train_data)
    write_json(eval_path, eval_data)
    report = build_report(records, train_data, eval_data, skipped)
    report_path = train_path.with_name(train_path.stem.replace("_train", "") + "_report.json")
    write_json(report_path, report)
    write_policy_report(output_path.parent / "alignment_policy_report.json")

    logger.info("=" * 60)
    logger.info("GRPO 数据准备完成")
    logger.info(f"训练集: {train_path} ({len(train_data)} 条)")
    logger.info(f"验证集: {eval_path} ({len(eval_data)} 条)")
    logger.info(f"报告: {report_path}")
    logger.info(f"任务分布: {report['task_distribution']}")
    if skipped:
        logger.info(f"跳过样本: {dict(skipped)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
