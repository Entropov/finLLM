#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-model, task-aware RLVF evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluation.eval_task_specific import (
    ALL_TASKS,
    OPEN_GENERATION_TASKS,
    _get_question,
    _get_reference,
    evaluate_task_predictions,
    generate_response,
    load_eval_data,
    load_model_and_tokenizer,
    run_judge_evaluation,
)
from scripts.rlhf.alignment_policy import (
    ALIGNMENT_POLICIES,
    compute_reward,
    get_policy,
    summarize_rewards,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_ADAPTERS = {
    "sft": "saves/qwen3-8b/lora/sft",
    "dpo": "saves/qwen3-8b/lora/dpo",
    "rlvf": "saves/qwen3-8b/lora/rlvf",
}


def adapter_exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def load_rlvf_eval_data(task: str, max_samples: Optional[int], seed: int) -> dict[str, list[dict]]:
    task_groups = load_eval_data(task)
    rng = random.Random(seed)
    for task_name, items in list(task_groups.items()):
        if max_samples and len(items) > max_samples:
            task_groups[task_name] = rng.sample(items, max_samples)
    return task_groups


def load_preference_eval(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        fallback = p.with_name(p.name.replace("_eval", "_train"))
        if fallback.exists():
            logger.warning(f"DPO eval data not found, using train subset: {fallback}")
            p = fallback
        else:
            return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def quick_quality_score(response: str, task_type: str) -> float:
    reward = compute_reward(task_type, response, response).get("reward", 0.0)
    score = 4.0 + reward * 4.0
    if len(response) > 200:
        score += 0.8
    if re.search(r"^\s*(?:\d+[.、]|[-*])", response, re.MULTILINE):
        score += 0.6
    if "风险" in response or "仅供参考" in response:
        score += 0.4
    if any(phrase in response for phrase in ["无法回答", "不清楚", "请咨询专业人士"]):
        score -= 1.2
    return round(max(0.0, min(10.0, score)), 3)


def compare_quality(responses_by_model: dict[str, list[str]], items: list[dict], task_type: str) -> dict:
    model_scores = {}
    for name, responses in responses_by_model.items():
        scores = [quick_quality_score(resp, task_type) for resp in responses]
        model_scores[name] = {
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "scores": scores,
        }
    wins = Counter()
    model_names = list(responses_by_model)
    for idx in range(len(items)):
        per_item = {
            name: model_scores[name]["scores"][idx]
            for name in model_names
            if idx < len(model_scores[name]["scores"])
        }
        if not per_item:
            continue
        best = max(per_item.values())
        winners = [name for name, score in per_item.items() if score == best]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            wins["tie"] += 1
    total = len(items)
    return {
        "total": total,
        "model_avg_scores": {name: data["avg_score"] for name, data in model_scores.items()},
        "wins": dict(wins),
        "win_rates": {name: round(wins.get(name, 0) / total, 4) if total else 0.0 for name in model_names},
        "tie_rate": round(wins.get("tie", 0) / total, 4) if total else 0.0,
    }


def dpo_preference_metrics(pref_items: list[dict]) -> dict:
    if not pref_items:
        return {"total": 0, "chosen_reward_win_rate": 0.0, "avg_reward_gap": 0.0}
    wins = 0
    gaps = []
    by_task = defaultdict(list)
    for item in pref_items:
        task_type = item.get("task_type", "financial_qa")
        chosen = item.get("chosen", {})
        rejected = item.get("rejected", {})
        chosen_text = chosen.get("value", "") if isinstance(chosen, dict) else str(chosen)
        rejected_text = rejected.get("value", "") if isinstance(rejected, dict) else str(rejected)
        chosen_reward = compute_reward(task_type, chosen_text, chosen_text).get("reward", 0.0)
        rejected_reward = compute_reward(task_type, rejected_text, chosen_text).get("reward", 0.0)
        gap = chosen_reward - rejected_reward
        gaps.append(gap)
        if gap > 0:
            wins += 1
        by_task[task_type].append({"reward": chosen_reward, "gap": gap})

    task_summary = {}
    for task_type, rows in by_task.items():
        task_summary[task_type] = {
            "count": len(rows),
            "chosen_avg_reward": round(sum(r["reward"] for r in rows) / len(rows), 4),
            "avg_reward_gap": round(sum(r["gap"] for r in rows) / len(rows), 4),
        }
    return {
        "total": len(pref_items),
        "chosen_reward_win_rate": round(wins / len(pref_items), 4),
        "avg_reward_gap": round(sum(gaps) / len(gaps), 4) if gaps else 0.0,
        "by_task": dict(sorted(task_summary.items())),
    }


def grpo_reward_metrics(task_groups: dict[str, list[dict]], predictions_by_model: dict[str, dict[str, list[str]]]) -> dict:
    summary = {}
    for model_name, task_predictions in predictions_by_model.items():
        model_summary = {}
        for task_type, items in task_groups.items():
            rows = []
            for item, pred in zip(items, task_predictions.get(task_type, [])):
                rows.append(compute_reward(task_type, pred, _get_reference(item), item.get("answer_key", "")))
            model_summary[task_type] = summarize_rewards(rows)
        summary[model_name] = model_summary
    return summary


def offline_reward_report(task_groups: dict[str, list[dict]], pref_items: list[dict]) -> dict:
    reference_rewards = {}
    for task_type, items in task_groups.items():
        rows = [
            compute_reward(task_type, _get_reference(item), _get_reference(item), item.get("answer_key", ""))
            for item in items
        ]
        reference_rewards[task_type] = summarize_rewards(rows)
    return {
        "mode": "offline_data_validation",
        "task_policy": {
            task: {
                "primary_method": policy.primary_method,
                "secondary_methods": list(policy.secondary_methods),
                "reward_name": policy.reward_name,
                "metrics": list(policy.metrics),
            }
            for task, policy in ALIGNMENT_POLICIES.items()
        },
        "reference_reward_summary": reference_rewards,
        "dpo_preference_metrics": dpo_preference_metrics(pref_items),
    }


def load_models(args, adapters: dict[str, Optional[str]]) -> dict:
    loaded = {}
    tokenizer = None
    for name, adapter in adapters.items():
        if adapter and not Path(adapter).exists():
            logger.warning(f"Skip {name}: adapter path not found: {adapter}")
            continue
        logger.info(f"Loading model for {name} (adapter={adapter or 'none'})")
        model, tokenizer = load_model_and_tokenizer(args.model_path, adapter, quantize=not args.no_quantize)
        loaded[name] = (model, tokenizer)
    return loaded


def run_inference_for_models(args, task_groups: dict[str, list[dict]], models: dict) -> dict[str, dict[str, list[str]]]:
    predictions_by_model = {}
    for model_name, (model, tokenizer) in models.items():
        predictions_by_model[model_name] = {}
        for task_type, items in sorted(task_groups.items()):
            logger.info(f"Inference: model={model_name}, task={task_type}, samples={len(items)}")
            preds = [
                generate_response(model, tokenizer, item, max_new_tokens=args.max_new_tokens)
                for item in items
            ]
            predictions_by_model[model_name][task_type] = preds
    return predictions_by_model


def evaluate_predictions(
    task_groups: dict[str, list[dict]],
    predictions_by_model: dict[str, dict[str, list[str]]],
    judge_mode: str,
    max_judge_samples: int,
) -> dict:
    results = {}
    for model_name, task_predictions in predictions_by_model.items():
        model_results = {}
        for task_type, items in task_groups.items():
            preds = task_predictions.get(task_type, [])
            metrics = evaluate_task_predictions(task_type, items, preds)
            if judge_mode in {"llm", "both"} and task_type in OPEN_GENERATION_TASKS:
                judged = run_judge_evaluation(task_type, items, preds, max_judge_samples=max_judge_samples)
                if judged is not None:
                    metrics["judge"] = judged
            model_results[task_type] = metrics
        results[model_name] = model_results
    return results


def build_alignment_comparison(task_groups: dict[str, list[dict]], predictions_by_model: dict[str, dict[str, list[str]]]) -> dict:
    by_task = {}
    for task_type, items in task_groups.items():
        responses_by_model = {
            model_name: task_predictions.get(task_type, [])
            for model_name, task_predictions in predictions_by_model.items()
        }
        by_task[task_type] = compare_quality(responses_by_model, items, task_type)
    return {
        "quick_quality_by_task": by_task,
        "grpo_reward_by_model": grpo_reward_metrics(task_groups, predictions_by_model),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# FinLLM RLVF 对齐评估报告",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 模式: {report.get('mode', 'model_comparison')}",
        "",
        "## 任务策略",
        "",
        "| 任务 | 主方法 | 辅助方法 | 奖励 |",
        "|------|--------|----------|------|",
    ]
    for task, policy in report.get("task_policy", {}).items():
        lines.append(
            f"| {task} | {policy.get('primary_method')} | {','.join(policy.get('secondary_methods', [])) or '-'} | {policy.get('reward_name')} |"
        )

    if report.get("mode") == "offline_data_validation":
        lines += [
            "",
            "## 离线奖励摘要",
            "",
            "| 任务 | 样本数 | 平均奖励 | P50 |",
            "|------|--------|----------|-----|",
        ]
        for task, summary in report.get("reference_reward_summary", {}).items():
            lines.append(f"| {task} | {summary['count']} | {summary['avg_reward']:.4f} | {summary['p50_reward']:.4f} |")
        dpo = report.get("dpo_preference_metrics", {})
        lines += [
            "",
            "## DPO 偏好数据",
            "",
            f"- 样本数: {dpo.get('total', 0)}",
            f"- chosen 奖励胜率: {dpo.get('chosen_reward_win_rate', 0):.4f}",
            f"- 平均奖励差: {dpo.get('avg_reward_gap', 0):.4f}",
        ]
        return "\n".join(lines)

    lines += [
        "",
        "## 模型任务指标",
        "",
    ]
    for model_name, task_results in report.get("task_results", {}).items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| 任务 | 核心指标 |")
        lines.append("|------|----------|")
        for task, metrics in task_results.items():
            compact = []
            for key, value in metrics.items():
                if key in {"task", "per_class", "judge"}:
                    continue
                if isinstance(value, (int, float)):
                    compact.append(f"{key}={value}")
            lines.append(f"| {task} | {'; '.join(compact[:6])} |")
        lines.append("")

    lines += [
        "## 对齐专项指标",
        "",
        "| 任务 | 模型均分 | 胜率 |",
        "|------|----------|------|",
    ]
    quick = report.get("alignment_comparison", {}).get("quick_quality_by_task", {})
    for task, metrics in quick.items():
        score_str = ", ".join(f"{k}:{v}" for k, v in metrics.get("model_avg_scores", {}).items())
        win_str = ", ".join(f"{k}:{v}" for k, v in metrics.get("win_rates", {}).items())
        lines.append(f"| {task} | {score_str} | {win_str} |")

    dpo = report.get("dpo_preference_metrics", {})
    lines += [
        "",
        "## DPO 偏好数据",
        "",
        f"- 样本数: {dpo.get('total', 0)}",
        f"- chosen 奖励胜率: {dpo.get('chosen_reward_win_rate', 0):.4f}",
        f"- 平均奖励差: {dpo.get('avg_reward_gap', 0):.4f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-aware SFT/DPO/RLVF comparison")
    parser.add_argument("--model-path", default=None, help="Base model path. Omit to run offline data/reward validation.")
    parser.add_argument("--sft-adapter", default=DEFAULT_ADAPTERS["sft"])
    parser.add_argument("--dpo-adapter", default=DEFAULT_ADAPTERS["dpo"])
    parser.add_argument("--rlvf-adapter", default=DEFAULT_ADAPTERS["rlvf"])
    parser.add_argument("--adapters", nargs="*", default=None, help="Optional name=path adapters; overrides default adapter set")
    parser.add_argument("--eval-data", default="data/sft/fin_instruct_eval.json", help="Kept for CLI compatibility; task loader is used")
    parser.add_argument("--preference-eval-data", default="data/rlhf/fin_dpo_preference_eval.json")
    parser.add_argument("--task", default="all", choices=ALL_TASKS + ["all"])
    parser.add_argument("--output", default="logs/rlvf_comparison.md")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--num-samples", type=int, default=100, help="Max samples per task")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-mode", choices=["off", "quick", "llm", "both"], default="quick")
    parser.add_argument("--max-judge-samples", type=int, default=50)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    task_groups = load_rlvf_eval_data(args.task, args.num_samples, args.seed)
    pref_items = load_preference_eval(args.preference_eval_data)

    task_policy = {
        task: {
            "primary_method": get_policy(task).primary_method,
            "secondary_methods": list(get_policy(task).secondary_methods),
            "reward_name": get_policy(task).reward_name,
            "metrics": list(get_policy(task).metrics),
        }
        for task in ALL_TASKS
    }

    if args.model_path is None:
        report = offline_reward_report(task_groups, pref_items)
        report["task_policy"] = task_policy
    else:
        if args.adapters:
            adapters = {}
            for raw in args.adapters:
                if "=" not in raw:
                    raise ValueError(f"--adapters expects name=path, got {raw}")
                name, path = raw.split("=", 1)
                adapters[name] = path
        else:
            adapters = {
                "sft": args.sft_adapter,
                "dpo": args.dpo_adapter,
                "rlvf": args.rlvf_adapter if adapter_exists(args.rlvf_adapter) else args.dpo_adapter,
            }
        models = load_models(args, adapters)
        if not models:
            raise RuntimeError("No model/adapters could be loaded")
        predictions = run_inference_for_models(args, task_groups, models)
        task_results = evaluate_predictions(task_groups, predictions, args.judge_mode, args.max_judge_samples)
        report = {
            "mode": "model_comparison",
            "model_path": args.model_path,
            "task_policy": task_policy,
            "task_results": task_results,
            "alignment_comparison": build_alignment_comparison(task_groups, predictions),
            "dpo_preference_metrics": dpo_preference_metrics(pref_items),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    json_output = Path(args.json_output) if args.json_output else output.with_suffix(".json")
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Report saved: {output}")
    logger.info(f"JSON saved: {json_output}")
    print(f"RLVF evaluation complete: {output}")


if __name__ == "__main__":
    main()
