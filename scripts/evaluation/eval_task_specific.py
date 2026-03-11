#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
各任务专项评测

对 6 类金融任务分别进行评测:
  1. 情感分析: 准确率、F1-score
  2. 股票分析: ROUGE-L、内容覆盖度
  3. 财报解读: ROUGE-L、关键指标提取率
  4. 量化策略: 代码可执行率、语法正确率
  5. 金融问答: ROUGE-L、答案准确率
  6. 风险评估: 关键要素覆盖度、风险等级准确率

用法:
  python scripts/evaluation/eval_task_specific.py --task sentiment_analysis
  python scripts/evaluation/eval_task_specific.py --task all
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

from tqdm import tqdm

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SFT_DIR = PROJECT_ROOT / "data" / "sft"
RESULTS_DIR = PROJECT_ROOT / "saves" / "eval_results"


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    计算 ROUGE-L F1 分数。

    基于最长公共子序列 (LCS) 计算:
      - Recall = LCS长度 / 参考文本长度
      - Precision = LCS长度 / 生成文本长度
      - F1 = 2 * P * R / (P + R)

    参数:
        reference:  参考文本
        hypothesis: 生成文本

    返回:
        ROUGE-L F1 分数 (0~1)
    """
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)
    m, n = len(ref_tokens), len(hyp_tokens)

    if m == 0 or n == 0:
        return 0.0

    # 动态规划求 LCS 长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    recall = lcs_len / m
    precision = lcs_len / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def evaluate_sentiment(eval_data: list, predictions: list) -> dict:
    """
    情感分析评测。

    评测指标:
      - 准确率 (Accuracy)
      - 各类别 F1 (Precision、Recall、F1)

    参数:
        eval_data:   评测数据列表
        predictions: 模型预测列表 (每条包含 predicted_label)

    返回:
        评测结果字典
    """
    # 情感标签规范化
    label_map = {
        "积极": "positive", "正面": "positive", "利好": "positive", "positive": "positive",
        "消极": "negative", "负面": "negative", "利空": "negative", "negative": "negative",
        "中性": "neutral", "中立": "neutral", "neutral": "neutral",
    }

    correct = 0
    total = 0
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for ref, pred in zip(eval_data, predictions):
        ref_label = label_map.get(ref.get("label", "").strip(), "unknown")
        pred_label = label_map.get(pred.get("predicted_label", "").strip(), "unknown")

        if ref_label == "unknown":
            continue

        total += 1
        if ref_label == pred_label:
            correct += 1
            label_stats[ref_label]["tp"] += 1
        else:
            label_stats[pred_label]["fp"] += 1
            label_stats[ref_label]["fn"] += 1

    accuracy = correct / total if total > 0 else 0.0

    # 计算各类别 F1
    per_class = {}
    for label, stats in label_stats.items():
        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "task": "sentiment_analysis",
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
    }


def evaluate_text_generation(eval_data: list, predictions: list, task_name: str) -> dict:
    """
    文本生成类任务评测（股票分析、财报解读、金融问答、风险评估）。

    评测指标:
      - 平均 ROUGE-L
      - 平均回答长度
      - 内容覆盖度（关键词命中率）

    参数:
        eval_data:   评测数据（含参考答案）
        predictions: 模型预测列表
        task_name:   任务名称

    返回:
        评测结果字典
    """
    rouge_scores = []
    lengths = []
    keyword_hits = []

    # 各任务的关键词检查列表
    task_keywords = {
        "stock_analysis": ["趋势", "支撑", "压力", "MACD", "成交量", "均线"],
        "financial_report": ["营收", "净利润", "ROE", "同比", "现金流", "资产负债"],
        "financial_qa": [],  # 金融问答不检查特定关键词
        "risk_assessment": ["风险", "评级", "波动", "建议", "因素"],
    }
    keywords = task_keywords.get(task_name, [])

    for ref, pred in zip(eval_data, predictions):
        ref_text = ref.get("reference", "")
        pred_text = pred.get("prediction", "")

        if not ref_text or not pred_text:
            continue

        # ROUGE-L
        rouge = compute_rouge_l(ref_text, pred_text)
        rouge_scores.append(rouge)

        # 回答长度
        lengths.append(len(pred_text))

        # 关键词命中
        if keywords:
            hits = sum(1 for kw in keywords if kw in pred_text)
            keyword_hits.append(hits / len(keywords))

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0
    avg_keyword_hit = sum(keyword_hits) / len(keyword_hits) if keyword_hits else 0.0

    return {
        "task": task_name,
        "total_evaluated": len(rouge_scores),
        "avg_rouge_l": round(avg_rouge, 4),
        "avg_response_length": round(avg_length, 1),
        "avg_keyword_coverage": round(avg_keyword_hit, 4) if keywords else "N/A",
    }


def evaluate_code_generation(predictions: list) -> dict:
    """
    量化策略代码生成评测。

    评测指标:
      - 语法正确率 (能否通过 ast.parse)
      - 代码完整率 (包含必要组件: import、函数定义、返回值)
      - 平均代码长度

    参数:
        predictions: 模型生成的代码列表

    返回:
        评测结果字典
    """
    import ast

    total = 0
    syntax_correct = 0
    has_import = 0
    has_function = 0
    code_lengths = []

    for pred in predictions:
        code = pred.get("prediction", "")

        # 提取代码块（如果嵌在 markdown 中）
        code_match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1)

        if not code.strip():
            continue

        total += 1
        code_lengths.append(len(code))

        # 语法检查
        try:
            ast.parse(code)
            syntax_correct += 1
        except SyntaxError:
            pass

        # 检查组件
        if re.search(r"^import\s|^from\s", code, re.MULTILINE):
            has_import += 1
        if re.search(r"^def\s", code, re.MULTILINE):
            has_function += 1

    return {
        "task": "quant_strategy",
        "total_evaluated": total,
        "syntax_correct_rate": round(syntax_correct / total, 4) if total > 0 else 0,
        "has_import_rate": round(has_import / total, 4) if total > 0 else 0,
        "has_function_rate": round(has_function / total, 4) if total > 0 else 0,
        "avg_code_length": round(sum(code_lengths) / len(code_lengths), 1) if code_lengths else 0,
    }


def run_task_evaluation(task: str = "all") -> None:
    """
    运行指定任务的评测。

    参数:
        task: 任务名称或 "all" 表示全部
    """
    logger.info("=" * 60)
    logger.info("Fin-Instruct 任务专项评测")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 加载评测数据
    eval_file = SFT_DIR / "fin_instruct_eval.json"
    if not eval_file.exists():
        logger.error(f"评测数据不存在: {eval_file}")
        logger.error("请先运行数据处理管道生成评测数据")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    logger.info(f"加载评测数据: {len(eval_data)} 条")

    # 按任务分组
    task_groups = defaultdict(list)
    for item in eval_data:
        task_type = item.get("task_type", "general")
        task_groups[task_type].append(item)

    logger.info("评测数据任务分布:")
    for t, items in task_groups.items():
        logger.info(f"  {t}: {len(items)} 条")

    logger.info("\n提示: 完整评测需要加载模型进行推理。")
    logger.info("当前版本提供评测框架，具体推理逻辑需配合模型运行。")
    logger.info("可使用 eval_finance_bench.py 进行多选题评测。")

    # 保存评测数据分布报告
    report = {
        "eval_data_count": len(eval_data),
        "task_distribution": {t: len(items) for t, items in task_groups.items()},
        "available_tasks": list(task_groups.keys()),
    }

    report_file = RESULTS_DIR / "task_eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n评测报告已保存: {report_file}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    all_tasks = [
        "stock_analysis", "quant_strategy", "financial_report",
        "sentiment_analysis", "financial_qa", "risk_assessment",
    ]

    parser = argparse.ArgumentParser(description="Fin-Instruct 任务专项评测")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=all_tasks + ["all"],
        help="指定评测任务（默认: all 全部评测）",
    )
    args = parser.parse_args()

    run_task_evaluation(task=args.task)
