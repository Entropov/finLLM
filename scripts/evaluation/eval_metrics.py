#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算工具模块

提供各种评估指标的统一计算接口:
  - ROUGE-L          (文本生成质量)
  - BLEU             (文本生成质量)
  - 准确率/F1        (分类任务)
  - 关键词覆盖率     (内容完整性)
  - 代码可执行率     (代码生成)

可被其他评估脚本导入使用。
"""

import re
import ast
import math
import logging
from typing import Union
from collections import Counter

# ============================================================
# 日志配置
# ============================================================
logger = logging.getLogger(__name__)


# ============================================================
# 文本生成指标
# ============================================================

def rouge_l_score(reference: str, hypothesis: str) -> dict:
    """
    计算 ROUGE-L 指标（Precision、Recall、F1）。

    参数:
        reference:  参考文本
        hypothesis: 生成文本

    返回:
        {"precision": float, "recall": float, "f1": float}
    """
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)
    m, n = len(ref_tokens), len(hyp_tokens)

    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # LCS 动态规划
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    计算 BLEU 分数（简化版，不含 brevity penalty 的修正）。

    参数:
        reference:  参考文本
        hypothesis: 生成文本
        max_n:      最大 n-gram 阶数

    返回:
        BLEU 分数 (0~1)
    """
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)

    if not hyp_tokens:
        return 0.0

    # 计算各阶 n-gram 精度
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        )

        # Clipped precision
        clipped = sum(
            min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items()
        )
        total = sum(hyp_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    # 避免 log(0)
    if any(p == 0 for p in precisions):
        return 0.0

    # 几何平均
    log_avg = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return round(bp * math.exp(log_avg), 4)


# ============================================================
# 分类指标
# ============================================================

def accuracy_score(y_true: list, y_pred: list) -> float:
    """
    计算准确率。

    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    返回:
        准确率 (0~1)
    """
    if len(y_true) != len(y_pred) or not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4)


def f1_score(y_true: list, y_pred: list, average: str = "macro") -> dict:
    """
    计算 F1 分数。

    参数:
        y_true:  真实标签列表
        y_pred:  预测标签列表
        average: 平均方式 ("macro" / "micro" / "weighted")

    返回:
        {"precision": float, "recall": float, "f1": float, "per_class": dict}
    """
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    if average == "macro":
        avg_p = sum(v["precision"] for v in per_class.values()) / len(per_class) if per_class else 0
        avg_r = sum(v["recall"] for v in per_class.values()) / len(per_class) if per_class else 0
        avg_f = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0
    else:
        # micro
        tp_all = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        avg_p = avg_r = avg_f = tp_all / len(y_true) if y_true else 0.0

    return {
        "precision": round(avg_p, 4),
        "recall": round(avg_r, 4),
        "f1": round(avg_f, 4),
        "per_class": per_class,
    }


# ============================================================
# 内容质量指标
# ============================================================

def keyword_coverage(text: str, keywords: list) -> float:
    """
    计算关键词覆盖率。

    参数:
        text:     待检测文本
        keywords: 关键词列表

    返回:
        覆盖率 (0~1)
    """
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw in text)
    return round(hits / len(keywords), 4)


def code_syntax_check(code: str) -> dict:
    """
    检查 Python 代码语法正确性。

    参数:
        code: Python 代码字符串

    返回:
        {"valid": bool, "error": str or None}
    """
    # 提取 markdown 代码块中的代码
    match = re.search(r"```python\n(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1)

    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": str(e)}


def response_quality_score(
    response: str,
    min_length: int = 50,
    max_length: int = 5000,
) -> float:
    """
    启发式评估回答质量分数（0~1）。

    评分维度:
      - 长度适中性 (0.3)
      - 结构化程度 (0.3): 是否包含分点、段落等
      - 专业性 (0.2): 是否包含专业术语
      - 完成度 (0.2): 是否有完整的开头和结尾

    参数:
        response:   回答文本
        min_length: 理想最小长度
        max_length: 理想最大长度

    返回:
        质量分数 (0~1)
    """
    if not response:
        return 0.0

    score = 0.0
    text_len = len(response)

    # 长度适中性 (0.3)
    if min_length <= text_len <= max_length:
        score += 0.3
    elif text_len < min_length:
        score += 0.3 * (text_len / min_length)
    else:
        score += 0.3 * max(0, 1 - (text_len - max_length) / max_length)

    # 结构化程度 (0.3)
    structure_score = 0
    if re.search(r"^\d+[.、]", response, re.MULTILINE):
        structure_score += 0.5  # 有序号列表
    if "\n" in response:
        structure_score += 0.3  # 多段落
    if re.search(r"[：:]\s*\n", response):
        structure_score += 0.2  # 有标题/小标题
    score += 0.3 * min(structure_score, 1.0)

    # 专业性 (0.2)
    finance_terms = [
        "收益率", "市盈率", "波动率", "资产", "负债", "净利润",
        "ROE", "MACD", "均线", "支撑", "压力", "策略",
        "风险", "收益", "估值", "因子", "回测", "Alpha",
    ]
    term_hits = sum(1 for t in finance_terms if t in response)
    score += 0.2 * min(term_hits / 3, 1.0)  # 命中3个以上得满分

    # 完成度 (0.2)
    has_conclusion = any(kw in response[-200:] for kw in ["综上", "总结", "总的来说", "建议", "注意"])
    score += 0.2 * (0.5 + 0.5 * has_conclusion)

    return round(score, 4)
