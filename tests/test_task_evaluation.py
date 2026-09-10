#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务专项评估测试。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _item(task: str, question: str, answer: str) -> dict:
    return {
        "task_type": task,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }


def test_sentiment_metrics_extract_labels():
    from scripts.evaluation.eval_task_specific import evaluate_sentiment

    items = [
        _item("sentiment_analysis", "新闻A", "情感倾向：积极"),
        _item("sentiment_analysis", "新闻B", "情感倾向：消极"),
        _item("sentiment_analysis", "新闻C", "情感倾向：中性"),
    ]
    preds = ["积极，利好", "消极，利空", "中性影响有限"]
    result = evaluate_sentiment(items, preds)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_code_generation_metrics_cover_strategy_elements():
    from scripts.evaluation.eval_task_specific import evaluate_code_generation

    items = [_item("quant_strategy", "设计策略", "参考答案")]
    preds = [
        """策略包含入场、出场、仓位和风控，并计算夏普和最大回撤。
```python
import pandas as pd

def generate_signals(df):
    return df
```"""
    ]
    result = evaluate_code_generation(items, preds)
    assert result["syntax_correct_rate"] == 1.0
    assert result["has_import_rate"] == 1.0
    assert result["has_function_rate"] == 1.0
    assert result["has_backtest_metric_rate"] == 1.0
    assert result["has_strategy_element_rate"] == 1.0


def test_risk_level_extraction_and_metrics():
    from scripts.evaluation.eval_task_specific import evaluate_risk_assessment, extract_risk_level

    assert extract_risk_level("综合风险等级：高。") == "high"
    items = [_item("risk_assessment", "评估风险", "综合风险等级：高。信用风险、市场风险、缓释建议。")]
    preds = ["综合风险等级：高。信用风险和市场风险较高，波动率30%，最大回撤20%，建议止损和对冲。"]
    result = evaluate_risk_assessment(items, preds)
    assert result["risk_level_extraction_rate"] == 1.0
    assert result["quantitative_risk_indicator_rate"] == 1.0


def test_judge_skips_without_api_key(monkeypatch):
    from scripts.evaluation.eval_task_specific import run_judge_evaluation

    monkeypatch.delenv("EVAL_JUDGE_API_KEY", raising=False)
    items = [_item("financial_report", "分析财报", "参考答案")]
    result = run_judge_evaluation("financial_report", items, ["模型回答"])
    assert result["skipped"] is True
