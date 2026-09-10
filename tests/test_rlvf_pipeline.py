#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RLVF task policy, data, and reward tests."""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _item(task: str, question: str, answer: str) -> dict:
    return {
        "task_type": task,
        "system": "你是金融助手。",
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }


def test_alignment_policy_covers_six_tasks():
    from scripts.rlhf.alignment_policy import ALIGNMENT_POLICIES, ALL_TASKS

    assert set(ALL_TASKS) == set(ALIGNMENT_POLICIES)
    assert ALIGNMENT_POLICIES["sentiment_analysis"].primary_method == "grpo"
    assert ALIGNMENT_POLICIES["quant_strategy"].primary_method == "grpo"
    assert ALIGNMENT_POLICIES["financial_report"].primary_method == "dpo"


def test_reward_functions_cover_task_specific_signals():
    from scripts.rlhf.alignment_policy import compute_reward

    assert compute_reward("sentiment_analysis", "情感倾向：积极", "积极")["reward"] > 0.8

    code_resp = """策略包含入场、出场、仓位和止损，并计算夏普、最大回撤和年化收益。
```python
import pandas as pd

def generate_signals(df):
    return df
```"""
    q = compute_reward("quant_strategy", code_resp)
    assert q["syntax_correct"] == 1.0
    assert q["has_import"] == 1.0
    assert q["has_function"] == 1.0
    assert q["reward"] > 0.7

    risk = compute_reward("risk_assessment", "综合风险等级：高。信用风险和市场风险较高，波动率30%，建议止损和对冲。")
    assert risk["risk_level_present"] == 1.0
    assert risk["quantitative_indicator"] == 1.0


def test_preference_generation_keeps_sentiment_pairs():
    from scripts.data_processing.synthesize_preference_data import process_rules_mode

    samples = [
        _item("sentiment_analysis", "新闻A", "情感倾向：积极"),
        _item("sentiment_analysis", "新闻B", "情感倾向：积极"),
    ]
    pairs = process_rules_mode(samples)
    assert len(pairs) == 2
    assert pairs[0]["task_type"] == "sentiment_analysis"
    assert pairs[0]["alignment_method"] == "dpo"
    assert pairs[0]["chosen"]["value"] != pairs[0]["rejected"]["value"]


def test_grpo_record_contains_reward_spec():
    from scripts.data_processing.prepare_grpo_data import build_grpo_record

    record = build_grpo_record(_item("sentiment_analysis", "新闻A", "情感倾向：积极"), "sentiment_analysis")
    assert record is not None
    assert record["alignment_method"] == "grpo"
    assert record["answer_key"] == "positive"
    assert record["reward_spec"]["name"] == "sentiment_label_reward"
    assert "offline_reference_reward" in record


def test_train_rlhf_help_mentions_new_methods():
    result = subprocess.run(
        ["bash", "scripts/training/train_rlhf.sh", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--method dpo|grpo|all" in result.stdout
    assert "--skip-train" in result.stdout
    assert "--grpo-config" in result.stdout


def test_trl_grpo_config_exists_and_points_to_rlvf_output():
    import yaml

    config_path = PROJECT_ROOT / "configs" / "qwen3_8b_qlora_grpo_trl.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["train_file"] == "data/rlhf/fin_grpo_prompts_train.json"
    assert config["eval_file"] == "data/rlhf/fin_grpo_prompts_eval.json"
    assert config["adapter_name_or_path"] == "saves/qwen3-8b/lora/dpo"
    assert config["output_dir"] == "saves/qwen3-8b/lora/rlvf"
    assert config["num_generations"] >= 2


def test_trl_grpo_reward_functions_accept_metadata():
    from scripts.training.train_grpo_trl import format_reward, length_reward, task_reward

    completions = ["Yes", "综合风险等级：高。信用风险较高，波动率30%，建议止损和对冲。"]
    task_type = ["sentiment_analysis", "risk_assessment"]
    reference = ["Yes", "综合风险等级：高。信用风险、市场风险、缓释建议。"]
    answer_key = ["yes", ""]
    rewards = task_reward(completions, task_type, reference, answer_key)
    assert rewards[0] > 0.8
    assert rewards[1] > 0.5
    assert len(format_reward(completions, task_type)) == 2
    assert len(length_reward(completions, task_type)) == 2


def test_offline_rlvf_report_builds(tmp_path):
    from scripts.evaluation.eval_rlhf_comparison import dpo_preference_metrics, offline_reward_report

    task_groups = {
        "sentiment_analysis": [_item("sentiment_analysis", "新闻A", "情感倾向：积极")],
        "risk_assessment": [_item("risk_assessment", "评估风险", "综合风险等级：高。信用风险、市场风险、缓释建议。")],
    }
    pref_items = [
        {
            "task_type": "sentiment_analysis",
            "conversations": [{"from": "human", "value": "新闻A"}],
            "chosen": {"from": "gpt", "value": "情感倾向：积极"},
            "rejected": {"from": "gpt", "value": "情感倾向：消极"},
        }
    ]
    report = offline_reward_report(task_groups, pref_items)
    assert report["mode"] == "offline_data_validation"
    assert report["reference_reward_summary"]["sentiment_analysis"]["count"] == 1
    assert dpo_preference_metrics(pref_items)["total"] == 1
