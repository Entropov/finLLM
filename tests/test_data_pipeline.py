#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管道单元测试

覆盖:
  - ShareGPT 格式校验
  - dataset_info.json 结构正确性
  - system_prompts.json 完整性
  - 数据清洗函数
  - 质量过滤函数
  - 格式转换函数
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 路径常量
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "system_prompts.json"
DATASET_INFO_FILE = DATA_DIR / "dataset_info.json"


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def system_prompts():
    """加载 system prompts。"""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def dataset_info():
    """加载 dataset_info.json。"""
    with open(DATASET_INFO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_sharegpt_data():
    """ShareGPT 格式样例数据。"""
    return [
        {
            "conversations": [
                {"from": "human", "value": "什么是市盈率？"},
                {"from": "gpt", "value": "市盈率（PE Ratio）是股票每股市价与每股盈利的比率。"},
            ],
            "system": "你是一位专业的金融分析师。",
        },
        {
            "conversations": [
                {"from": "human", "value": "分析贵州茅台的投资价值"},
                {"from": "gpt", "value": "贵州茅台作为A股白酒龙头..."},
                {"from": "human", "value": "请从估值角度分析"},
                {"from": "gpt", "value": "从估值角度来看..."},
            ],
            "system": "你是一位专业的金融分析师。",
        },
    ]


@pytest.fixture
def sample_alpaca_data():
    """Alpaca 格式样例数据。"""
    return [
        {
            "instruction": "什么是市盈率？",
            "input": "",
            "output": "市盈率（PE Ratio）是股票每股市价与每股盈利的比率。",
        },
        {
            "instruction": "分析以下公司的财务状况",
            "input": "公司名称：贵州茅台",
            "output": "贵州茅台财务状况良好...",
        },
    ]


# ============================================================
# 测试: system_prompts.json
# ============================================================
class TestSystemPrompts:
    """测试 system prompts 完整性。"""

    EXPECTED_TASKS = [
        "stock_analysis",
        "quant_strategy",
        "financial_report",
        "sentiment_analysis",
        "financial_qa",
        "risk_assessment",
        "general",
    ]

    def test_prompts_file_exists(self):
        """提示词文件存在。"""
        assert PROMPTS_FILE.exists(), f"找不到 {PROMPTS_FILE}"

    def test_all_tasks_present(self, system_prompts):
        """所有任务类型都有对应的 prompt。"""
        for task in self.EXPECTED_TASKS:
            assert task in system_prompts, f"缺少任务: {task}"

    def test_prompt_structure(self, system_prompts):
        """每个 prompt 包含必要字段。"""
        for task, data in system_prompts.items():
            assert "task_name" in data, f"{task} 缺少 task_name"
            assert "system_prompt" in data, f"{task} 缺少 system_prompt"
            assert len(data["system_prompt"]) > 50, f"{task} 的 prompt 太短"

    def test_prompt_contains_disclaimer(self, system_prompts):
        """每个 prompt 应包含免责声明。"""
        for task, data in system_prompts.items():
            prompt = data["system_prompt"]
            # 检查是否含有某种形式的免责提示
            has_disclaimer = any(
                kw in prompt for kw in ["免责", "不构成", "风险", "仅供参考", "自行判断"]
            )
            assert has_disclaimer, f"{task} 的 prompt 缺少免责声明"


# ============================================================
# 测试: dataset_info.json
# ============================================================
class TestDatasetInfo:
    """测试 dataset_info.json 正确性。"""

    def test_dataset_info_exists(self):
        """数据集配置文件存在。"""
        assert DATASET_INFO_FILE.exists()

    def test_entries_have_required_fields(self, dataset_info):
        """每个数据集条目包含必要字段。"""
        for name, info in dataset_info.items():
            assert "file_name" in info, f"{name} 缺少 file_name"
            assert "formatting" in info, f"{name} 缺少 formatting"

    def test_sharegpt_formatting(self, dataset_info):
        """ShareGPT 格式的数据集配置正确。"""
        for name, info in dataset_info.items():
            if info.get("formatting") == "sharegpt":
                columns = info.get("columns", {})
                assert "messages" in columns, f"{name} 缺少 messages 列映射"
                tags = info.get("tags", {})
                assert "role_tag" in tags, f"{name} 缺少 role_tag"
                assert "content_tag" in tags, f"{name} 缺少 content_tag"

    def test_train_eval_datasets_exist(self, dataset_info):
        """训练集和验证集定义存在。"""
        assert "fin_instruct_train" in dataset_info, "缺少训练集定义"
        assert "fin_instruct_eval" in dataset_info, "缺少验证集定义"


# ============================================================
# 测试: ShareGPT 格式校验
# ============================================================
class TestShareGPTFormat:
    """测试 ShareGPT 数据格式。"""

    def test_valid_structure(self, sample_sharegpt_data):
        """有效数据结构。"""
        for item in sample_sharegpt_data:
            assert "conversations" in item
            assert isinstance(item["conversations"], list)
            assert len(item["conversations"]) >= 2

    def test_alternating_roles(self, sample_sharegpt_data):
        """对话角色交替出现。"""
        for item in sample_sharegpt_data:
            convs = item["conversations"]
            for i, turn in enumerate(convs):
                expected_role = "human" if i % 2 == 0 else "gpt"
                assert turn["from"] == expected_role, (
                    f"第 {i} 轮角色应为 {expected_role}，实际为 {turn['from']}"
                )

    def test_non_empty_values(self, sample_sharegpt_data):
        """对话内容非空。"""
        for item in sample_sharegpt_data:
            for turn in item["conversations"]:
                assert "value" in turn
                assert len(turn["value"].strip()) > 0

    def test_system_field_is_string(self, sample_sharegpt_data):
        """system 字段为字符串。"""
        for item in sample_sharegpt_data:
            if "system" in item:
                assert isinstance(item["system"], str)


# ============================================================
# 测试: 数据清洗函数
# ============================================================
class TestDataCleaning:
    """测试数据清洗函数。"""

    def test_html_removal(self):
        """HTML 标签移除。"""
        from scripts.data_processing.clean_data import clean_text
        text = "<p>这是一段<b>测试</b>文本</p>"
        cleaned = clean_text(text)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "测试" in cleaned

    def test_url_removal(self):
        """URL 移除。"""
        from scripts.data_processing.clean_data import clean_text
        text = "访问 https://www.example.com 了解详情"
        cleaned = clean_text(text)
        assert "https://" not in cleaned

    def test_phone_masking(self):
        """手机号脱敏。"""
        from scripts.data_processing.clean_data import clean_text
        text = "联系电话：13812345678"
        cleaned = clean_text(text)
        assert "13812345678" not in cleaned

    def test_id_card_masking(self):
        """身份证号脱敏。"""
        from scripts.data_processing.clean_data import clean_text
        text = "身份证号：110101199001011234"
        cleaned = clean_text(text)
        assert "110101199001011234" not in cleaned


# ============================================================
# 测试: 质量过滤函数
# ============================================================
class TestQualityFilter:
    """测试质量过滤函数。"""

    def test_length_filter(self):
        """长度过滤。"""
        from scripts.data_processing.quality_filter import filter_by_length
        short_item = {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好"},
            ]
        }
        long_item = {
            "conversations": [
                {"from": "human", "value": "请详细分析贵州茅台的投资价值"},
                {"from": "gpt", "value": "贵州茅台是A股市场的标杆企业。" * 20},
            ]
        }
        # 短文本应被过滤
        assert not filter_by_length(short_item, min_len=50)
        # 长文本应通过
        assert filter_by_length(long_item, min_len=50)

    def test_dedup_exact(self):
        """精确去重。"""
        from scripts.data_processing.quality_filter import deduplicate_exact
        items = [
            {"conversations": [{"from": "human", "value": "问题A"}, {"from": "gpt", "value": "回答A"}]},
            {"conversations": [{"from": "human", "value": "问题A"}, {"from": "gpt", "value": "回答A"}]},
            {"conversations": [{"from": "human", "value": "问题B"}, {"from": "gpt", "value": "回答B"}]},
        ]
        deduped = deduplicate_exact(items)
        assert len(deduped) == 2


# ============================================================
# 测试: 格式转换
# ============================================================
class TestFormatConversion:
    """测试格式转换函数。"""

    def test_alpaca_to_sharegpt(self, sample_alpaca_data):
        """Alpaca → ShareGPT 格式转换。"""
        from scripts.data_processing.convert_to_sharegpt import convert_alpaca_to_sharegpt
        result = convert_alpaca_to_sharegpt(sample_alpaca_data)
        for item in result:
            assert "conversations" in item
            assert item["conversations"][0]["from"] == "human"
            assert item["conversations"][1]["from"] == "gpt"

    def test_alpaca_with_input(self, sample_alpaca_data):
        """Alpaca 带 input 字段的转换。"""
        from scripts.data_processing.convert_to_sharegpt import convert_alpaca_to_sharegpt
        result = convert_alpaca_to_sharegpt(sample_alpaca_data)
        # 第二条数据有 input，应合并到 human 消息中
        human_msg = result[1]["conversations"][0]["value"]
        assert "分析" in human_msg
        assert "贵州茅台" in human_msg


# ============================================================
# 测试: 评估指标
# ============================================================
class TestEvalMetrics:
    """测试评估指标函数。"""

    def test_rouge_l(self):
        """ROUGE-L 计算。"""
        from scripts.evaluation.eval_metrics import compute_rouge_l
        hypothesis = "今天天气真不错啊"
        reference = "今天天气很好"
        score = compute_rouge_l(hypothesis, reference)
        assert 0 <= score <= 1

    def test_accuracy(self):
        """准确率计算。"""
        from scripts.evaluation.eval_metrics import compute_accuracy
        preds = ["A", "B", "C", "A"]
        refs = ["A", "B", "A", "A"]
        acc = compute_accuracy(preds, refs)
        assert acc == pytest.approx(0.75)

    def test_keyword_coverage(self):
        """关键词覆盖率。"""
        from scripts.evaluation.eval_metrics import compute_keyword_coverage
        text = "贵州茅台的市盈率为30倍，净利润增长15%"
        keywords = ["市盈率", "净利润", "营收"]
        coverage = compute_keyword_coverage(text, keywords)
        assert coverage == pytest.approx(2 / 3)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
