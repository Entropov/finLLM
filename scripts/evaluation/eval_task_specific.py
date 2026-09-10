#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
各任务专项评测

对金融任务分别进行评测:
  1. 情感分析 (sentiment_analysis): 准确率、F1-score
  2. 股票分析 (stock_analysis): ROUGE-L、关键词覆盖率
  3. 金融问答 (financial_qa): ROUGE-L、答案长度
  4. 财报解读 (financial_report): ROUGE-L、关键指标提取率
  5. 量化策略 (quant_strategy): 代码语法正确率、完整性
  6. 风险评估 (risk_assessment): 关键要素覆盖率

数据格式 (conversations 格式):
  {
      "conversations": [
          {"from": "human", "value": "问题"},
          {"from": "gpt",   "value": "参考答案"}
      ],
      "system": "系统提示",
      "task_type": "financial_qa"
  }

用法:
  # 仅统计数据分布（不加载模型）
  python scripts/evaluation/eval_task_specific.py --task all

  # 加载模型进行推理评测
  python scripts/evaluation/eval_task_specific.py \\
      --model-path Qwen/Qwen3-8B --task all

  # 评测微调后的模型
  python scripts/evaluation/eval_task_specific.py \\
      --model-path saves/qwen3-8b/lora/sft --task all

  # 调试模式（只跑少量样本）
  python scripts/evaluation/eval_task_specific.py \\
      --model-path Qwen/Qwen3-8B --task financial_qa --max-samples 20
"""

import ast
import os
import json
import re
import logging
import sys
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

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
sys.path.append(str(PROJECT_ROOT))
from scripts.inference.chat_template import apply_chat_template

SFT_DIR = PROJECT_ROOT / "data" / "sft"
RESULTS_DIR = PROJECT_ROOT / "saves" / "eval_results"

ALL_TASKS = [
    "stock_analysis", "quant_strategy", "financial_report",
    "sentiment_analysis", "financial_qa", "risk_assessment",
]
MIN_EVAL_PER_TASK = 300
OPEN_GENERATION_TASKS = {
    "stock_analysis",
    "quant_strategy",
    "financial_report",
    "financial_qa",
    "risk_assessment",
}

TASK_FILE_MAP = {
    "stock_analysis": "fin_stock_analysis",
    "quant_strategy": "fin_quant_strategy",
    "financial_report": "fin_financial_report",
    "sentiment_analysis": "fin_sentiment_analysis",
    "financial_qa": "fin_financial_qa",
    "risk_assessment": "fin_risk_assessment",
}

TASK_KEYWORDS = {
    "stock_analysis": ["趋势", "支撑", "压力", "成交量", "均线", "MACD", "RSI", "K线"],
    "financial_report": ["营收", "净利润", "ROE", "现金流", "毛利率", "资产负债率", "流动比率"],
    "financial_qa": ["定义", "公式", "风险", "例", "适用", "区别", "原理"],
    "risk_assessment": ["风险", "评级", "波动", "建议", "因素", "违约", "VaR", "回撤", "对冲"],
}


def _preferred_task_file(task_name: str) -> Optional[Path]:
    base_name = TASK_FILE_MAP[task_name]
    raw_path = SFT_DIR / f"{base_name}.json"
    filtered_path = SFT_DIR / f"{base_name}_filtered.json"
    if filtered_path.exists() and raw_path.exists():
        return filtered_path if filtered_path.stat().st_mtime >= raw_path.stat().st_mtime else raw_path
    if filtered_path.exists():
        return filtered_path
    if raw_path.exists():
        return raw_path
    return None


# ============================================================
# 数据加载
# ============================================================

def load_eval_data(task: str = "all") -> dict:
    """
    加载评测数据，按任务分组。

    数据源优先级:
    1. data/sft/fin_instruct_eval.json（主评测集）
    2. 各个任务专用 JSON 文件（补充）

    返回:
        {task_name: [样本列表]} 形式的字典
    """
    task_groups = defaultdict(list)

    # 主评测文件
    eval_file = SFT_DIR / "fin_instruct_eval.json"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        for item in eval_data:
            task_type = item.get("task_type", "general")
            task_groups[task_type].append(item)
        logger.info(f"主评测集加载: {len(eval_data)} 条")
    else:
        logger.warning(f"主评测集不存在: {eval_file}")

    target_tasks = ALL_TASKS if task == "all" else [task]
    for task_name in target_tasks:
        fpath = _preferred_task_file(task_name)
        if fpath and len(task_groups[task_name]) < MIN_EVAL_PER_TASK:
            with open(fpath, "r", encoding="utf-8") as f:
                extra = json.load(f)
            # 只取尚未加入的部分（用 conversations 内容去重）
            existing_q = {
                _get_question(item) for item in task_groups[task_name]
            }
            added = 0
            for item in extra:
                if _get_question(item) not in existing_q:
                    item["task_type"] = task_name
                    task_groups[task_name].append(item)
                    existing_q.add(_get_question(item))
                    added += 1
                    if len(task_groups[task_name]) >= MIN_EVAL_PER_TASK:
                        break
            if added:
                logger.info(f"  从 {fpath.name} 补充 {task_name}: +{added} 条")

    # 过滤
    if task != "all":
        return {task: task_groups.get(task, [])}
    return {task_name: task_groups.get(task_name, []) for task_name in ALL_TASKS}


def _get_question(item: dict) -> str:
    """从评测样本中提取问题文本（用于去重）"""
    convs = item.get("conversations", [])
    for c in convs:
        if c.get("from") == "human":
            return c.get("value", "")[:200]
    return item.get("input", item.get("question", ""))[:200]


def _get_reference(item: dict) -> str:
    """从评测样本中提取参考答案"""
    convs = item.get("conversations", [])
    for c in convs:
        if c.get("from") == "gpt":
            return c.get("value", "")
    return item.get("output", item.get("answer", ""))


# ============================================================
# 模型推理
# ============================================================

def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
    quantize: bool = True,
):
    """
    加载模型和分词器。

    参数:
        model_path:   基座模型路径或 HuggingFace ID
        adapter_path: LoRA adapter 路径（可选）
        quantize:     是否使用 4-bit 量化

    返回:
        (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger.info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    if quantize and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if adapter_path and Path(adapter_path).exists():
        logger.info(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    item: dict,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """
    对单条样本进行推理，返回模型生成的文本。

    参数:
        model:          语言模型
        tokenizer:      分词器
        item:           评测数据条目
        max_new_tokens: 最大生成 token 数
        temperature:    采样温度（低温更确定）

    返回:
        模型生成的文本
    """
    import torch

    question = _get_question(item)
    system_prompt = item.get("system", "你是一个金融专家助手，请认真回答以下问题。")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        text = apply_chat_template(messages=messages, tokenizer=tokenizer)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        logger.debug(f"推理失败: {e}")
        return ""


# ============================================================
# 评测指标计算
# ============================================================

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    计算 ROUGE-L F1 分数（基于字符级 LCS）。

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


def keyword_coverage(text: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    return round(sum(1 for kw in keywords if kw in text) / len(keywords), 4)


def regex_presence_rate(text: str, patterns: list) -> float:
    if not patterns:
        return 1.0
    hits = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
    return round(hits / len(patterns), 4)


def response_quality_score(text: str) -> float:
    if not text:
        return 0.0
    score = 0.0
    if len(text) >= 80:
        score += 0.25
    if re.search(r"^\s*\d+[.、]", text, re.MULTILINE) or "\n" in text:
        score += 0.25
    if any(term in text for term in ["风险", "收益", "估值", "现金流", "波动", "策略", "指标"]):
        score += 0.25
    if any(term in text[-200:] for term in ["建议", "综上", "关注", "不构成投资建议", "仅供"]):
        score += 0.25
    return round(score, 4)


def extract_risk_level(text: str) -> str:
    patterns = [
        ("very_high", ["极高风险", "极高"]),
        ("high", ["高风险", "较高", "高"]),
        ("medium", ["中等风险", "中风险", "中"]),
        ("low", ["低风险", "较低", "低"]),
    ]
    for label, keywords in patterns:
        if any(keyword in text for keyword in keywords):
            return label
    return "unknown"


def _extract_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def extract_sentiment_label(text: str) -> str:
    """
    从模型输出中提取情感标签（积极/消极/中性）。

    参数:
        text: 模型生成文本

    返回:
        标准化的情感标签: "positive" / "negative" / "neutral" / "unknown"
    """
    label_map = {
        "积极": "positive", "正面": "positive", "利好": "positive",
        "看多": "positive", "bullish": "positive", "positive": "positive",
        "消极": "negative", "负面": "negative", "利空": "negative",
        "看空": "negative", "bearish": "negative", "negative": "negative",
        "中性": "neutral", "中立": "neutral", "neutral": "neutral",
    }
    text_lower = text.lower()
    for keyword, label in label_map.items():
        if keyword.lower() in text_lower:
            return label
    return "unknown"


def evaluate_sentiment(items: list, predictions: list) -> dict:
    """
    情感分析评测（准确率、各类别 F1）。

    情感标签从参考答案和预测文本中自动解析。
    """
    label_map = {
        "积极": "positive", "正面": "positive", "利好": "positive",
        "positive": "positive", "看多": "positive",
        "消极": "negative", "负面": "negative", "利空": "negative",
        "negative": "negative", "看空": "negative",
        "中性": "neutral", "中立": "neutral", "neutral": "neutral",
    }

    correct = 0
    total = 0
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for item, pred_text in zip(items, predictions):
        ref_text = _get_reference(item)

        # 从参考答案提取真实标签
        ref_label = "unknown"
        for kw, lbl in label_map.items():
            if kw.lower() in ref_text.lower():
                ref_label = lbl
                break

        # 从预测中提取预测标签
        pred_label = extract_sentiment_label(pred_text)

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

    per_class = {}
    for label, stats in label_stats.items():
        p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
        r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    # 宏平均 F1
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0

    return {
        "task": "sentiment_analysis",
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def evaluate_text_generation(items: list, predictions: list, task_name: str) -> dict:
    """
    文本生成类任务评测（股票分析、财报解读、金融问答、风险评估）。

    评测指标:
      - 平均 ROUGE-L
      - 平均回答长度
      - 关键词覆盖率（任务相关关键词）
    """
    keywords = TASK_KEYWORDS.get(task_name, [])
    rouge_scores = []
    lengths = []
    keyword_hits = []
    quality_scores = []

    for item, pred_text in zip(items, predictions):
        ref_text = _get_reference(item)

        if not ref_text or not pred_text:
            continue

        rouge = compute_rouge_l(ref_text, pred_text)
        rouge_scores.append(rouge)
        lengths.append(len(pred_text))
        quality_scores.append(response_quality_score(pred_text))

        if keywords:
            keyword_hits.append(keyword_coverage(pred_text, keywords))

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0
    avg_keyword = sum(keyword_hits) / len(keyword_hits) if keyword_hits else None

    result = {
        "task": task_name,
        "total_evaluated": len(rouge_scores),
        "avg_rouge_l": round(avg_rouge, 4),
        "avg_response_length": round(avg_length, 1),
        "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.0,
    }
    if avg_keyword is not None:
        result["avg_keyword_coverage"] = round(avg_keyword, 4)
    return result


def evaluate_stock_analysis(items: list, predictions: list) -> dict:
    result = evaluate_text_generation(items, predictions, "stock_analysis")
    structure_patterns = [r"支撑", r"压力", r"趋势", r"MACD", r"RSI", r"成交量|量能", r"风险|止损"]
    rates = [regex_presence_rate(pred, structure_patterns) for pred in predictions if pred]
    result["avg_technical_structure_coverage"] = round(sum(rates) / len(rates), 4) if rates else 0.0
    return result


def evaluate_financial_report(items: list, predictions: list) -> dict:
    result = evaluate_text_generation(items, predictions, "financial_report")
    metric_patterns = [r"营收|营业收入", r"净利润", r"ROE", r"现金流", r"资产负债率", r"流动比率"]
    risk_patterns = [r"亮点|优势|改善", r"风险|压力|异常"]
    number_pattern = r"\d+(\.\d+)?%|\d+(\.\d+)?(亿|万)"
    metric_rates = [regex_presence_rate(pred, metric_patterns) for pred in predictions if pred]
    risk_rates = [regex_presence_rate(pred, risk_patterns) for pred in predictions if pred]
    number_rates = [1.0 if re.search(number_pattern, pred) else 0.0 for pred in predictions if pred]
    result["avg_financial_metric_coverage"] = round(sum(metric_rates) / len(metric_rates), 4) if metric_rates else 0.0
    result["avg_risk_highlight_coverage"] = round(sum(risk_rates) / len(risk_rates), 4) if risk_rates else 0.0
    result["numeric_reference_rate"] = round(sum(number_rates) / len(number_rates), 4) if number_rates else 0.0
    return result


def evaluate_risk_assessment(items: list, predictions: list) -> dict:
    result = evaluate_text_generation(items, predictions, "risk_assessment")
    factor_patterns = [r"信用风险", r"市场风险", r"波动|VaR|回撤", r"缓释|对冲|止损", r"等级|评级"]
    rates = [regex_presence_rate(pred, factor_patterns) for pred in predictions if pred]
    level_hits = [1.0 if extract_risk_level(pred) != "unknown" else 0.0 for pred in predictions if pred]
    quantitative_hits = [1.0 if re.search(r"\d+(\.\d+)?%|VaR|Beta|回撤|波动率", pred) else 0.0 for pred in predictions if pred]
    result["avg_risk_factor_coverage"] = round(sum(rates) / len(rates), 4) if rates else 0.0
    result["risk_level_extraction_rate"] = round(sum(level_hits) / len(level_hits), 4) if level_hits else 0.0
    result["quantitative_risk_indicator_rate"] = round(sum(quantitative_hits) / len(quantitative_hits), 4) if quantitative_hits else 0.0
    return result


def evaluate_code_generation(items: list, predictions: list) -> dict:
    """
    量化策略代码生成评测。

    评测指标:
      - 语法正确率 (ast.parse)
      - import 存在率
      - 函数定义存在率
      - 平均代码长度
    """
    total = 0
    syntax_correct = 0
    has_import = 0
    has_function = 0
    has_backtest_metric = 0
    has_strategy_element = 0
    code_lengths = []
    rouge_scores = []

    for item, pred_text in zip(items, predictions):
        # 提取代码块（markdown 格式）
        code = _extract_code(pred_text)

        if not code.strip():
            continue

        total += 1
        code_lengths.append(len(code))

        try:
            ast.parse(code)
            syntax_correct += 1
        except SyntaxError:
            pass

        if re.search(r"^import\s|^from\s", code, re.MULTILINE):
            has_import += 1
        if re.search(r"^def\s", code, re.MULTILINE):
            has_function += 1
        if re.search(r"sharpe|夏普|drawdown|回撤|annual_return|年化", pred_text, re.IGNORECASE):
            has_backtest_metric += 1
        if re.search(r"入场|出场|止损|仓位|风控|信号", pred_text):
            has_strategy_element += 1

        ref_text = _get_reference(item)
        if ref_text:
            rouge_scores.append(compute_rouge_l(ref_text, pred_text))

    return {
        "task": "quant_strategy",
        "total_evaluated": total,
        "syntax_correct_rate": round(syntax_correct / total, 4) if total > 0 else 0.0,
        "has_import_rate": round(has_import / total, 4) if total > 0 else 0.0,
        "has_function_rate": round(has_function / total, 4) if total > 0 else 0.0,
        "has_backtest_metric_rate": round(has_backtest_metric / total, 4) if total > 0 else 0.0,
        "has_strategy_element_rate": round(has_strategy_element / total, 4) if total > 0 else 0.0,
        "avg_code_length": round(sum(code_lengths) / len(code_lengths), 1) if code_lengths else 0.0,
        "avg_rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else 0.0,
    }


def evaluate_task_predictions(task_name: str, items: list, predictions: list) -> dict:
    if task_name == "sentiment_analysis":
        return evaluate_sentiment(items, predictions)
    if task_name == "quant_strategy":
        return evaluate_code_generation(items, predictions)
    if task_name == "stock_analysis":
        return evaluate_stock_analysis(items, predictions)
    if task_name == "financial_report":
        return evaluate_financial_report(items, predictions)
    if task_name == "risk_assessment":
        return evaluate_risk_assessment(items, predictions)
    return evaluate_text_generation(items, predictions, task_name)


# ============================================================
# 数据分布报告（不需要加载模型）
# ============================================================

def report_data_distribution(task_groups: dict) -> dict:
    """生成数据分布统计报告（无需模型推理）"""
    report = {
        "total_samples": sum(len(v) for v in task_groups.values()),
        "task_distribution": {},
        "sample_examples": {},
    }

    for task_name, items in task_groups.items():
        report["task_distribution"][task_name] = len(items)
        if items:
            sample = items[0]
            report["sample_examples"][task_name] = {
                "question_preview": _get_question(sample)[:100] + "...",
                "reference_preview": _get_reference(sample)[:100] + "...",
            }

    return report


async def judge_one_sample(item: dict, prediction: str, task_name: str) -> Optional[dict]:
    """调用 OpenAI 兼容 API 对开放生成任务打分。缺少配置时由上层跳过。"""
    import aiohttp

    api_key = os.environ.get("EVAL_JUDGE_API_KEY", "")
    api_base = os.environ.get("EVAL_JUDGE_API_BASE", "https://api.openai.com/v1").rstrip("/")
    model_name = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    if not api_key:
        return None

    prompt = (
        "你是金融大模型评测员。请基于题目、参考答案和模型回答评分，返回严格 JSON，"
        "字段为 professionalism, factual_consistency, structure_completeness, risk_disclaimer，"
        "每项为1到5的整数。\n\n"
        f"任务:{task_name}\n"
        f"题目:{_get_question(item)}\n"
        f"参考答案:{_get_reference(item)[:1200]}\n"
        f"模型回答:{prediction[:1200]}"
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    logger.warning(f"Judge 调用失败 HTTP {resp.status}: {await resp.text()}")
                    return None
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"Judge 调用异常: {e}")
        return None

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    result = {}
    for key in ["professionalism", "factual_consistency", "structure_completeness", "risk_disclaimer"]:
        try:
            result[key] = max(1, min(5, int(raw.get(key, 0))))
        except (TypeError, ValueError):
            result[key] = 0
    return result


def run_judge_evaluation(task_name: str, items: list, predictions: list, max_judge_samples: int = 50) -> Optional[dict]:
    """可选 LLM-as-judge；未配置 API Key 时返回 skipped。"""
    if task_name not in OPEN_GENERATION_TASKS:
        return None
    if not os.environ.get("EVAL_JUDGE_API_KEY"):
        return {"skipped": True, "reason": "EVAL_JUDGE_API_KEY not set"}

    import asyncio

    async def _run():
        selected = list(zip(items, predictions))[:max_judge_samples]
        results = []
        for item, pred in selected:
            judged = await judge_one_sample(item, pred, task_name)
            if judged:
                results.append(judged)
        return results

    judged_results = asyncio.run(_run())
    if not judged_results:
        return {"skipped": True, "reason": "no judge result"}
    summary = {"samples": len(judged_results)}
    for key in ["professionalism", "factual_consistency", "structure_completeness", "risk_disclaimer"]:
        values = [r[key] for r in judged_results if r.get(key)]
        summary[f"avg_{key}"] = round(sum(values) / len(values), 3) if values else 0.0
    return summary


# ============================================================
# 主评测流程
# ============================================================

def run_task_evaluation(
    task: str = "all",
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    quantize: bool = True,
    max_new_tokens: int = 512,
    use_judge: bool = False,
    max_judge_samples: int = 50,
) -> None:
    """
    运行指定任务的评测。

    参数:
        task:           任务名称或 "all"
        model_path:     模型路径（None 则只统计数据分布）
        adapter_path:   LoRA adapter 路径（可选）
        max_samples:    每个任务最大评测样本数（调试用）
        quantize:       是否使用 4-bit 量化
        max_new_tokens: 模型最大生成 token 数
    """
    logger.info("=" * 60)
    logger.info("Fin-Instruct 任务专项评测")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载评测数据
    task_groups = load_eval_data(task)
    if not task_groups:
        logger.error("未找到任何评测数据")
        return

    total_samples = sum(len(v) for v in task_groups.values())
    logger.info(f"评测数据总量: {total_samples} 条")
    logger.info("各任务分布:")
    for t, items in sorted(task_groups.items()):
        logger.info(f"  {t}: {len(items)} 条")

    # 2. 无模型模式：仅统计分布
    if model_path is None:
        logger.info("\n" + "=" * 40)
        logger.info("模式: 仅统计数据分布（未指定 --model-path）")
        logger.info("提示: 使用 --model-path 参数指定模型路径以进行完整推理评测")
        logger.info("=" * 40)

        report = report_data_distribution(task_groups)
        report_file = RESULTS_DIR / "task_eval_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"\n数据分布报告已保存: {report_file}")
        return

    # 3. 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        quantize=quantize,
    )

    # 4. 对各任务进行推理 & 评测
    all_results = {}

    for task_name, items in sorted(task_groups.items()):
        if not items:
            continue

        # 限制样本数（调试模式）
        if max_samples and max_samples > 0:
            items = items[:max_samples]

        logger.info(f"\n{'='*40}")
        logger.info(f"开始评测任务: {task_name} ({len(items)} 条)")

        # 推理
        predictions = []
        for item in tqdm(items, desc=f"{task_name} 推理中", unit="条"):
            pred = generate_response(model, tokenizer, item, max_new_tokens=max_new_tokens)
            predictions.append(pred)

        # 评测
        result = evaluate_task_predictions(task_name, items, predictions)
        if use_judge:
            judge_result = run_judge_evaluation(task_name, items, predictions, max_judge_samples=max_judge_samples)
            if judge_result is not None:
                result["judge"] = judge_result

        all_results[task_name] = result

        # 打印当前任务结果
        logger.info(f"任务 {task_name} 结果:")
        for k, v in result.items():
            if k not in ("task", "per_class"):
                logger.info(f"  {k}: {v}")
        if "per_class" in result:
            for label, stats in result["per_class"].items():
                logger.info(f"  [{label}] P={stats['precision']:.4f}  R={stats['recall']:.4f}  F1={stats['f1']:.4f}")

        # 保存预测详情（每个任务单独存）
        detail_file = RESULTS_DIR / f"task_{task_name}_predictions.json"
        details = []
        for item, pred in zip(items, predictions):
            details.append({
                "question": _get_question(item)[:300],
                "reference": _get_reference(item)[:500],
                "prediction": pred[:500],
            })
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

    # 5. 汇总报告
    model_name = Path(model_path).name or "unknown"
    if adapter_path:
        model_name += "_lora"

    summary = {
        "model": model_path,
        "adapter": adapter_path,
        "max_samples_per_task": max_samples,
        "judge_enabled": use_judge,
        "task_results": all_results,
    }

    summary_file = RESULTS_DIR / f"task_eval_{model_name}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 评测总结")
    logger.info("=" * 60)
    logger.info(f"模型: {model_path}")
    for task_name, result in sorted(all_results.items()):
        logger.info(f"\n  [{task_name}]")
        for k, v in result.items():
            if k not in ("task", "per_class"):
                logger.info(f"    {k}: {v}")
    logger.info(f"\n汇总报告: {summary_file}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fin-Instruct 任务专项评测")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=ALL_TASKS + ["all"],
        help="指定评测任务（默认: all 全部评测）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型路径或 HuggingFace ID（不指定则仅统计数据分布）",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter 路径（可选，评测微调模型时使用）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每任务最大评测样本数（调试用，默认全量）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="模型最大生成 token 数（默认: 512）",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="禁用 4-bit 量化（GPU 显存充足时使用）",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="启用可选 LLM-as-judge 评估（需 EVAL_JUDGE_API_KEY）",
    )
    parser.add_argument(
        "--max-judge-samples",
        type=int,
        default=50,
        help="每任务最多 Judge 样本数（默认: 50）",
    )
    args = parser.parse_args()

    run_task_evaluation(
        task=args.task,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_samples=args.max_samples,
        quantize=not args.no_quantize,
        max_new_tokens=args.max_new_tokens,
        use_judge=args.judge,
        max_judge_samples=args.max_judge_samples,
    )
