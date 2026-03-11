#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融基准测试评估

使用 FinEval 数据集对模型进行金融知识多选题评测:
  - 加载 FinEval 评测数据（金融、经济、会计、证书类题目）
  - 批量推理，提取模型选择的答案
  - 计算各科目准确率
  - 支持对比微调前后的表现

用法:
  python scripts/evaluation/eval_finance_bench.py --model_path saves/qwen2.5-7b/lora/sft
  python scripts/evaluation/eval_finance_bench.py --model_path Qwen/Qwen2.5-7B-Instruct  # 基线
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

import torch
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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "saves" / "eval_results"


def load_fineval_data(data_dir: Optional[Path] = None) -> list:
    """
    加载 FinEval 评测数据。

    数据格式（多选题）:
    {
        "question": "题目文本",
        "A": "选项A",
        "B": "选项B",
        "C": "选项C",
        "D": "选项D",
        "answer": "A"  // 正确答案
    }

    参数:
        data_dir: FinEval 数据目录，默认为 data/raw/fineval/

    返回:
        评测题目列表
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR / "fineval"

    if not data_dir.exists():
        logger.warning(f"FinEval 数据目录不存在: {data_dir}")
        logger.info("请先运行: python scripts/data_collection/download_open_datasets.py --datasets fineval")
        return []

    all_questions = []
    for json_file in sorted(data_dir.rglob("*.json")):
        if json_file.name.startswith("_"):
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "question" in item or "input" in item:
                        all_questions.append(item)
            elif isinstance(data, dict):
                # 可能是按 split/subject 组织的
                for key, items in data.items():
                    if isinstance(items, list):
                        all_questions.extend(items)
        except Exception as e:
            logger.warning(f"读取 {json_file} 失败: {e}")

    logger.info(f"加载 FinEval 题目: {len(all_questions)} 道")
    return all_questions


def format_mcq_prompt(question_data: dict) -> str:
    """
    将多选题数据格式化为模型输入 Prompt。

    参数:
        question_data: 单道题目的字典

    返回:
        格式化后的 prompt 字符串
    """
    question = question_data.get("question", question_data.get("input", ""))
    option_a = question_data.get("A", "")
    option_b = question_data.get("B", "")
    option_c = question_data.get("C", "")
    option_d = question_data.get("D", "")

    prompt = (
        f"请回答以下单选题，直接给出答案选项字母（A/B/C/D），不需要解释。\n\n"
        f"题目：{question}\n"
        f"A. {option_a}\n"
        f"B. {option_b}\n"
        f"C. {option_c}\n"
        f"D. {option_d}\n\n"
        f"答案："
    )
    return prompt


def extract_answer(response: str) -> str:
    """
    从模型回答中提取答案选项。

    参数:
        response: 模型的原始回答文本

    返回:
        提取到的答案（A/B/C/D），未能提取返回空字符串
    """
    response = response.strip()

    # 策略1: 直接匹配开头的单字母
    if response and response[0] in "ABCD":
        return response[0]

    # 策略2: 正则匹配 "答案是X"、"选X"、"答案为X" 等模式
    patterns = [
        r"答案[是为：:\s]*([ABCD])",
        r"选[择项]?\s*([ABCD])",
        r"^([ABCD])[.、\)\s]",
        r"正确答案[是为：:\s]*([ABCD])",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    # 策略3: 在整个回答中寻找唯一出现的选项字母
    found = re.findall(r"\b([ABCD])\b", response)
    if len(found) == 1:
        return found[0]

    return ""


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
    quantize: bool = True,
):
    """
    加载模型和分词器。

    参数:
        model_path:   基座模型路径或 HuggingFace ID
        adapter_path: LoRA adapter 路径 (可选)
        quantize:     是否使用 4-bit 量化

    返回:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    logger.info(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    # 量化配置
    if quantize:
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

    # 加载 LoRA adapter
    if adapter_path and Path(adapter_path).exists():
        logger.info(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    questions: list,
    batch_size: int = 1,
    max_new_tokens: int = 32,
) -> dict:
    """
    在 FinEval 数据集上评估模型。

    参数:
        model:          语言模型
        tokenizer:      分词器
        questions:      评测题目列表
        batch_size:     批处理大小
        max_new_tokens: 最大生成 token 数

    返回:
        评估结果字典
    """
    correct = 0
    total = 0
    results = []

    for item in tqdm(questions, desc="评测进度", unit="题"):
        prompt = format_mcq_prompt(item)
        ground_truth = item.get("answer", "").strip().upper()

        if not ground_truth or ground_truth not in "ABCD":
            continue

        # 构建 ChatML 格式输入
        messages = [
            {"role": "system", "content": "你是一个金融知识专家，请直接回答选择题的正确选项。"},
            {"role": "user", "content": prompt},
        ]

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,  # 低温度，接近贪心解码
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # 解码生成的部分（去掉输入）
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)

            predicted = extract_answer(response)
            is_correct = predicted == ground_truth

            total += 1
            if is_correct:
                correct += 1

            results.append({
                "question": item.get("question", item.get("input", "")),
                "ground_truth": ground_truth,
                "predicted": predicted,
                "raw_response": response[:200],
                "correct": is_correct,
            })

        except Exception as e:
            logger.debug(f"推理失败: {e}")
            total += 1
            results.append({
                "question": item.get("question", ""),
                "ground_truth": ground_truth,
                "predicted": "",
                "raw_response": str(e),
                "correct": False,
            })

    accuracy = correct / total if total > 0 else 0.0

    eval_result = {
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": results,
    }

    return eval_result


def run_evaluation(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: Optional[str] = None,
    max_questions: Optional[int] = None,
) -> None:
    """
    运行完整的 FinEval 评测流程。

    参数:
        model_path:    模型路径
        adapter_path:  LoRA adapter 路径
        max_questions: 最大评测题数（调试用）
    """
    logger.info("=" * 60)
    logger.info("FinEval 金融基准测试")
    logger.info("=" * 60)

    # 1. 加载数据
    questions = load_fineval_data()
    if not questions:
        return

    if max_questions:
        questions = questions[:max_questions]
        logger.info(f"调试模式: 仅评测前 {max_questions} 题")

    # 2. 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
    )

    # 3. 评测
    result = evaluate_model(model, tokenizer, questions)

    # 4. 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).name or "unknown"
    if adapter_path:
        model_name += "_lora"

    result_file = RESULTS_DIR / f"fineval_{model_name}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 5. 输出结果
    logger.info(f"\n{'='*60}")
    logger.info(f"评测结果:")
    logger.info(f"  模型: {model_path}")
    if adapter_path:
        logger.info(f"  Adapter: {adapter_path}")
    logger.info(f"  总题数: {result['total_questions']}")
    logger.info(f"  正确数: {result['correct']}")
    logger.info(f"  准确率: {result['accuracy']:.2%}")
    logger.info(f"  结果保存: {result_file}")
    logger.info(f"{'='*60}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinEval 金融基准测试")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型路径或 HuggingFace ID",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter 路径 (可选，评测微调模型时使用)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="最大评测题数（调试用）",
    )
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_questions=args.max_questions,
    )
