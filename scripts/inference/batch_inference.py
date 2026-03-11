#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理脚本

从 CSV/JSON 文件读取问题列表，批量推理并输出结果。

支持:
  - CSV 格式输入（需包含 "question" 列）
  - JSON 格式输入（列表，每条含 "question" 字段）
  - 自动添加 system prompt（按任务类型）
  - 结果保存为 JSON/CSV

用法:
  python scripts/inference/batch_inference.py --input questions.json --output results.json
  python scripts/inference/batch_inference.py --input questions.csv --task stock_analysis
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
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
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "system_prompts.json"
RESULTS_DIR = PROJECT_ROOT / "saves" / "batch_results"


def load_system_prompts() -> dict:
    """加载 system prompt 模板。"""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return {k: v["system_prompt"] for k, v in prompts.items() if "system_prompt" in v}


def load_questions(input_file: str) -> list:
    """
    加载问题列表。

    支持 JSON 和 CSV 格式:
      - JSON: [{"question": "..."}, ...] 或 ["问题1", "问题2", ...]
      - CSV:  需包含 "question" 列

    参数:
        input_file: 输入文件路径

    返回:
        问题字典列表 [{"question": ..., "task_type": ..., ...}, ...]
    """
    path = Path(input_file)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return [{"question": q} for q in data]
            return data
        raise ValueError("JSON 文件格式不支持，需要列表格式")

    elif path.suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "question" not in df.columns:
            raise ValueError("CSV 文件需包含 'question' 列")
        return df.to_dict("records")

    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}，支持 .json 和 .csv")


def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    quantize: bool = True,
):
    """加载模型和分词器。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

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

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        logger.info(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def batch_generate(
    model,
    tokenizer,
    questions: list,
    system_prompt: str = "",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list:
    """
    批量推理。

    参数:
        model:          语言模型
        tokenizer:      分词器
        questions:       问题字典列表
        system_prompt:  系统提示词
        max_new_tokens: 最大生成 token 数
        temperature:    温度系数
        top_p:          核采样概率

    返回:
        结果列表 [{"question": ..., "answer": ..., "tokens": ...}, ...]
    """
    results = []

    for item in tqdm(questions, desc="批量推理", unit="条"):
        question = item.get("question", "")
        if not question:
            results.append({**item, "answer": "", "error": "问题为空"})
            continue

        # 使用问题级别的 system prompt（如果有），否则用默认
        sp = item.get("system_prompt", system_prompt)

        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": question},
        ]

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_length = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            generated = outputs[0][input_length:]
            answer = tokenizer.decode(generated, skip_special_tokens=True)

            results.append({
                **item,
                "answer": answer,
                "input_tokens": input_length,
                "output_tokens": len(generated),
            })

        except Exception as e:
            logger.error(f"推理失败: {e}")
            results.append({**item, "answer": "", "error": str(e)})

    return results


def save_results(results: list, output_file: str) -> None:
    """
    保存推理结果。

    参数:
        results:     结果列表
        output_file: 输出文件路径（.json 或 .csv）
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    elif path.suffix == ".csv":
        df = pd.DataFrame(results)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        # 默认 JSON
        path = path.with_suffix(".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存: {path}")


def run_batch_inference(
    input_file: str,
    output_file: Optional[str] = None,
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: Optional[str] = None,
    task: str = "general",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> None:
    """
    运行批量推理完整流程。

    参数:
        input_file:     输入文件路径
        output_file:    输出文件路径
        model_path:     模型路径
        adapter_path:   LoRA adapter 路径
        task:           任务类型
        max_new_tokens: 最大生成 token 数
        temperature:    温度系数
    """
    logger.info("=" * 60)
    logger.info("Fin-Instruct 批量推理")
    logger.info("=" * 60)

    # 1. 加载问题
    questions = load_questions(input_file)
    logger.info(f"加载问题: {len(questions)} 条")

    # 2. 获取 system prompt
    prompts = load_system_prompts()
    system_prompt = prompts.get(task, prompts.get("general", ""))
    logger.info(f"任务类型: {task}")

    # 3. 加载模型
    model, tokenizer = load_model(model_path, adapter_path)

    # 4. 批量推理
    results = batch_generate(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # 5. 保存结果
    if output_file is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = str(RESULTS_DIR / f"batch_{task}_results.json")

    save_results(results, output_file)

    # 6. 统计
    success = sum(1 for r in results if r.get("answer"))
    logger.info(f"\n推理完成: 成功 {success}/{len(results)}")
    total_tokens = sum(r.get("output_tokens", 0) for r in results)
    logger.info(f"总生成 tokens: {total_tokens}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    task_choices = [
        "stock_analysis", "quant_strategy", "financial_report",
        "sentiment_analysis", "financial_qa", "risk_assessment",
        "general",
    ]

    parser = argparse.ArgumentParser(description="Fin-Instruct 批量推理")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入文件路径 (.json 或 .csv，需含 'question' 字段)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认: saves/batch_results/）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型路径",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter 路径",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="general",
        choices=task_choices,
        help="任务类型（决定 system prompt）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度系数",
    )
    args = parser.parse_args()

    run_batch_inference(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
