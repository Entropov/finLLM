#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速推理测试 - 加载 LoRA 微调模型并测试多种金融任务"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ADAPTER_PATH = PROJECT_ROOT / "saves" / "qwen2.5-7b" / "lora" / "sft"
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "system_prompts.json"

TEST_CASES = [
    {
        "task": "sentiment_analysis",
        "question": "请分析以下新闻的情感倾向：'央行宣布降准0.5个百分点，释放长期资金约1万亿元，支持实体经济发展。'"
    },
    {
        "task": "financial_qa",
        "question": "什么是市盈率（PE）？如何用市盈率来判断一只股票是否被高估？"
    },
    {
        "task": "stock_analysis",
        "question": "一只股票连续三天放量上涨，MACD金叉，KDJ超买区，请分析后续可能的走势。"
    },
    {
        "task": "general",
        "question": "请简要比较A股和港股市场的主要区别。"
    },
]


def main():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    prompts = {k: v["system_prompt"] for k, v in prompts_data.items() if "system_prompt" in v}

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"加载基座模型 (bf16): {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )

    print(f"加载 LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()
    print("模型加载完成！\n")

    for i, case in enumerate(TEST_CASES, 1):
        task = case["task"]
        question = case["question"]
        system_prompt = prompts.get(task, prompts.get("general", ""))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        print(f"{'='*60}")
        print(f"测试 {i}/{len(TEST_CASES)} | 任务: {task}")
        print(f"问题: {question}")
        print(f"{'-'*60}")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        generated = outputs[0][input_len:]
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        out_tokens = len(generated)

        print(f"回答:\n{answer}")
        print(f"\n[输入 {input_len} tokens | 生成 {out_tokens} tokens | 耗时 {elapsed:.1f}s | {out_tokens/elapsed:.1f} tok/s]")
        print()

    print("=" * 60)
    print("推理测试完成！")


if __name__ == "__main__":
    main()
