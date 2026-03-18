#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 自动标注脚本 (Auto-Labeler)
================================
利用已训练好的 SFT 模型为同一问题生成多个候选回答，
再通过质量评分或外部 LLM Judge 自动选出 chosen / rejected，
从而批量生成高质量偏好数据对，无需人工逐条标注。

核心流程：
  1. 读取原始问题（来自 SFT 数据或自定义问题列表）
  2. 用 SFT 模型以不同温度/采样参数生成 N 个候选回答
  3. 对候选回答打分（启发式 or LLM Judge）
  4. 选最高分为 chosen，最低分为 rejected
  5. 过滤太相似的对，输出 DPO 格式数据

用法:
  # 使用 SFT 模型生成候选，启发式打分
  python scripts/rlhf/auto_label.py \\
      --model-path saves/qwen2.5-7b/merged \\
      --input data/sft/fin_instruct_train.json \\
      --output data/rlhf/auto_labeled.json \\
      --num-candidates 4 \\
      --num-samples 1000

  # 配合 LLM Judge 精选
  export SYNTH_API_KEY="..."
  python scripts/rlhf/auto_label.py \\
      --model-path saves/qwen2.5-7b/merged \\
      --input data/sft/fin_instruct_train.json \\
      --output data/rlhf/auto_labeled.json \\
      --judge-mode llm --num-candidates 3 --num-samples 500
"""

import argparse
import json
import os
import re
import random
import logging
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 质量评分（复用 synthesize_preference_data 的逻辑）
# ──────────────────────────────────────────────────────────────
def quick_quality_score(text: str) -> float:
    """启发式质量评分，0~10。"""
    score = 5.0
    length = len(text)
    if length > 600:
        score += 1.5
    elif length > 300:
        score += 0.8
    elif length < 80:
        score -= 2.5

    pro_patterns = [
        r"\d+\.?\d*%",
        r"(?:ROE|ROA|PE|PB|MACD|RSI|KDJ|VaR|夏普)",
        r"(?:支撑位|压力位|均线|趋势线)",
        r"(?:营收|净利润|毛利率|现金流)",
    ]
    for pat in pro_patterns:
        if re.search(pat, text):
            score += 0.3

    if re.search(r"^#{1,3}\s", text, re.MULTILINE):
        score += 0.8
    if re.search(r"^\s*[-*•]\s", text, re.MULTILINE):
        score += 0.5
    if re.search(r"\|.*\|.*\|", text):
        score += 0.7

    lazy = ["无法回答", "不清楚", "建议咨询专业人士", "这是个复杂问题"]
    for phrase in lazy:
        if phrase in text:
            score -= 1.5

    return round(max(0, min(10, score)), 2)


def text_similarity(a: str, b: str) -> float:
    """粗略的文本相似度（基于词重叠）。"""
    words_a = set(a[:300])
    words_b = set(b[:300])
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / max(len(words_a), len(words_b))


# ──────────────────────────────────────────────────────────────
# 多候选生成
# ──────────────────────────────────────────────────────────────
def load_model(model_path: str):
    """加载已合并的模型（merged 或完整路径）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_candidates(
    model,
    tokenizer,
    question: str,
    system: str,
    num_candidates: int = 4,
    max_new_tokens: int = 512,
    temperatures: Optional[list[float]] = None,
) -> list[str]:
    """
    为一个问题生成 N 个候选回答。
    使用不同温度以增加多样性。
    """
    import torch

    if temperatures is None:
        # 默认温度梯度：从保守到探索
        temp_pool = [0.1, 0.5, 0.8, 1.0, 1.2]
        temperatures = temp_pool[:num_candidates]
        # 不够就循环补充
        while len(temperatures) < num_candidates:
            temperatures.append(random.uniform(0.3, 1.0))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    candidates = []
    for temp in temperatures[:num_candidates]:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=(temp > 0.01),
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if resp:
            candidates.append(resp)

    return candidates


# ──────────────────────────────────────────────────────────────
# LLM Judge 选取 best/worst
# ──────────────────────────────────────────────────────────────
RANK_SYSTEM = """你是金融AI回答质量评估专家。请对以下多个回答按质量从高到低排序，并返回排名。
评估围绕：专业准确性、分析深度、实用性、结构清晰度。
输出格式（严格遵守，只输出这一行）：
RANK: 2,1,3  （数字为回答编号，从最好到最差）"""


def llm_rank_candidates(
    question: str,
    candidates: list[str],
    api_key: str,
    api_base: str,
    model_name: str,
) -> Optional[list[int]]:
    """用 LLM 对候选回答排序，返回索引列表（从好到差）。"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)

        candidates_text = "\n\n".join(
            f"回答{i+1}：\n{c[:500]}" for i, c in enumerate(candidates)
        )
        user_msg = f"问题：{question[:300]}\n\n{candidates_text}"

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": RANK_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        text = resp.choices[0].message.content or ""
        match = re.search(r"RANK:\s*([\d,\s]+)", text)
        if not match:
            return None
        order = [int(x.strip()) - 1 for x in match.group(1).split(",") if x.strip().isdigit()]
        # 验证索引合法
        if len(order) == len(candidates) and all(0 <= idx < len(candidates) for idx in order):
            return order
        return None
    except Exception as e:
        logger.warning(f"LLM rank 失败: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 主标注逻辑
# ──────────────────────────────────────────────────────────────
def auto_label_sample(
    model,
    tokenizer,
    item: dict,
    num_candidates: int,
    judge_mode: str,
    api_key: str,
    api_base: str,
    llm_model: str,
    min_score_gap: float = 1.0,
    min_length: int = 60,
    max_new_tokens: int = 512,
) -> Optional[dict]:
    """对单个样本执行自动标注，返回 DPO 格式数据或 None（无法标注时）。"""
    convs = item.get("conversations", [])
    human_turn = next((c for c in convs if c.get("from") == "human"), None)
    if not human_turn:
        return None

    question = human_turn["value"]
    system = item.get("system", "")

    # 生成候选
    candidates = generate_candidates(
        model, tokenizer, question, system, num_candidates, max_new_tokens
    )

    # 过滤太短的候选
    candidates = [c for c in candidates if len(c) >= min_length]
    if len(candidates) < 2:
        return None

    # 打分 / 排序
    if judge_mode == "llm" and api_key:
        order = llm_rank_candidates(question, candidates, api_key, api_base, llm_model)
        if order is None:
            # 回退到启发式
            scores = [quick_quality_score(c) for c in candidates]
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
        scores = [quick_quality_score(c) for c in candidates]
    else:
        scores = [quick_quality_score(c) for c in candidates]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])

    best_idx = order[0]
    worst_idx = order[-1]

    chosen = candidates[best_idx]
    rejected = candidates[worst_idx]

    # 质量差距过小，跳过（避免噪音对）
    score_gap = abs(scores[best_idx] - scores[worst_idx])
    if score_gap < min_score_gap:
        return None

    # 文本相似度过高，跳过
    if text_similarity(chosen, rejected) > 0.85:
        return None

    return {
        "conversations": [{"from": "human", "value": question}],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
        "system": system,
        "task_type": item.get("task_type", ""),
        "_source": "auto_label",
        "_score_gap": round(score_gap, 2),
        "_chosen_score": round(scores[best_idx], 2),
        "_rejected_score": round(scores[worst_idx], 2),
    }


# ──────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI 自动标注偏好数据")
    parser.add_argument(
        "--model-path",
        default="saves/qwen2.5-7b/merged",
        help="已合并的 SFT 模型路径（推荐使用 merged 以获得最佳质量）",
    )
    parser.add_argument("--input", default="data/sft/fin_instruct_train.json")
    parser.add_argument("--output", default="data/rlhf/auto_labeled.json")
    parser.add_argument(
        "--num-candidates", type=int, default=4,
        help="每个问题生成的候选回答数（建议 3~5）",
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument(
        "--judge-mode",
        choices=["quick", "llm"],
        default="quick",
        help="打分模式: quick(启发式) | llm(LLM排序)",
    )
    parser.add_argument("--min-score-gap", type=float, default=1.0,
                        help="chosen/rejected 分数差阈值（低于此值丢弃）")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--merge-existing", default=None,
                        help="与已有偏好数据合并（路径）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 读取输入
    with open(args.input, encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"读取 {len(raw_data)} 条原始数据")

    if len(raw_data) > args.num_samples:
        raw_data = random.sample(raw_data, args.num_samples)
        logger.info(f"采样 {len(raw_data)} 条")

    api_key = os.environ.get("SYNTH_API_KEY", "")
    api_base = os.environ.get("SYNTH_API_BASE", "https://api.openai.com/v1")
    llm_model = os.environ.get("SYNTH_API_MODEL", "gpt-4o-mini")

    if args.judge_mode == "llm" and not api_key:
        logger.warning("未设置 SYNTH_API_KEY，自动切换 judge-mode 到 quick")
        args.judge_mode = "quick"

    # 加载模型
    model, tokenizer = load_model(args.model_path)

    # 标注
    results = []
    failed = 0
    for i, item in enumerate(raw_data):
        labeled = auto_label_sample(
            model=model,
            tokenizer=tokenizer,
            item=item,
            num_candidates=args.num_candidates,
            judge_mode=args.judge_mode,
            api_key=api_key,
            api_base=api_base,
            llm_model=llm_model,
            min_score_gap=args.min_score_gap,
            max_new_tokens=args.max_new_tokens,
        )
        if labeled:
            results.append(labeled)
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            logger.info(f"进度: {i+1}/{len(raw_data)}, 成功: {len(results)}, 跳过: {failed}")

    logger.info(f"自动标注完成: {len(results)} 对，跳过 {failed} 条")

    # 合并已有数据
    if args.merge_existing and Path(args.merge_existing).exists():
        with open(args.merge_existing, encoding="utf-8") as f:
            existing = json.load(f)
        results = existing + results
        logger.info(f"合并后共 {len(results)} 条")

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 统计
    if results:
        avg_gap = sum(r.get("_score_gap", 0) for r in results) / len(results)
        avg_chosen = sum(r.get("_chosen_score", 0) for r in results) / len(results)
        avg_rejected = sum(r.get("_rejected_score", 0) for r in results) / len(results)
        logger.info(f"\n✅ 输出: {out_path}")
        logger.info(f"   总量: {len(results)} 对")
        logger.info(f"   平均分差: {avg_gap:.2f}")
        logger.info(f"   chosen 均分: {avg_chosen:.2f}/10")
        logger.info(f"   rejected 均分: {avg_rejected:.2f}/10")


if __name__ == "__main__":
    main()
