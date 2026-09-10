#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好数据格式转换工具
====================
将各类格式的偏好标注数据统一转换为 LLaMA-Factory DPO 格式。

支持输入格式:
1. 已有的 SFT + 手动标注 (manual_annotation) — CSV 格式
2. FinGPT 偏好数据格式
3. Anthropic HH (Helpful/Harmless) 格式
4. 自定义简单格式（JSON: 含 prompt/chosen/rejected 字段）

输出格式 (LLaMA-Factory DPO):
{
    "conversations": [{"from": "human", "value": "<问题>"}],
    "chosen": {"from": "gpt", "value": "<优质回答>"},
    "rejected": {"from": "gpt", "value": "<劣质回答>"},
    "system": "<系统提示>"  # 可选
}

用法:
  python scripts/data_processing/convert_to_preference.py \\
      --input data/rlhf/raw_annotations.json \\
      --output data/rlhf/fin_preference_train.json \\
      --format simple

  # 合并多个来源
  python scripts/data_processing/convert_to_preference.py \\
      --input data/rlhf/source1.json data/rlhf/source2.json \\
      --output data/rlhf/fin_preference_train.json \\
      --format simple --merge-mode concat --shuffle
"""

import argparse
import json
import random
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 格式转换器
# ──────────────────────────────────────────────────────────────
def convert_simple(data: list[dict], system: str = "") -> list[dict]:
    """
    简单格式转换：输入含 prompt/chosen/rejected 字段。

    支持字段别名:
    - question / instruction / input → prompt
    - answer_chosen / better / response_chosen → chosen
    - answer_rejected / worse / response_rejected → rejected
    """
    results = []
    prompt_aliases = ["prompt", "question", "instruction", "input", "query"]
    chosen_aliases = ["chosen", "answer_chosen", "better", "response_chosen", "response_a"]
    rejected_aliases = ["rejected", "answer_rejected", "worse", "response_rejected", "response_b"]

    for item in data:
        prompt = next((item[k] for k in prompt_aliases if k in item), None)
        chosen = next((item[k] for k in chosen_aliases if k in item), None)
        rejected = next((item[k] for k in rejected_aliases if k in item), None)

        if not all([prompt, chosen, rejected]):
            continue

        result = {
            "conversations": [{"from": "human", "value": str(prompt)}],
            "chosen": {"from": "gpt", "value": str(chosen)},
            "rejected": {"from": "gpt", "value": str(rejected)},
        }
        sys_prompt = item.get("system", system)
        if sys_prompt:
            result["system"] = sys_prompt

        task_type = item.get("task_type", "")
        if task_type:
            result["task_type"] = task_type

        results.append(result)

    return results


def convert_sharegpt_preference(data: list[dict]) -> list[dict]:
    """
    从两条 ShareGPT 格式回答合并为偏好对。
    
    输入格式:
    {
      "conversations": [...],
      "chosen_response": "...",
      "rejected_response": "..."
    }
    """
    results = []
    for item in data:
        convs = item.get("conversations", [])
        human_turn = next((c for c in convs if c.get("from") == "human"), None)
        if not human_turn:
            continue

        chosen_resp = item.get("chosen_response") or item.get("chosen")
        rejected_resp = item.get("rejected_response") or item.get("rejected")

        if not chosen_resp or not rejected_resp:
            continue

        # 解包字典格式
        if isinstance(chosen_resp, dict):
            chosen_resp = chosen_resp.get("value", str(chosen_resp))
        if isinstance(rejected_resp, dict):
            rejected_resp = rejected_resp.get("value", str(rejected_resp))

        result = {
            "conversations": [{"from": "human", "value": human_turn["value"]}],
            "chosen": {"from": "gpt", "value": str(chosen_resp)},
            "rejected": {"from": "gpt", "value": str(rejected_resp)},
        }
        if item.get("system"):
            result["system"] = item["system"]
        results.append(result)

    return results


def convert_hh_format(data: list[dict]) -> list[dict]:
    """
    Anthropic HH (Helpful/Harmless) 格式转换。
    
    输入格式:
    {
      "chosen": "Human: ...\n\nAssistant: ...",
      "rejected": "Human: ...\n\nAssistant: ..."
    }
    """
    results = []
    for item in data:
        chosen_text = item.get("chosen", "")
        rejected_text = item.get("rejected", "")

        if not chosen_text or not rejected_text:
            continue

        def extract_last_turn(text: str) -> tuple[str, str]:
            """提取最后一轮 Human/Assistant 对话。"""
            turns = text.split("\n\nHuman:")
            if len(turns) < 2:
                return "", text
            last = "Human:" + turns[-1]
            parts = last.split("\n\nAssistant:")
            if len(parts) < 2:
                return last, ""
            return parts[0].replace("Human:", "").strip(), parts[1].strip()

        question, chosen_resp = extract_last_turn(chosen_text)
        _, rejected_resp = extract_last_turn(rejected_text)

        if not question or not chosen_resp or not rejected_resp:
            continue

        results.append({
            "conversations": [{"from": "human", "value": question}],
            "chosen": {"from": "gpt", "value": chosen_resp},
            "rejected": {"from": "gpt", "value": rejected_resp},
        })

    return results


def convert_csv_annotation(path: str) -> list[dict]:
    """
    从 CSV 人工标注文件转换。
    
    期望列: question, response_a, response_b, winner (A/B/tie), system
    """
    import csv
    results = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            winner = row.get("winner", "").strip().upper()
            if winner == "TIE":
                continue  # 跳过平局

            question = row.get("question", "").strip()
            resp_a = row.get("response_a", "").strip()
            resp_b = row.get("response_b", "").strip()

            if not all([question, resp_a, resp_b]):
                continue

            if winner == "A":
                chosen, rejected = resp_a, resp_b
            elif winner == "B":
                chosen, rejected = resp_b, resp_a
            else:
                continue

            result = {
                "conversations": [{"from": "human", "value": question}],
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": rejected},
            }
            if row.get("system"):
                result["system"] = row["system"]
            results.append(result)

    return results


# ──────────────────────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────────────────────
def validate_sample(item: dict) -> Optional[str]:
    """验证单个偏好样本，返回错误信息或 None（合法）。"""
    if "conversations" not in item or not isinstance(item["conversations"], list):
        return "missing conversations"
    if len(item["conversations"]) == 0:
        return "empty conversations"
    if item["conversations"][0].get("from") != "human":
        return "first turn must be human"

    for key in ("chosen", "rejected"):
        if key not in item:
            return f"missing {key}"
        v = item[key]
        if not isinstance(v, dict) or "value" not in v:
            return f"{key} must be dict with 'value' field"
        if not v["value"] or len(v["value"].strip()) < 5:
            return f"{key} value too short"

    chosen_val = item["chosen"]["value"].strip()
    rejected_val = item["rejected"]["value"].strip()
    if chosen_val == rejected_val:
        return "chosen == rejected"

    return None


def quality_filter(data: list[dict], min_chosen_len: int = 50) -> list[dict]:
    """过滤低质量样本。"""
    valid = []
    invalid_count = 0

    for item in data:
        error = validate_sample(item)
        if error:
            invalid_count += 1
            continue

        chosen_len = len(item["chosen"]["value"])
        if chosen_len < min_chosen_len:
            invalid_count += 1
            continue

        valid.append(item)

    if invalid_count:
        logger.info(f"质量过滤: 移除 {invalid_count} 条不合格样本，保留 {len(valid)} 条")
    return valid


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="将各类偏好数据转换为 LLaMA-Factory DPO 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="输入文件路径（支持多个，空格分隔）",
    )
    parser.add_argument(
        "--output",
        default="data/rlhf/fin_preference_train.json",
        help="输出路径",
    )
    parser.add_argument(
        "--format",
        choices=["simple", "sharegpt_pref", "hh", "csv_annotation", "auto"],
        default="auto",
        help="输入格式（auto 自动检测）",
    )
    parser.add_argument(
        "--merge-mode",
        choices=["concat", "interleave"],
        default="concat",
        help="多文件合并方式",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="随机打乱输出数据",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大输出样本数",
    )
    parser.add_argument(
        "--system",
        default="",
        help="默认 system prompt（当数据无 system 字段时使用）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    all_results = []

    for input_path in args.input:
        path = Path(input_path)
        logger.info(f"处理文件: {path}")

        if not path.exists():
            logger.error(f"文件不存在: {path}")
            continue

        fmt = args.format
        if fmt == "auto":
            if path.suffix == ".csv":
                fmt = "csv_annotation"
            else:
                # JSON 格式：读入后自动检测
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                if raw and isinstance(raw[0], dict):
                    keys = set(raw[0].keys())
                    if "chosen" in keys and "rejected" in keys and "conversations" in keys:
                        fmt = "sharegpt_pref"
                    elif any(k in keys for k in ["question", "prompt", "instruction"]):
                        fmt = "simple"
                    elif "chosen" in keys and "rejected" in keys:
                        # Check if it's HH format (string values)
                        if isinstance(raw[0].get("chosen"), str) and "\n\nAssistant:" in raw[0].get("chosen", ""):
                            fmt = "hh"
                        else:
                            fmt = "simple"
                    else:
                        fmt = "simple"
                logger.info(f"  自动检测格式: {fmt}")

        if fmt == "csv_annotation":
            converted = convert_csv_annotation(str(path))
        else:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if fmt == "simple":
                converted = convert_simple(raw, args.system)
            elif fmt == "sharegpt_pref":
                converted = convert_sharegpt_preference(raw)
            elif fmt == "hh":
                converted = convert_hh_format(raw)
            else:
                converted = convert_simple(raw, args.system)

        logger.info(f"  转换结果: {len(converted)} 条")
        all_results.extend(converted)

    # 质量过滤
    all_results = quality_filter(all_results)

    # 合并后去重
    seen = set()
    deduped = []
    for item in all_results:
        key = item["conversations"][0]["value"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    if len(deduped) < len(all_results):
        logger.info(f"去重: {len(all_results)} → {len(deduped)} 条")
    all_results = deduped

    if args.shuffle:
        random.shuffle(all_results)

    if args.max_samples and len(all_results) > args.max_samples:
        all_results = all_results[:args.max_samples]
        logger.info(f"截断至 {args.max_samples} 条")

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info(f"✅ 格式转换完成！")
    logger.info(f"   输出: {out_path}")
    logger.info(f"   样本数: {len(all_results)}")

    if all_results:
        sample = all_results[0]
        logger.info(f"\n样本示例:")
        logger.info(f"  Human: {sample['conversations'][0]['value'][:60]}...")
        logger.info(f"  Chosen: {sample['chosen']['value'][:60]}...")
        logger.info(f"  Rejected: {sample['rejected']['value'][:60]}...")


if __name__ == "__main__":
    main()
