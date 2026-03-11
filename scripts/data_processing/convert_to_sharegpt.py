#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换模块

将各种来源的数据统一转换为 LLaMA-Factory 要求的 ShareGPT 格式:

ShareGPT 格式示例:
{
    "conversations": [
        {"from": "human", "value": "用户输入"},
        {"from": "gpt", "value": "模型回答"}
    ],
    "system": "系统提示词"
}

支持的输入格式:
  - Alpaca 格式（instruction/input/output）
  - DISC-FinLLM 格式（各种子集）
  - FinGPT 格式（input/output 或 instruction/output）
  - 自定义合成数据（已经是 ShareGPT 格式）

输出: data/sft/ 目录下的 JSON 文件
"""

import json
import logging
from pathlib import Path
from typing import Optional

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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SFT_DIR = PROJECT_ROOT / "data" / "sft"
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "system_prompts.json"


def load_system_prompts() -> dict:
    """加载 system prompt 模板。"""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_sharegpt_entry(
    user_text: str,
    assistant_text: str,
    system_prompt: str = "",
    task_type: str = "",
) -> Optional[dict]:
    """
    构造单条 ShareGPT 格式数据。

    参数:
        user_text:      用户输入文本
        assistant_text: 助手回答文本
        system_prompt:  系统提示词
        task_type:      任务类型标签

    返回:
        ShareGPT 格式字典，输入无效时返回 None
    """
    if not user_text or not assistant_text:
        return None
    if not user_text.strip() or not assistant_text.strip():
        return None

    entry = {
        "conversations": [
            {"from": "human", "value": user_text.strip()},
            {"from": "gpt", "value": assistant_text.strip()},
        ],
        "system": system_prompt,
    }
    if task_type:
        entry["task_type"] = task_type
    return entry


def convert_alpaca_format(
    data: list,
    system_prompt: str = "",
    task_type: str = "",
) -> list:
    """
    将 Alpaca 格式数据转换为 ShareGPT 格式。

    Alpaca 格式: {"instruction": ..., "input": ..., "output": ...}

    参数:
        data:          Alpaca 格式数据列表
        system_prompt: 系统提示词
        task_type:     任务类型

    返回:
        ShareGPT 格式数据列表
    """
    results = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        # 如果有 input，拼接到 instruction 后面
        if input_text:
            user_text = f"{instruction}\n\n{input_text}"
        else:
            user_text = instruction

        entry = _make_sharegpt_entry(
            user_text=user_text,
            assistant_text=output_text,
            system_prompt=system_prompt,
            task_type=task_type,
        )
        if entry:
            results.append(entry)

    return results


def convert_fingpt_format(
    data: list,
    system_prompt: str = "",
    task_type: str = "",
) -> list:
    """
    将 FinGPT 格式数据转换为 ShareGPT 格式。

    FinGPT 格式通常为: {"input": ..., "output": ...}
    或: {"instruction": ..., "input": ..., "output": ...}

    参数:
        data:          FinGPT 格式数据列表
        system_prompt: 系统提示词
        task_type:     任务类型

    返回:
        ShareGPT 格式数据列表
    """
    results = []
    for item in data:
        # 兼容多种字段名
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        if instruction and input_text:
            user_text = f"{instruction}\n\n{input_text}"
        elif input_text:
            user_text = input_text
        elif instruction:
            user_text = instruction
        else:
            continue

        entry = _make_sharegpt_entry(
            user_text=user_text,
            assistant_text=output_text,
            system_prompt=system_prompt,
            task_type=task_type,
        )
        if entry:
            results.append(entry)

    return results


def convert_disc_finllm_format(
    data: list,
    system_prompt: str = "",
) -> list:
    """
    将 DISC-FinLLM 格式数据转换为 ShareGPT 格式。

    DISC-FinLLM 包含多种子数据集，格式可能为:
      - {"input": ..., "output": ...}
      - {"instruction": ..., "input": ..., "output": ...}
      - {"conversations": [...]} (已是对话格式)

    参数:
        data:          DISC-FinLLM 格式数据列表
        system_prompt: 系统提示词

    返回:
        ShareGPT 格式数据列表
    """
    results = []
    for item in data:
        # 如果已经是对话格式
        if "conversations" in item:
            convs = item["conversations"]
            if isinstance(convs, list) and len(convs) >= 2:
                entry = {
                    "conversations": [],
                    "system": system_prompt or item.get("system", ""),
                }
                for turn in convs:
                    role = turn.get("from", turn.get("role", ""))
                    content = turn.get("value", turn.get("content", ""))
                    if role in ("human", "user"):
                        entry["conversations"].append(
                            {"from": "human", "value": content.strip()}
                        )
                    elif role in ("gpt", "assistant"):
                        entry["conversations"].append(
                            {"from": "gpt", "value": content.strip()}
                        )
                if len(entry["conversations"]) >= 2:
                    results.append(entry)
        else:
            # Alpaca-like 格式
            converted = convert_alpaca_format(
                [item], system_prompt=system_prompt, task_type="disc_finllm"
            )
            results.extend(converted)

    return results



def _convert_messages_format(
    data: list,
    system_prompt: str = "",
    task_type: str = "",
) -> list:
    """
    将 OpenAI messages 格式数据转换为 ShareGPT 格式。

    Messages 格式: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    参数:
        data:          messages 格式数据列表
        system_prompt: 系统提示词
        task_type:     任务类型

    返回:
        ShareGPT 格式数据列表
    """
    results = []
    for item in data:
        messages = item.get("messages", [])
        if not messages or len(messages) < 2:
            continue

        conversations = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content or not content.strip():
                continue
            if role == "user":
                conversations.append({"from": "human", "value": content.strip()})
            elif role == "assistant":
                conversations.append({"from": "gpt", "value": content.strip()})
            elif role == "system":
                system_prompt = content.strip()

        if len(conversations) >= 2:
            entry = {
                "conversations": conversations,
                "system": system_prompt,
            }
            if task_type:
                entry["task_type"] = task_type
            results.append(entry)

    return results


def process_open_dataset(
    dataset_name: str,
    task_type: str,
    system_prompt: str,
) -> list:
    """
    处理单个公开数据集，自动识别格式并转换。

    参数:
        dataset_name:  数据集名称（对应 data/raw/ 下的子目录）
        task_type:     任务类型
        system_prompt: 系统提示词

    返回:
        转换后的 ShareGPT 格式数据列表
    """
    dataset_dir = RAW_DATA_DIR / dataset_name
    if not dataset_dir.exists():
        logger.warning(f"数据集目录不存在: {dataset_dir}")
        return []

    all_data = []

    # 跳过的文件名（元数据 / 标记文件）
    SKIP_NAMES = {"meta.json"}

    # 加载所有 JSON 文件
    for json_file in sorted(dataset_dir.rglob("*.json")):
        if json_file.name.startswith("_") or json_file.name in SKIP_NAMES:
            continue  # 跳过标记文件和元数据
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            all_data.extend(data)
        except json.JSONDecodeError:
            # 尝试 JSONL 格式（每行一个 JSON 对象）
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    jsonl_data = [json.loads(line) for line in f if line.strip()]
                if jsonl_data:
                    all_data.extend(jsonl_data)
                    logger.info(f"  {json_file.name}: JSONL 格式，加载 {len(jsonl_data)} 条")
            except Exception as e2:
                logger.warning(f"读取 {json_file} 失败: {e2}")
        except Exception as e:
            logger.warning(f"读取 {json_file} 失败: {e}")

    if not all_data:
        return []

    logger.info(f"  {dataset_name}: 加载 {len(all_data)} 条原始数据")

    # 自动检测格式并转换
    sample = all_data[0]

    if "conversations" in sample:
        # 已是对话格式（ShareGPT / DISC-FinLLM）
        results = convert_disc_finllm_format(all_data, system_prompt)
    elif "messages" in sample:
        # OpenAI messages 格式（fingpt_finred / fingpt_fiqa_qa 等）
        results = _convert_messages_format(all_data, system_prompt, task_type)
    elif "instruction" in sample:
        # Alpaca 格式
        results = convert_alpaca_format(all_data, system_prompt, task_type)
    elif "input" in sample and "output" in sample:
        # FinGPT 简洁格式
        results = convert_fingpt_format(all_data, system_prompt, task_type)
    else:
        logger.warning(f"  {dataset_name}: 未识别的数据格式，字段: {list(sample.keys())}")
        results = []

    logger.info(f"  {dataset_name}: 转换得到 {len(results)} 条 ShareGPT 数据")
    return results


def run_conversion() -> None:
    """
    执行完整的格式转换流程。

    将所有数据源转换为 ShareGPT 格式，按任务类型分别保存。
    """
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_system_prompts()

    # ============================================================
    # 数据集 -> 任务类型的映射关系
    # ============================================================
    DATASET_TASK_MAP = {
        # FinGPT 系列
        "fingpt_sentiment": ("sentiment_analysis", "金融新闻情感分析"),
        "fingpt_fiqa_qa": ("financial_qa", "金融问答/知识"),
        "fingpt_finred": ("financial_qa", "金融关系抽取"),
        "fingpt_headline": ("sentiment_analysis", "金融标题分析"),
        "fingpt_fineval": ("financial_qa", "金融评测"),
        # DISC-FinLLM
        "disc_finllm": ("general", "DISC-FinLLM综合"),
    }

    # 按任务类型收集数据
    task_data = {}

    logger.info("=" * 60)
    logger.info("开始数据格式转换")
    logger.info("=" * 60)

    # 1. 处理公开数据集
    logger.info("\n--- 处理公开数据集 ---")
    for dataset_name, (task_type, desc) in DATASET_TASK_MAP.items():
        prompt_info = prompts.get(task_type, prompts["general"])
        system_prompt = prompt_info["system_prompt"]

        results = process_open_dataset(
            dataset_name=dataset_name,
            task_type=task_type,
            system_prompt=system_prompt,
        )

        if results:
            if task_type not in task_data:
                task_data[task_type] = []
            task_data[task_type].extend(results)

    # 2. 处理合成数据
    logger.info("\n--- 处理合成数据 ---")
    synth_dir = PROCESSED_DIR / "synthesized"
    if synth_dir.exists():
        for json_file in sorted(synth_dir.glob("*.json")):
            task_type = json_file.stem
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if task_type not in task_data:
                    task_data[task_type] = []
                # 合成数据已经是 ShareGPT 格式
                task_data[task_type].extend(data)
                logger.info(f"  合成数据 {task_type}: {len(data)} 条")
            except Exception as e:
                logger.warning(f"  读取合成数据失败 {json_file}: {e}")

    # 3. 按任务类型保存
    logger.info("\n--- 保存分任务数据 ---")
    total_count = 0
    for task_type, data in task_data.items():
        output_file = SFT_DIR / f"fin_{task_type}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"  {task_type}: {len(data)} 条 -> {output_file.name}")
        total_count += len(data)

    logger.info(f"\n格式转换完成，共 {total_count} 条数据")
    logger.info(f"输出目录: {SFT_DIR}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    run_conversion()
