#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集合并模块

将多个来源的 ShareGPT 格式数据合并为最终的训练集和验证集:
  1. 加载所有分任务数据文件
  2. 按任务类型进行均衡采样
  3. 随机打乱
  4. 按 95:5 比例划分训练集/验证集
  5. 输出 fin_instruct_train.json / fin_instruct_eval.json

输出: data/sft/fin_instruct_train.json, data/sft/fin_instruct_eval.json
"""

import json
import random
import logging
from pathlib import Path
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
SFT_DIR = PROJECT_ROOT / "data" / "sft"

# ============================================================
# 合并参数
# ============================================================
TRAIN_RATIO = 0.95           # 训练集比例
RANDOM_SEED = 42             # 随机种子
TRAIN_OUTPUT = "fin_instruct_train.json"
EVAL_OUTPUT = "fin_instruct_eval.json"

# 各任务数据的最大采样量（防止单一任务数据过多导致不均衡）
MAX_SAMPLES_PER_TASK = {
    "stock_analysis": 20000,
    "quant_strategy": 15000,
    "financial_report": 20000,
    "sentiment_analysis": 15000,
    "financial_qa": 20000,
    "risk_assessment": 10000,
    "general": 15000,
    "disc_finllm": 30000,
}
DEFAULT_MAX_SAMPLES = 20000


def load_sft_files() -> dict:
    """
    加载 data/sft/ 目录下所有分任务数据文件。

    返回:
        {任务名称: 数据列表} 的字典
    """
    task_data = {}

    if not SFT_DIR.exists():
        logger.error(f"SFT 数据目录不存在: {SFT_DIR}")
        return task_data

    # 优先加载过滤后的文件（*_filtered.json），否则加载原始文件（fin_*.json）
    for json_file in sorted(SFT_DIR.glob("*.json")):
        # 跳过最终输出文件，避免循环
        if json_file.name in (TRAIN_OUTPUT, EVAL_OUTPUT):
            continue
        # 跳过未过滤版本（如果有过滤版本的话）
        stem = json_file.stem
        if not stem.endswith("_filtered"):
            filtered_version = SFT_DIR / f"{stem}_filtered.json"
            if filtered_version.exists():
                continue  # 优先使用 filtered 版本

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue

            # 提取任务名称
            task_name = stem.replace("fin_", "").replace("_filtered", "")
            task_data[task_name] = data
            logger.info(f"  加载 {json_file.name}: {len(data)} 条")

        except Exception as e:
            logger.warning(f"  读取 {json_file.name} 失败: {e}")

    return task_data


def sample_and_merge(task_data: dict) -> list:
    """
    对各任务数据进行采样并合并。

    参数:
        task_data: {任务名称: 数据列表} 字典

    返回:
        合并后的数据列表
    """
    random.seed(RANDOM_SEED)
    merged = []

    logger.info("\n--- 任务采样 ---")
    for task_name, data in task_data.items():
        max_samples = MAX_SAMPLES_PER_TASK.get(task_name, DEFAULT_MAX_SAMPLES)

        if len(data) > max_samples:
            sampled = random.sample(data, max_samples)
            logger.info(
                f"  {task_name}: 采样 {max_samples}/{len(data)} 条"
            )
        else:
            sampled = data
            logger.info(f"  {task_name}: 全部保留 {len(data)} 条")

        merged.extend(sampled)

    # 随机打乱
    random.shuffle(merged)
    logger.info(f"\n合并后总计: {len(merged)} 条")

    return merged


def split_train_eval(data: list) -> tuple:
    """
    将数据划分为训练集和验证集。

    参数:
        data: 合并后的数据列表

    返回:
        (训练集列表, 验证集列表)
    """
    random.seed(RANDOM_SEED)

    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    logger.info(f"训练集: {len(train_data)} 条 ({TRAIN_RATIO:.0%})")
    logger.info(f"验证集: {len(eval_data)} 条 ({1-TRAIN_RATIO:.0%})")

    return train_data, eval_data


def print_statistics(data: list, label: str) -> None:
    """
    打印数据集统计信息。

    参数:
        data:  数据列表
        label: 数据集标签（如 "训练集" / "验证集"）
    """
    # 任务分布
    task_counter = Counter()
    length_stats = []

    for entry in data:
        task_type = entry.get("task_type", "unknown")
        task_counter[task_type] += 1

        # 计算回答长度
        convs = entry.get("conversations", [])
        for c in convs:
            if c.get("from") == "gpt":
                length_stats.append(len(c.get("value", "")))

    logger.info(f"\n{label} 统计:")
    logger.info(f"  总条数: {len(data)}")

    if task_counter:
        logger.info("  任务分布:")
        for task, count in task_counter.most_common():
            pct = count / len(data) * 100
            logger.info(f"    {task}: {count} ({pct:.1f}%)")

    if length_stats:
        import numpy as np

        arr = np.array(length_stats)
        logger.info(f"  回答长度: 均值={arr.mean():.0f}, "
                     f"中位数={np.median(arr):.0f}, "
                     f"最小={arr.min()}, 最大={arr.max()}")


def run_merge() -> None:
    """执行完整的数据合并流程。"""
    logger.info("=" * 60)
    logger.info("Fin-Instruct 数据集合并")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("\n--- 加载数据文件 ---")
    task_data = load_sft_files()

    if not task_data:
        logger.error("未找到任何可用数据文件！")
        logger.error("请先运行: python scripts/data_processing/convert_to_sharegpt.py")
        return

    # 2. 采样并合并
    merged = sample_and_merge(task_data)

    if not merged:
        logger.error("合并后数据为空！")
        return

    # 3. 划分训练集/验证集
    train_data, eval_data = split_train_eval(merged)

    # 4. 保存
    train_file = SFT_DIR / TRAIN_OUTPUT
    eval_file = SFT_DIR / EVAL_OUTPUT

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\n训练集已保存: {train_file}")

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    logger.info(f"验证集已保存: {eval_file}")

    # 5. 打印统计
    print_statistics(train_data, "训练集")
    print_statistics(eval_data, "验证集")

    logger.info("\n" + "=" * 60)
    logger.info("数据合并完成！")
    logger.info(f"下一步: 使用以下命令开始训练:")
    logger.info(f"  llamafactory-cli train configs/qwen2.5_7b_qlora_sft.yaml")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    run_merge()
