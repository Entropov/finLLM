#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量过滤模块

对 ShareGPT 格式的训练数据执行多维度质量检查:
  1. 基础过滤: 回答长度、格式完整性
  2. 重复检测: 基于 n-gram 的文本去重
  3. 内容质量: 回答质量打分（规则+启发式）
  4. 任务分布均衡: 确保各任务比例合理

输出: 过滤后的高质量数据 + 统计报告
"""

import json
import logging
import hashlib
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
SFT_DIR = PROJECT_ROOT / "data" / "sft"

# ============================================================
# 质量过滤参数
# ============================================================
# 最小回答长度（字符数）
MIN_RESPONSE_LENGTH = 30
# 最大回答长度（字符数）
MAX_RESPONSE_LENGTH = 8000
# 最小问题长度（字符数）
MIN_QUESTION_LENGTH = 5
# n-gram 去重的 n 值
DEDUP_NGRAM_SIZE = 5
# 相似度阈值（Jaccard 相似度高于此值视为重复）
DEDUP_SIMILARITY_THRESHOLD = 0.8
# 低质量标记词（回答中出现这些通常表示生成失败）
LOW_QUALITY_MARKERS = [
    "作为AI", "作为一个AI", "作为人工智能",
    "我无法", "我不能", "抱歉，我",
    "很抱歉", "Sorry", "I cannot",
    "对不起，我无法",
]
# 各任务目标比例（用于均衡采样）
TASK_TARGET_RATIO = {
    "stock_analysis": 0.20,
    "quant_strategy": 0.15,
    "financial_report": 0.20,
    "sentiment_analysis": 0.15,
    "financial_qa": 0.20,
    "risk_assessment": 0.10,
}


def _get_ngrams(text: str, n: int = DEDUP_NGRAM_SIZE) -> set:
    """
    提取文本的 character n-gram 集合。

    参数:
        text: 输入文本
        n:    n-gram 大小

    返回:
        n-gram 集合
    """
    text = text.strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """计算两个集合的 Jaccard 相似度。"""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _content_hash(text: str) -> str:
    """计算文本的 MD5 哈希（用于精确去重）。"""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def check_basic_quality(entry: dict) -> tuple:
    """
    检查单条数据的基础质量。

    参数:
        entry: ShareGPT 格式的数据条目

    返回:
        (是否通过, 失败原因)
    """
    conversations = entry.get("conversations", [])

    # 检查对话轮数
    if len(conversations) < 2:
        return False, "对话轮数不足"

    # 检查必要的角色
    has_human = any(c.get("from") == "human" for c in conversations)
    has_gpt = any(c.get("from") == "gpt" for c in conversations)
    if not has_human or not has_gpt:
        return False, "缺少必要角色(human/gpt)"

    # 检查问题长度
    first_human = next(c for c in conversations if c.get("from") == "human")
    if len(first_human.get("value", "")) < MIN_QUESTION_LENGTH:
        return False, "问题过短"

    # 检查回答长度
    first_gpt = next(c for c in conversations if c.get("from") == "gpt")
    response = first_gpt.get("value", "")

    if len(response) < MIN_RESPONSE_LENGTH:
        return False, f"回答过短({len(response)}字)"

    if len(response) > MAX_RESPONSE_LENGTH:
        return False, f"回答过长({len(response)}字)"

    # 检查低质量标记
    for marker in LOW_QUALITY_MARKERS:
        if marker in response[:100]:  # 仅检查开头 100 字
            return False, f"包含低质量标记: {marker}"

    # 检查回答是否为空白或重复字符
    unique_chars = len(set(response))
    if unique_chars < 10:
        return False, "回答内容过于单一"

    return True, "OK"


def deduplicate_data(data: list) -> list:
    """
    对数据进行去重（精确去重 + 近似去重）。

    参数:
        data: ShareGPT 格式数据列表

    返回:
        去重后的数据列表
    """
    logger.info("开始数据去重...")

    # 第一步: MD5 精确去重
    seen_hashes = set()
    stage1_data = []
    exact_dup_count = 0

    for entry in data:
        # 基于第一轮问答的 hash 去重
        convs = entry.get("conversations", [])
        if len(convs) >= 2:
            key_text = convs[0].get("value", "") + convs[1].get("value", "")
            h = _content_hash(key_text)
            if h not in seen_hashes:
                seen_hashes.add(h)
                stage1_data.append(entry)
            else:
                exact_dup_count += 1
        else:
            stage1_data.append(entry)

    logger.info(f"  精确去重: 移除 {exact_dup_count} 条完全重复数据")

    # 第二步: n-gram 近似去重（仅对回答进行）
    # 注意: 数据量大时此步骤较慢，可酌情跳过
    if len(stage1_data) > 50000:
        logger.info("  数据量较大，跳过近似去重（可通过参数启用）")
        return stage1_data

    ngram_cache = []
    stage2_data = []
    approx_dup_count = 0

    for entry in tqdm(stage1_data, desc="近似去重", unit="条"):
        convs = entry.get("conversations", [])
        gpt_responses = [c.get("value", "") for c in convs if c.get("from") == "gpt"]
        response_text = " ".join(gpt_responses)
        ngrams = _get_ngrams(response_text)

        is_dup = False
        for cached_ngrams in ngram_cache:
            if _jaccard_similarity(ngrams, cached_ngrams) > DEDUP_SIMILARITY_THRESHOLD:
                is_dup = True
                approx_dup_count += 1
                break

        if not is_dup:
            ngram_cache.append(ngrams)
            stage2_data.append(entry)

    logger.info(f"  近似去重: 移除 {approx_dup_count} 条相似数据")
    return stage2_data


def balance_tasks(data: list, total_target: Optional[int] = None) -> list:
    """
    按任务类型进行均衡采样。

    参数:
        data:         去重后的数据列表
        total_target: 目标总数据量，None=保持原数量

    返回:
        均衡采样后的数据列表
    """
    import random

    # 按任务分组
    task_groups = defaultdict(list)
    no_task = []

    for entry in data:
        task_type = entry.get("task_type", "")
        if task_type and task_type in TASK_TARGET_RATIO:
            task_groups[task_type].append(entry)
        else:
            no_task.append(entry)

    if not task_groups:
        logger.info("数据中无任务类型标签，跳过均衡采样")
        return data

    total = total_target or len(data)
    logger.info(f"\n任务分布均衡采样 (目标总量: {total})")

    balanced = []
    for task_type, ratio in TASK_TARGET_RATIO.items():
        target_count = int(total * ratio)
        available = task_groups.get(task_type, [])

        if len(available) >= target_count:
            sampled = random.sample(available, target_count)
        else:
            # 数据不足时全部使用
            sampled = available
            logger.warning(
                f"  {task_type}: 目标 {target_count}, 实际仅 {len(available)}"
            )

        balanced.extend(sampled)
        logger.info(
            f"  {task_type}: {len(sampled)}/{len(available)} (目标比例 {ratio:.0%})"
        )

    # 加入无任务标签的数据
    balanced.extend(no_task)

    random.shuffle(balanced)
    return balanced


def filter_and_report(input_file: Path, output_file: Path) -> dict:
    """
    对单个数据文件执行完整的质量过滤流程。

    参数:
        input_file:  输入文件路径
        output_file: 输出文件路径

    返回:
        过滤统计报告
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {
        "input_count": len(data),
        "quality_pass": 0,
        "quality_fail": 0,
        "fail_reasons": Counter(),
        "dedup_removed": 0,
        "output_count": 0,
    }

    # 1. 基础质量过滤
    quality_passed = []
    for entry in tqdm(data, desc="质量检查", unit="条"):
        passed, reason = check_basic_quality(entry)
        if passed:
            quality_passed.append(entry)
            stats["quality_pass"] += 1
        else:
            stats["quality_fail"] += 1
            stats["fail_reasons"][reason] += 1

    # 2. 去重
    before_dedup = len(quality_passed)
    deduped = deduplicate_data(quality_passed)
    stats["dedup_removed"] = before_dedup - len(deduped)

    # 3. 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    stats["output_count"] = len(deduped)
    return stats


def run_quality_filter() -> None:
    """执行完整的质量过滤流程。"""
    logger.info("=" * 60)
    logger.info("Fin-Instruct 数据质量过滤")
    logger.info("=" * 60)

    if not SFT_DIR.exists():
        logger.error(f"数据目录不存在: {SFT_DIR}")
        logger.error("请先运行 convert_to_sharegpt.py 生成 SFT 数据")
        return

    total_stats = {
        "input_count": 0,
        "output_count": 0,
        "fail_reasons": Counter(),
    }

    for json_file in sorted(SFT_DIR.glob("fin_*.json")):
        logger.info(f"\n处理: {json_file.name}")
        output_file = SFT_DIR / f"{json_file.stem}_filtered.json"

        stats = filter_and_report(json_file, output_file)

        total_stats["input_count"] += stats["input_count"]
        total_stats["output_count"] += stats["output_count"]
        total_stats["fail_reasons"] += stats["fail_reasons"]

        logger.info(
            f"  输入: {stats['input_count']}, "
            f"质量通过: {stats['quality_pass']}, "
            f"去重移除: {stats['dedup_removed']}, "
            f"输出: {stats['output_count']}"
        )

    # 打印总统计
    logger.info("\n" + "=" * 60)
    logger.info("质量过滤总统计:")
    logger.info(f"  总输入: {total_stats['input_count']}")
    logger.info(f"  总输出: {total_stats['output_count']}")
    logger.info(
        f"  过滤率: {1 - total_stats['output_count'] / max(total_stats['input_count'], 1):.1%}"
    )

    if total_stats["fail_reasons"]:
        logger.info("\n过滤原因分布:")
        for reason, count in total_stats["fail_reasons"].most_common():
            logger.info(f"  {reason}: {count}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    run_quality_filter()
