#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块

对采集的原始数据进行统一清洗:
  1. 统一编码（全部转为 UTF-8）
  2. 去除 HTML 标签和广告内容
  3. 文本长度过滤（过短/过长）
  4. 敏感信息脱敏（手机号、身份证、邮箱等）
  5. 去除无意义的特殊字符

输出: data/processed/ 目录
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Union

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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================
# 清洗参数
# ============================================================
MIN_TEXT_LENGTH = 20       # 最小文本长度（字符数），低于此值视为无效
MAX_TEXT_LENGTH = 8000     # 最大文本长度（字符数），超过此值截断
TRUNCATE_SUFFIX = "..."    # 截断后缀

# ============================================================
# 正则表达式（预编译，提升性能）
# ============================================================
# HTML 标签
RE_HTML_TAG = re.compile(r"<[^>]+>")
# 手机号（中国大陆 11 位）
RE_PHONE = re.compile(r"1[3-9]\d{9}")
# 身份证号（18 位）
RE_ID_CARD = re.compile(r"\d{17}[\dXx]")
# 邮箱
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# 银行卡号（16-19 位数字）
RE_BANK_CARD = re.compile(r"\d{16,19}")
# 多余空白（连续空格、制表符等）
RE_WHITESPACE = re.compile(r"[ \t]+")
# 连续空行
RE_BLANK_LINES = re.compile(r"\n{3,}")
# 广告关键词模式（常见网页广告词）
RE_AD_PATTERN = re.compile(
    r"(点击查看|立即下载|扫码关注|免费领取|限时优惠|加微信|加QQ|"
    r"点击进入|复制链接|长按识别|阅读原文|更多精彩|"
    r"广告|推广|赞助商)",
    re.IGNORECASE,
)
# URL
RE_URL = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")


def clean_html(text: str) -> str:
    """去除 HTML 标签。"""
    return RE_HTML_TAG.sub("", text)


def clean_whitespace(text: str) -> str:
    """规范化空白字符。"""
    text = RE_WHITESPACE.sub(" ", text)
    text = RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def mask_sensitive_info(text: str) -> str:
    """
    脱敏处理：将敏感信息替换为占位符。

    替换规则:
      - 手机号 -> [手机号]
      - 身份证 -> [身份证号]
      - 邮箱   -> [邮箱]
      - 银行卡 -> [银行卡号]
    """
    text = RE_PHONE.sub("[手机号]", text)
    text = RE_ID_CARD.sub("[身份证号]", text)
    text = RE_EMAIL.sub("[邮箱]", text)
    # 银行卡号需谨慎处理，避免误伤普通数字（如股票代码、金额等）
    # 仅替换独立出现的 16-19 位数字
    text = re.sub(r"(?<!\d)\d{16,19}(?!\d)", "[银行卡号]", text)
    return text


def remove_urls(text: str) -> str:
    """移除 URL 链接。"""
    return RE_URL.sub("", text)


def is_ad_content(text: str, threshold: int = 3) -> bool:
    """
    判断文本是否为广告内容。

    参数:
        text:      待检测文本
        threshold: 广告关键词出现次数阈值

    返回:
        是否为广告内容
    """
    matches = RE_AD_PATTERN.findall(text)
    return len(matches) >= threshold


def clean_text(
    text: str,
    min_length: int = MIN_TEXT_LENGTH,
    max_length: int = MAX_TEXT_LENGTH,
    remove_ads: bool = True,
    mask_pii: bool = True,
) -> Union[str, None]:
    """
    对单条文本执行完整的清洗流程。

    参数:
        text:       原始文本
        min_length: 最小长度阈值
        max_length: 最大长度阈值（超过则截断）
        remove_ads: 是否过滤广告内容
        mask_pii:   是否脱敏个人信息

    返回:
        清洗后的文本，如果文本无效则返回 None
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # 1. 去除 HTML 标签
    text = clean_html(text)

    # 2. 移除 URL
    text = remove_urls(text)

    # 3. 规范化空白
    text = clean_whitespace(text)

    # 4. 广告检测
    if remove_ads and is_ad_content(text):
        return None

    # 5. 长度过滤
    if len(text) < min_length:
        return None

    # 6. 截断过长文本
    if len(text) > max_length:
        text = text[:max_length] + TRUNCATE_SUFFIX

    # 7. 敏感信息脱敏
    if mask_pii:
        text = mask_sensitive_info(text)

    return text


def clean_news_data(input_dir: Path, output_dir: Path) -> dict:
    """
    清洗新闻数据（JSON 格式）。

    参数:
        input_dir:  原始新闻目录（data/raw/news/）
        output_dir: 清洗后输出目录

    返回:
        清洗统计信息字典
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "valid": 0, "filtered": 0}

    for json_file in input_dir.rglob("*.json"):
        logger.info(f"清洗新闻数据: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                news_list = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"  文件读取失败: {e}")
            continue

        cleaned = []
        for item in news_list:
            stats["total"] += 1
            title = clean_text(item.get("title", ""), min_length=5, max_length=200)
            content = clean_text(item.get("content", ""), min_length=20)

            if title is None:
                stats["filtered"] += 1
                continue

            item["title"] = title
            item["content"] = content or ""
            cleaned.append(item)
            stats["valid"] += 1

        # 保存清洗结果
        rel_path = json_file.relative_to(input_dir)
        out_file = output_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

        logger.info(f"  保留 {len(cleaned)}/{len(news_list)} 条")

    return stats


def clean_csv_data(input_dir: Path, output_dir: Path) -> dict:
    """
    清洗 CSV 格式的股票/财务数据。

    主要操作:
      - 去除空行和全 NaN 列
      - 统一编码为 UTF-8
      - 数值列类型规范化

    参数:
        input_dir:  原始数据目录
        output_dir: 清洗后输出目录

    返回:
        清洗统计信息字典
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "rows_before": 0, "rows_after": 0}

    for csv_file in input_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, encoding="gbk")
            except Exception as e:
                logger.warning(f"  读取失败 {csv_file.name}: {e}")
                continue

        stats["files"] += 1
        stats["rows_before"] += len(df)

        # 去除全空行和全空列
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        stats["rows_after"] += len(df)

        # 保存
        rel_path = csv_file.relative_to(input_dir)
        out_file = output_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False, encoding="utf-8-sig")

    return stats


def run_full_cleaning() -> None:
    """执行完整的数据清洗流程。"""
    logger.info("=" * 60)
    logger.info("Fin-Instruct 数据清洗开始")
    logger.info("=" * 60)

    # 1. 清洗新闻数据
    news_input = RAW_DATA_DIR / "news"
    news_output = PROCESSED_DIR / "news"
    if news_input.exists():
        logger.info("\n--- 清洗新闻数据 ---")
        stats = clean_news_data(news_input, news_output)
        logger.info(f"新闻清洗统计: {stats}")

    # 2. 清洗股票行情数据
    stock_input = RAW_DATA_DIR / "stock_prices"
    stock_output = PROCESSED_DIR / "stock_prices"
    if stock_input.exists():
        logger.info("\n--- 清洗股票行情数据 ---")
        stats = clean_csv_data(stock_input, stock_output)
        logger.info(f"股票数据清洗统计: {stats}")

    # 3. 清洗财务报表数据
    report_input = RAW_DATA_DIR / "financial_reports"
    report_output = PROCESSED_DIR / "financial_reports"
    if report_input.exists():
        logger.info("\n--- 清洗财务报表数据 ---")
        stats = clean_csv_data(report_input, report_output)
        logger.info(f"财务报表清洗统计: {stats}")

    logger.info("\n" + "=" * 60)
    logger.info("数据清洗完成！")
    logger.info(f"输出目录: {PROCESSED_DIR}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    run_full_cleaning()
