#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融新闻数据采集

使用 AKShare 采集:
  1. 个股新闻（东方财富源）
  2. 财经要闻（新浪/东方财富）

支持基于标题+日期的自动去重，增量采集。
输出: data/raw/news/ 目录下的 JSON 文件
"""

import os
import json
import time
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Set

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
NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"
STOCK_NEWS_DIR = NEWS_DIR / "stock_news"
GENERAL_NEWS_DIR = NEWS_DIR / "general_news"

# 请求间隔（秒）
REQUEST_INTERVAL_MIN = 1.0
REQUEST_INTERVAL_MAX = 2.5

# 连续失败阈值与冷却
FAILURE_THRESHOLD = 3
COOLDOWN_AFTER_FAILURES = 30


def _reset_ak_session():
    """强制关闭并重建 requests 连接池。"""
    import requests
    try:
        for attr_name in dir(requests):
            obj = getattr(requests, attr_name, None)
            if isinstance(obj, requests.Session):
                try:
                    obj.close()
                except Exception:
                    pass
    except Exception:
        pass


def _random_sleep():
    """随机间隔休眠。"""
    time.sleep(random.uniform(REQUEST_INTERVAL_MIN, REQUEST_INTERVAL_MAX))


def _generate_news_id(title: str, date: str = "") -> str:
    """
    基于标题和日期生成新闻唯一ID，用于去重。

    参数:
        title: 新闻标题
        date:  发布日期

    返回:
        MD5 哈希字符串
    """
    content = f"{title.strip()}_{date.strip()}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _load_existing_ids(file_path: Path) -> Set[str]:
    """
    从已有的新闻文件中加载已存在的新闻ID集合（用于增量去重）。

    参数:
        file_path: JSON 新闻文件路径

    返回:
        已有新闻ID集合
    """
    if not file_path.exists():
        return set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {item.get("news_id", "") for item in existing}
    except (json.JSONDecodeError, KeyError):
        return set()


def _clean_text(text: str) -> str:
    """
    清洗新闻文本：去除多余空白、HTML残留等。

    参数:
        text: 原始文本

    返回:
        清洗后的文本
    """
    import re

    if not isinstance(text, str):
        return ""
    # 去除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_stock_news(symbol: str, max_retries: int = 5) -> Optional[pd.DataFrame]:
    """
    获取单只股票的相关新闻（东方财富源），带手动重试。

    参数:
        symbol:      股票代码（6位数字，如 "000001"）
        max_retries: 最大重试次数

    返回:
        新闻 DataFrame，失败返回 None
    """
    import akshare as ak
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_news_em(symbol=symbol)
            if df is not None and not df.empty:
                return df
            return None
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(kw in error_str for kw in [
                "connection", "remote", "disconnected", "reset",
                "aborted", "broken pipe", "errno 104", "timeout",
                "prematurely", "chunked", "ssl", "max retries",
            ])
            if retryable and attempt < max_retries:
                wait = min(10 * attempt, 60)
                logger.warning(
                    f"获取 {symbol} 新闻失败 ({attempt}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                logger.info(f"  等待 {wait}s 后重试...")
                _reset_ak_session()
                time.sleep(wait)
            else:
                raise

    raise last_error


def fetch_general_financial_news(max_retries: int = 5) -> Optional[pd.DataFrame]:
    """
    获取财经要闻（东方财富全球财经快讯），带手动重试。

    返回:
        新闻 DataFrame，失败返回 None
    """
    import akshare as ak
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_info_global_em()
            if df is not None and not df.empty:
                return df
            return None
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(kw in error_str for kw in [
                "connection", "remote", "disconnected", "reset",
                "aborted", "broken pipe", "errno 104", "timeout",
                "prematurely", "chunked", "ssl", "max retries",
            ])
            if retryable and attempt < max_retries:
                wait = min(10 * attempt, 60)
                logger.warning(
                    f"获取财经要闻失败 ({attempt}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                logger.info(f"  等待 {wait}s 后重试...")
                _reset_ak_session()
                time.sleep(wait)
            else:
                raise

    raise last_error


def process_stock_news(
    symbol: str,
    stock_name: str = "",
) -> list:
    """
    采集并处理单只股票的新闻数据。

    参数:
        symbol:     股票代码
        stock_name: 股票名称

    返回:
        处理后的新闻记录列表
    """
    df = fetch_stock_news(symbol)
    if df is None:
        return []

    news_list = []
    for _, row in df.iterrows():
        # 提取标题和内容（列名可能因 AKShare 版本而异）
        title = str(row.get("新闻标题", row.get("title", "")))
        content = str(row.get("新闻内容", row.get("content", "")))
        date = str(row.get("发布时间", row.get("publish_time", "")))
        source = str(row.get("文章来源", row.get("source", "东方财富")))

        title = _clean_text(title)
        content = _clean_text(content)

        if not title or len(title) < 5:
            continue  # 跳过无效标题

        news_id = _generate_news_id(title, date)
        news_list.append(
            {
                "news_id": news_id,
                "stock_code": symbol,
                "stock_name": stock_name,
                "title": title,
                "content": content,
                "date": date,
                "source": source,
            }
        )

    return news_list


def fetch_all_stock_news(
    stock_codes: list,
    stock_names: Optional[dict] = None,
) -> None:
    """
    批量采集个股新闻。

    参数:
        stock_codes:  股票代码列表
        stock_names:  {代码: 名称} 映射字典
    """
    STOCK_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    stock_names = stock_names or {}

    output_file = STOCK_NEWS_DIR / "all_stock_news.json"
    existing_ids = _load_existing_ids(output_file)

    # 加载已有数据（增量追加）
    all_news = []
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            all_news = json.load(f)

    new_count = 0
    consecutive_failures = 0

    for code in tqdm(stock_codes, desc="采集个股新闻", unit="只"):
        code = str(code).zfill(6)
        name = stock_names.get(code, "")

        try:
            news_list = process_stock_news(code, name)
            for item in news_list:
                if item["news_id"] not in existing_ids:
                    all_news.append(item)
                    existing_ids.add(item["news_id"])
                    new_count += 1
            consecutive_failures = 0
        except Exception as e:
            logger.error(f"  {code} 新闻采集失败: {e}")
            consecutive_failures += 1
            if consecutive_failures >= FAILURE_THRESHOLD:
                logger.warning(
                    f"连续失败 {consecutive_failures} 次，冷却 {COOLDOWN_AFTER_FAILURES}s..."
                )
                _reset_ak_session()
                time.sleep(COOLDOWN_AFTER_FAILURES)
                consecutive_failures = 0

        _random_sleep()

    # 保存合并结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)

    logger.info(f"个股新闻采集完成: 新增 {new_count} 条, 总计 {len(all_news)} 条")


def fetch_general_news() -> None:
    """采集并保存财经要闻。"""
    GENERAL_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = GENERAL_NEWS_DIR / "financial_news.json"

    existing_ids = _load_existing_ids(output_file)
    existing_news = []
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_news = json.load(f)

    logger.info("正在采集财经要闻...")
    try:
        df = fetch_general_financial_news()
        if df is None:
            logger.warning("未获取到财经要闻")
            return

        new_count = 0
        for _, row in df.iterrows():
            title = _clean_text(str(row.iloc[0] if len(row) > 0 else ""))
            content = _clean_text(str(row.iloc[1] if len(row) > 1 else ""))
            date = str(row.iloc[2] if len(row) > 2 else "")

            if not title or len(title) < 5:
                continue

            news_id = _generate_news_id(title, date)
            if news_id not in existing_ids:
                existing_news.append(
                    {
                        "news_id": news_id,
                        "title": title,
                        "content": content,
                        "date": date,
                        "source": "东方财富全球财经",
                    }
                )
                existing_ids.add(news_id)
                new_count += 1

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_news, f, ensure_ascii=False, indent=2)

        logger.info(f"财经要闻采集完成: 新增 {new_count} 条, 总计 {len(existing_news)} 条")

    except Exception as e:
        logger.error(f"财经要闻采集失败: {e}")


def get_hs300_stock_info() -> tuple:
    """获取沪深300股票代码和名称映射（复用 fetch_stock_data 的重试+缓存逻辑）。"""
    try:
        from scripts.data_collection.fetch_stock_data import get_hs300_stocks
        df = get_hs300_stocks()
        codes = df["code"].tolist()
        names = dict(zip(df["code"], df["name"]))
        return codes, names
    except ImportError:
        pass

    # fallback: 直接调用 akshare
    import akshare as ak
    df = ak.index_stock_cons(symbol="000300")
    if "品种代码" in df.columns:
        codes = df["品种代码"].tolist()
        names = dict(zip(df["品种代码"], df["品种名称"]))
    elif "stock_code" in df.columns:
        codes = df["stock_code"].tolist()
        names = dict(zip(df["stock_code"], df["stock_name"]))
    else:
        codes = df.iloc[:, 0].tolist()
        names = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return codes, names


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="金融新闻数据采集")
    parser.add_argument(
        "--type",
        choices=["stock", "general", "all"],
        default="all",
        help="采集类型: stock(个股新闻), general(财经要闻), all(全部)",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="最大采集股票数量（调试用，默认全部沪深300）",
    )
    args = parser.parse_args()

    if args.type in ("stock", "all"):
        codes, names = get_hs300_stock_info()
        if args.max_stocks:
            codes = codes[: args.max_stocks]
        fetch_all_stock_news(stock_codes=codes, stock_names=names)

    if args.type in ("general", "all"):
        fetch_general_news()
