#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上市公司财务报表数据采集

使用 AKShare 采集沪深300成分股的:
  1. 三大财务报表（资产负债表、利润表、现金流量表）
  2. 关键财务指标（ROE、毛利率、净利率、资产负债率等）

输出: data/raw/financial_reports/ 目录下的 CSV 文件
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "financial_reports"

# 请求间隔（秒）
REQUEST_INTERVAL = 1.0

# ============================================================
# 财务报表类型映射
# ============================================================
# AKShare 新浪财经接口支持的报表类型
REPORT_TYPES = {
    "balance_sheet": "资产负债表",
    "income_statement": "利润表",
    "cash_flow": "现金流量表",
}


def get_hs300_stocks() -> pd.DataFrame:
    """获取沪深300成分股列表（复用 fetch_stock_data 中的逻辑）。"""
    import akshare as ak

    logger.info("正在获取沪深300成分股列表...")
    try:
        df = ak.index_stock_cons(symbol="000300")
        if "品种代码" in df.columns:
            df = df.rename(columns={"品种代码": "code", "品种名称": "name"})
        elif "stock_code" in df.columns:
            df = df.rename(columns={"stock_code": "code", "stock_name": "name"})
        else:
            df.columns = ["code", "name"] + list(df.columns[2:])
        logger.info(f"获取到 {len(df)} 只沪深300成分股")
        return df[["code", "name"]]
    except Exception as e:
        logger.error(f"获取成分股列表失败: {e}")
        raise


def _build_stock_symbol(code: str) -> str:
    """
    将6位股票代码转换为新浪财务接口需要的格式。

    规则:
      - 6开头 -> sh600xxx (上海)
      - 0/3开头 -> sz000xxx / sz300xxx (深圳)
    """
    code = str(code).zfill(6)
    if code.startswith("6"):
        return f"sh{code}"
    else:
        return f"sz{code}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_financial_report(
    stock_symbol: str,
    report_type: str,
) -> Optional[pd.DataFrame]:
    """
    获取单只股票的某类财务报表（带自动重试）。

    参数:
        stock_symbol: 新浪格式的股票代码（如 "sh600519"）
        report_type:  报表中文名称（"资产负债表" / "利润表" / "现金流量表"）

    返回:
        财务报表 DataFrame，失败返回 None
    """
    import akshare as ak

    try:
        df = ak.stock_financial_report_sina(
            stock=stock_symbol,
            symbol=report_type,
        )
        if df is not None and not df.empty:
            return df
        return None
    except Exception as e:
        logger.warning(f"  获取 {stock_symbol} {report_type} 失败: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_financial_indicators(code: str) -> Optional[pd.DataFrame]:
    """
    获取单只股票的主要财务指标。

    使用 AKShare 的财务指标接口，获取以下指标:
      - 净资产收益率 (ROE)
      - 毛利率、净利率
      - 资产负债率
      - 流动比率、速动比率
      - 营收/净利润增长率 等

    参数:
        code: 6位股票代码（如 "600519"）

    返回:
        财务指标 DataFrame，失败返回 None
    """
    import akshare as ak

    try:
        # 尝试使用东方财富的财务指标接口
        sina_symbol = _build_stock_symbol(code)
        df = ak.stock_financial_analysis_indicator(symbol=sina_symbol)
        if df is not None and not df.empty:
            return df
        return None
    except Exception as e:
        logger.warning(f"  获取 {code} 财务指标失败: {e}")
        raise


def fetch_single_stock_reports(
    code: str,
    name: str,
    save_dir: Path,
) -> bool:
    """
    采集单只股票的所有财务报表和指标。

    参数:
        code:     股票代码
        name:     股票名称
        save_dir: 保存目录

    返回:
        是否成功
    """
    stock_dir = save_dir / code
    stock_dir.mkdir(parents=True, exist_ok=True)

    stock_symbol = _build_stock_symbol(code)
    success = True

    # 1. 采集三大财务报表
    for report_key, report_name in REPORT_TYPES.items():
        output_file = stock_dir / f"{report_key}.csv"
        if output_file.exists():
            continue

        try:
            df = fetch_financial_report(stock_symbol, report_name)
            if df is not None:
                df.to_csv(output_file, index=False, encoding="utf-8-sig")
                logger.debug(f"  {code}({name}) {report_name}: {len(df)} 行")
            else:
                logger.warning(f"  {code}({name}) {report_name} 无数据")
        except Exception as e:
            logger.error(f"  {code}({name}) {report_name} 采集失败: {e}")
            success = False

        time.sleep(REQUEST_INTERVAL)

    # 2. 采集财务指标
    indicators_file = stock_dir / "financial_indicators.csv"
    if not indicators_file.exists():
        try:
            df = fetch_financial_indicators(code)
            if df is not None:
                df.to_csv(indicators_file, index=False, encoding="utf-8-sig")
                logger.debug(f"  {code}({name}) 财务指标: {len(df)} 行")
        except Exception as e:
            logger.error(f"  {code}({name}) 财务指标采集失败: {e}")
            success = False

        time.sleep(REQUEST_INTERVAL)

    return success


def fetch_all_reports(max_stocks: Optional[int] = None) -> None:
    """
    批量采集沪深300成分股的财务数据。

    参数:
        max_stocks: 最大采集数量（调试用），None 表示全部
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stocks_df = get_hs300_stocks()
    if max_stocks:
        stocks_df = stocks_df.head(max_stocks)

    logger.info(f"开始采集 {len(stocks_df)} 只股票的财务数据")
    logger.info(f"报表类型: {list(REPORT_TYPES.values())}")
    logger.info(f"保存目录: {RAW_DATA_DIR}\n")

    success_count = 0
    fail_count = 0

    for _, row in tqdm(
        stocks_df.iterrows(),
        total=len(stocks_df),
        desc="采集财务报表",
        unit="只",
    ):
        code = str(row["code"]).zfill(6)
        name = row["name"]

        ok = fetch_single_stock_reports(
            code=code,
            name=name,
            save_dir=RAW_DATA_DIR,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"\n采集完成: 成功 {success_count}, 失败 {fail_count}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="上市公司财务报表数据采集")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="最大采集股票数量（调试用，默认全部）",
    )
    args = parser.parse_args()

    fetch_all_reports(max_stocks=args.max_stocks)
