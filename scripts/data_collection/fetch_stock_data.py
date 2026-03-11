#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股行情与技术指标数据采集

使用 AKShare 采集:
  1. 沪深 300 成分股的日K线/周K线数据（近3年）
  2. 技术指标计算（MACD、KDJ、RSI、布林带等）

输出: data/raw/stock_prices/ 目录下的 CSV 文件
"""

import os
import time
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "stock_prices"

# ============================================================
# 数据采集参数
# ============================================================
DEFAULT_END_DATE = datetime.now().strftime("%Y%m%d")
DEFAULT_START_DATE = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y%m%d")

# ★ 关键参数：请求间隔（秒）— 太短会被东方财富限流
REQUEST_INTERVAL_MIN = 3.0   # 最小间隔 (调大以防屏蔽)
REQUEST_INTERVAL_MAX = 6.0   # 最大间隔（随机化避免固定模式被识别）

# 连续失败后的冷却时间（秒）
COOLDOWN_AFTER_FAILURES = 30
# 连续失败多少次触发冷却
FAILURE_THRESHOLD = 3

# ============================================================
# AKShare 延迟加载（仅加载一次）
# ============================================================
_ak = None


def _get_ak():
    """延迟加载 akshare 模块，并重置其内部 requests Session。"""
    global _ak
    if _ak is None:
        try:
            import akshare as ak
            _ak = ak
            logger.info(f"akshare 版本: {ak.__version__}")
        except ImportError:
            logger.error("未安装 akshare！请执行: pip install akshare")
            raise
    return _ak


def _reset_ak_session():
    """
    ★ 强制关闭并重建 requests 连接池，并为后续请求注入反爬 Header。
    """
    import requests
    import urllib3
    try:
        urllib3.disable_warnings()
        
        # 1. 关闭已有的 session
        for attr_name in dir(requests):
            obj = getattr(requests, attr_name, None)
            if isinstance(obj, requests.Session):
                try:
                    obj.close()
                except Exception:
                    pass
                    
        # 2. 全局 Monkey Patch：强制 requests 使用真实 User-Agent 并关闭 keep-alive
        if not hasattr(requests.Session, '_patched_for_akshare'):
            original_request = requests.Session.request
            
            def new_request(self, method, url, **kwargs):
                kwargs.setdefault('headers', {})
                # 随机 User-Agent 防止被封
                ua_list = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
                ]
                kwargs['headers']['User-Agent'] = random.choice(ua_list)
                # 关闭 Keep-Alive，避免 urllib3 复用被东方财富服务端悄悄断开的连接
                kwargs['headers']['Connection'] = 'close'
                return original_request(self, method, url, **kwargs)
                
            requests.Session.request = new_request
            requests.Session._patched_for_akshare = True
            
    except Exception as e:
        logger.error(f"重置连接池时出错: {e}")
        pass
    logger.debug("已重置 HTTP Session 连接池并注入反爬 Header")


def _random_sleep():
    """随机间隔休眠，模拟人类访问模式，减少被封概率。"""
    interval = random.uniform(REQUEST_INTERVAL_MIN, REQUEST_INTERVAL_MAX)
    time.sleep(interval)


# ============================================================
# 数据采集
# ============================================================

HS300_CACHE_FILE = RAW_DATA_DIR / "_hs300_stocks.csv"


def _fetch_hs300_from_api(max_retries: int = 5) -> pd.DataFrame:
    """带重试的沪深300成分股 API 调用。"""
    ak = _get_ak()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            _reset_ak_session()
            time.sleep(random.uniform(1, 3))
            return ak.index_stock_cons(symbol="000300")
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(kw in error_str for kw in [
                "connection", "remote", "disconnected", "reset",
                "aborted", "broken pipe", "errno 104", "timeout",
                "prematurely", "chunked", "ssl", "max retries",
            ])
            if retryable and attempt < max_retries:
                wait = min(15 * attempt, 90)
                logger.warning(
                    f"获取沪深300列表失败 ({attempt}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )
                logger.info(f"  等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise

    raise last_error


def get_hs300_stocks() -> pd.DataFrame:
    """
    获取沪深300成分股列表（带重试 + 本地缓存 fallback）。

    返回:
        包含 code(代码) 和 name(名称) 列的 DataFrame
    """
    logger.info("正在获取沪深300成分股列表...")

    try:
        df = _fetch_hs300_from_api()
        if "品种代码" in df.columns:
            df = df.rename(columns={"品种代码": "code", "品种名称": "name"})
        elif "stock_code" in df.columns:
            df = df.rename(columns={"stock_code": "code", "stock_name": "name"})
        else:
            df.columns = ["code", "name"] + list(df.columns[2:])
        df = df[["code", "name"]]
        logger.info(f"获取到 {len(df)} 只沪深300成分股")

        # 缓存到本地，下次网络故障时可用
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(HS300_CACHE_FILE, index=False, encoding="utf-8-sig")
        return df

    except Exception as e:
        logger.error(f"在线获取沪深300成分股列表失败: {e}")

        # fallback: 使用本地缓存
        if HS300_CACHE_FILE.exists():
            logger.info(f"使用本地缓存: {HS300_CACHE_FILE}")
            df = pd.read_csv(HS300_CACHE_FILE, dtype={"code": str})
            logger.info(f"从缓存加载 {len(df)} 只成分股")
            return df

        raise


def _get_secid(symbol: str) -> str:
    """根据股票代码判断交易所前缀 (secid): 0=深圳, 1=上海。"""
    if symbol.startswith(("6", "9")):
        return f"1.{symbol}"  # 上海
    else:
        return f"0.{symbol}"  # 深圳


_PERIOD_MAP = {"daily": "101", "weekly": "102", "monthly": "103"}

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


def _fetch_stock_direct(
    symbol: str,
    period: str = "daily",
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    adjust: str = "qfq",
) -> Optional[pd.DataFrame]:
    """
    ★ 直接调用东方财富 API 获取历史行情，绕过 akshare。

    与 akshare 使用完全相同的 API 端点 (push2his.eastmoney.com)，
    但由我们自行控制 HTTP header / timeout / 连接方式，
    从根本上解决 RemoteDisconnected 问题。
    """
    import requests

    secid = _get_secid(symbol)
    klt = _PERIOD_MAP.get(period, "101")

    # 前复权=1, 后复权=2, 不复权=0
    fqt = {"qfq": "1", "hfq": "2", "": "0"}.get(adjust, "1")

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9004f1e12c8b0f2",
        "klt": klt,
        "fqt": fqt,
        "secid": secid,
        "beg": start_date,
        "end": end_date,
        "_": str(int(time.time() * 1000)),
    }
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json, text/plain, */*",
        "Connection": "close",  # ★ 不复用连接，彻底避免 RemoteDisconnected
    }

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if not data.get("data") or not data["data"].get("klines"):
        return None

    rows = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        # 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        rows.append(parts)

    df = pd.DataFrame(rows, columns=[
        "date", "open", "close", "high", "low",
        "volume", "amount", "amplitude", "pct_change", "price_change", "turnover",
    ])
    # 转换数值列
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_stock_history(
    symbol: str,
    period: str = "daily",
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    max_retries: int = 5,
) -> Optional[pd.DataFrame]:
    """
    获取单只股票的历史行情数据。

    策略：优先直连东方财富 API（绕过 akshare），失败后回退到 akshare。

    参数:
        symbol:      股票代码（如 "000001"）
        period:      周期，"daily"(日线) / "weekly"(周线)
        start_date:  起始日期，格式 YYYYMMDD
        end_date:    截止日期，格式 YYYYMMDD
        max_retries: 最大重试次数

    返回:
        包含 OHLCV 数据的 DataFrame，失败返回 None
    """
    last_error = None

    # ★ 方式 1：直连东方财富 API（推荐，成功率高）
    for attempt in range(1, max_retries + 1):
        try:
            df = _fetch_stock_direct(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
            )
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
                    f"直连获取 {symbol} {period} 失败 ({attempt}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)[:80]}"
                )
                logger.info(f"  等待 {wait}s 后重试...")
                time.sleep(wait)
            elif not retryable:
                # 非网络错误，直接跳到 akshare fallback
                break
            else:
                break  # 网络错误重试耗尽，跳到 akshare fallback

    # ★ 方式 2：fallback 到 akshare
    logger.info(f"  {symbol} 直连失败，尝试 akshare fallback...")
    try:
        ak = _get_ak()
        _reset_ak_session()
        time.sleep(random.uniform(2, 5))

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            column_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "price_change",
                "换手率": "turnover",
            }
            df = df.rename(columns=column_map)
            return df
        return None
    except Exception as fallback_error:
        logger.error(
            f"  {symbol}({period}) 直连 + akshare 均失败: "
            f"直连={type(last_error).__name__}, "
            f"akshare={type(fallback_error).__name__}"
        )
        raise last_error from fallback_error


# ============================================================
# 技术指标计算
# ============================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    为行情数据计算常用技术指标。

    计算的指标包括:
      - MA5, MA10, MA20, MA60: 移动平均线
      - MACD_DIF, MACD_DEA, MACD_HIST: MACD 指标
      - RSI_6, RSI_12, RSI_24: 相对强弱指数
      - KDJ_K, KDJ_D, KDJ_J: KDJ 随机指标
      - BOLL_upper, BOLL_mid, BOLL_lower: 布林带

    参数:
        df: 包含 OHLCV 数据的 DataFrame

    返回:
        添加了技术指标列的 DataFrame
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ---- 移动平均线 (MA) ----
    for window in [5, 10, 20, 60]:
        df[f"MA{window}"] = close.rolling(window=window).mean()

    # ---- MACD (12, 26, 9) ----
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_DIF"] = ema12 - ema26
    df["MACD_DEA"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = 2 * (df["MACD_DIF"] - df["MACD_DEA"])

    # ---- RSI (6, 12, 24) ----
    for period in [6, 12, 24]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

    # ---- KDJ (9, 3, 3) ----
    low_min = low.rolling(window=9).min()
    high_max = high.rolling(window=9).max()
    rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    df["KDJ_K"] = rsv.ewm(com=2, adjust=False).mean()
    df["KDJ_D"] = df["KDJ_K"].ewm(com=2, adjust=False).mean()
    df["KDJ_J"] = 3 * df["KDJ_K"] - 2 * df["KDJ_D"]

    # ---- 布林带 (20日, 2倍标准差) ----
    df["BOLL_mid"] = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    df["BOLL_upper"] = df["BOLL_mid"] + 2 * std_20
    df["BOLL_lower"] = df["BOLL_mid"] - 2 * std_20

    return df


# ============================================================
# 批量采集主函数
# ============================================================

def fetch_all_stocks(
    max_stocks: Optional[int] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    skip_existing: bool = True,
) -> None:
    """
    批量采集沪深300成分股的行情与技术指标数据。

    参数:
        max_stocks:    最大采集数量（调试用），None 表示全部
        start_date:    起始日期
        end_date:      截止日期
        skip_existing: 是否跳过已采集的股票
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stocks_df = get_hs300_stocks()
    if max_stocks:
        stocks_df = stocks_df.head(max_stocks)

    logger.info(f"开始采集 {len(stocks_df)} 只股票行情数据")
    logger.info(f"时间范围: {start_date} ~ {end_date}")
    logger.info(f"保存目录: {RAW_DATA_DIR}\n")

    # 在请求成分股之后、大批量请求历史数据之前，增加缓冲时间
    logger.info("等待 5 秒，以防止接口频率限制...")
    time.sleep(5)

    success_count = 0
    fail_count = 0
    skip_count = 0
    consecutive_failures = 0  # ★ 连续失败计数器

    for _, row in tqdm(
        stocks_df.iterrows(),
        total=len(stocks_df),
        desc="采集股票行情",
        unit="只",
    ):
        code = str(row["code"]).zfill(6)
        name = row["name"]
        stock_success = True

        for period in ["daily", "weekly"]:
            output_file = RAW_DATA_DIR / f"{code}_{period}.csv"

            # 跳过已存在的文件
            if skip_existing and output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 100:  # 文件大于100字节才视为有效
                    skip_count += 1
                    continue

            try:
                df = fetch_stock_history(
                    symbol=code,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                )
                if df is not None:
                    # 日线数据计算技术指标
                    if period == "daily":
                        df = calculate_technical_indicators(df)
                    df.to_csv(output_file, index=False, encoding="utf-8-sig")
                    logger.debug(f"  {code}({name}) {period}: {len(df)} 行")
                    consecutive_failures = 0  # ★ 成功后重置计数器
                else:
                    logger.warning(f"  {code}({name}) {period} 无数据")

            except Exception as e:
                logger.error(f"  {code}({name}) {period} 采集失败: {e}")
                stock_success = False
                consecutive_failures += 1

                # ★ 连续失败超过阈值，进入冷却期
                if consecutive_failures >= FAILURE_THRESHOLD:
                    logger.warning(
                        f"  连续失败 {consecutive_failures} 次，"
                        f"冷却 {COOLDOWN_AFTER_FAILURES} 秒..."
                    )
                    _reset_ak_session()
                    time.sleep(COOLDOWN_AFTER_FAILURES)
                    consecutive_failures = 0

            # ★ 每次请求后随机休眠
            _random_sleep()

        if stock_success:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"\n采集完成: 成功 {success_count}, 失败 {fail_count}, 跳过 {skip_count} 个文件")
    logger.info(f"数据保存目录: {RAW_DATA_DIR}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse
    from http.client import RemoteDisconnected

    parser = argparse.ArgumentParser(description="A股行情数据采集")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="最大采集股票数量（调试用，默认全部）",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"起始日期 YYYYMMDD (默认: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help=f"截止日期 YYYYMMDD (默认: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="不跳过已存在的文件，全部重新采集",
    )
    args = parser.parse_args()

    fetch_all_stocks(
        max_stocks=args.max_stocks,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_existing=not args.no_skip,
    )
