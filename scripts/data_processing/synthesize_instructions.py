#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 指令数据合成模块

基于采集的原始数据（K线、新闻、财报），调用大模型 API 生成高质量指令-回答对。
为 6 类金融任务分别设计了 Prompt 模板:
  1. 股票市场分析
  2. 量化策略研发
  3. 财报解读
  4. 金融新闻情感分析
  5. 金融问答/知识
  6. 风险评估

支持:
  - 批量调用 OpenAI 兼容 API（Qwen API / DeepSeek / OpenAI 等）
  - 异步并发提升吞吐
  - 结果缓存与断点续传
  - 输出 ShareGPT 格式

输出: data/processed/synthesized/ 目录
"""

import os
import re
import json
import time
import asyncio
import hashlib
import logging
import random
import math
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

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
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHESIZED_DIR = PROCESSED_DIR / "synthesized"
CACHE_DIR = SYNTHESIZED_DIR / ".cache"  # 请求结果缓存目录
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "system_prompts.json"

# ============================================================
# API 配置（从环境变量读取，保护密钥安全）
# ============================================================
API_BASE_URL = os.environ.get("SYNTH_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
API_KEY = os.environ.get("SYNTH_API_KEY", "")
API_MODEL = os.environ.get("SYNTH_API_MODEL", "GLM-4.7-Flash")
MAX_CONCURRENT = int(os.environ.get("SYNTH_MAX_CONCURRENT", "5"))  # 最大并发数
REQUEST_TIMEOUT = int(os.environ.get("SYNTH_TIMEOUT", "60"))  # 请求超时（秒）


def _load_system_prompts() -> dict:
    """加载 system prompt 模板。"""
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts


def _cache_key(task: str, content: str) -> str:
    """生成缓存键（基于任务类型和输入内容的 MD5）。"""
    raw = f"{task}_{content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _get_cached(cache_key: str) -> Optional[str]:
    """从缓存中获取已有结果。"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f).get("response")
    return None


def _save_cache(cache_key: str, response: str) -> None:
    """保存请求结果到缓存。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"response": response}, f, ensure_ascii=False)


TASK_NAMES = [
    "stock_analysis",
    "quant_strategy",
    "financial_report",
    "sentiment_analysis",
    "financial_qa",
    "risk_assessment",
]


# ============================================================
# 各任务的 Prompt 模板
# ============================================================

TASK_TEMPLATES = {
    "stock_analysis": {
        "instruction_template": (
            "以下是{stock_name}（{stock_code}）最近的行情数据摘要:\n"
            "{market_data}\n\n"
            "请从技术面角度分析该股票的走势，包括:\n"
            "1. K线形态分析\n"
            "2. 关键技术指标解读（MACD、KDJ、RSI等）\n"
            "3. 支撑位和压力位判断\n"
            "4. 短期趋势研判和操作建议"
        ),
        "data_source": "stock_prices",
    },
    "quant_strategy": {
        "instruction_template": (
            "基于以下A股市场数据特征:\n"
            "{market_context}\n\n"
            "请设计一个{strategy_type}量化交易策略，要求:\n"
            "1. 说明策略核心逻辑和因子选择依据\n"
            "2. 给出完整的 Python 实现代码（使用 pandas/numpy）\n"
            "3. 说明入场/出场条件和仓位管理规则\n"
            "4. 分析策略的风险点和适用市场环境"
        ),
        "data_source": "stock_prices",
    },
    "financial_report": {
        "instruction_template": (
            "以下是{company_name}（{stock_code}）最新一期的关键财务数据:\n"
            "{financial_data}\n\n"
            "请对该公司进行深度财务分析，包括:\n"
            "1. 盈利能力分析（ROE、毛利率、净利率等）\n"
            "2. 偿债能力评估（资产负债率、流动比率等）\n"
            "3. 运营效率分析（应收账款周转率等）\n"
            "4. 同比/环比变化趋势及异常信号\n"
            "5. 综合投资价值评估"
        ),
        "data_source": "financial_reports",
    },
    "sentiment_analysis": {
        "instruction_template": (
            "请分析以下金融新闻的情感倾向和市场影响:\n\n"
            "标题: {title}\n"
            "内容: {content}\n\n"
            "要求:\n"
            "1. 判断情感倾向（积极/中性/消极）并给出置信度\n"
            "2. 识别涉及的公司/行业/政策\n"
            "3. 评估对相关股票/板块的潜在影响（方向和程度）\n"
            "4. 给出关键依据"
        ),
        "data_source": "news",
    },
    "financial_qa": {
        "instruction_template": "{question}",
        "data_source": "knowledge",
    },
    "risk_assessment": {
        "instruction_template": (
            "以下是{company_name}（{stock_code}）的财务和市场数据:\n"
            "{risk_data}\n\n"
            "请进行全面的风险评估:\n"
            "1. 信用风险评估（违约概率、财务健康度）\n"
            "2. 市场风险评估（波动率、Beta、VaR估算）\n"
            "3. 行业风险因素\n"
            "4. 综合风险等级评定（低/中/高/极高）\n"
            "5. 风险缓释建议"
        ),
        "data_source": "financial_reports",
    },
}

# 金融知识问题库（用于生成金融问答数据）
FINANCE_QUESTIONS = [
    "什么是市盈率（PE）？如何用市盈率判断股票估值是否合理？",
    "请解释股票市场中的做空机制是如何运作的？",
    "什么是夏普比率？如何用它来评估投资组合的表现？",
    "请解释A股市场的涨跌停板制度及其设计目的。",
    "什么是可转债？它的投资价值和风险有哪些？",
    "请解释什么是量化交易中的alpha和beta。",
    "CPI和PPI有什么区别？它们对股市有怎样的影响？",
    "什么是北向资金？为什么它被视为A股的风向标？",
    "请解释期权的四种基本策略（买入看涨、卖出看涨、买入看跌、卖出看跌）。",
    "什么是杜邦分析法？如何用它分析企业的ROE？",
    "什么是资产负债表中的商誉？商誉减值意味着什么？",
    "请解释什么是股票的技术分析和基本面分析，各自的优缺点是什么？",
    "什么是ETF基金？它与普通开放式基金有什么区别？",
    "请解释DCF（自由现金流折现）估值模型的基本原理和步骤。",
    "什么是融资融券？融资融券余额的变化对市场意味着什么？",
]

FINANCE_ANSWER_MAP = {
    "市盈率": (
        "市盈率（PE）= 每股价格 / 每股收益，也可以理解为总市值 / 净利润。"
        "它反映投资者愿意为 1 元盈利支付多少价格。使用时要结合行业、成长性、"
        "盈利周期和一次性损益判断，不能只看绝对高低。"
    ),
    "做空": (
        "做空是先借入证券卖出，再在未来买回归还，若价格下跌则获得差价收益。"
        "A股主要通过融券、股指期货或期权表达空头观点。风险在于价格上涨会放大亏损，"
        "还会面临保证金、强平和借券成本。"
    ),
    "夏普比率": (
        "夏普比率 =（组合收益率 - 无风险收益率）/ 组合波动率，用来衡量单位风险带来的超额收益。"
        "数值越高通常代表风险调整后表现越好，但它依赖历史收益分布，遇到极端尾部风险时需要结合最大回撤等指标。"
    ),
    "涨跌停板": (
        "涨跌停板是交易所对单日价格波动幅度设置的限制，A股主板通常为10%，部分板块或风险警示股票规则不同。"
        "它的目的在于抑制异常波动、给市场冷静时间，但也可能造成流动性不足和连续涨跌停。"
    ),
    "可转债": (
        "可转债兼具债券和股票期权属性，持有人可按约定价格转为股票。"
        "其价值由纯债价值、转股价值、转股溢价率、信用风险和利率环境共同决定。"
        "优势是下有债底、上有弹性，风险包括正股下跌、赎回、回售和信用风险。"
    ),
    "alpha": (
        "Alpha 表示投资组合相对基准或风险模型解释之外的超额收益，Beta 表示组合对市场系统性风险的敏感度。"
        "量化策略通常希望获取稳定 alpha，同时控制 beta 暴露、行业暴露和风格暴露。"
    ),
    "CPI": (
        "CPI 衡量居民消费价格变化，PPI 衡量工业生产者出厂价格变化。"
        "CPI 更接近终端消费通胀，PPI 更反映上游成本压力。二者会影响货币政策预期、企业利润率和市场估值。"
    ),
    "北向资金": (
        "北向资金指通过沪股通、深股通从香港市场流入A股的境外资金。"
        "它常被视为外资风险偏好和配置方向的观察指标，但短期流入流出也会受汇率、指数调整和全球风险偏好影响。"
    ),
    "期权": (
        "买入看涨适合看多并控制最大亏损，卖出看涨适合收取权利金但承担上涨风险；"
        "买入看跌用于看空或保护持仓，卖出看跌用于收取权利金但承担下跌接货风险。"
        "期权策略必须关注波动率、时间价值和保证金。"
    ),
    "杜邦": (
        "杜邦分析把 ROE 拆解为净利率、总资产周转率和权益乘数。"
        "它能说明企业回报来自盈利能力、运营效率还是财务杠杆，进一步帮助识别 ROE 改善的质量。"
    ),
    "商誉": (
        "商誉来自企业并购中支付价格超过可辨认净资产公允价值的部分。"
        "商誉减值通常意味着被收购资产盈利能力不及预期，会直接减少利润，并提示并购整合或估值过高风险。"
    ),
    "技术分析": (
        "技术分析侧重价格、成交量和技术指标，适合观察趋势和交易时点；基本面分析关注商业模式、财务质量、行业格局和估值。"
        "前者反应快但容易受噪声影响，后者逻辑更稳但验证周期较长。"
    ),
    "ETF": (
        "ETF 是在交易所上市交易的指数基金，通常跟踪特定指数或资产。"
        "相比普通开放式基金，ETF 可盘中交易、费率较低、持仓透明，但也存在跟踪误差和流动性差异。"
    ),
    "DCF": (
        "DCF 估值通过预测企业未来自由现金流，并用合适折现率折现到当前价值。"
        "核心步骤包括预测收入和利润、估算自由现金流、确定 WACC、计算终值并折现。"
        "模型对长期增速和折现率非常敏感。"
    ),
    "融资融券": (
        "融资是借钱买证券，融券是借证券卖出。融资融券余额反映杠杆资金参与程度，"
        "余额快速上升可能说明风险偏好增强，也可能积累回撤压力；余额下降通常代表杠杆收缩。"
    ),
}


def _to_float(value) -> Optional[float]:
    """将常见数值/百分比字符串转为 float。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"--", "-", "nan", "None"}:
        return None
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _fmt_num(value: Optional[float], suffix: str = "") -> str:
    """格式化财务/行情数值。"""
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1e8:
        return f"{value / 1e8:.2f}亿{suffix}"
    if abs_value >= 1e4:
        return f"{value / 1e4:.2f}万{suffix}"
    return f"{value:.2f}{suffix}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _pct_change(new_value: Optional[float], old_value: Optional[float]) -> Optional[float]:
    if new_value is None or old_value in (None, 0):
        return None
    return (new_value / old_value - 1) * 100


def _first_value(row: pd.Series, candidates: list) -> Optional[float]:
    for name in candidates:
        if name in row.index:
            value = _to_float(row.get(name))
            if value is not None:
                return value
    return None


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df.dropna(how="all")
    except Exception as e:
        logger.debug(f"读取 CSV 失败 {path}: {e}")
        return None


def _load_stock_names() -> dict:
    names = {}
    for path in [
        PROCESSED_DIR / "stock_prices" / "_hs300_stocks.csv",
        PROJECT_ROOT / "data" / "raw" / "stock_prices" / "_hs300_stocks.csv",
    ]:
        df = _safe_read_csv(path)
        if df is None:
            continue
        code_col = "code" if "code" in df.columns else "股票代码" if "股票代码" in df.columns else None
        name_col = "name" if "name" in df.columns else "股票名称" if "股票名称" in df.columns else None
        if not code_col or not name_col:
            continue
        for _, row in df.iterrows():
            code = str(row.get(code_col, "")).zfill(6)
            name = str(row.get(name_col, "")).strip()
            if code and name:
                names[code] = name
    return names


def _stock_daily_files(stock_dir: Path) -> list:
    files = []
    if not stock_dir.exists():
        return files
    for path in sorted(stock_dir.iterdir()):
        if path.is_dir() and (path / "daily.csv").exists():
            files.append((path.name.zfill(6), path / "daily.csv"))
        elif path.is_file() and path.name.endswith("_daily.csv"):
            files.append((path.stem.replace("_daily", "").zfill(6), path))
    return files


def _prepare_stock_window(df: pd.DataFrame, end_idx: int, window: int = 60) -> Optional[pd.DataFrame]:
    if end_idx < 30:
        return None
    start_idx = max(0, end_idx - window)
    recent = df.iloc[start_idx:end_idx].copy()
    if len(recent) < 30 or "close" not in recent.columns:
        return None
    recent["close"] = pd.to_numeric(recent["close"], errors="coerce")
    recent = recent.dropna(subset=["close"])
    if len(recent) < 30:
        return None
    return recent


def _stock_metrics(recent: pd.DataFrame) -> dict:
    last = recent.iloc[-1]
    first = recent.iloc[0]
    returns = recent["close"].pct_change().dropna()
    high = pd.to_numeric(recent.get("high", recent["close"]), errors="coerce")
    low = pd.to_numeric(recent.get("low", recent["close"]), errors="coerce")
    close = _to_float(last.get("close"))
    first_close = _to_float(first.get("close"))
    ma20 = _to_float(last.get("MA20"))
    macd = _to_float(last.get("MACD_DIF"))
    rsi = _to_float(last.get("RSI_6"))
    pct_period = _pct_change(close, first_close)
    volatility = float(returns.std() * math.sqrt(252) * 100) if len(returns) > 2 else None
    rolling_max = recent["close"].cummax()
    drawdown = ((recent["close"] / rolling_max) - 1).min() * 100 if len(recent) else None
    return {
        "close": close,
        "pct_period": pct_period,
        "volatility": volatility,
        "max_drawdown": drawdown,
        "support": _to_float(low.tail(20).min()),
        "resistance": _to_float(high.tail(20).max()),
        "ma20": ma20,
        "macd": macd,
        "rsi": rsi,
        "turnover": _to_float(last.get("turnover")),
    }


def _market_summary(recent: pd.DataFrame, metrics: dict) -> str:
    lines = []
    for _, row in recent.tail(10).iterrows():
        lines.append(
            "日期:{date}, 开:{open}, 高:{high}, 低:{low}, 收:{close}, 量:{volume}, 涨跌幅:{pct}%".format(
                date=row.get("date", row.get("日期", "")),
                open=row.get("open", ""),
                high=row.get("high", ""),
                low=row.get("low", ""),
                close=row.get("close", ""),
                volume=row.get("volume", ""),
                pct=row.get("pct_change", row.get("涨跌幅", "")),
            )
        )
    lines.append(
        "窗口指标: 区间涨跌幅={pct}, 年化波动率={vol}, 最大回撤={dd}, "
        "MA20={ma20}, MACD_DIF={macd}, RSI_6={rsi}, 支撑位={support}, 压力位={resistance}".format(
            pct=_fmt_pct(metrics["pct_period"]),
            vol=_fmt_pct(metrics["volatility"]),
            dd=_fmt_pct(metrics["max_drawdown"]),
            ma20=_fmt_num(metrics["ma20"]),
            macd=_fmt_num(metrics["macd"]),
            rsi=_fmt_num(metrics["rsi"]),
            support=_fmt_num(metrics["support"]),
            resistance=_fmt_num(metrics["resistance"]),
        )
    )
    return "\n".join(lines)


def _trend_label(metrics: dict) -> str:
    pct = metrics.get("pct_period")
    close = metrics.get("close")
    ma20 = metrics.get("ma20")
    if pct is not None and pct > 5 and (ma20 is None or close is None or close >= ma20):
        return "偏强上行"
    if pct is not None and pct < -5 and (ma20 is None or close is None or close < ma20):
        return "偏弱下行"
    return "震荡整理"


def _rule_stock_analysis(item: dict) -> str:
    metrics = item.get("_metrics", {})
    trend = _trend_label(metrics)
    return (
        f"1. K线与趋势：{item['stock_name']}（{item['stock_code']}）近窗口走势为{trend}，"
        f"区间涨跌幅约{_fmt_pct(metrics.get('pct_period'))}，当前收盘价约{_fmt_num(metrics.get('close'))}。\n"
        f"2. 技术指标：MA20约{_fmt_num(metrics.get('ma20'))}，MACD_DIF为{_fmt_num(metrics.get('macd'))}，"
        f"RSI_6为{_fmt_num(metrics.get('rsi'))}。若价格位于均线之上且MACD改善，说明多头动能较强；"
        f"若RSI过高则需要防范短线回落。\n"
        f"3. 关键价位：近20日支撑位可参考{_fmt_num(metrics.get('support'))}，压力位可参考"
        f"{_fmt_num(metrics.get('resistance'))}。跌破支撑位说明趋势转弱，突破压力位并放量则验证强势。\n"
        f"4. 操作参考：适合采用分批和止损纪律，关注成交量、均线斜率和市场风险偏好变化。"
        f"以上仅为基于历史行情的分析，不构成投资建议。"
    )


def _rule_quant_strategy(item: dict) -> str:
    strategy = item.get("strategy_type", "均线动量")
    return (
        f"策略名称：{strategy}策略。\n"
        "核心逻辑：使用趋势、波动率和成交量过滤信号，价格站上中期均线且短期均线上穿长期均线时做多，"
        "跌破长期均线或触发止损时退出。\n\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "def generate_signals(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    data = df.copy()\n"
        "    data['ret'] = data['close'].pct_change()\n"
        "    data['ma10'] = data['close'].rolling(10).mean()\n"
        "    data['ma20'] = data['close'].rolling(20).mean()\n"
        "    data['volatility'] = data['ret'].rolling(20).std()\n"
        "    data['signal'] = 0\n"
        "    long_cond = (data['ma10'] > data['ma20']) & (data['close'] > data['ma20'])\n"
        "    risk_cond = data['volatility'] < data['volatility'].rolling(60).quantile(0.8)\n"
        "    data.loc[long_cond & risk_cond, 'signal'] = 1\n"
        "    data['position'] = data['signal'].shift(1).fillna(0)\n"
        "    data['strategy_ret'] = data['position'] * data['ret']\n"
        "    return data\n\n"
        "def backtest_metrics(result: pd.DataFrame) -> dict:\n"
        "    ret = result['strategy_ret'].dropna()\n"
        "    nav = (1 + ret).cumprod()\n"
        "    annual_return = nav.iloc[-1] ** (252 / max(len(ret), 1)) - 1 if len(ret) else 0\n"
        "    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() and ret.std() > 0 else 0\n"
        "    max_drawdown = (nav / nav.cummax() - 1).min() if len(nav) else 0\n"
        "    return {'annual_return': annual_return, 'sharpe': sharpe, 'max_drawdown': max_drawdown}\n"
        "```\n\n"
        "入场条件：MA10上穿MA20、收盘价位于MA20上方且波动率未明显放大。出场条件：MA10下穿MA20、"
        "收盘价跌破MA20或回撤超过预设阈值。仓位管理建议单标的不超过20%，并设置固定止损。"
    )


def _rule_sentiment(item: dict) -> str:
    text = f"{item.get('title', '')} {item.get('content', '')}"
    positive_words = ["增长", "利好", "上涨", "中标", "突破", "回购", "增持", "盈利", "创新高", "走稳"]
    negative_words = ["下跌", "亏损", "处罚", "风险", "减持", "违约", "调查", "暴跌", "下滑"]
    pos = sum(1 for word in positive_words if word in text)
    neg = sum(1 for word in negative_words if word in text)
    if pos > neg:
        label, confidence, direction = "积极", "0.72", "偏正面"
    elif neg > pos:
        label, confidence, direction = "消极", "0.72", "偏负面"
    else:
        label, confidence, direction = "中性", "0.60", "影响有限"
    return (
        f"情感倾向：{label}，置信度：{confidence}。\n"
        f"关键依据：标题和正文中积极词命中{pos}次、消极词命中{neg}次，事件信息对市场解读为{direction}。\n"
        "涉及对象：需结合新闻中的公司、行业或政策主体进一步确认。\n"
        "市场影响：短期可能影响相关股票或板块情绪，但仍需结合成交量、基本面和大盘环境验证。"
    )


def _rule_financial_qa(item: dict) -> str:
    question = item.get("question", "")
    for keyword, answer in FINANCE_ANSWER_MAP.items():
        if keyword in question:
            return answer
    return (
        "这个问题需要从定义、计算方法、适用场景和风险限制四个层面理解。"
        "金融指标不能孤立使用，应结合行业特征、宏观环境、企业质量和估值水平综合判断。"
    )


def _financial_snapshot(stock_dir: Path) -> Optional[dict]:
    income = _safe_read_csv(stock_dir / "income_statement.csv")
    balance = _safe_read_csv(stock_dir / "balance_sheet.csv")
    cash_flow = _safe_read_csv(stock_dir / "cash_flow.csv")
    if income is None and balance is None and cash_flow is None:
        return None

    latest_income = income.iloc[0] if income is not None and len(income) else pd.Series(dtype=object)
    prev_income = income.iloc[1] if income is not None and len(income) > 1 else pd.Series(dtype=object)
    latest_balance = balance.iloc[0] if balance is not None and len(balance) else pd.Series(dtype=object)
    latest_cash = cash_flow.iloc[0] if cash_flow is not None and len(cash_flow) else pd.Series(dtype=object)

    revenue = _first_value(latest_income, ["营业收入", "营业总收入"])
    prev_revenue = _first_value(prev_income, ["营业收入", "营业总收入"])
    net_profit = _first_value(latest_income, ["净利润", "归属于母公司所有者的净利润"])
    prev_net_profit = _first_value(prev_income, ["净利润", "归属于母公司所有者的净利润"])
    total_assets = _first_value(latest_balance, ["资产总计", "资产合计"])
    total_liabilities = _first_value(latest_balance, ["负债合计", "负债总计"])
    equity = _first_value(latest_balance, ["所有者权益(或股东权益)合计", "所有者权益合计", "归属于母公司所有者权益合计"])
    current_assets = _first_value(latest_balance, ["流动资产合计"])
    current_liabilities = _first_value(latest_balance, ["流动负债合计"])
    operating_cash_flow = _first_value(latest_cash, ["经营活动产生的现金流量净额"])

    debt_ratio = total_liabilities / total_assets * 100 if total_liabilities is not None and total_assets else None
    roe = net_profit / equity * 100 if net_profit is not None and equity else None
    net_margin = net_profit / revenue * 100 if net_profit is not None and revenue else None
    current_ratio = current_assets / current_liabilities if current_assets is not None and current_liabilities else None
    report_date = str(
        latest_income.get("报告日", latest_balance.get("报告日", latest_cash.get("报告日", "")))
    )

    return {
        "report_date": report_date,
        "revenue": revenue,
        "revenue_growth": _pct_change(revenue, prev_revenue),
        "net_profit": net_profit,
        "net_profit_growth": _pct_change(net_profit, prev_net_profit),
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "equity": equity,
        "debt_ratio": debt_ratio,
        "roe": roe,
        "net_margin": net_margin,
        "current_ratio": current_ratio,
        "operating_cash_flow": operating_cash_flow,
    }


def _financial_data_text(snapshot: dict) -> str:
    return (
        f"报告期:{snapshot['report_date']}\n"
        f"营业收入:{_fmt_num(snapshot['revenue'])}, 收入环比/同比参考:{_fmt_pct(snapshot['revenue_growth'])}\n"
        f"净利润:{_fmt_num(snapshot['net_profit'])}, 净利润变化:{_fmt_pct(snapshot['net_profit_growth'])}\n"
        f"总资产:{_fmt_num(snapshot['total_assets'])}, 总负债:{_fmt_num(snapshot['total_liabilities'])}\n"
        f"资产负债率:{_fmt_pct(snapshot['debt_ratio'])}, ROE:{_fmt_pct(snapshot['roe'])}, "
        f"净利率:{_fmt_pct(snapshot['net_margin'])}, 流动比率:{_fmt_num(snapshot['current_ratio'])}\n"
        f"经营现金流净额:{_fmt_num(snapshot['operating_cash_flow'])}"
    )


def _rule_financial_report(item: dict) -> str:
    data = item.get("_snapshot", {})
    cash = data.get("operating_cash_flow")
    cash_comment = "经营现金流为正，利润质量有支撑" if cash is not None and cash >= 0 else "经营现金流偏弱，需要关注利润兑现质量"
    return (
        f"1. 盈利能力：报告期{data.get('report_date', '')}营业收入为{_fmt_num(data.get('revenue'))}，"
        f"净利润为{_fmt_num(data.get('net_profit'))}，净利率约{_fmt_pct(data.get('net_margin'))}，"
        f"ROE约{_fmt_pct(data.get('roe'))}。\n"
        f"2. 偿债能力：总资产{_fmt_num(data.get('total_assets'))}，总负债{_fmt_num(data.get('total_liabilities'))}，"
        f"资产负债率约{_fmt_pct(data.get('debt_ratio'))}，流动比率约{_fmt_num(data.get('current_ratio'))}。\n"
        f"3. 运营和现金流：{cash_comment}，经营活动现金流净额为{_fmt_num(cash)}。\n"
        "4. 关注点：若收入和净利润同步改善且现金流匹配，财务质量较好；若负债率上升、现金流与利润背离，"
        "需要警惕偿债压力和盈利质量下降。\n"
        "5. 综合评价：该结论基于公开财务数据的结构化分析，仅供研究参考。"
    )


def _rule_risk_assessment(item: dict) -> str:
    data = item.get("_snapshot", {})
    market = item.get("_market_metrics", {})
    risk_points = 0
    if data.get("debt_ratio") is not None and data["debt_ratio"] > 70:
        risk_points += 2
    if data.get("current_ratio") is not None and data["current_ratio"] < 1:
        risk_points += 1
    if data.get("operating_cash_flow") is not None and data["operating_cash_flow"] < 0:
        risk_points += 1
    if market.get("volatility") is not None and market["volatility"] > 40:
        risk_points += 1
    if market.get("max_drawdown") is not None and market["max_drawdown"] < -25:
        risk_points += 1
    level = "低" if risk_points <= 1 else "中" if risk_points <= 3 else "高"
    return (
        f"综合风险等级：{level}。\n"
        f"1. 信用风险：资产负债率约{_fmt_pct(data.get('debt_ratio'))}，流动比率约"
        f"{_fmt_num(data.get('current_ratio'))}，需关注短期偿债能力和债务结构。\n"
        f"2. 市场风险：年化波动率约{_fmt_pct(market.get('volatility'))}，最大回撤约"
        f"{_fmt_pct(market.get('max_drawdown'))}，价格波动会影响持仓风险敞口。\n"
        f"3. 经营风险：经营现金流净额为{_fmt_num(data.get('operating_cash_flow'))}，"
        "若持续弱于净利润，应关注回款和利润质量。\n"
        "4. 风险缓释：建议设置止损线、控制单一标的仓位、跟踪现金流和负债率变化，必要时用分散配置或对冲工具降低组合波动。"
    )


def _rule_response(task_name: str, item: dict) -> str:
    handlers = {
        "stock_analysis": _rule_stock_analysis,
        "quant_strategy": _rule_quant_strategy,
        "financial_report": _rule_financial_report,
        "sentiment_analysis": _rule_sentiment,
        "financial_qa": _rule_financial_qa,
        "risk_assessment": _rule_risk_assessment,
    }
    return handlers[task_name](item)


async def call_llm_api(
    system_prompt: str,
    user_prompt: str,
    cache_key: str,
) -> Optional[str]:
    """
    调用 LLM API 生成回答（异步，带缓存和重试）。

    参数:
        system_prompt: 系统提示词
        user_prompt:   用户输入
        cache_key:     缓存键

    返回:
        模型生成的回答文本，失败返回 None
    """
    # 先查缓存
    cached = _get_cached(cache_key)
    if cached:
        return cached

    import aiohttp

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        _save_cache(cache_key, content)
                        return content
                    elif resp.status == 429:
                        # 限流，等待后重试
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"API限流，等待 {wait_time}s 后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await resp.text()
                        logger.error(f"API调用失败 (HTTP {resp.status}): {error_text}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"API请求超时 (第{attempt+1}次)")
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return None

    return None


def prepare_stock_analysis_inputs(data_dir: Path, max_samples: int = 500) -> list:
    """
    从行情数据中准备股票分析任务的输入。

    参数:
        data_dir:    清洗后的行情数据目录
        max_samples: 最大样本数

    返回:
        [{"stock_code": ..., "stock_name": ..., "market_data": ...}, ...]
    """
    inputs = []
    stock_dir = data_dir / "stock_prices"
    if not stock_dir.exists():
        logger.warning(f"行情数据目录不存在: {stock_dir}")
        return inputs

    stock_names = _load_stock_names()
    for stock_code, daily_file in _stock_daily_files(stock_dir):

        try:
            df = pd.read_csv(daily_file)
            if len(df) < 30:
                continue  # 数据太少跳过
            step = max(20, len(df) // 4)
            end_indices = list(range(60, len(df) + 1, step))
            if len(df) not in end_indices:
                end_indices.append(len(df))

            for end_idx in end_indices:
                recent = _prepare_stock_window(df, end_idx, window=60)
                if recent is None:
                    continue
                metrics = _stock_metrics(recent)
                inputs.append({
                    "stock_code": stock_code,
                    "stock_name": stock_names.get(stock_code, stock_code),
                    "market_data": _market_summary(recent, metrics),
                    "_metrics": metrics,
                })
                if len(inputs) >= max_samples:
                    break

        except Exception as e:
            logger.debug(f"处理 {stock_code} 时出错: {e}")

        if len(inputs) >= max_samples:
            break

    return inputs


def prepare_quant_strategy_inputs(data_dir: Path, max_samples: int = 500) -> list:
    """从行情数据中准备量化策略任务输入。"""
    inputs = []
    stock_dir = data_dir / "stock_prices"
    strategy_types = ["均线动量", "波动率过滤", "趋势跟踪", "均值回归", "量价配合"]
    for idx, (stock_code, daily_file) in enumerate(_stock_daily_files(stock_dir)):
        try:
            df = pd.read_csv(daily_file)
            recent = _prepare_stock_window(df, len(df), window=120)
            if recent is None:
                continue
            metrics = _stock_metrics(recent)
            context = (
                f"标的代码:{stock_code}\n"
                f"样本长度:{len(recent)}个交易日\n"
                f"区间涨跌幅:{_fmt_pct(metrics['pct_period'])}, 年化波动率:{_fmt_pct(metrics['volatility'])}, "
                f"最大回撤:{_fmt_pct(metrics['max_drawdown'])}\n"
                f"MA20:{_fmt_num(metrics['ma20'])}, MACD_DIF:{_fmt_num(metrics['macd'])}, RSI_6:{_fmt_num(metrics['rsi'])}"
            )
            inputs.append({
                "market_context": context,
                "strategy_type": strategy_types[idx % len(strategy_types)],
                "_metrics": metrics,
            })
        except Exception as e:
            logger.debug(f"准备量化策略输入失败 {daily_file}: {e}")
        if len(inputs) >= max_samples:
            break
    return inputs


def prepare_financial_report_inputs(data_dir: Path, max_samples: int = 500) -> list:
    """从财务报表目录准备财报解读任务输入。"""
    inputs = []
    report_dir = data_dir / "financial_reports"
    stock_names = _load_stock_names()
    if not report_dir.exists():
        logger.warning(f"财务报表目录不存在: {report_dir}")
        return inputs

    for stock_dir in sorted(report_dir.iterdir()):
        if not stock_dir.is_dir():
            continue
        snapshot = _financial_snapshot(stock_dir)
        if not snapshot:
            continue
        stock_code = stock_dir.name.zfill(6)
        inputs.append({
            "stock_code": stock_code,
            "company_name": stock_names.get(stock_code, stock_code),
            "financial_data": _financial_data_text(snapshot),
            "_snapshot": snapshot,
        })
        if len(inputs) >= max_samples:
            break
    return inputs


def prepare_risk_assessment_inputs(data_dir: Path, max_samples: int = 500) -> list:
    """从财报和行情数据中准备风险评估任务输入。"""
    inputs = []
    report_dir = data_dir / "financial_reports"
    stock_dir = data_dir / "stock_prices"
    stock_names = _load_stock_names()
    if not report_dir.exists():
        logger.warning(f"财务报表目录不存在: {report_dir}")
        return inputs

    stock_file_map = dict(_stock_daily_files(stock_dir))
    for company_dir in sorted(report_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        stock_code = company_dir.name.zfill(6)
        snapshot = _financial_snapshot(company_dir)
        if not snapshot:
            continue
        market_metrics = {}
        daily_file = stock_file_map.get(stock_code)
        if daily_file:
            df = _safe_read_csv(daily_file)
            recent = _prepare_stock_window(df, len(df), window=120) if df is not None else None
            market_metrics = _stock_metrics(recent) if recent is not None else {}
        risk_data = _financial_data_text(snapshot)
        if market_metrics:
            risk_data += (
                f"\n市场波动率:{_fmt_pct(market_metrics.get('volatility'))}, "
                f"最大回撤:{_fmt_pct(market_metrics.get('max_drawdown'))}, "
                f"支撑位:{_fmt_num(market_metrics.get('support'))}, 压力位:{_fmt_num(market_metrics.get('resistance'))}"
            )
        inputs.append({
            "stock_code": stock_code,
            "company_name": stock_names.get(stock_code, stock_code),
            "risk_data": risk_data,
            "_snapshot": snapshot,
            "_market_metrics": market_metrics,
        })
        if len(inputs) >= max_samples:
            break
    return inputs


def prepare_sentiment_inputs(data_dir: Path, max_samples: int = 500) -> list:
    """从新闻数据中准备情感分析任务的输入。"""
    inputs = []
    news_dir = data_dir / "news"
    if not news_dir.exists():
        return inputs

    for json_file in news_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                news_list = json.load(f)
            for item in news_list:
                title = item.get("title", "")
                content = item.get("content", "")
                if title and len(title) > 10:
                    inputs.append({"title": title, "content": content[:500]})
                if len(inputs) >= max_samples:
                    break
        except Exception:
            continue
        if len(inputs) >= max_samples:
            break

    return inputs


async def synthesize_task(
    task_name: str,
    inputs: list,
    system_prompt: str,
    template: str,
    output_file: Path,
) -> int:
    """
    为单个任务批量合成指令数据。

    参数:
        task_name:     任务名称
        inputs:        输入数据列表
        system_prompt: 系统提示词
        template:      指令模板
        output_file:   输出文件路径

    返回:
        成功合成的条数
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = []

    async def process_one(item: dict, idx: int):
        async with semaphore:
            try:
                user_prompt = template.format(**item)
            except KeyError as e:
                logger.debug(f"模板填充失败: {e}")
                return None

            ck = _cache_key(task_name, user_prompt)
            response = await call_llm_api(system_prompt, user_prompt, ck)
            if response:
                return {
                    "conversations": [
                        {"from": "human", "value": user_prompt},
                        {"from": "gpt", "value": response},
                    ],
                    "system": system_prompt,
                    "task_type": task_name,
                }
            return None

    tasks = [process_one(item, i) for i, item in enumerate(inputs)]
    results_raw = await tqdm_asyncio.gather(*tasks, desc=f"合成-{task_name}")

    results = [r for r in results_raw if r is not None]

    # 保存到文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_name}] 合成完成: {len(results)}/{len(inputs)} 条")
    return len(results)


def build_rule_based_task(
    task_name: str,
    inputs: list,
    system_prompt: str,
    template: str,
    output_file: Path,
) -> int:
    """使用本地规则生成 ShareGPT 样本，作为无 API 时的默认路径。"""
    results = []
    for item in inputs:
        try:
            user_prompt = template.format(**item)
            assistant_text = _rule_response(task_name, item)
        except Exception as e:
            logger.debug(f"[{task_name}] 规则样本生成失败: {e}")
            continue
        results.append({
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_text},
            ],
            "system": system_prompt,
            "task_type": task_name,
            "source": "rule_synthesis",
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_name}] 规则生成完成: {len(results)}/{len(inputs)} 条 -> {output_file.name}")
    return len(results)


def prepare_task_inputs(task_name: str, max_per_task: int) -> list:
    """按任务准备输入数据。"""
    if task_name == "stock_analysis":
        return prepare_stock_analysis_inputs(PROCESSED_DIR, max_per_task)
    if task_name == "quant_strategy":
        return prepare_quant_strategy_inputs(PROCESSED_DIR, max_per_task)
    if task_name == "financial_report":
        return prepare_financial_report_inputs(PROCESSED_DIR, max_per_task)
    if task_name == "sentiment_analysis":
        return prepare_sentiment_inputs(PROCESSED_DIR, max_per_task)
    if task_name == "financial_qa":
        return [{"question": q} for q in FINANCE_QUESTIONS[:max_per_task]]
    if task_name == "risk_assessment":
        return prepare_risk_assessment_inputs(PROCESSED_DIR, max_per_task)
    return []


async def run_synthesis(
    tasks: Optional[list] = None,
    max_per_task: int = 500,
    mode: str = "auto",
) -> None:
    """
    运行完整的数据合成流程。

    参数:
        tasks:        要合成的任务列表，None=全部
        max_per_task: 每个任务最大合成数量
    """
    if mode not in {"auto", "rules", "llm"}:
        raise ValueError("mode 必须为 auto / rules / llm")
    use_llm = mode == "llm" or (mode == "auto" and bool(API_KEY))
    if mode == "llm" and not API_KEY:
        logger.error("指定 --mode llm 但未设置 SYNTH_API_KEY")
        return
    if not API_KEY and mode == "auto":
        logger.info("未设置 SYNTH_API_KEY，使用本地规则生成样本")

    prompts = _load_system_prompts()
    all_tasks = list(TASK_TEMPLATES.keys()) if tasks is None else tasks

    logger.info(f"开始合成数据，任务列表: {all_tasks}")
    logger.info(f"每任务最大样本数: {max_per_task}")
    logger.info(f"生成模式: {'llm' if use_llm else 'rules'}")
    if use_llm:
        logger.info(f"API: {API_BASE_URL}, 模型: {API_MODEL}\n")

    total = 0
    for task_name in all_tasks:
        if task_name not in TASK_TEMPLATES:
            logger.warning(f"未知任务: {task_name}，跳过")
            continue

        config = TASK_TEMPLATES[task_name]
        prompt_info = prompts.get(task_name, prompts["general"])
        system_prompt = prompt_info["system_prompt"]
        template = config["instruction_template"]

        inputs = prepare_task_inputs(task_name, max_per_task)

        if not inputs:
            logger.warning(f"[{task_name}] 无可用输入数据，跳过")
            continue

        output_file = SYNTHESIZED_DIR / f"{task_name}.json"
        if use_llm:
            count = await synthesize_task(
                task_name=task_name,
                inputs=inputs,
                system_prompt=system_prompt,
                template=template,
                output_file=output_file,
            )
        else:
            count = build_rule_based_task(
                task_name=task_name,
                inputs=inputs,
                system_prompt=system_prompt,
                template=template,
                output_file=output_file,
            )
        total += count

    logger.info(f"\n合成全部完成，共生成 {total} 条指令数据")
    logger.info(f"输出目录: {SYNTHESIZED_DIR}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM 指令数据合成")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list(TASK_TEMPLATES.keys()),
        help="指定要合成的任务（默认全部）",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=500,
        help="每个任务最大合成样本数 (默认: 500)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "rules", "llm"],
        default="auto",
        help="生成模式: auto=有API走LLM否则规则, rules=本地规则, llm=强制API",
    )
    args = parser.parse_args()

    asyncio.run(run_synthesis(tasks=args.tasks, max_per_task=args.max_per_task, mode=args.mode))
