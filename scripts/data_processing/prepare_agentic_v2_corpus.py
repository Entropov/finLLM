#!/usr/bin/env python3
"""Prepare issuer-disjoint audit documents, collection prompts, and trusted gold."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag.audit_schema import EvidenceRecord, canonicalize_url, content_digest  # noqa: E402


TASKS = (
    "stock_analysis",
    "quant_strategy",
    "financial_report",
    "sentiment_analysis",
    "financial_qa",
    "risk_assessment",
)


@dataclass(frozen=True)
class IssuerSnapshot:
    code: str
    name: str
    report_date: str
    previous_report_date: str
    announcement_at: str
    fetched_at: str
    revenue: float
    previous_revenue: float
    net_profit: float
    previous_net_profit: float
    operating_cash_flow: float
    assets: float
    liabilities: float
    current_assets: float
    current_liabilities: float
    latest_market_date: str
    latest_close: float
    return_20d: float
    annualized_volatility_60d: float
    max_drawdown_60d: float
    ma5: float
    ma20: float
    rsi6: float | None
    market_fetched_at: str

    @property
    def revenue_growth(self) -> float:
        return _growth(self.revenue, self.previous_revenue)

    @property
    def profit_growth(self) -> float:
        return _growth(self.net_profit, self.previous_net_profit)

    @property
    def debt_ratio(self) -> float:
        return self.liabilities / self.assets if self.assets else math.nan

    @property
    def current_ratio(self) -> float:
        return self.current_assets / self.current_liabilities if self.current_liabilities else math.nan

    @property
    def sentiment(self) -> str:
        profit_positive = (self.previous_net_profit <= 0 < self.net_profit) or self.profit_growth > 0.03
        profit_negative = (self.previous_net_profit >= 0 > self.net_profit) or self.profit_growth < -0.03
        positives = sum((self.revenue_growth > 0.03, profit_positive, self.operating_cash_flow > 0))
        negatives = sum((self.revenue_growth < -0.03, profit_negative, self.operating_cash_flow < 0))
        if positives >= 2 and negatives == 0:
            return "positive"
        if negatives >= 2 and positives == 0:
            return "negative"
        return "neutral"


def _growth(current: float, previous: float) -> float:
    if previous <= 0 or not math.isfinite(previous):
        return math.nan
    return current / previous - 1.0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(row: pd.Series, candidates: tuple[str, ...]) -> float | None:
    for name in candidates:
        if name in row:
            number = _finite(row[name])
            if number is not None:
                return number
    return None


def _iso_date(value: Any, *, end_of_day: bool = False) -> str:
    digits = re.sub(r"\D", "", str(value or ""))[:8]
    if len(digits) != 8:
        raise ValueError(f"invalid date: {value!r}")
    suffix = "T23:59:59+08:00" if end_of_day else "T00:00:00+08:00"
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}{suffix}"


def _file_observed_at(paths: list[Path]) -> str:
    observed = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(observed, tz=timezone.utc).isoformat(timespec="seconds")


def _matching_previous(frame: pd.DataFrame, report_date: str) -> pd.Series | None:
    target = str(int(report_date) - 10000)
    matches = frame[frame["报告日"].astype(str).str.replace(r"\.0$", "", regex=True) == target]
    if not matches.empty:
        return matches.iloc[0]
    return frame.iloc[1] if len(frame) > 1 else None


def load_snapshot(code: str, name: str, report_dir: Path, price_dir: Path) -> IssuerSnapshot | None:
    paths = {
        "income": report_dir / code / "income_statement.csv",
        "balance": report_dir / code / "balance_sheet.csv",
        "cash": report_dir / code / "cash_flow.csv",
        "price": price_dir / f"{code}_daily.csv",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        income = pd.read_csv(paths["income"])
        balance = pd.read_csv(paths["balance"])
        cash = pd.read_csv(paths["cash"])
        price = pd.read_csv(paths["price"])
        if min(len(income), len(balance), len(cash)) < 2 or len(price) < 60:
            return None
        for frame in (income, balance, cash):
            frame.sort_values("报告日", ascending=False, inplace=True)
        current_income = income.iloc[0]
        report_date = str(current_income["报告日"]).replace(".0", "")
        previous_income = _matching_previous(income, report_date)
        current_balance = balance[balance["报告日"].astype(str).str.replace(r"\.0$", "", regex=True) == report_date]
        current_cash = cash[cash["报告日"].astype(str).str.replace(r"\.0$", "", regex=True) == report_date]
        if previous_income is None or current_balance.empty or current_cash.empty:
            return None
        current_balance_row = current_balance.iloc[0]
        current_cash_row = current_cash.iloc[0]
        fields = {
            "revenue": _first_number(current_income, ("营业总收入", "营业收入")),
            "previous_revenue": _first_number(previous_income, ("营业总收入", "营业收入")),
            "net_profit": _first_number(current_income, ("归属于母公司所有者的净利润", "净利润")),
            "previous_net_profit": _first_number(previous_income, ("归属于母公司所有者的净利润", "净利润")),
            "operating_cash_flow": _first_number(
                current_cash_row,
                ("经营活动产生的现金流量净额", "经营活动现金流量净额"),
            ),
            "assets": _first_number(current_balance_row, ("资产总计",)),
            "liabilities": _first_number(current_balance_row, ("负债合计",)),
            "current_assets": _first_number(current_balance_row, ("流动资产合计",)),
            "current_liabilities": _first_number(current_balance_row, ("流动负债合计",)),
        }
        if any(value is None for value in fields.values()):
            return None
        price["date"] = pd.to_datetime(price["date"], errors="coerce")
        price["close"] = pd.to_numeric(price["close"], errors="coerce")
        price = price.dropna(subset=["date", "close"]).sort_values("date").tail(60)
        if len(price) < 60:
            return None
        returns = price["close"].pct_change().dropna()
        running_peak = price["close"].cummax()
        drawdowns = price["close"] / running_peak - 1.0
        latest = price.iloc[-1]
        close_20_sessions_ago = float(price.iloc[-21]["close"])
        report_files = [paths["income"], paths["balance"], paths["cash"]]
        announcement = current_income.get("公告日期") or current_balance_row.get("公告日期")
        return IssuerSnapshot(
            code=code,
            name=name,
            report_date=_iso_date(report_date),
            previous_report_date=_iso_date(previous_income["报告日"]),
            announcement_at=_iso_date(announcement, end_of_day=True),
            fetched_at=_file_observed_at(report_files),
            latest_market_date=latest["date"].strftime("%Y-%m-%dT15:00:00+08:00"),
            latest_close=float(latest["close"]),
            return_20d=float(latest["close"] / close_20_sessions_ago - 1.0),
            annualized_volatility_60d=float(returns.std(ddof=1) * math.sqrt(252)),
            max_drawdown_60d=float(drawdowns.min()),
            ma5=float(price["close"].tail(5).mean()),
            ma20=float(price["close"].tail(20).mean()),
            rsi6=_finite(latest.get("RSI_6")),
            market_fetched_at=_file_observed_at([paths["price"]]),
            **fields,
        )
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _billions(value: float) -> str:
    return f"{value / 1e8:.4f}亿元"


def _percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def _profit_change(snapshot: IssuerSnapshot) -> str:
    if snapshot.previous_net_profit <= 0 < snapshot.net_profit:
        return "净利润由亏损转为盈利"
    if snapshot.previous_net_profit >= 0 > snapshot.net_profit:
        return "净利润由盈利转为亏损"
    if math.isfinite(snapshot.profit_growth):
        return f"净利润同比变化{_percent(snapshot.profit_growth)}"
    return "净利润同比百分比因比较基数不适用而未计算"


def financial_quote(snapshot: IssuerSnapshot) -> str:
    return (
        f"{snapshot.name}（证券代码{snapshot.code}）报告期{snapshot.report_date[:10]}的营业收入为"
        f"{_billions(snapshot.revenue)}，归母口径净利润为{_billions(snapshot.net_profit)}，经营活动现金流量净额为"
        f"{_billions(snapshot.operating_cash_flow)}；同比可比期{snapshot.previous_report_date[:10]}的营业收入为"
        f"{_billions(snapshot.previous_revenue)}，归母口径净利润为{_billions(snapshot.previous_net_profit)}。"
        f"营业收入同比变化{_percent(snapshot.revenue_growth)}，{_profit_change(snapshot)}。"
        f"同期资产总计{_billions(snapshot.assets)}、负债合计{_billions(snapshot.liabilities)}、流动资产合计"
        f"{_billions(snapshot.current_assets)}、流动负债合计{_billions(snapshot.current_liabilities)}。"
        f"按负债合计除以资产总计计算，资产负债率为{_percent(snapshot.debt_ratio)}；按流动资产合计除以流动负债合计计算，"
        f"流动比率为{snapshot.current_ratio:.4f}。"
    )


def market_quote(snapshot: IssuerSnapshot) -> str:
    rsi = f"，RSI6为{snapshot.rsi6:.4f}" if snapshot.rsi6 is not None else ""
    return (
        f"{snapshot.name}（证券代码{snapshot.code}）截至{snapshot.latest_market_date[:10]}收盘价为"
        f"{snapshot.latest_close:.4f}元，近20个交易日收益率为{_percent(snapshot.return_20d)}。"
        f"最近60个交易日按日收益率标准差乘以根号252计算的年化波动率为{_percent(snapshot.annualized_volatility_60d)}，"
        f"按收盘价相对历史滚动高点计算的最大回撤为{_percent(snapshot.max_drawdown_60d)}。"
        f"MA5为{snapshot.ma5:.4f}元，MA20为{snapshot.ma20:.4f}元{rsi}。"
    )


def _document(snapshot: IssuerSnapshot, kind: str) -> tuple[str, dict[str, str]]:
    if kind == "financial":
        url = f"https://money.finance.sina.com.cn/corp/go.php/vFD_FinancialGuideLine/stockid/{snapshot.code}/displaytype/4.phtml"
        title = f"{snapshot.name} {snapshot.code} 财务报表可审计摘要"
        published_at = snapshot.announcement_at
        effective_at = snapshot.report_date
        fetched_at = snapshot.fetched_at
        publisher = "新浪财经"
        quote = financial_quote(snapshot)
    else:
        market = "1" if snapshot.code.startswith(("6", "9")) else "0"
        url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{snapshot.code}&klt=101"
        title = f"{snapshot.name} {snapshot.code} 行情指标可审计摘要"
        published_at = snapshot.latest_market_date
        effective_at = snapshot.latest_market_date
        fetched_at = snapshot.market_fetched_at
        publisher = "东方财富"
        quote = market_quote(snapshot)
    content = (
        f"# {title}\n\n"
        f"- 来源: [{publisher}]({url})\n"
        f"- 发布者: {publisher}\n"
        f"- 发布时间: {published_at}\n"
        f"- 数据截至: {effective_at}\n"
        f"- 抓取时间: {fetched_at}\n\n"
        f"## 可引用事实\n\n{quote}\n"
    )
    return content, {
        "url": canonicalize_url(url),
        "title": title,
        "publisher": publisher,
        "published_at": published_at,
        "effective_at": effective_at,
        "fetched_at": fetched_at,
        "quote": quote,
        "document_version": content_digest(content)[:16],
    }


def _write_document(output_dir: Path, snapshot: IssuerSnapshot, kind: str) -> dict[str, str]:
    content, metadata = _document(snapshot, kind)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{snapshot.code}_{kind}.md"
    path.write_text(content, encoding="utf-8")
    return {**metadata, "path": str(path.relative_to(PROJECT_ROOT))}


def _training_question(task: str, snapshot: IssuerSnapshot, variant: int) -> str:
    instructions = {
        "stock_analysis": (
            "结合最新收盘价、20日收益、均线、波动与财务基本面，分析趋势和主要风险。",
            "从价格趋势、波动率、回撤和盈利变化四个角度评价当前状态。",
            "比较短期均线位置与20日表现，并结合盈利、现金流解释走势是否得到基本面支持。",
            "按数据时间列出价格、均线和财务事实，再判断短期趋势、波动和回撤风险。",
            "分别核对行情证据与财报证据，说明趋势信号是否一致并披露主要不确定性。",
            "只依据可引用指标分析价格动量、盈利变化和流动性风险，不补充行业或估值假设。",
        ),
        "quant_strategy": (
            "基于MA5、MA20和近60日风险指标设计研究用策略；给出可运行Python函数、signal、入场出场、仓位止损、Sharpe与回撤评估。",
            "设计一个避免前视偏差的均线研究策略，必须包含Python代码、信号、仓位、交易成本、夏普和最大回撤。",
            "以MA5和MA20为输入构造可复现回测框架，给出Python函数及signal、换仓、成本、止损、Sharpe和drawdown定义。",
            "构建只使用历史DataFrame的均线策略函数，完整定义滞后信号、仓位、成本、止损、Sharpe和最大回撤。",
            "根据可用行情字段设计无前视的研究策略，代码不得硬编码证券价格或虚构回测绩效。",
            "给出可运行的Python回测骨架，明确入场、出场、仓位约束及风险指标计算，并披露证据边界。",
        ),
        "financial_report": (
            "解读最新报告期的收入、净利润、经营现金流、杠杆与流动性，并指出同比变化和局限。",
            "概括最新财报核心指标，对比同比收入和净利润，评价现金流、资产负债率与流动比率。",
            "核对最新期收入、利润、现金流与资产负债结构，区分报告期和公告时间并披露同比基数问题。",
            "逐项复核营收、利润、经营现金流、杠杆与流动性，不推测证据没有披露的变化原因。",
            "按报告期整理损益、现金流和资产负债指标，并说明同比方向与数据适用边界。",
            "形成简明财务摘要，确保每个数值保持原精度并将解释性判断与事实分开。",
        ),
        "sentiment_analysis": (
            "根据收入、净利润和经营现金流证据，将基本面情绪判为positive、neutral或negative并说明依据。",
            "只在positive、neutral、negative中选择一个基本面情绪标签，再用数据解释。",
            "依据收入同比、盈利方向和经营现金流输出英文情绪标签positive/neutral/negative，并列明判定证据。",
            "先列收入、利润、现金流及同比事实，再从positive、neutral、negative中给出唯一标签。",
            "用可核验财务指标判断基本面情绪，不使用新闻语气或股价涨跌替代财务证据。",
            "对盈利与现金流信号进行一致性检查后输出英文情绪标签，并明确混合信号。",
        ),
        "financial_qa": (
            "解释资产负债率与流动比率的公式、含义和局限，并用最新报告期数据核对结果。",
            "什么是资产负债率和流动比率？用该公司最新数据举例，说明两者不能单独代表偿债能力的原因。",
            "核验负债合计/资产总计与流动资产/流动负债两个计算，解释结果对长期和短期偿债分析的边界。",
            "使用最新负债、资产和流动项目复算两项比率，并区分公式事实与解释性结论。",
            "说明资产负债率和流动比率各自回答什么问题，逐项引用本期输入值与计算结果。",
            "核对长期杠杆和短期流动性指标，解释在缺少资产质量信息时应如何限制结论。",
        ),
        "risk_assessment": (
            "基于盈利、现金流、资产负债率、流动比率、波动率和最大回撤评估风险等级与缓释措施。",
            "识别基本面、流动性和市场风险，给出有数据依据的风险排序及控制建议。",
            "建立高、中、低风险判断，覆盖盈利质量、杠杆、流动性、年化波动和最大回撤，并提出缓释动作。",
            "按财务风险、市场风险和证据局限组织评估，给出可监测但不含买卖指令的缓释措施。",
            "核验现金流、偿债和价格风险指标后给出风险等级，每个判断都绑定对应证据。",
            "对盈利、杠杆、流动性、波动和回撤进行分项审计，避免使用证据外阈值或行业均值。",
        ),
    }[task]
    request = instructions[variant % len(instructions)]
    return (
        f"针对{snapshot.name}（{snapshot.code}），{request}"
        "每项事实或计算结论必须引用检索返回的Evidence ID；优先复述证据原句，证据不足时明确拒答，不作收益保证。"
    )


def build_collection_requests(snapshots: list[IssuerSnapshot], per_task: int) -> list[dict[str, Any]]:
    rows = []
    for task_index, task in enumerate(TASKS):
        for index in range(per_task):
            snapshot = snapshots[(index * 17 + task_index * 11) % len(snapshots)]
            variant = index // len(snapshots)
            rows.append(
                {
                    "id": f"train-{task}-{index + 1:04d}",
                    "task_type": task,
                    "issuer_code": snapshot.code,
                    "issuer_name": snapshot.name,
                    "question": _training_question(task, snapshot, variant),
                }
            )
    return rows


def _gold_evidence(snapshot: IssuerSnapshot, task: str, question: str) -> list[dict[str, Any]]:
    kinds = {
        "financial_report": ("financial",),
        "sentiment_analysis": ("financial",),
        "financial_qa": ("financial",),
        "stock_analysis": ("financial", "market"),
        "quant_strategy": ("market",),
        "risk_assessment": ("financial", "market"),
    }[task]
    records = []
    for rank, kind in enumerate(kinds, start=1):
        _, metadata = _document(snapshot, kind)
        record = EvidenceRecord.from_retrieved_doc(
            {
                "content": metadata["quote"],
                "metadata": {
                    "canonical_url": metadata["url"],
                    "publisher": metadata["publisher"],
                    "source_type": "news",
                    "reliability_tier": "unverified_secondary",
                    "published_at": metadata["published_at"],
                    "effective_at": metadata["effective_at"],
                    "fetched_at": metadata["fetched_at"],
                    "document_version": metadata["document_version"],
                    "title": metadata["title"],
                    "chunk_index": 0,
                    "start_char": 0,
                    "end_char": len(metadata["quote"]),
                },
            },
            rank=rank,
            retrieval_query=question,
            observed_at="2026-09-01T00:00:00+08:00",
        )
        records.append(record.to_dict())
    return records


def _gold_scoring(snapshot: IssuerSnapshot, task: str) -> dict[str, Any]:
    if task == "sentiment_analysis":
        return {"mode": "label", "accepted_labels": [snapshot.sentiment]}
    patterns = {
        "financial_report": [re.escape(snapshot.code), re.escape(_billions(snapshot.revenue)), re.escape(_billions(snapshot.net_profit))],
        "stock_analysis": [re.escape(f"{snapshot.latest_close:.4f}"), r"趋势|均线", r"风险|波动|回撤"],
        "quant_strategy": [r"```(?:python)?", r"def\s+", r"signal|信号", r"Sharpe|夏普", r"drawdown|回撤"],
        "financial_qa": [r"资产负债率", re.escape(_percent(snapshot.debt_ratio)), r"流动比率", re.escape(f"{snapshot.current_ratio:.4f}")],
        "risk_assessment": [r"风险", re.escape(_percent(snapshot.debt_ratio)), re.escape(_percent(snapshot.annualized_volatility_60d)), r"缓释|控制|降低"],
    }[task]
    return {"mode": "patterns", "required_patterns": patterns, "forbidden_patterns": [r"保证收益|稳赚|必涨"], "pass_threshold": 0.66}


def build_trusted_gold(snapshots: list[IssuerSnapshot], per_task: int) -> list[dict[str, Any]]:
    rows = []
    for task_index, task in enumerate(TASKS):
        for index in range(per_task):
            snapshot = snapshots[(index * 7 + task_index * 3) % len(snapshots)]
            question = _training_question(task, snapshot, index // len(snapshots))
            rows.append(
                {
                    "id": f"heldout-{task}-{index + 1:04d}",
                    "task_type": task,
                    "question": question,
                    "request_as_of": "2026-09-01T00:00:00+08:00",
                    "source_group": f"heldout-issuer-{snapshot.code}",
                    "requires_audit": True,
                    "evidence": _gold_evidence(snapshot, task, question),
                    "scoring": _gold_scoring(snapshot, task),
                }
            )
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    report_dir = Path(args.report_dir).resolve()
    price_dir = Path(args.price_dir).resolve()
    names = pd.read_csv(price_dir / "_hs300_stocks.csv", dtype={"code": str})
    name_by_code = {str(row.code).zfill(6): str(row.name) for row in names.itertuples()}
    common_codes = sorted(
        code
        for code in name_by_code
        if (report_dir / code).is_dir() and (price_dir / f"{code}_daily.csv").is_file()
    )
    snapshots = [
        snapshot
        for code in common_codes
        if (snapshot := load_snapshot(code, name_by_code[code], report_dir, price_dir)) is not None
    ]
    if len(snapshots) < args.heldout_issuers + 20:
        raise RuntimeError(f"not enough valid issuer snapshots: {len(snapshots)}")
    shuffled = list(snapshots)
    random.Random(args.seed).shuffle(shuffled)
    heldout = sorted(shuffled[: args.heldout_issuers], key=lambda item: item.code)
    train = sorted(shuffled[args.heldout_issuers :], key=lambda item: item.code)

    train_dir = Path(args.train_knowledge_dir).resolve()
    heldout_dir = Path(args.heldout_corpus_dir).resolve()
    for directory in (train_dir, heldout_dir):
        directory.mkdir(parents=True, exist_ok=True)
    documents = []
    for split, rows, output_dir in (("train", train, train_dir), ("heldout", heldout, heldout_dir)):
        for snapshot in rows:
            for kind in ("financial", "market"):
                documents.append({"split": split, "issuer_code": snapshot.code, "kind": kind, **_write_document(output_dir, snapshot, kind)})

    requests = build_collection_requests(train, args.collection_per_task)
    gold = build_trusted_gold(heldout, args.gold_per_task)
    outputs = {
        "collection_requests": Path(args.collection_requests).resolve(),
        "trusted_gold": Path(args.trusted_gold).resolve(),
        "split_manifest": Path(args.split_manifest).resolve(),
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["collection_requests"].write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["trusted_gold"].write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "version": "agentic-corpus-v2.0",
        "seed": args.seed,
        "valid_issuers": len(snapshots),
        "train_issuers": [item.code for item in train],
        "heldout_issuers": [item.code for item in heldout],
        "issuer_overlap": sorted({item.code for item in train} & {item.code for item in heldout}),
        "documents": documents,
        "collection_requests": len(requests),
        "collection_by_task": {task: sum(row["task_type"] == task for row in requests) for task in TASKS},
        "trusted_gold": len(gold),
        "gold_by_task": {task: sum(row["task_type"] == task for row in gold) for task in TASKS},
        "source_note": "Financial CSVs were fetched from Sina Finance; price CSVs were fetched from Eastmoney. Both remain secondary sources.",
    }
    outputs["split_manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "documents"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare auditable SFT v2 corpus and issuer-disjoint trusted gold")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "data/raw/financial_reports"))
    parser.add_argument("--price-dir", default=str(PROJECT_ROOT / "data/raw/stock_prices"))
    parser.add_argument("--train-knowledge-dir", default=str(PROJECT_ROOT / "data/knowledge/v2_train"))
    parser.add_argument("--heldout-corpus-dir", default=str(PROJECT_ROOT / "data/evaluation/v2_heldout_corpus"))
    parser.add_argument("--collection-requests", default=str(PROJECT_ROOT / "data/rag/v2_collection_requests.json"))
    parser.add_argument("--trusted-gold", default=str(PROJECT_ROOT / "data/evaluation/trusted_finance_v2.json"))
    parser.add_argument("--split-manifest", default=str(PROJECT_ROOT / "data/evaluation/v2_split_manifest.json"))
    parser.add_argument("--heldout-issuers", type=int, default=30)
    parser.add_argument("--collection-per-task", type=int, default=120)
    parser.add_argument("--gold-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    report = prepare(parse_args())
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
