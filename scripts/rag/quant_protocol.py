#!/usr/bin/env python3
"""Strict quant action protocol and deterministic artifact renderer."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


QUANT_ACTION_VERSION = "quant.action.v2.5"
QUANT_ARTIFACT_VERSION = "quant.v2.5"
QUANT_TEMPLATE_ID = "ma5_ma20_long_only"
QUANT_ACTION = {
    "action": "render_quant_artifact",
    "artifact_version": QUANT_ARTIFACT_VERSION,
    "template_id": QUANT_TEMPLATE_ID,
}

QUANT_CODE = '''```python
import numpy as np
import pandas as pd

def quant_artifact(df):
    if not isinstance(df, pd.DataFrame) or "close" not in df:
        raise ValueError("external DataFrame with close is required")
    close = pd.to_numeric(df["close"], errors="raise")
    fast, slow = close.rolling(5).mean(), close.rolling(20).mean()
    entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exit_ = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    signal = np.select([entry, exit_], [1, -1], default=0)
    position = pd.Series(signal, index=df.index).replace(0, np.nan).ffill().fillna(0).clip(0, 1).shift(1).fillna(0)
    stop_loss = close.cummax() * 0.95
    position = position.mask(close <= stop_loss, 0)
    strategy_return = position * close.pct_change()
    volatility = strategy_return.std()
    sharpe = np.nan if pd.isna(volatility) or volatility == 0 else np.sqrt(252) * strategy_return.mean() / volatility
    equity = (1 + strategy_return.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {"signal": signal, "entry": entry, "exit": exit_, "position": position, "stop_loss": stop_loss, "sharpe": sharpe, "drawdown": drawdown}
```'''


def canonical_quant_action() -> str:
    """Return the only model completion accepted by the v2.5 quant route."""
    return json.dumps(QUANT_ACTION, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parse_quant_action(raw_output: str) -> tuple[dict[str, str] | None, list[str]]:
    """Parse a model action without repairing malformed or extra output."""
    text = raw_output.strip()
    errors: list[str] = []
    if not text:
        return None, ["quant_action_empty"]
    if len(text) > 240:
        errors.append("quant_action_too_long")
    if re.search(r"</?think>|```", text, re.IGNORECASE):
        errors.append("quant_action_extra_markup")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, sorted(set([*errors, "quant_action_invalid_json"]))
    if not isinstance(payload, dict):
        return None, sorted(set([*errors, "quant_action_not_object"]))
    if set(payload) != set(QUANT_ACTION):
        errors.append("quant_action_schema")
    for key, expected in QUANT_ACTION.items():
        if payload.get(key) != expected:
            errors.append(f"quant_action_value:{key}")
    if errors:
        return None, sorted(set(errors))
    return {key: str(payload[key]) for key in QUANT_ACTION}, []


def _market_evidence(evidence: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in evidence
            if "收盘价" in str(item.get("exact_quote", ""))
            and "年化波动率" in str(item.get("exact_quote", ""))
        ),
        None,
    )


def _provenance(record: dict[str, Any]) -> str:
    evidence_id = str(record.get("evidence_id", "")).upper()
    publisher = str(record.get("publisher") or record.get("source") or "unknown")
    tier = str(record.get("reliability_tier") or "unknown")
    published = str(record.get("published_at") or "unknown")
    effective = str(record.get("effective_at") or "unknown")
    fetched = str(record.get("fetched_at") or "unknown")
    return (
        f"### 证据 [{evidence_id}] | 来源：{publisher}（{tier}） | 发布时间：{published} | "
        f"生效时间：{effective} | 获取时间：{fetched}"
    )


def _fact(record: dict[str, Any], label: str) -> str | None:
    quote = str(record.get("exact_quote", ""))
    match = re.search(rf"{re.escape(label)}为([^，。；]+)", quote)
    if not match:
        return None
    evidence_id = str(record.get("evidence_id", "")).upper()
    return f"- {label}为{match.group(1).strip()} [{evidence_id}]。"


def render_quant_artifact(evidence: Iterable[dict[str, Any]]) -> str:
    """Materialize the canonical audited artifact from trusted state, not model prose."""
    market = _market_evidence(evidence)
    lines: list[str] = []
    if market is not None:
        lines.append(_provenance(market))
    lines.extend(
        [
            f"### artifact_version: {QUANT_ARTIFACT_VERSION}",
            "### template_id: ma5_ma20_long_only",
            "### input_schema: pandas.DataFrame[close]",
            "### output_fields: signal, entry, exit, position, stop_loss, sharpe, drawdown",
        ]
    )
    if market is not None:
        facts = [
            _fact(market, label)
            for label in ("收盘价", "近20个交易日收益率", "年化波动率", "最大回撤")
        ]
        lines.extend(["### 可审计行情输入", *(fact for fact in facts if fact)])
    lines.extend(["### 固定代码", QUANT_CODE, "### 边界"])
    if market is None:
        lines.append("- 证据不足，无法确认该 artifact 的实测 Sharpe、drawdown 或未来收益。")
    else:
        evidence_id = str(market.get("evidence_id", "")).upper()
        lines.append(
            f"- 证据不足，无法确认该 artifact 的实测 Sharpe、drawdown 或未来收益 [{evidence_id}]。"
        )
    return "\n".join(lines)


def materialize_quant_output(
    raw_output: str,
    evidence: Iterable[dict[str, Any]],
) -> tuple[str | None, list[str]]:
    """Validate the action before invoking the deterministic renderer."""
    _, errors = parse_quant_action(raw_output)
    if errors:
        return None, errors
    return render_quant_artifact(evidence), []
