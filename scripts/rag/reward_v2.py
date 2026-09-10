#!/usr/bin/env python3
"""Constraint-first rewards for auditable financial agent trajectories."""

from __future__ import annotations

import ast
import math
import re
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from scripts.rag.audit_schema import CalculationRecord, ClaimRecord, EvidenceRecord, extract_citation_ids, parse_timestamp, strip_thinking


ABSTENTION_TERMS = ("无法确认", "资料不足", "证据不足", "未检索到", "无法给出确定", "需要补充")
FRESHNESS_TERMS = ("最新", "今天", "今日", "当前", "目前", "实时", "近期", "本周", "本月", "本季度", "刚刚")
PROHIBITED_ADVICE_PATTERNS = (
    r"保证(?:收益|盈利)", r"稳赚", r"必涨", r"绝对不会亏", r"无风险收益", r"全仓(?:买入|卖出)", r"梭哈",
)
TASK_TERMS = {
    "stock_analysis": ("趋势", "估值", "基本面", "风险", "成交量", "波动"),
    "quant_strategy": ("策略", "信号", "入场", "出场", "仓位", "止损", "回测"),
    "financial_report": ("营收", "净利润", "现金流", "资产", "负债", "风险"),
    "sentiment_analysis": ("积极", "消极", "中性", "positive", "negative", "neutral", "yes", "no"),
    "financial_qa": ("定义", "公式", "原因", "风险", "区别", "适用"),
    "risk_assessment": ("风险", "等级", "波动", "回撤", "敞口", "缓释"),
}
_STOP_TERMS = {
    "请问", "请", "分析", "一下", "根据", "以下", "说明", "给出", "要求", "用户", "问题", "相关", "金融",
    "the", "and", "for", "with", "from", "that", "this", "what", "please", "using", "input", "output",
}


@dataclass
class RewardV2Config:
    min_retrieval_relevance: float = 0.18
    min_claim_support: float = 0.72
    min_citation_coverage: float = 1.0
    min_citation_precision: float = 1.0
    require_timestamp_for_fresh_queries: bool = True
    market_reward_cap: float = 0.05
    require_complete_trajectory: bool = True
    component_weights: dict[str, float] = field(
        default_factory=lambda: {
            "retrieval_relevance": 0.14,
            "claim_support": 0.20,
            "numeric_consistency": 0.12,
            "citation_coverage": 0.12,
            "citation_precision": 0.10,
            "source_quality": 0.10,
            "temporal_validity": 0.08,
            "task_validity": 0.09,
            "trajectory_quality": 0.05,
        }
    )


def _terms(text: str) -> set[str]:
    text = re.sub(r"\s+", " ", text or "").lower()
    ascii_terms = set(re.findall(r"[a-z0-9_.%+-]{2,}", text))
    chinese_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    chinese_terms: set[str] = set()
    for run in chinese_runs:
        if len(run) <= 4:
            chinese_terms.add(run)
        chinese_terms.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return {term for term in ascii_terms | chinese_terms if term not in _STOP_TERMS}


def _overlap(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    precision = len(left_terms & right_terms) / len(left_terms)
    recall = len(left_terms & right_terms) / len(right_terms)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _query_relevance(query: str, evidence: list[EvidenceRecord]) -> tuple[float, bool]:
    if not evidence:
        return 0.0, True
    scores = sorted((_overlap(query, item.exact_quote) for item in evidence), reverse=True)
    top_scores = scores[: min(3, len(scores))]
    lexical_relevance = sum(top_scores) / len(top_scores)

    requested_codes = set(re.findall(r"(?<!\d)\d{6}(?!\d)", query))
    evidence_codes = set(re.findall(r"(?<!\d)\d{6}(?!\d)", " ".join(item.exact_quote for item in evidence)))
    entity_match = not requested_codes or bool(requested_codes & evidence_codes)
    if requested_codes:
        entity_coverage = len(requested_codes & evidence_codes) / len(requested_codes)
        relevance = 0.6 * lexical_relevance + 0.4 * entity_coverage
    else:
        relevance = lexical_relevance
    return round(relevance, 4), entity_match


def _numeric_tokens(text: str) -> set[str]:
    without_dates = re.sub(r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?:T[^\s，。；;]*)?", " ", text)
    without_dates = re.sub(r"(?<!\d)\d{4}年\d{1,2}月(?:\d{1,2}日)?", " ", without_dates)
    without_dates = re.sub(r"(?<!\d)\d{4}年(?:第)?[一二三四1234]季度", " ", without_dates)
    without_dates = re.sub(r"(?<!\d)\d{4}[ -]?[Qq][1-4]", " ", without_dates)
    # Ignore ordinals embedded in field-like identifiers (for example
    # ``fact_1``); they are formatting metadata, not financial values.
    raw_tokens = re.findall(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?%?", without_dates)
    normalized = set()
    for token in raw_tokens:
        suffix = "%" if token.endswith("%") else ""
        value = token[:-1] if suffix else token
        try:
            normalized.add(f"{Decimal(value).normalize()}{suffix}")
        except InvalidOperation:
            normalized.add(token.lstrip("+"))
    return normalized


def _claim_support(
    claim: ClaimRecord,
    evidence_by_id: dict[str, EvidenceRecord],
    valid_calculation_ids: set[str],
) -> float:
    if claim.claim_type == "opinion" and not claim.supporting_evidence_ids:
        return 1.0
    if claim.calculation_id in valid_calculation_ids:
        return 1.0
    cited = [evidence_by_id[item] for item in claim.supporting_evidence_ids if item in evidence_by_id]
    if not cited:
        return 0.0
    contradiction_pairs = (
        ("增长", "下降"), ("增加", "减少"), ("上升", "下降"), ("盈利", "亏损"),
        ("高于", "低于"), ("利好", "利空"), ("改善", "恶化"), ("positive", "negative"),
    )
    for item in cited:
        for left, right in contradiction_pairs:
            if (left in claim.statement and right in item.exact_quote and left not in item.exact_quote) or (
                right in claim.statement and left in item.exact_quote and right not in item.exact_quote
            ):
                return 0.0
    claim_terms = _terms(claim.statement)
    if not claim_terms:
        return 0.0
    return max(len(claim_terms & _terms(item.exact_quote)) / len(claim_terms) for item in cited)


def _numeric_consistency(
    claim: ClaimRecord,
    evidence_by_id: dict[str, EvidenceRecord],
    valid_calculation_ids: set[str],
) -> float:
    if claim.calculation_id in valid_calculation_ids:
        return 1.0
    numbers = _numeric_tokens(claim.statement)
    if not numbers:
        return 1.0
    cited_text = " ".join(
        evidence_by_id[item].exact_quote
        for item in claim.supporting_evidence_ids
        if item in evidence_by_id
    )
    cited_numbers = _numeric_tokens(cited_text)
    return round(len(numbers & cited_numbers) / len(numbers), 4)


def _task_validity(task_type: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    if any(term in answer for term in ABSTENTION_TERMS):
        return 1.0
    if task_type == "quant_strategy":
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL | re.IGNORECASE)
        code = code_match.group(1) if code_match else ""
        syntax = 0.0
        if code:
            try:
                ast.parse(code)
                syntax = 1.0
            except SyntaxError:
                pass
        checks = (
            syntax,
            1.0 if re.search(r"^\s*(?:import|from)\s", code, re.MULTILINE) else 0.0,
            1.0 if re.search(r"^\s*def\s", code, re.MULTILINE) else 0.0,
            1.0 if re.search(r"入场|出场|仓位|止损|signal", answer, re.IGNORECASE) else 0.0,
            1.0 if re.search(r"sharpe|夏普|drawdown|回撤|annual_return|年化", answer, re.IGNORECASE) else 0.0,
        )
        return round(sum(checks) / len(checks), 4)
    terms = TASK_TERMS.get(task_type, ())
    if not terms:
        return 1.0
    return round(min(1.0, sum(1 for term in terms if term.lower() in answer.lower()) / max(1, min(3, len(terms)))), 4)


def _source_quality(evidence: Iterable[EvidenceRecord]) -> float:
    scores = {
        "primary": 1.0,
        "verified_secondary": 0.75,
        "internal": 0.55,
        "unverified_secondary": 0.35,
        "unknown": 0.0,
    }
    values = [scores.get(item.reliability_tier, 0.0) for item in evidence]
    return round(sum(values) / len(values), 4) if values else 0.0


def _temporal_validity(evidence: Iterable[EvidenceRecord], request_as_of: str) -> tuple[float, list[str]]:
    as_of = parse_timestamp(request_as_of)
    if as_of is None:
        return 0.0, ["invalid_request_as_of"]
    failures = []
    checked = 0
    valid = 0
    for item in evidence:
        published = parse_timestamp(item.published_at)
        effective = parse_timestamp(item.effective_at)
        fetched = parse_timestamp(item.fetched_at)
        if fetched is None:
            failures.append(f"invalid_fetched_at:{item.evidence_id}")
            continue
        checked += 1
        if published and published > as_of:
            failures.append(f"future_publication:{item.evidence_id}")
            continue
        if effective and effective > as_of:
            failures.append(f"future_evidence:{item.evidence_id}")
            continue
        if fetched > as_of:
            failures.append(f"not_available_as_of:{item.evidence_id}")
            continue
        valid += 1
    return (round(valid / checked, 4) if checked else 0.0), failures


def _trajectory_quality(trajectory: list[dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    nodes = {str(item.get("node", "")) for item in trajectory}
    expected = {"plan_queries", "retrieve_local", "normalize_evidence", "build_claim_graph", "verify_grounding"}
    node_score = len(nodes & expected) / len(expected)
    action_score = sum(1 for item in trajectory if item.get("action")) / len(trajectory)
    return round(0.7 * node_score + 0.3 * action_score, 4)


def _trajectory_integrity(trajectory: list[dict[str, Any]], evidence_ids: set[str]) -> list[str]:
    if not trajectory:
        return ["trajectory_missing"]
    failures = []
    nodes = {str(item.get("node", "")) for item in trajectory}
    required_nodes = {"plan_queries", "retrieve_local", "normalize_evidence", "build_claim_graph"}
    if required_nodes - nodes:
        failures.append("trajectory_incomplete")

    step_ids = [item.get("step_id") for item in trajectory]
    if step_ids != list(range(1, len(trajectory) + 1)):
        failures.append("trajectory_step_order_invalid")

    plans = [item for item in trajectory if item.get("node") == "plan_queries"]
    if not any((item.get("action_args") or item.get("metrics") or {}).get("queries") for item in plans):
        failures.append("trajectory_query_plan_missing")

    normalization_events = [item for item in trajectory if item.get("node") == "normalize_evidence"]
    normalized_ids = {
        str(evidence_id).upper()
        for item in normalization_events
        for evidence_id in item.get("observation_ids", ())
    }
    if normalized_ids != {item.upper() for item in evidence_ids}:
        failures.append("trajectory_evidence_mismatch")

    retrieval_events = [item for item in trajectory if item.get("node") in {"retrieve_local", "maybe_web_collect"}]
    if not retrieval_events or not any(item.get("action") not in {None, "", "skipped"} for item in retrieval_events):
        failures.append("trajectory_retrieval_missing")
    return sorted(set(failures))


def _compliance_failures(answer: str) -> list[str]:
    return [f"prohibited_advice:{pattern}" for pattern in PROHIBITED_ADVICE_PATTERNS if re.search(pattern, answer)]


def compute_delayed_market_reward(
    feedback: dict[str, Any],
    *,
    request_as_of: str,
    task_type: str,
    cap: float,
) -> tuple[float, list[str]]:
    """Compute a capped, shrunk posterior signal from a preregistered forecast."""
    required = {
        "prediction_id",
        "made_at",
        "matures_at",
        "evaluated_at",
        "direction",
        "horizon_days",
        "benchmark",
        "gross_return",
        "benchmark_return",
        "realized_volatility",
        "max_drawdown",
        "transaction_cost",
        "sample_size",
    }
    failures = [f"market_missing:{name}" for name in sorted(required - set(feedback))]
    if not feedback.get("eligible") or not feedback.get("pre_registered"):
        failures.append("market_not_preregistered")
    if not feedback.get("forecast_task") or task_type not in {"stock_analysis", "quant_strategy"}:
        failures.append("market_ineligible_task")
    request_time = parse_timestamp(request_as_of)
    made_at = parse_timestamp(feedback.get("made_at"))
    matures_at = parse_timestamp(feedback.get("matures_at"))
    evaluated_at = parse_timestamp(feedback.get("evaluated_at"))
    if not request_time or not made_at or made_at > request_time:
        failures.append("market_prediction_time_invalid")
    if not matures_at or not evaluated_at or evaluated_at < matures_at:
        failures.append("market_not_matured")
    direction = str(feedback.get("direction", "")).lower()
    if direction not in {"long", "short", "up", "down"}:
        failures.append("market_direction_invalid")
    try:
        horizon = int(feedback.get("horizon_days", 0))
        sample_size = int(feedback.get("sample_size", 0))
        gross_return = float(feedback.get("gross_return", 0.0))
        benchmark_return = float(feedback.get("benchmark_return", 0.0))
        volatility = float(feedback.get("realized_volatility", 0.0))
        drawdown = float(feedback.get("max_drawdown", 0.0))
        transaction_cost = float(feedback.get("transaction_cost", 0.0))
    except (TypeError, ValueError):
        failures.append("market_numeric_field_invalid")
        return 0.0, sorted(set(failures))
    values = (gross_return, benchmark_return, volatility, drawdown, transaction_cost)
    if horizon <= 0 or sample_size <= 0 or not all(math.isfinite(value) for value in values):
        failures.append("market_numeric_field_invalid")
    if failures:
        return 0.0, sorted(set(failures))

    directional_sign = 1.0 if direction in {"long", "up"} else -1.0
    if not str(feedback.get("prediction_id", "")).strip() or not str(feedback.get("benchmark", "")).strip():
        failures.append("market_identity_missing")
    if made_at and matures_at and matures_at <= made_at:
        failures.append("market_horizon_invalid")
    if volatility < 0 or transaction_cost < 0:
        failures.append("market_numeric_field_invalid")
    if failures:
        return 0.0, sorted(set(failures))

    net_excess = directional_sign * (gross_return - benchmark_return) - transaction_cost
    volatility = abs(volatility)
    drawdown = abs(drawdown)
    downside_risk = max(volatility, drawdown, 0.01)
    risk_adjusted = math.tanh(net_excess / downside_risk)
    posterior_shrinkage = sample_size / (sample_size + 30.0)
    return max(-cap, min(cap, cap * risk_adjusted * posterior_shrinkage)), []


def compute_auditable_reward(
    *,
    query: str,
    answer: str,
    evidence: list[EvidenceRecord],
    claims: list[ClaimRecord],
    calculations: Optional[list[CalculationRecord]] = None,
    task_type: str,
    request_as_of: str,
    trajectory: Optional[list[dict[str, Any]]] = None,
    market_feedback: Optional[dict[str, Any]] = None,
    unresolved_conflicts: Optional[list[dict[str, Any]]] = None,
    config: Optional[RewardV2Config] = None,
) -> dict[str, Any]:
    """Score a complete trajectory, applying non-compensable hard gates first."""
    config = config or RewardV2Config()
    trajectory = trajectory or []
    calculations = calculations or []
    unresolved_conflicts = unresolved_conflicts or []
    answer = strip_thinking(answer)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    evidence_ids = set(evidence_by_id)
    calculation_errors = {
        item.calculation_id: item.validation_errors(evidence_ids)
        for item in calculations
        if item.validation_errors(evidence_ids)
    }
    valid_calculation_ids = {item.calculation_id for item in calculations if item.calculation_id not in calculation_errors}
    answer_citations = extract_citation_ids(answer)
    auditable_claims = [
        item for item in claims if not any(term in item.statement for term in ABSTENTION_TERMS)
    ]
    verifiable_claims = [item for item in auditable_claims if item.claim_type != "opinion"]
    cited_claims = [item for item in auditable_claims if item.supporting_evidence_ids or item.calculation_id]

    relevance, entity_match = _query_relevance(query, evidence)
    claim_scores = [_claim_support(item, evidence_by_id, valid_calculation_ids) for item in verifiable_claims]
    numeric_scores = [_numeric_consistency(item, evidence_by_id, valid_calculation_ids) for item in auditable_claims]
    claim_support = round(sum(claim_scores) / len(claim_scores), 4) if claim_scores else 1.0
    numeric_consistency = round(sum(numeric_scores) / len(numeric_scores), 4) if numeric_scores else 1.0
    citation_coverage = round(len(cited_claims) / len(auditable_claims), 4) if auditable_claims else 1.0
    citation_precision = round(len(answer_citations & evidence_ids) / len(answer_citations), 4) if answer_citations else (1.0 if not auditable_claims else 0.0)
    source_quality = _source_quality(evidence)
    temporal_validity, temporal_failures = _temporal_validity(evidence, request_as_of)
    task_validity = _task_validity(task_type, answer)
    trajectory_quality = _trajectory_quality(trajectory)
    trajectory_failures = _trajectory_integrity(trajectory, evidence_ids) if config.require_complete_trajectory else []
    freshness_required = any(term in query for term in FRESHNESS_TERMS)
    abstained = any(term in answer for term in ABSTENTION_TERMS)

    hard_failures: list[str] = []
    hard_failures.extend(_compliance_failures(answer))
    hard_failures.extend(temporal_failures)
    hard_failures.extend(trajectory_failures)
    if answer_citations - evidence_ids:
        hard_failures.append("fabricated_citation")
    if evidence and relevance < config.min_retrieval_relevance:
        hard_failures.append("irrelevant_retrieval")
    if not entity_match:
        hard_failures.append("entity_mismatch")
    if auditable_claims and citation_coverage < config.min_citation_coverage:
        hard_failures.append("insufficient_citation_coverage")
    if auditable_claims and citation_precision < config.min_citation_precision:
        hard_failures.append("invalid_citation")
    if verifiable_claims and claim_support < config.min_claim_support:
        hard_failures.append("unsupported_claim")
    if auditable_claims and numeric_consistency < 1.0:
        hard_failures.append("numeric_mismatch")
    if any(item.contradicting_evidence_ids and item.confidence > 0.5 for item in claims):
        hard_failures.append("unresolved_evidence_conflict")
    if unresolved_conflicts:
        hard_failures.append("unresolved_evidence_conflict")
    if any(item.validation_errors() for item in evidence):
        hard_failures.append("invalid_evidence_metadata")
    if calculation_errors:
        hard_failures.append("invalid_calculation")
    if auditable_claims and evidence and source_quality <= 0.0:
        hard_failures.append("unreliable_source")
    if not evidence and auditable_claims and not abstained:
        hard_failures.append("evidence_missing")
    if freshness_required and config.require_timestamp_for_fresh_queries:
        if evidence and any(not (item.published_at or item.effective_at) for item in evidence):
            hard_failures.append("freshness_timestamp_missing")
        if not evidence and not abstained:
            hard_failures.append("fresh_evidence_missing")
    if task_type == "quant_strategy" and not abstained and task_validity < 0.8:
        hard_failures.append("invalid_quant_artifact")

    components = {
        "retrieval_relevance": relevance if evidence else (1.0 if abstained else 0.0),
        "claim_support": claim_support,
        "numeric_consistency": numeric_consistency,
        "citation_coverage": citation_coverage,
        "citation_precision": citation_precision,
        "source_quality": source_quality if evidence else (1.0 if abstained else 0.0),
        "temporal_validity": temporal_validity if evidence else (1.0 if abstained else 0.0),
        "task_validity": task_validity,
        "trajectory_quality": trajectory_quality,
    }
    total_weight = sum(config.component_weights.values()) or 1.0
    base_reward = sum(components[name] * weight for name, weight in config.component_weights.items()) / total_weight

    market_reward = 0.0
    market_applied = False
    market_feedback_failures: list[str] = []
    if market_feedback:
        market_reward, market_feedback_failures = compute_delayed_market_reward(
            market_feedback,
            request_as_of=request_as_of,
            task_type=task_type,
            cap=config.market_reward_cap,
        )
        market_applied = not hard_failures and not market_feedback_failures
        if not market_applied:
            market_reward = 0.0

    total_reward = max(0.0, min(1.0, base_reward + market_reward)) if not hard_failures else 0.0
    result: dict[str, Any] = {
        **{key: round(value, 4) for key, value in components.items()},
        "groundedness": claim_support,
        "task_coverage": task_validity,
        "abstention_quality": 1.0 if abstained and (not evidence or bool(hard_failures)) else (0.5 if abstained else 1.0),
        "cost_efficiency": round(max(0.0, 1.0 - 0.04 * max(0, len(evidence) - 3) - 0.02 * max(0, len(trajectory) - 12)), 4),
        "base_reward": round(base_reward, 4),
        "market_reward": round(market_reward, 4),
        "market_reward_applied": market_applied,
        "market_feedback_failures": market_feedback_failures,
        "hard_gate_passed": not hard_failures,
        "hard_failures": sorted(set(hard_failures)),
        "total_reward": round(total_reward, 4),
        "reward_version": "2.0",
    }
    return result
