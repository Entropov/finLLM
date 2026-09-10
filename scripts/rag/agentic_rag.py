#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LangGraph-based agentic RAG orchestration with deterministic RL-style rewards.

The module keeps the existing Chroma retriever and web knowledge agent as tools.
If LangGraph is not installed, the same node sequence is executed directly so
the inference path can degrade gracefully.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from scripts.rag.audit_schema import (
    AuditEnvelope,
    CalculationRecord,
    ClaimRecord,
    EvidenceRecord,
    TrajectoryEvent,
    build_claims_from_answer,
    normalize_evidence,
    strip_thinking,
    utc_now,
)
from scripts.rag.reward_v2 import RewardV2Config, compute_auditable_reward
from scripts.rag.quant_protocol import materialize_quant_output

try:
    import yaml
except ImportError:  # pragma: no cover - dependency may be absent in minimal envs
    yaml = None

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - exercised through fallback tests
    END = "__end__"
    StateGraph = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_agentic.yaml"
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "data" / "rag" / "trajectories"
DEFAULT_PREFERENCE_DIR = PROJECT_ROOT / "data" / "rag" / "preferences"
DEFAULT_EVAL_SET_DIR = PROJECT_ROOT / "data" / "rag" / "eval_sets"

ALL_TASKS = [
    "stock_analysis",
    "quant_strategy",
    "financial_report",
    "sentiment_analysis",
    "financial_qa",
    "risk_assessment",
]

TASK_KEYWORDS = {
    "stock_analysis": ["趋势", "支撑", "压力", "成交量", "均线", "MACD", "RSI", "风险"],
    "quant_strategy": ["策略", "信号", "入场", "出场", "仓位", "止损", "回测", "收益"],
    "financial_report": ["营收", "净利润", "毛利率", "现金流", "ROE", "资产负债率", "风险"],
    "sentiment_analysis": ["积极", "消极", "中性", "利好", "利空", "影响"],
    "financial_qa": ["定义", "公式", "适用", "风险", "区别", "例子"],
    "risk_assessment": ["风险", "等级", "因素", "缓释", "波动", "回撤", "VaR"],
}

FRESHNESS_TERMS = [
    "最新", "今天", "今日", "昨天", "昨日", "明天", "当前", "目前", "近期", "实时",
    "公告", "新闻", "政策", "财报发布", "2025", "2026", "未来", "过去", "年初", "年末", "季度", "月初", "月末",
]


class AgenticRAGState(TypedDict, total=False):
    query: str
    system_prompt: str
    task_type: str
    freshness_required: bool
    retrieval_plan: dict[str, Any]
    query_variants: list[str]
    local_docs: list[dict[str, Any]]
    web_docs: list[dict[str, Any]]
    selected_context: list[dict[str, Any]]
    answer: str
    raw_answer: str
    quant_action_errors: list[str]
    reference: str
    verification: dict[str, Any]
    reward_breakdown: dict[str, Any]
    request_as_of: str
    claim_plan: dict[str, Any]
    evidence_records: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    compliance: dict[str, Any]
    evidence_quality: dict[str, Any]
    unresolved_conflicts: list[dict[str, Any]]
    audit: dict[str, Any]
    trajectory: list[dict[str, Any]]
    errors: list[str]
    runtime_options: dict[str, Any]
    refine_count: int
    needs_refine: bool
    answer_locked: bool
    execution_backend: str
    answer_guard_applied: bool
    query_id: str
    started_at: float
    finished_at: float


@dataclass
class AgenticRAGConfig:
    """Runtime configuration for agentic RAG."""

    top_k: int = 3
    max_context_docs: int = 5
    max_context_chars: int = 6000
    multi_query_count: int = 4
    allow_web: bool = False
    web_search_top_n: int = 5
    web_fetch_top_n: int = 3
    max_refine_steps: int = 1
    policy_type: str = "rules"
    min_total_reward_for_finish: float = 0.35
    trajectory_log_dir: str = str(DEFAULT_TRAJECTORY_DIR)
    preference_log_dir: str = str(DEFAULT_PREFERENCE_DIR)
    eval_set_dir: str = str(DEFAULT_EVAL_SET_DIR)
    enable_trajectory_logging: bool = True
    require_langgraph: bool = False
    reward_version: str = "2.0"
    min_retrieval_relevance: float = 0.18
    min_claim_support: float = 0.72
    min_citation_coverage: float = 1.0
    min_citation_precision: float = 1.0
    market_reward_cap: float = 0.05
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "retrieval_relevance": 0.20,
            "citation_coverage": 0.15,
            "groundedness": 0.25,
            "task_coverage": 0.20,
            "abstention_quality": 0.10,
            "cost_efficiency": 0.10,
        }
    )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AgenticRAGConfig":
        defaults = cls()
        payload = asdict(defaults)
        for key, value in data.items():
            if key == "reward_weights" and isinstance(value, dict):
                weights = dict(payload["reward_weights"])
                weights.update({k: float(v) for k, v in value.items()})
                payload[key] = weights
            elif key in payload:
                payload[key] = value
        return cls(**payload)


@dataclass(frozen=True)
class RAGPolicyAction:
    use_local: bool
    use_web: bool
    top_k: int
    multi_query_count: int
    refine_allowed: bool
    query_rewrite: str
    policy_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_agentic_rag_config(path: Optional[str | Path] = None) -> AgenticRAGConfig:
    """Load YAML config, falling back to defaults when the file is absent."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AgenticRAGConfig()
    if yaml is None:
        raise ImportError("PyYAML is required to load agentic RAG YAML config.")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AgenticRAGConfig.from_mapping(data)


def classify_financial_task(query: str) -> str:
    text = query.lower()
    if re.search(r"\b\d{6}\b|k线|macd|rsi|均线|支撑|压力|股价|走势|技术面", query, re.IGNORECASE):
        return "stock_analysis"
    if any(term in text for term in ["量化", "策略", "回测", "因子", "python", "代码", "仓位", "止损"]):
        return "quant_strategy"
    if any(term in query for term in ["财报", "年报", "季报", "营收", "净利润", "现金流", "资产负债率", "ROE"]):
        return "financial_report"
    if any(term in query for term in ["情感", "情绪", "利好", "利空", "积极", "消极", "中性", "新闻影响"]):
        return "sentiment_analysis"
    if any(term in query for term in ["风险", "违约", "信用", "评级", "敞口", "回撤", "VaR"]):
        return "risk_assessment"
    return "financial_qa"


def needs_fresh_knowledge(query: str) -> bool:
    return any(term in query for term in FRESHNESS_TERMS)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize_for_overlap(text: str) -> set[str]:
    text = normalize_text(text)
    chinese_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    ascii_terms = set(re.findall(r"[a-zA-Z0-9_.%+-]{2,}", text.lower()))
    compact_terms = set()
    for term in chinese_terms:
        if len(term) <= 6:
            compact_terms.add(term)
        else:
            compact_terms.update(term[i : i + 2] for i in range(len(term) - 1))
    return compact_terms | ascii_terms


def keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    return sum(1 for kw in keywords if re.search(re.escape(kw), text, re.IGNORECASE)) / len(keywords)


def task_coverage_score(text: str, task_type: str) -> float:
    return keyword_coverage(text, TASK_KEYWORDS.get(task_type, TASK_KEYWORDS["financial_qa"]))


def compute_context_relevance(query: str, docs: list[dict[str, Any]]) -> float:
    if not docs:
        return 0.0
    query_terms = tokenize_for_overlap(query)
    if not query_terms:
        return 0.0
    scores = []
    for doc in docs:
        content_terms = tokenize_for_overlap(doc.get("content", ""))
        overlap = len(query_terms & content_terms) / max(1, len(query_terms))
        distance = doc.get("distance")
        distance_bonus = 0.0
        if isinstance(distance, (int, float)) and math.isfinite(distance):
            distance_bonus = max(0.0, 1.0 - min(distance, 2.0) / 2.0) * 0.2
        scores.append(min(1.0, overlap + distance_bonus))
    return round(sum(scores) / len(scores), 4)


def compute_groundedness(answer: str, docs: list[dict[str, Any]]) -> float:
    if not answer or not docs:
        return 0.0
    answer_terms = tokenize_for_overlap(answer)
    context_terms = tokenize_for_overlap(" ".join(doc.get("content", "") for doc in docs))
    if not answer_terms:
        return 0.0
    return round(len(answer_terms & context_terms) / max(1, len(answer_terms)), 4)


def compute_citation_coverage(answer: str, docs: list[dict[str, Any]]) -> float:
    if not docs:
        return 0.0
    if re.search(r"\[[^\]]+\]|来源|参考|资料|知识库", answer):
        return 1.0
    source_mentions = 0
    for doc in docs:
        metadata = doc.get("metadata") or {}
        source = str(metadata.get("source", "")).strip()
        if source and source in answer:
            source_mentions += 1
    return min(1.0, source_mentions / len(docs))


def compute_abstention_quality(answer: str, docs: list[dict[str, Any]]) -> float:
    uncertainty = any(term in answer for term in ["无法确认", "资料不足", "未检索到", "需要进一步", "不确定"])
    if docs:
        return 1.0 if not uncertainty else 0.4
    return 1.0 if uncertainty else 0.2


def compute_rag_reward(
    query: str,
    answer: str,
    docs: list[dict[str, Any]],
    task_type: str,
    web_doc_count: int = 0,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Compatibility wrapper around the constraint-first v2 reward."""
    del weights, web_doc_count
    request_as_of = utc_now()
    evidence = normalize_evidence(docs, query=query, observed_at=request_as_of)
    evidence_ids = [item.evidence_id for item in evidence]
    normalized_answer = answer
    # Legacy callers cite file names. They remain visible but are intentionally
    # not treated as valid v2 evidence IDs.
    claims = build_claims_from_answer(normalized_answer, request_as_of=request_as_of)
    result = compute_auditable_reward(
        query=query,
        answer=normalized_answer,
        evidence=evidence,
        claims=claims,
        task_type=task_type,
        request_as_of=request_as_of,
        config=RewardV2Config(require_complete_trajectory=False),
    )
    result["legacy_evidence_ids"] = evidence_ids
    return result


class RuleBasedRAGPolicy:
    """Default deterministic policy used before learned policies are available."""

    def __init__(self, config: AgenticRAGConfig):
        self.config = config

    def select(self, task_type: str, freshness_required: bool, runtime_options: dict[str, Any]) -> RAGPolicyAction:
        top_k = int(runtime_options.get("top_k", self.config.top_k))
        allow_web = bool(runtime_options.get("allow_web", self.config.allow_web))
        multi_query_count = int(runtime_options.get("multi_query_count", self.config.multi_query_count))

        if task_type in {"stock_analysis", "financial_report", "risk_assessment"}:
            top_k = max(top_k, 5)
        elif task_type == "sentiment_analysis":
            top_k = min(top_k, 3)
            multi_query_count = min(multi_query_count, 2)
        elif task_type == "quant_strategy":
            top_k = max(top_k, 4)

        use_web = allow_web and freshness_required
        return RAGPolicyAction(
            use_local=True,
            use_web=use_web,
            top_k=max(1, top_k),
            multi_query_count=max(1, multi_query_count),
            refine_allowed=task_type != "sentiment_analysis",
            query_rewrite="finance_keywords",
            policy_name="rules",
        )


class ContextualBanditRAGPolicy(RuleBasedRAGPolicy):
    """Simple offline bandit policy using persisted average rewards per context/action."""

    def __init__(self, config: AgenticRAGConfig, stats_path: Optional[Path] = None):
        super().__init__(config)
        self.stats_path = stats_path or PROJECT_ROOT / "data" / "rag" / "policy_stats.json"
        self.stats = self._load_stats()

    def _load_stats(self) -> dict[str, Any]:
        if not self.stats_path.exists():
            return {}
        try:
            return json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Agentic RAG policy stats not readable: {exc}")
            return {}

    def select(self, task_type: str, freshness_required: bool, runtime_options: dict[str, Any]) -> RAGPolicyAction:
        fallback = super().select(task_type, freshness_required, runtime_options)
        context_key = f"{task_type}:{'fresh' if freshness_required else 'static'}"
        candidates = self.stats.get(context_key, {})
        if not candidates:
            return RAGPolicyAction(**{**fallback.to_dict(), "policy_name": "bandit_rules_fallback"})
        best_name = max(
            candidates,
            key=lambda name: candidates[name].get("reward_sum", 0.0) / max(1, candidates[name].get("count", 0)),
        )
        if best_name == "broad_local":
            return RAGPolicyAction(True, False, max(fallback.top_k, 5), max(fallback.multi_query_count, 4), True, "finance_keywords", "bandit:broad_local")
        if best_name == "fresh_web":
            return RAGPolicyAction(True, bool(runtime_options.get("allow_web", self.config.allow_web)), max(fallback.top_k, 5), 4, True, "finance_keywords", "bandit:fresh_web")
        return RAGPolicyAction(**{**fallback.to_dict(), "policy_name": f"bandit:{best_name}"})


class LearnedRAGPolicy(RuleBasedRAGPolicy):
    """Placeholder loader for offline DPO/GRPO-trained routing policies.

    The first implementation keeps inference reproducible: if no exported policy
    artifact is available, it falls back to rule actions while preserving the
    requested method in the trajectory for evaluation comparisons.
    """

    def __init__(self, config: AgenticRAGConfig, method: str):
        super().__init__(config)
        self.method = method
        self.policy_path = PROJECT_ROOT / "data" / "rag" / f"{method}_policy.json"
        self.policy = self._load_policy()

    def _load_policy(self) -> dict[str, Any]:
        if not self.policy_path.exists():
            return {}
        try:
            return json.loads(self.policy_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Agentic RAG {self.method} policy not readable: {exc}")
            return {}

    def select(self, task_type: str, freshness_required: bool, runtime_options: dict[str, Any]) -> RAGPolicyAction:
        fallback = super().select(task_type, freshness_required, runtime_options)
        key = f"{task_type}:{'fresh' if freshness_required else 'static'}"
        action = self.policy.get(key) or self.policy.get(task_type) or {}
        if not action:
            return RAGPolicyAction(**{**fallback.to_dict(), "policy_name": f"{self.method}_policy_fallback"})
        return RAGPolicyAction(
            use_local=bool(action.get("use_local", fallback.use_local)),
            use_web=bool(action.get("use_web", fallback.use_web)) and bool(runtime_options.get("allow_web", self.config.allow_web)),
            top_k=int(action.get("top_k", fallback.top_k)),
            multi_query_count=int(action.get("multi_query_count", fallback.multi_query_count)),
            refine_allowed=bool(action.get("refine_allowed", fallback.refine_allowed)),
            query_rewrite=str(action.get("query_rewrite", fallback.query_rewrite)),
            policy_name=f"{self.method}_policy",
        )


class AgenticRAGPipeline:
    """Agentic RAG graph wrapper.

    The graph returns selected context and reward diagnostics. It can be used as
    a pure prompt enricher in chat/API inference or as a standalone evaluator.
    """

    def __init__(
        self,
        retriever: Any = None,
        web_agent: Any = None,
        config: Optional[AgenticRAGConfig] = None,
        config_path: Optional[str | Path] = None,
        answer_generator: Optional[Callable[[AgenticRAGState], str]] = None,
    ):
        self.retriever = retriever
        self.web_agent = web_agent
        self.config = config or load_agentic_rag_config(config_path)
        self.answer_generator = answer_generator
        self.policy = self._build_policy()
        self.langgraph_available = StateGraph is not None
        if self.config.require_langgraph and not self.langgraph_available:
            raise ImportError("langgraph is required by config but is not installed.")
        self.graph = self._build_graph() if self.langgraph_available else None
        self.prepare_graph = self._build_prepare_graph() if self.langgraph_available else None
        self.execution_backend = "langgraph" if self.graph is not None else "sequential_fallback"
        if not self.langgraph_available:
            logger.warning("LangGraph not installed; Agentic RAG uses sequential fallback execution.")

    def _build_policy(self):
        if self.config.policy_type in {"bandit", "contextual_bandit"}:
            return ContextualBanditRAGPolicy(self.config)
        if self.config.policy_type in {"dpo", "grpo", "dpo_policy", "grpo_policy"}:
            return LearnedRAGPolicy(self.config, self.config.policy_type.replace("_policy", ""))
        return RuleBasedRAGPolicy(self.config)

    def _build_graph(self):
        graph = StateGraph(AgenticRAGState)
        graph.add_node("classify_intent", self._classify_intent_node)
        graph.add_node("build_claim_plan", self._build_claim_plan_node)
        graph.add_node("select_policy", self._select_policy_node)
        graph.add_node("plan_queries", self._plan_queries_node)
        graph.add_node("retrieve_local", self._retrieve_local_node)
        graph.add_node("maybe_web_collect", self._maybe_web_collect_node)
        graph.add_node("rerank_context", self._rerank_context_node)
        graph.add_node("normalize_evidence", self._normalize_evidence_node)
        graph.add_node("source_quality_gate", self._source_quality_gate_node)
        graph.add_node("generate_answer", self._generate_answer_node)
        graph.add_node("materialize_quant_artifact", self._materialize_quant_artifact_node)
        graph.add_node("build_claim_graph", self._build_claim_graph_node)
        graph.add_node("resolve_contradictions", self._resolve_contradictions_node)
        graph.add_node("compliance_gate", self._compliance_gate_node)
        graph.add_node("verify_grounding", self._verify_grounding_node)
        graph.add_node("refine_or_finish", self._refine_or_finish_node)
        graph.add_node("log_trajectory", self._log_trajectory_node)

        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "build_claim_plan")
        graph.add_edge("build_claim_plan", "select_policy")
        graph.add_edge("select_policy", "plan_queries")
        graph.add_edge("plan_queries", "retrieve_local")
        graph.add_edge("retrieve_local", "maybe_web_collect")
        graph.add_edge("maybe_web_collect", "rerank_context")
        graph.add_edge("rerank_context", "normalize_evidence")
        graph.add_edge("normalize_evidence", "source_quality_gate")
        graph.add_edge("source_quality_gate", "generate_answer")
        graph.add_edge("generate_answer", "materialize_quant_artifact")
        graph.add_edge("materialize_quant_artifact", "build_claim_graph")
        graph.add_edge("build_claim_graph", "resolve_contradictions")
        graph.add_edge("resolve_contradictions", "compliance_gate")
        graph.add_edge("compliance_gate", "verify_grounding")
        graph.add_edge("verify_grounding", "refine_or_finish")
        graph.add_conditional_edges(
            "refine_or_finish",
            self._route_after_refine,
            {"refine": "plan_queries", "finish": "log_trajectory"},
        )
        graph.add_edge("log_trajectory", END)
        return graph.compile()

    def _build_prepare_graph(self):
        graph = StateGraph(AgenticRAGState)
        graph.add_node("classify_intent", self._classify_intent_node)
        graph.add_node("build_claim_plan", self._build_claim_plan_node)
        graph.add_node("select_policy", self._select_policy_node)
        graph.add_node("plan_queries", self._plan_queries_node)
        graph.add_node("retrieve_local", self._retrieve_local_node)
        graph.add_node("maybe_web_collect", self._maybe_web_collect_node)
        graph.add_node("rerank_context", self._rerank_context_node)
        graph.add_node("normalize_evidence", self._normalize_evidence_node)
        graph.add_node("source_quality_gate", self._source_quality_gate_node)
        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "build_claim_plan")
        graph.add_edge("build_claim_plan", "select_policy")
        graph.add_edge("select_policy", "plan_queries")
        graph.add_edge("plan_queries", "retrieve_local")
        graph.add_edge("retrieve_local", "maybe_web_collect")
        graph.add_edge("maybe_web_collect", "rerank_context")
        graph.add_edge("rerank_context", "normalize_evidence")
        graph.add_edge("normalize_evidence", "source_quality_gate")
        graph.add_edge("source_quality_gate", END)
        return graph.compile()

    def run(
        self,
        query: str,
        system_prompt: str = "",
        task_type: Optional[str] = None,
        answer: str = "",
        reference: str = "",
        runtime_options: Optional[dict[str, Any]] = None,
    ) -> AgenticRAGState:
        state: AgenticRAGState = {
            "query": query,
            "system_prompt": system_prompt,
            "task_type": task_type or "",
            "answer": strip_thinking(answer),
            "reference": reference,
            "runtime_options": runtime_options or {},
            "trajectory": [],
            "errors": [],
            "refine_count": 0,
            "needs_refine": False,
            "answer_locked": bool(answer),
            "query_id": uuid.uuid4().hex,
            "started_at": time.time(),
            "execution_backend": self.execution_backend,
            "request_as_of": str((runtime_options or {}).get("as_of") or utc_now()),
            "evidence_records": [],
            "claims": [],
            "calculations": [],
        }
        if self.graph is not None:
            return self.graph.invoke(state)
        return self._run_sequential(state)

    def build_augmented_prompt(
        self,
        system_prompt: str,
        query: str,
        task_type: Optional[str] = None,
        runtime_options: Optional[dict[str, Any]] = None,
    ) -> str:
        prompt, _ = self.build_augmented_prompt_with_state(
            system_prompt=system_prompt,
            query=query,
            task_type=task_type,
            runtime_options=runtime_options,
        )
        return prompt

    def build_augmented_prompt_with_state(
        self,
        system_prompt: str,
        query: str,
        task_type: Optional[str] = None,
        runtime_options: Optional[dict[str, Any]] = None,
    ) -> tuple[str, AgenticRAGState]:
        """Prepare evidence for generation and return the state for final audit."""
        state = self.prepare(
            query=query,
            system_prompt=system_prompt,
            task_type=task_type,
            runtime_options=runtime_options,
        )
        context = state.get("selected_context", [])
        if not context:
            if self.retriever is not None and getattr(self.retriever, "is_ready", lambda: False)():
                top_k = int((runtime_options or {}).get("top_k", self.config.top_k))
                return self.retriever.format_prompt_with_context(system_prompt, query, top_k=top_k), state
            return system_prompt, state
        return format_agentic_prompt(system_prompt, context, state), state

    def prepare(
        self,
        query: str,
        system_prompt: str = "",
        task_type: Optional[str] = None,
        runtime_options: Optional[dict[str, Any]] = None,
    ) -> AgenticRAGState:
        """Run the observable environment up to evidence normalization."""
        state: AgenticRAGState = {
            "query": query,
            "system_prompt": system_prompt,
            "task_type": task_type or "",
            "answer": "",
            "reference": "",
            "runtime_options": runtime_options or {},
            "trajectory": [],
            "errors": [],
            "refine_count": 0,
            "needs_refine": False,
            "answer_locked": False,
            "query_id": uuid.uuid4().hex,
            "started_at": time.time(),
            "execution_backend": self.execution_backend,
            "request_as_of": str((runtime_options or {}).get("as_of") or utc_now()),
            "evidence_records": [],
            "claims": [],
            "calculations": [],
        }
        if self.prepare_graph is not None:
            return self.prepare_graph.invoke(state)
        for node in (
            self._classify_intent_node,
            self._build_claim_plan_node,
            self._select_policy_node,
            self._plan_queries_node,
            self._retrieve_local_node,
            self._maybe_web_collect_node,
            self._rerank_context_node,
            self._normalize_evidence_node,
            self._source_quality_gate_node,
        ):
            state = node(state)
        return state

    def finalize(self, prepared_state: AgenticRAGState, answer: str) -> AgenticRAGState:
        """Attach the model's actual completion, verify it, and persist its audit."""
        state = dict(prepared_state)
        state["answer"] = strip_thinking(answer)
        state["answer_locked"] = True
        for node in (
            self._generate_answer_node,
            self._materialize_quant_artifact_node,
            self._build_claim_graph_node,
            self._resolve_contradictions_node,
            self._compliance_gate_node,
            self._verify_grounding_node,
        ):
            state = node(state)
        guard_enabled = bool(state.get("runtime_options", {}).get("enable_answer_guard"))
        raw_failures = list(state.get("reward_breakdown", {}).get("hard_failures", []))
        if guard_enabled and raw_failures:
            guarded_answer = self._guarded_answer(state)
            if guarded_answer and guarded_answer != state.get("answer"):
                state = self._append_event(
                    state,
                    "answer_guard",
                    "evidence_extractive_repair",
                    {
                        "original_hard_failures": raw_failures,
                        "evidence_ids": [
                            item.get("evidence_id", "") for item in state.get("evidence_records", [])
                        ],
                    },
                )
                state["answer"] = guarded_answer
                state["answer_guard_applied"] = True
                for node in (
                    self._build_claim_graph_node,
                    self._resolve_contradictions_node,
                    self._compliance_gate_node,
                    self._verify_grounding_node,
                ):
                    state = node(state)
        state["needs_refine"] = False
        state = self._append_event(
            state,
            "refine_or_finish",
            "finish_guarded_completion" if state.get("answer_guard_applied") else "finish_external_completion",
            {"reward": state.get("reward_breakdown", {}).get("total_reward", 0.0)},
        )
        return self._log_trajectory_node(state)

    def _guarded_answer(self, state: AgenticRAGState) -> str:
        task_type = str(state.get("task_type", ""))
        evidence = list(state.get("evidence_records", []))
        if task_type not in {"financial_qa", "sentiment_analysis", "risk_assessment"} or not evidence:
            return ""

        financial = next(
            (item for item in evidence if "营业收入" in str(item.get("exact_quote", ""))),
            None,
        )
        market = next(
            (item for item in evidence if "年化波动率" in str(item.get("exact_quote", ""))),
            None,
        )
        if task_type == "financial_qa":
            lines = _cited_evidence_fragments(
                financial,
                keywords=("资产总计", "负债合计", "资产负债率", "流动资产", "流动负债", "流动比率"),
            )
            lines.append("观点：证据不足，无法确认指标的其他适用边界。")
            return "\n".join(lines)

        if task_type == "sentiment_analysis":
            lines = _cited_evidence_fragments(
                financial,
                keywords=("营业收入", "净利润", "经营活动现金流量净额", "同比变化"),
            )
            if financial:
                label = _fundamental_sentiment_label(str(financial.get("exact_quote", "")))
                lines.append(f"观点：基本面情绪标签为{label} [{financial.get('evidence_id', '')}]。")
            else:
                lines.append("证据不足，无法确认基本面情绪标签。")
            return "\n".join(lines)

        lines = _cited_evidence_fragments(
            financial,
            keywords=("净利润", "经营活动现金流量净额", "资产负债率", "流动比率"),
        )
        lines.extend(
            _cited_evidence_fragments(
                market,
                keywords=("年化波动率", "最大回撤"),
            )
        )
        cited_ids = [
            str(item.get("evidence_id", "")) for item in (financial, market) if item and item.get("evidence_id")
        ]
        if cited_ids:
            label = _risk_level(
                str(financial.get("exact_quote", "")) if financial else "",
                str(market.get("exact_quote", "")) if market else "",
            )
            citations = "".join(f"[{item}]" for item in cited_ids)
            lines.append(f"观点：基于上述财务与市场指标，风险等级为{label} {citations}。")
        else:
            lines.append("证据不足，无法确认风险等级。")
        lines.append("观点：证据不足，无法确认可验证的具体缓释措施。")
        return "\n".join(lines)

    def _run_sequential(self, state: AgenticRAGState) -> AgenticRAGState:
        nodes = [
            self._classify_intent_node,
            self._build_claim_plan_node,
            self._select_policy_node,
            self._plan_queries_node,
            self._retrieve_local_node,
            self._maybe_web_collect_node,
            self._rerank_context_node,
            self._normalize_evidence_node,
            self._source_quality_gate_node,
            self._generate_answer_node,
            self._materialize_quant_artifact_node,
            self._build_claim_graph_node,
            self._resolve_contradictions_node,
            self._compliance_gate_node,
            self._verify_grounding_node,
            self._refine_or_finish_node,
        ]
        state = dict(state)
        while True:
            for node in nodes:
                state = node(state)
            if self._route_after_refine(state) != "refine":
                break
            state["needs_refine"] = False
        return self._log_trajectory_node(state)

    def _option(self, state: AgenticRAGState, name: str) -> Any:
        return state.get("runtime_options", {}).get(name, getattr(self.config, name))

    def _append_event(
        self,
        state: AgenticRAGState,
        node: str,
        action: str,
        metrics: Optional[dict[str, Any]] = None,
    ) -> AgenticRAGState:
        new_state = dict(state)
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "timestamp": utc_now(),
            "step_id": len(new_state.get("trajectory", [])) + 1,
            "node": node,
            "action": action,
            "action_args": metrics or {},
            "observation_ids": list((metrics or {}).get("evidence_ids", [])),
            "verifier_scores": (metrics or {}).get("verifier_scores", {}),
            "metrics": metrics or {},
        }
        new_state.setdefault("trajectory", [])
        new_state["trajectory"] = list(new_state["trajectory"]) + [event]
        return new_state

    def _classify_intent_node(self, state: AgenticRAGState) -> AgenticRAGState:
        task_type = state.get("task_type") or classify_financial_task(state.get("query", ""))
        freshness = needs_fresh_knowledge(state.get("query", ""))
        state = dict(state)
        state["task_type"] = task_type
        state["freshness_required"] = freshness
        return self._append_event(
            state,
            "classify_intent",
            "classified",
            {"task_type": task_type, "freshness_required": freshness},
        )

    def _build_claim_plan_node(self, state: AgenticRAGState) -> AgenticRAGState:
        task_type = state.get("task_type", "financial_qa")
        query = state.get("query", "")
        requested_entities = sorted(set(re.findall(r"(?<!\d)\d{6}(?!\d)|[A-Z]{2,6}", query)))
        required = {
            "stock_analysis": ["point_in_time_market_data", "fundamental_or_event_facts", "risk_factors"],
            "financial_report": ["reported_period", "financial_metrics", "comparatives", "risk_factors"],
            "risk_assessment": ["risk_factors", "quantitative_indicators", "mitigants"],
            "quant_strategy": ["input_assumptions", "signal_definition", "risk_controls", "backtest_protocol"],
            "sentiment_analysis": ["target_text", "label_basis"],
            "financial_qa": ["authoritative_definition", "scope_and_limitations"],
        }.get(task_type, ["answerable_facts"])
        plan = {
            "task_type": task_type,
            "requested_entities": requested_entities,
            "required_claim_groups": required,
            "requires_point_in_time": bool(state.get("freshness_required")),
            "complex_task": bool(
                len(query) >= 80
                or task_type in {"stock_analysis", "quant_strategy", "financial_report", "risk_assessment"}
                or state.get("freshness_required")
            ),
            "reasoning_format": "structured_verifiable_summary",
        }
        state = dict(state)
        state["claim_plan"] = plan
        return self._append_event(state, "build_claim_plan", "planned", plan)

    def _select_policy_node(self, state: AgenticRAGState) -> AgenticRAGState:
        action = self.policy.select(
            state.get("task_type", "financial_qa"),
            bool(state.get("freshness_required", False)),
            state.get("runtime_options", {}),
        )
        state = dict(state)
        state["retrieval_plan"] = action.to_dict()
        return self._append_event(state, "select_policy", action.policy_name, action.to_dict())

    def _plan_queries_node(self, state: AgenticRAGState) -> AgenticRAGState:
        plan = state.get("retrieval_plan", {})
        limit = int(plan.get("multi_query_count", self._option(state, "multi_query_count")))
        base_query = state.get("query", "")
        if state.get("refine_count", 0) > 0:
            task_terms = " ".join(TASK_KEYWORDS.get(state.get("task_type", ""), []))
            base_query = f"{base_query} {task_terms} 依据 风险"

        variants = self._generate_query_variants(base_query, limit)
        state = dict(state)
        state["query_variants"] = variants
        return self._append_event(state, "plan_queries", "generated_variants", {"count": len(variants), "queries": variants})

    def _generate_query_variants(self, query: str, limit: int) -> list[str]:
        if self.retriever is not None and hasattr(self.retriever, "_generate_query_variants"):
            try:
                return self.retriever._generate_query_variants(query, limit)  # noqa: SLF001 - reuse local retriever logic
            except Exception as exc:
                logger.debug(f"Agentic RAG retriever query expansion failed: {exc}")
        variants = [normalize_text(query)]
        keywords = " ".join(sorted(tokenize_for_overlap(query))[:8])
        if keywords and keywords not in variants:
            variants.append(keywords)
        if len(variants) < limit:
            variants.append(f"{query} 财务指标 风险 影响因素")
        deduped = []
        for item in variants:
            if item and item not in deduped:
                deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _retrieve_local_node(self, state: AgenticRAGState) -> AgenticRAGState:
        plan = state.get("retrieval_plan", {})
        if not plan.get("use_local", True):
            state = dict(state)
            state["local_docs"] = []
            return self._append_event(state, "retrieve_local", "skipped")
        if self.retriever is None or not getattr(self.retriever, "is_ready", lambda: False)():
            state = dict(state)
            state["local_docs"] = []
            state["errors"] = list(state.get("errors", [])) + ["local_retriever_unavailable"]
            return self._append_event(state, "retrieve_local", "unavailable")

        top_k = int(plan.get("top_k", self._option(state, "top_k")))
        docs: list[dict[str, Any]] = []
        seen = set()
        for query in state.get("query_variants") or [state.get("query", "")]:
            try:
                results = self.retriever.retrieve(query, top_k=top_k, use_multi_query=False)
            except TypeError:
                results = self.retriever.retrieve(query, top_k=top_k)
            except Exception as exc:
                state = dict(state)
                state["errors"] = list(state.get("errors", [])) + [f"local_retrieve_failed:{exc}"]
                continue
            for item in results or []:
                key = _doc_key(item)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(item)
        requested_codes = sorted(set(re.findall(r"(?<!\d)\d{6}(?!\d)", state.get("query", ""))))
        entity_docs = []
        if requested_codes and hasattr(self.retriever, "retrieve_by_entity_codes"):
            try:
                entity_docs = self.retriever.retrieve_by_entity_codes(requested_codes)
            except Exception as exc:
                state = dict(state)
                state["errors"] = list(state.get("errors", [])) + [f"entity_retrieve_failed:{exc}"]
            for item in entity_docs:
                key = _doc_key(item)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(item)
        state = dict(state)
        state["local_docs"] = docs
        retrieved_evidence = normalize_evidence(
            docs,
            query=state.get("query", ""),
            observed_at=state.get("request_as_of") or utc_now(),
        )
        return self._append_event(
            state,
            "retrieve_local",
            "retrieved",
            {
                "count": len(docs),
                "queries": list(state.get("query_variants", [])),
                "top_k": top_k,
                "entity_lookup_codes": requested_codes,
                "entity_lookup_count": len(entity_docs),
                "evidence_ids": [item.evidence_id for item in retrieved_evidence],
            },
        )

    def _maybe_web_collect_node(self, state: AgenticRAGState) -> AgenticRAGState:
        plan = state.get("retrieval_plan", {})
        if not plan.get("use_web", False):
            state = dict(state)
            state["web_docs"] = []
            return self._append_event(state, "maybe_web_collect", "skipped")
        if self.web_agent is None:
            state = dict(state)
            state["web_docs"] = []
            state["errors"] = list(state.get("errors", [])) + ["web_agent_unavailable"]
            return self._append_event(state, "maybe_web_collect", "unavailable")

        query = state.get("query", "")
        try:
            saved_files = self.web_agent.collect(
                query,
                search_top_n=int(self._option(state, "web_search_top_n")),
                fetch_top_n=int(self._option(state, "web_fetch_top_n")),
            )
            if self.retriever is not None and hasattr(self.retriever, "refresh"):
                self.retriever.refresh()
            web_docs = []
            if self.retriever is not None and getattr(self.retriever, "is_ready", lambda: False)():
                web_docs = self.retriever.retrieve(query, top_k=int(plan.get("top_k", self._option(state, "top_k"))))
            state = dict(state)
            state["web_docs"] = web_docs
            return self._append_event(
                state,
                "maybe_web_collect",
                "collected",
                {"saved_files": len(saved_files or []), "web_docs": len(web_docs)},
            )
        except Exception as exc:
            state = dict(state)
            state["web_docs"] = []
            state["errors"] = list(state.get("errors", [])) + [f"web_collect_failed:{exc}"]
            return self._append_event(state, "maybe_web_collect", "failed", {"error": str(exc)})

    def _rerank_context_node(self, state: AgenticRAGState) -> AgenticRAGState:
        docs = list(state.get("local_docs", [])) + list(state.get("web_docs", []))
        requested_codes = set(re.findall(r"(?<!\d)\d{6}(?!\d)", state.get("query", "")))
        if requested_codes:
            entity_docs = [
                doc
                for doc in docs
                if requested_codes
                & set(
                    re.findall(
                        r"(?<!\d)\d{6}(?!\d)",
                        f"{doc.get('content', '')} {(doc.get('metadata') or {}).get('title', '')}",
                    )
                )
            ]
            if entity_docs:
                docs = entity_docs
        fact_docs = [
            doc for doc in docs if (doc.get("metadata") or {}).get("section_title") == "可引用事实"
        ]
        if fact_docs:
            docs = fact_docs
        query_terms = tokenize_for_overlap(state.get("query", ""))
        ranked = []
        seen = set()
        for doc in docs:
            key = _doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            content_terms = tokenize_for_overlap(doc.get("content", ""))
            overlap = len(query_terms & content_terms) / max(1, len(query_terms))
            score = float(doc.get("score") or 0.0)
            distance = doc.get("distance")
            distance_bonus = 0.0
            if isinstance(distance, (int, float)) and math.isfinite(distance):
                distance_bonus = max(0.0, 1.0 - min(distance, 2.0) / 2.0)
            item = dict(doc)
            item["agentic_score"] = round(0.55 * overlap + 0.25 * score + 0.20 * distance_bonus, 4)
            ranked.append(item)
        ranked.sort(key=lambda item: item.get("agentic_score", 0.0), reverse=True)

        selected = []
        total_chars = 0
        max_docs = int(self._option(state, "max_context_docs"))
        max_chars = int(self._option(state, "max_context_chars"))
        for doc in ranked:
            content = doc.get("content", "")
            if not content:
                continue
            if total_chars + len(content) > max_chars and selected:
                break
            selected.append(doc)
            total_chars += len(content)
            if len(selected) >= max_docs:
                break
        state = dict(state)
        state["selected_context"] = selected
        selected_evidence = normalize_evidence(
            selected,
            query=state.get("query", ""),
            observed_at=state.get("request_as_of") or utc_now(),
        )
        return self._append_event(
            state,
            "rerank_context",
            "selected",
            {
                "count": len(selected),
                "evidence_ids": [item.evidence_id for item in selected_evidence],
                "ranking_scores": [item.get("agentic_score", 0.0) for item in selected],
            },
        )

    def _normalize_evidence_node(self, state: AgenticRAGState) -> AgenticRAGState:
        records = normalize_evidence(
            state.get("selected_context", []),
            query=state.get("query", ""),
            observed_at=state.get("request_as_of") or utc_now(),
        )
        state = dict(state)
        state["evidence_records"] = [item.to_dict() for item in records]
        invalid = {
            item.evidence_id: item.validation_errors()
            for item in records
            if item.validation_errors()
        }
        return self._append_event(
            state,
            "normalize_evidence",
            "normalized",
            {
                "count": len(records),
                "evidence_ids": [item.evidence_id for item in records],
                "invalid_records": invalid,
            },
        )

    def _source_quality_gate_node(self, state: AgenticRAGState) -> AgenticRAGState:
        records = state.get("evidence_records", [])
        counts: dict[str, int] = {}
        for item in records:
            tier = str(item.get("reliability_tier", "unknown"))
            counts[tier] = counts.get(tier, 0) + 1
        invalid_ids = [
            item.get("evidence_id", "")
            for item in records
            if item.get("reliability_tier") == "unknown" or not item.get("source_uri")
        ]
        quality = {
            "passed": not invalid_ids,
            "tier_counts": counts,
            "invalid_evidence_ids": invalid_ids,
        }
        state = dict(state)
        state["evidence_quality"] = quality
        return self._append_event(
            state,
            "source_quality_gate",
            "passed" if quality["passed"] else "failed",
            {**quality, "evidence_ids": [item.get("evidence_id", "") for item in records]},
        )

    def _generate_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        if state.get("answer") and state.get("answer_locked"):
            return self._append_event(state, "generate_answer", "provided_answer")
        if self.answer_generator is not None:
            try:
                answer = self.answer_generator(state)
            except Exception as exc:
                answer = ""
                state = dict(state)
                state["errors"] = list(state.get("errors", [])) + [f"answer_generator_failed:{exc}"]
        else:
            answer = self._extractive_answer(state)
        state = dict(state)
        state["answer"] = answer
        return self._append_event(state, "generate_answer", "extractive" if self.answer_generator is None else "generated")

    def _materialize_quant_artifact_node(self, state: AgenticRAGState) -> AgenticRAGState:
        if state.get("task_type") != "quant_strategy":
            return self._append_event(state, "materialize_quant_artifact", "skipped")
        raw_answer = str(state.get("answer", ""))
        rendered, errors = materialize_quant_output(raw_answer, state.get("evidence_records", []))
        state = dict(state)
        state["raw_answer"] = raw_answer
        state["quant_action_errors"] = errors
        if errors:
            state["answer"] = "证据不足，无法确认量化 artifact；模型未返回有效 quant.action.v2.5。"
            state["errors"] = list(state.get("errors", [])) + errors
            return self._append_event(
                state,
                "materialize_quant_artifact",
                "rejected",
                {"errors": errors},
            )
        state["answer"] = str(rendered)
        return self._append_event(
            state,
            "materialize_quant_artifact",
            "rendered",
            {"artifact_version": "quant.v2.5"},
        )

    def _extractive_answer(self, state: AgenticRAGState) -> str:
        evidence = state.get("evidence_records", [])
        if not evidence:
            return "未检索到足够可靠的参考资料，建议补充信息后再判断。"
        lines = ["基于检索资料，可审计要点如下："]
        for idx, item in enumerate(evidence[:3], start=1):
            source = item.get("canonical_url") or item.get("source_uri") or "unknown"
            fetched_at = item.get("fetched_at") or "unknown"
            preview = normalize_text(item.get("exact_quote", ""))[:180]
            lines.append(f"{idx}. [{item['evidence_id']}] {preview}（来源: {source}；获取时间: {fetched_at}）")
        if state.get("task_type") in {"stock_analysis", "risk_assessment", "financial_report"}:
            lines.append("请结合最新数据复核，并注意风险提示；以上内容不构成投资建议。")
        return "\n".join(lines)

    def _build_claim_graph_node(self, state: AgenticRAGState) -> AgenticRAGState:
        claims = build_claims_from_answer(
            state.get("answer", ""),
            request_as_of=state.get("request_as_of", ""),
        )
        state = dict(state)
        state["claims"] = [item.to_dict() for item in claims]
        return self._append_event(
            state,
            "build_claim_graph",
            "mapped",
            {
                "claim_count": len(claims),
                "claim_ids": [item.claim_id for item in claims],
                "evidence_ids": sorted({evidence_id for claim in claims for evidence_id in claim.supporting_evidence_ids}),
            },
        )

    def _resolve_contradictions_node(self, state: AgenticRAGState) -> AgenticRAGState:
        records = state.get("evidence_records", [])
        conflicts = []
        opposites = (("增长", "下降"), ("盈利", "亏损"), ("利好", "利空"), ("改善", "恶化"))
        for left_index, left in enumerate(records):
            left_text = str(left.get("exact_quote", ""))
            left_entities = set(re.findall(r"(?<!\d)\d{6}(?!\d)", left_text))
            left_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", left_text))
            for right in records[left_index + 1 :]:
                right_text = str(right.get("exact_quote", ""))
                right_entities = set(re.findall(r"(?<!\d)\d{6}(?!\d)", right_text))
                right_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", right_text))
                # Lexical opposites alone are not enough: they commonly describe
                # different issuers or reporting periods in the same result set.
                if not left_entities or not right_entities or not (left_entities & right_entities):
                    continue
                if left_years and right_years and not (left_years & right_years):
                    continue
                reasons = [
                    f"{positive}_vs_{negative}"
                    for positive, negative in opposites
                    if (positive in left_text and negative in right_text) or (negative in left_text and positive in right_text)
                ]
                if reasons:
                    conflicts.append(
                        {
                            "evidence_ids": [left.get("evidence_id"), right.get("evidence_id")],
                            "reasons": reasons,
                            "status": "unresolved",
                        }
                    )
        state = dict(state)
        state["unresolved_conflicts"] = conflicts
        return self._append_event(
            state,
            "resolve_contradictions",
            "passed" if not conflicts else "unresolved",
            {
                "conflict_count": len(conflicts),
                "evidence_ids": sorted({item for conflict in conflicts for item in conflict["evidence_ids"] if item}),
            },
        )

    def _compliance_gate_node(self, state: AgenticRAGState) -> AgenticRAGState:
        answer = state.get("answer", "")
        patterns = [r"保证(?:收益|盈利)", r"稳赚", r"必涨", r"绝对不会亏", r"无风险收益", r"全仓(?:买入|卖出)", r"梭哈"]
        violations = [pattern for pattern in patterns if re.search(pattern, answer)]
        state = dict(state)
        state["compliance"] = {"passed": not violations, "violations": violations}
        return self._append_event(
            state,
            "compliance_gate",
            "passed" if not violations else "failed",
            {"violations": violations},
        )

    def _verify_grounding_node(self, state: AgenticRAGState) -> AgenticRAGState:
        evidence = [EvidenceRecord(**item) for item in state.get("evidence_records", [])]
        claims = []
        for payload in state.get("claims", []):
            payload = dict(payload)
            for key in ("supporting_evidence_ids", "contradicting_evidence_ids", "assumptions"):
                payload[key] = tuple(payload.get(key, ()))
            claims.append(ClaimRecord(**payload))
        calculations = []
        for payload in state.get("calculations", []):
            payload = dict(payload)
            payload["evidence_ids"] = tuple(payload.get("evidence_ids", ()))
            calculations.append(CalculationRecord(**payload))
        reward = compute_auditable_reward(
            query=state.get("query", ""),
            answer=state.get("answer", ""),
            evidence=evidence,
            claims=claims,
            calculations=calculations,
            task_type=state.get("task_type", "financial_qa"),
            request_as_of=state.get("request_as_of") or utc_now(),
            trajectory=state.get("trajectory", []),
            market_feedback=state.get("runtime_options", {}).get("market_feedback"),
            unresolved_conflicts=state.get("unresolved_conflicts", []),
            config=RewardV2Config(
                min_retrieval_relevance=float(self.config.min_retrieval_relevance),
                min_claim_support=float(self.config.min_claim_support),
                min_citation_coverage=float(self.config.min_citation_coverage),
                min_citation_precision=float(self.config.min_citation_precision),
                market_reward_cap=float(self.config.market_reward_cap),
                require_complete_trajectory=True,
            ),
        )
        verification = {
            "grounded": reward["claim_support"] >= 0.72,
            "has_context": bool(evidence),
            "hard_gate_passed": reward["hard_gate_passed"],
            "reward_ok": reward["hard_gate_passed"] and reward["total_reward"] >= float(self._option(state, "min_total_reward_for_finish")),
        }
        state = dict(state)
        state["reward_breakdown"] = reward
        state["verification"] = verification
        return self._append_event(
            state,
            "verify_grounding",
            "passed" if verification["reward_ok"] else "failed",
            {
                **reward,
                "verifier_scores": {
                    key: value
                    for key, value in reward.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
            },
        )

    def _refine_or_finish_node(self, state: AgenticRAGState) -> AgenticRAGState:
        reward = state.get("reward_breakdown", {}).get("total_reward", 0.0)
        refine_count = int(state.get("refine_count", 0))
        max_refine = int(self._option(state, "max_refine_steps"))
        plan = state.get("retrieval_plan", {})
        should_refine = bool(plan.get("refine_allowed", True)) and reward < float(self._option(state, "min_total_reward_for_finish")) and refine_count < max_refine
        state = dict(state)
        state["needs_refine"] = should_refine
        if should_refine:
            state["refine_count"] = refine_count + 1
            if not state.get("answer_locked"):
                state["answer"] = ""
            return self._append_event(state, "refine_or_finish", "refine", {"refine_count": state["refine_count"], "reward": reward})
        return self._append_event(state, "refine_or_finish", "finish", {"reward": reward})

    def _route_after_refine(self, state: AgenticRAGState) -> str:
        return "refine" if state.get("needs_refine") else "finish"

    def _log_trajectory_node(self, state: AgenticRAGState) -> AgenticRAGState:
        state = dict(state)
        state["finished_at"] = time.time()
        state = self._append_event(
            state,
            "log_trajectory",
            "logged" if self.config.enable_trajectory_logging else "disabled",
            {"latency_sec": round(state["finished_at"] - state.get("started_at", state["finished_at"]), 4)},
        )
        try:
            evidence = [EvidenceRecord(**item) for item in state.get("evidence_records", [])]
            claims = []
            for payload in state.get("claims", []):
                payload = dict(payload)
                for key in ("supporting_evidence_ids", "contradicting_evidence_ids", "assumptions"):
                    payload[key] = tuple(payload.get(key, ()))
                claims.append(ClaimRecord(**payload))
            events = [
                TrajectoryEvent(
                    step_id=int(item.get("step_id", index)),
                    timestamp=str(item.get("timestamp") or item.get("ts") or utc_now()),
                    node=str(item.get("node", "")),
                    action=str(item.get("action", "")),
                    action_args=dict(item.get("action_args") or item.get("metrics") or {}),
                    observation_ids=tuple(item.get("observation_ids", ())),
                    verifier_scores=dict(item.get("verifier_scores") or {}),
                    errors=tuple(item.get("errors", ())),
                )
                for index, item in enumerate(state.get("trajectory", []), start=1)
            ]
            calculations = []
            for payload in state.get("calculations", []):
                payload = dict(payload)
                payload["evidence_ids"] = tuple(payload.get("evidence_ids", ()))
                calculations.append(CalculationRecord(**payload))
            envelope = AuditEnvelope(
                query_id=state.get("query_id", ""),
                query=state.get("query", ""),
                task_type=state.get("task_type", ""),
                request_as_of=state.get("request_as_of", ""),
                final_answer=state.get("answer", ""),
                evidence=evidence,
                claims=claims,
                calculations=calculations,
                trajectory=events,
            )
            audit_payload = envelope.to_dict()
            audit_payload["validation_errors"] = envelope.validation_errors()
            audit_payload["reward"] = state.get("reward_breakdown", {})
            state["audit"] = audit_payload
        except Exception as exc:
            state["errors"] = list(state.get("errors", [])) + [f"audit_envelope_failed:{exc}"]
        if not self.config.enable_trajectory_logging:
            return state
        try:
            log_dir = resolve_project_path(self.config.trajectory_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_serializable_state(state), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Agentic RAG trajectory logging failed: {exc}")
        return state


def _doc_key(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    source = metadata.get("source") or metadata.get("url") or ""
    chunk = metadata.get("chunk_index", "")
    content = doc.get("content", "")
    return f"{source}:{chunk}:{content[:120]}"


def _cited_evidence_fragments(
    evidence: Optional[dict[str, Any]],
    *,
    keywords: tuple[str, ...],
) -> list[str]:
    if not evidence or not evidence.get("evidence_id"):
        return []
    text = re.sub(r"^#+\s*可引用事实\s*", "", str(evidence.get("exact_quote", "")).strip())
    fragments = [
        item.strip()
        for item in re.split(r"(?<=[。！？!?；;])", text)
        if item.strip() and any(keyword in item for keyword in keywords)
    ]
    evidence_id = str(evidence["evidence_id"])
    return [f"{fragment} [{evidence_id}]" for fragment in fragments]


def _metric(text: str, name: str) -> Optional[float]:
    match = re.search(rf"{re.escape(name)}(?:为|变化)?\s*([-+]?\d+(?:\.\d+)?)%?", text)
    return float(match.group(1)) if match else None


def _fundamental_sentiment_label(text: str) -> str:
    revenue_yoy = _metric(text, "营业收入同比变化")
    profit_yoy = _metric(text, "净利润同比变化")
    operating_cash_flow = _metric(text, "经营活动现金流量净额")
    if revenue_yoy is not None and profit_yoy is not None and operating_cash_flow is not None:
        if revenue_yoy > 0 and profit_yoy > 0 and operating_cash_flow > 0:
            return "positive"
        if (revenue_yoy < 0 and profit_yoy < 0) or (profit_yoy < 0 and operating_cash_flow < 0):
            return "negative"
    return "neutral"


def _risk_level(financial_text: str, market_text: str) -> str:
    leverage = _metric(financial_text, "资产负债率")
    liquidity = _metric(financial_text, "流动比率")
    volatility = _metric(market_text, "年化波动率")
    drawdown = _metric(market_text, "最大回撤")
    if (
        (leverage is not None and leverage >= 70)
        or (liquidity is not None and liquidity < 1)
        or (volatility is not None and volatility >= 50)
        or (drawdown is not None and drawdown <= -20)
    ):
        return "高"
    if (
        (leverage is not None and leverage >= 50)
        or (liquidity is not None and liquidity < 1.5)
        or (volatility is not None and volatility >= 30)
        or (drawdown is not None and drawdown <= -10)
    ):
        return "中"
    return "低"


def _serializable_state(state: AgenticRAGState) -> dict[str, Any]:
    allowed = {
        "query_id", "query", "task_type", "freshness_required", "retrieval_plan",
        "query_variants", "local_docs", "web_docs", "selected_context", "answer",
        "raw_answer", "quant_action_errors",
        "reference", "verification", "reward_breakdown", "trajectory", "errors",
        "refine_count", "execution_backend", "started_at", "finished_at", "request_as_of",
        "claim_plan", "evidence_records", "claims", "calculations", "compliance", "audit",
        "evidence_quality", "unresolved_conflicts",
        "answer_guard_applied",
    }
    return {key: state.get(key) for key in allowed if key in state}


def format_agentic_prompt(system_prompt: str, context: list[dict[str, Any]], state: AgenticRAGState) -> str:
    lines = []
    evidence_records = state.get("evidence_records", [])
    if evidence_records:
        for item in evidence_records:
            source = item.get("canonical_url") or item.get("source_uri") or "unknown"
            prefix = (
                f"[{item.get('evidence_id')}] 来源: {source}; 发布/生效: "
                f"{item.get('effective_at') or item.get('published_at') or 'unknown'}; "
                f"获取: {item.get('fetched_at') or 'unknown'}"
            )
            lines.append(f"{prefix}\n{normalize_text(item.get('exact_quote', ''))}")
    else:
        for idx, doc in enumerate(context, start=1):
            metadata = doc.get("metadata") or {}
            source = metadata.get("source") or metadata.get("url") or "unknown"
            lines.append(f"[{idx}] 来源: {source}\n{normalize_text(doc.get('content', ''))}")
    diagnostics = state.get("reward_breakdown", {})
    diagnostic_text = ""
    if diagnostics:
        diagnostic_text = (
            "\n\n【检索诊断】"
            f"\n任务类型: {state.get('task_type', 'unknown')}"
            f"\n执行后端: {state.get('execution_backend', 'unknown')}"
            f"\n检索奖励: {diagnostics.get('total_reward', 0.0)}"
        )
    return (
        f"{system_prompt}\n\n"
        "【Agentic RAG 参考知识库】\n"
        "以下资料由多步检索和证据标准化得到。每项事实或计算结论必须引用对应的 [Evidence ID]；"
        "不得把获取时间当成发布时间。如果资料不足或互相冲突，请明确拒答或披露冲突。\n\n"
        + "\n\n".join(lines)
        + "\n\n【最终答案硬约束】\n"
        "自然语言部分最多八行，每行只写一个完整结论，不要输出无引用的标题、表头或引言。"
        "每个事实、数值、计算、趋势判断、情绪标签和风险结论必须在同一行末尾引用支持它的 [Evidence ID]。"
        "需要多个证据时写成 [Evidence ID][Evidence ID]，不得把多个 ID 放在同一个方括号内。"
        "事实句应尽量直接摘录证据；逐字保留证据中的数值精度和正负号，不得四舍五入。"
        "趋势、情绪、风险等级、策略选择等推断句必须以‘观点：’开头并附证据；不得补充证据未提及的原因、行业比较、阈值或公司数据。"
        "证据不足时只写‘证据不足，无法确认该项。’，该句无需引用。"
        "量化任务的 Python 代码置于一个 fenced code block 中；代码外仍遵守逐行引用要求。"
        "不要使用‘无风险收益’等绝对化措辞，即使是否定语境也不要使用。"
        + diagnostic_text
    )


def build_preference_pair_from_states(chosen: AgenticRAGState, rejected: AgenticRAGState) -> dict[str, Any]:
    """Create a DPO-style preference sample from two graph trajectories."""
    chosen_reward = chosen.get("reward_breakdown", {}).get("total_reward", 0.0)
    rejected_reward = rejected.get("reward_breakdown", {}).get("total_reward", 0.0)
    if rejected_reward > chosen_reward:
        chosen, rejected = rejected, chosen
    return {
        "prompt": chosen.get("query", ""),
        "task_type": chosen.get("task_type", "financial_qa"),
        "chosen": chosen.get("answer", ""),
        "rejected": rejected.get("answer", ""),
        "chosen_reward": chosen.get("reward_breakdown", {}),
        "rejected_reward": rejected.get("reward_breakdown", {}),
        "metadata": {
            "source": "agentic_rag_trajectory",
            "chosen_query_id": chosen.get("query_id"),
            "rejected_query_id": rejected.get("query_id"),
        },
    }


def judge_available() -> bool:
    return bool(os.getenv("EVAL_JUDGE_API_KEY"))


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
