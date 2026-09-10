#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for LangGraph/agentic-RL RAG orchestration."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def is_ready(self):
        return True

    def _generate_query_variants(self, query, limit):
        return [query, f"{query} 风险"][:limit]

    def retrieve(self, query, top_k=3, use_multi_query=None):
        self.calls.append((query, top_k, use_multi_query))
        return [
            {
                "content": "贵州茅台营收增长，净利润改善，现金流稳定，但估值和市场波动存在风险。",
                "metadata": {"source": "mock.md", "chunk_index": 0},
                "distance": 0.2,
                "score": 0.5,
            },
            {
                "content": "技术面显示趋势向上，成交量放大，MACD改善，RSI接近强势区间。",
                "metadata": {"source": "mock.md", "chunk_index": 1},
                "distance": 0.4,
                "score": 0.4,
            },
        ][:top_k]

    def retrieve_by_entity_codes(self, codes, limit_per_code=12):
        self.calls.append(("entity", tuple(codes), limit_per_code))
        return [
            {
                "content": "贵州茅台600519经营现金流稳定。",
                "metadata": {"source": "600519_financial.md", "chunk_index": 0},
                "score": 1.0,
            }
        ]

    def format_prompt_with_context(self, system_prompt, query, top_k=3):
        return f"{system_prompt}\nBASIC:{query}:{top_k}"


class EmptyRetriever:
    def is_ready(self):
        return False


def test_reward_scores_are_bounded_and_task_aware():
    from scripts.rag.agentic_rag import compute_rag_reward

    docs = [{"content": "风险 等级 波动 回撤 VaR 缓释 因素", "metadata": {"source": "risk.md"}}]
    reward = compute_rag_reward(
        query="评估该公司的风险等级",
        answer="风险等级为中等，关注波动、回撤和VaR，并通过分散和止损缓释风险。",
        docs=docs,
        task_type="risk_assessment",
    )
    assert 0.0 <= reward["total_reward"] <= 1.0
    assert reward["task_coverage"] > 0
    assert set(reward) >= {"retrieval_relevance", "groundedness", "cost_efficiency"}


def test_agentic_pipeline_sequential_fallback_logs_trajectory(tmp_path, monkeypatch):
    import scripts.rag.agentic_rag as agentic
    from scripts.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline

    monkeypatch.setattr(agentic, "StateGraph", None)
    config = AgenticRAGConfig(
        trajectory_log_dir=str(tmp_path),
        enable_trajectory_logging=True,
        allow_web=False,
        top_k=2,
        max_refine_steps=0,
    )
    pipeline = AgenticRAGPipeline(retriever=FakeRetriever(), config=config)
    state = pipeline.run("请分析贵州茅台600519的技术面和风险", system_prompt="你是金融助手")

    assert state["execution_backend"] == "sequential_fallback"
    assert state["selected_context"]
    assert any(call[0] == "entity" for call in pipeline.retriever.calls)
    assert state["reward_breakdown"]["total_reward"] >= 0
    trajectory_nodes = [event["node"] for event in state["trajectory"]]
    assert trajectory_nodes[0] == "classify_intent"
    assert "resolve_contradictions" in trajectory_nodes
    logs = list(tmp_path.glob("*.jsonl"))
    assert logs
    row = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[0])
    assert row["query"] == "请分析贵州茅台600519的技术面和风险"


def test_build_augmented_prompt_falls_back_when_no_context(monkeypatch):
    import scripts.rag.agentic_rag as agentic
    from scripts.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline

    monkeypatch.setattr(agentic, "StateGraph", None)
    pipeline = AgenticRAGPipeline(
        retriever=EmptyRetriever(),
        config=AgenticRAGConfig(enable_trajectory_logging=False),
    )
    prompt = pipeline.build_augmented_prompt("系统提示", "什么是市盈率")
    assert prompt == "系统提示"


def test_policy_uses_web_only_when_allowed_and_fresh():
    from scripts.rag.agentic_rag import AgenticRAGConfig, RuleBasedRAGPolicy

    policy = RuleBasedRAGPolicy(AgenticRAGConfig(allow_web=False))
    action = policy.select("financial_qa", True, {"allow_web": False})
    assert action.use_web is False

    action = policy.select("financial_qa", True, {"allow_web": True})
    assert action.use_web is True


def test_judge_available_false_without_key(monkeypatch):
    from scripts.rag.agentic_rag import judge_available

    monkeypatch.delenv("EVAL_JUDGE_API_KEY", raising=False)
    assert judge_available() is False


def test_contradiction_check_does_not_mix_entities_or_periods():
    from scripts.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline

    pipeline = AgenticRAGPipeline(
        retriever=EmptyRetriever(),
        config=AgenticRAGConfig(enable_trajectory_logging=False, require_langgraph=False),
    )
    state = pipeline._resolve_contradictions_node(
        {
            "trajectory": [],
            "evidence_records": [
                {"evidence_id": "E1111111111", "exact_quote": "公司600519在2025年营收增长。"},
                {"evidence_id": "E2222222222", "exact_quote": "公司300750在2025年营收下降。"},
                {"evidence_id": "E3333333333", "exact_quote": "公司600519在2024年营收下降。"},
            ],
        }
    )
    assert state["unresolved_conflicts"] == []


def test_reranker_prefers_target_entity_fact_chunks():
    from scripts.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline

    pipeline = AgenticRAGPipeline(
        retriever=EmptyRetriever(),
        config=AgenticRAGConfig(enable_trajectory_logging=False, require_langgraph=False),
    )
    state = pipeline._rerank_context_node(
        {
            "query": "分析目标公司600000",
            "request_as_of": "2026-09-01T00:00:00+08:00",
            "trajectory": [],
            "local_docs": [
                {
                    "content": "目标公司600000来源标题",
                    "metadata": {"title": "目标公司600000", "section_title": "标题"},
                    "score": 0.9,
                },
                {
                    "content": "目标公司600000营业收入100亿元。",
                    "metadata": {"title": "目标公司600000", "section_title": "可引用事实"},
                    "score": 0.5,
                },
                {
                    "content": "其他公司600001营业收入200亿元。",
                    "metadata": {"title": "其他公司600001", "section_title": "可引用事实"},
                    "score": 1.0,
                },
            ],
            "web_docs": [],
        }
    )
    assert [doc["content"] for doc in state["selected_context"]] == ["目标公司600000营业收入100亿元。"]


def test_langgraph_finalize_audits_actual_completion_without_freeform_thinking():
    import scripts.rag.agentic_rag as agentic
    from scripts.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline

    if agentic.StateGraph is None:
        pytest.skip("LangGraph is not installed")
    pipeline = AgenticRAGPipeline(
        retriever=FakeRetriever(),
        config=AgenticRAGConfig(
            enable_trajectory_logging=False,
            require_langgraph=True,
            allow_web=False,
        ),
    )
    prepared = pipeline.prepare("分析贵州茅台的营收、现金流、技术面趋势、成交量和风险")
    evidence_id = prepared["evidence_records"][0]["evidence_id"]
    completed = pipeline.finalize(
        prepared,
        f"<think>private free-form trace</think>贵州茅台营收增长，现金流稳定。[{evidence_id}]",
    )

    assert completed["execution_backend"] == "langgraph"
    assert "private free-form trace" not in completed["answer"]
    assert completed["audit"]["final_answer"] == completed["answer"]
    assert completed["audit"]["validation_errors"] == []
    assert completed["audit"]["reward"]["hard_gate_passed"] is True


def test_quant_completion_is_materialized_by_audited_graph_node(monkeypatch):
    import scripts.rag.agentic_rag as agentic
    from scripts.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline
    from scripts.rag.quant_protocol import canonical_quant_action

    monkeypatch.setattr(agentic, "StateGraph", None)
    pipeline = AgenticRAGPipeline(
        retriever=EmptyRetriever(),
        config=AgenticRAGConfig(enable_trajectory_logging=False),
    )
    prepared = pipeline.prepare("设计量化策略", task_type="quant_strategy")
    completed = pipeline.finalize(prepared, canonical_quant_action())

    assert completed["raw_answer"] == canonical_quant_action()
    assert "def quant_artifact(df):" in completed["answer"]
    assert completed["quant_action_errors"] == []
    assert any(
        event["node"] == "materialize_quant_artifact" and event["action"] == "rendered"
        for event in completed["trajectory"]
    )


def test_vector_builder_accepts_relative_data_directory(tmp_path, monkeypatch):
    import scripts.rag.build_vector_db as builder

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "source.md").write_text("# Source\n\nAuditable fact.", encoding="utf-8")
    observed = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builder, "chromadb", object())
    monkeypatch.setattr(builder, "SentenceTransformer", object())
    monkeypatch.setattr(
        builder,
        "index_documents",
        lambda **kwargs: observed.update(kwargs) or len(kwargs["docs"]),
    )

    builder.build_vector_db("knowledge", "vector-db", "fake-model", 500, 50)

    assert len(observed["docs"]) == 1
    assert observed["db_dir"] == str((tmp_path / "vector-db").resolve())
