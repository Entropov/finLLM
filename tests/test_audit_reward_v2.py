#!/usr/bin/env python3
"""Audit schema, reward red-team, SFT v2, and trusted-eval tests."""

import json
from dataclasses import replace
from types import SimpleNamespace

from scripts.data_processing.build_sft_v2_dataset import _supervision_target, build_dataset
from scripts.data_processing.build_sft_v2_1_dataset import (
    _answer_target_errors,
    _component_stats,
    _normalize_answer_sample,
)
from scripts.data_processing.prepare_agentic_v2_corpus import _training_question
from scripts.data_processing.collect_agentic_v2_trajectories import build_retry_system_prompt, select_tasks
from scripts.evaluation.eval_sft_v2_trusted import (
    dataset_fingerprint,
    evaluate,
    evidence_source_identities,
    validate_prediction_protocol,
)
from scripts.evaluation.validate_agentic_rewards import AS_OF, build_cases, run_validation, valid_trajectory
from scripts.rag.audit_schema import (
    AuditEnvelope,
    CalculationRecord,
    EvidenceRecord,
    TrajectoryEvent,
    build_claims_from_answer,
    content_digest,
    infer_source_quality,
)
from scripts.rag.reward_v2 import (
    RewardV2Config,
    _numeric_tokens,
    _query_relevance,
    compute_auditable_reward,
)


def _valid_audit_payload(query_id: str) -> dict:
    control = build_cases()[0]
    evidence = [
        replace(
            item,
            source_uri=f"{item.source_uri}/{query_id}",
            canonical_url=f"{item.canonical_url}/{query_id}",
            document_version=f"report-{query_id}",
        )
        for item in control.evidence
    ]
    envelope = AuditEnvelope(
        query_id=query_id,
        query=f"{control.query} case {query_id}",
        task_type=control.task_type,
        request_as_of=AS_OF,
        final_answer=control.answer,
        evidence=evidence,
        claims=list(control.claims),
        trajectory=[
            TrajectoryEvent(
                step_id=event["step_id"],
                timestamp=event["timestamp"],
                node=event["node"],
                action=event["action"],
                action_args=event["action_args"],
                observation_ids=tuple(event["observation_ids"]),
            )
            for event in valid_trajectory(tuple(item.evidence_id for item in evidence))
        ],
    )
    return {"audit": envelope.to_dict(), "claim_plan": {"required_claim_groups": ["financial_metrics"]}}


def test_reward_v2_rejects_all_known_attacks():
    report = run_validation()
    assert report["summary"]["release_gate_passed"] is True
    assert report["summary"]["attack_false_accept_rate"] == 0.0


def test_calculation_record_rejects_result_mismatch():
    calculation = CalculationRecord(
        calculation_id="CAL1",
        expression="(current - previous) / previous",
        inputs={"current": 1708, "previous": 1476},
        result=0.25,
        evidence_ids=("E1111111111",),
    )
    errors = calculation.validation_errors({"E1111111111"})
    assert "calculation_result_mismatch" in errors


def test_source_quality_rejects_exchange_suffix_spoof():
    source_type, tier, _ = infer_source_quality("https://evilsse.com.cn/fake", {})
    assert source_type == "news"
    assert tier == "unverified_secondary"


def test_claim_parser_ignores_markdown_structure_and_code_but_keeps_uncited_fact():
    claims = build_claims_from_answer(
        "### 核心指标\n营业收入为10亿元。\n```python\ndef signal(x):\n    return x > 0\n```\n"
        "净利润为2亿元 [E1111111111]。"
    )
    assert [claim.statement for claim in claims] == ["营业收入为10亿元。", "净利润为2亿元 。"]
    assert claims[0].supporting_evidence_ids == ()
    assert claims[1].supporting_evidence_ids == ("E1111111111",)


def test_explicit_opinion_with_numbers_remains_linked_to_evidence():
    claim = build_claims_from_answer("观点：资产负债率80%意味着杠杆风险较高 [E1111111111]。")[0]
    assert claim.claim_type == "opinion"
    assert claim.supporting_evidence_ids == ("E1111111111",)


def test_reward_requires_citations_and_numeric_consistency_for_opinions():
    control = build_cases()[0]
    evidence_id = control.evidence[0].evidence_id
    uncited_answer = "观点：风险等级较高。"
    uncited = compute_auditable_reward(
        query=control.query,
        answer=uncited_answer,
        evidence=control.evidence,
        claims=build_claims_from_answer(uncited_answer, AS_OF),
        task_type=control.task_type,
        request_as_of=AS_OF,
        config=RewardV2Config(require_complete_trajectory=False),
    )
    mismatch_answer = f"观点：资产负债率为999% [{evidence_id}]。"
    mismatch = compute_auditable_reward(
        query=control.query,
        answer=mismatch_answer,
        evidence=control.evidence,
        claims=build_claims_from_answer(mismatch_answer, AS_OF),
        task_type=control.task_type,
        request_as_of=AS_OF,
        config=RewardV2Config(require_complete_trajectory=False),
    )
    assert "insufficient_citation_coverage" in uncited["hard_failures"]
    assert "numeric_mismatch" in mismatch["hard_failures"]


def test_opinion_scope_covers_sentences_on_same_line_and_citations_are_per_sentence():
    claims = build_claims_from_answer(
        "观点：杠杆风险较高 [E1111111111]。短期流动性也需关注 [ E2222222222 ]。"
    )
    assert [claim.claim_type for claim in claims] == ["opinion", "opinion"]
    assert claims[0].supporting_evidence_ids == ("E1111111111",)
    assert claims[1].supporting_evidence_ids == ("E2222222222",)


def test_explicit_abstention_does_not_require_a_citation():
    claim = build_claims_from_answer("证据不足，无法确认该项。")[0]
    assert claim.claim_type == "opinion"
    assert claim.validation_errors(set(), set()) == []


def test_numeric_tokens_normalize_formatting_and_exclude_dates():
    assert _numeric_tokens("截至2025年9月30日，同比增长+0.3908%，金额93.1600元") == {"0.3908%", "93.16"}
    assert _numeric_tokens("截至2025-09-30，同比增长0.3908%，金额93.16元") == {"0.3908%", "93.16"}
    assert _numeric_tokens("2025年第三季度与2025 Q3的金额均为93.1600元") == {"93.16"}


def test_numeric_tokens_ignore_field_identifier_ordinals():
    claims = build_claims_from_answer("факт_1：营业收入为588.1426亿元 [E1111111111]")
    assert claims[0].statement == "факт_1：营业收入为588.1426亿元"
    assert _numeric_tokens(claims[0].statement) == {"588.1426"}


def test_query_relevance_uses_requested_issuer_identity():
    control = build_cases()[0]
    relevance, entity_match = _query_relevance(
        "针对示例公司（600000），分析最新指标并逐项引用证据，不作收益保证。",
        [replace(control.evidence[0], exact_quote="示例公司（证券代码600000）的营业收入为100亿元。")],
    )
    assert entity_match is True
    assert relevance >= 0.4


def test_structured_thinking_is_only_added_when_requested():
    assert _supervision_target({"step": "verify"}, "answer", False) == "answer"
    assert _supervision_target({"step": "verify"}, "answer", True).startswith("<think>\n")


def test_v2_1_answer_profile_aligns_contract_and_disables_thinking_for_sentiment():
    sample = {
        "id": "sentiment:answer",
        "group_id": "sentiment",
        "task_type": "sentiment_analysis",
        "sample_kind": "auditable_answer",
        "system": "old",
        "conversations": [
            {"from": "human", "value": '{"query":"label this","claim_plan":{"complex_task":true}}'},
            {"from": "gpt", "value": "<think>\n{\"step\":\"answer\"}\n</think>\npositive [E1111111111]"},
        ],
    }
    normalized = _normalize_answer_sample(sample)
    prompt = json.loads(normalized["conversations"][0]["value"])
    assert prompt["claim_plan"]["complex_task"] is False
    assert "output_contract" in prompt
    assert "<think>" not in normalized["conversations"][1]["value"]
    assert normalized["training_component"] == "answer"


def test_v2_1_component_stats_reject_cyrillic_targets():
    rows = [
        {
            "id": "x",
            "group_id": "g",
            "task_type": "sentiment_analysis",
            "sample_kind": "auditable_answer",
            "conversations": [
                {"from": "human", "value": "prompt"},
                {"from": "gpt", "value": "факт_1"},
            ],
        }
    ]
    assert _component_stats(rows)["cyrillic_targets"] == 1


def test_v2_1_answer_gate_requires_per_claim_citations_and_matching_numbers():
    base = {
        "id": "answer",
        "group_id": "g",
        "task_type": "financial_report",
        "sample_kind": "auditable_answer",
        "conversations": [
            {
                "from": "human",
                "value": json.dumps(
                    {
                        "request_as_of": AS_OF,
                        "evidence": [{"evidence_id": "E1111111111", "exact_quote": "营业收入为10亿元。"}],
                    },
                    ensure_ascii=False,
                ),
            },
            {"from": "gpt", "value": "营业收入为10亿元 [E1111111111]。"},
        ],
    }
    assert _answer_target_errors(base) == []
    uncited = {**base, "conversations": [base["conversations"][0], {"from": "gpt", "value": "营业收入为10亿元。"}]}
    mismatch = {
        **base,
        "conversations": [base["conversations"][0], {"from": "gpt", "value": "营业收入为11亿元 [E1111111111]。"}],
    }
    assert "claim_missing_citation" in _answer_target_errors(uncited)
    assert "claim_numeric_mismatch" in _answer_target_errors(mismatch)


def test_v2_1_request_variants_extend_without_changing_existing_variants():
    issuer = SimpleNamespace(name="示例公司", code="600000")
    variants = [_training_question("risk_assessment", issuer, index) for index in range(6)]
    assert len(set(variants)) == 6
    assert "基于盈利、现金流" in variants[0]
    assert "建立高、中、低风险判断" in variants[2]


def test_agentic_collection_retry_includes_previous_answer_and_task_repair_contract():
    prompt = build_retry_system_prompt(
        "base",
        task_type="financial_qa",
        previous_answer="18 / 100 = 18%",
        failures=["numeric_mismatch", "insufficient_citation_coverage"],
    )
    assert "18 / 100 = 18%" in prompt
    assert "numeric_mismatch" in prompt
    assert "不要写除法算式" in prompt
    assert "数字必须从证据逐字复制" in prompt


def test_agentic_collection_can_select_only_shortfall_tasks():
    rows = [
        {"id": "a", "task_type": "financial_qa"},
        {"id": "b", "task_type": "stock_analysis"},
        {"id": "c", "task_type": "risk_assessment"},
    ]
    assert [row["id"] for row in select_tasks(rows, ["financial_qa", "risk_assessment"])] == ["a", "c"]


def test_answer_guard_reaudits_evidence_extractive_risk_fallback():
    from scripts.rag.agentic_rag import AgenticRAGConfig, AgenticRAGPipeline

    financial_quote = (
        "## 可引用事实\n示例公司（证券代码600000）报告期2025-09-30的营业收入为100.0000亿元，"
        "归母口径净利润为10.0000亿元，经营活动现金流量净额为8.0000亿元；"
        "按负债合计除以资产总计计算，资产负债率为40.0000%；"
        "按流动资产合计除以流动负债合计计算，流动比率为2.0000。"
    )
    market_quote = (
        "## 可引用事实\n示例公司（证券代码600000）最近60个交易日按日收益率标准差乘以根号252计算的"
        "年化波动率为20.0000%，按收盘价相对历史滚动高点计算的最大回撤为-5.0000%。"
    )
    evidence = [
        EvidenceRecord(
            evidence_id="E1111111111",
            source_uri="https://example.test/financial",
            canonical_url="https://example.test/financial",
            publisher="example",
            source_type="internal",
            reliability_tier="internal",
            exact_quote=financial_quote,
            content_hash=content_digest(financial_quote),
            document_version="financial-v1",
            fetched_at=AS_OF,
            effective_at=AS_OF,
        ),
        EvidenceRecord(
            evidence_id="E2222222222",
            source_uri="https://example.test/market",
            canonical_url="https://example.test/market",
            publisher="example",
            source_type="internal",
            reliability_tier="internal",
            exact_quote=market_quote,
            content_hash=content_digest(market_quote),
            document_version="market-v1",
            fetched_at=AS_OF,
            effective_at=AS_OF,
        ),
    ]
    pipeline = AgenticRAGPipeline(
        retriever=None,
        config=AgenticRAGConfig(enable_trajectory_logging=False, require_langgraph=False),
    )
    state = {
        "query_id": "guard-test",
        "query": "评估示例公司600000的财务和市场风险",
        "task_type": "risk_assessment",
        "request_as_of": AS_OF,
        "runtime_options": {"enable_answer_guard": True},
        "trajectory": valid_trajectory(tuple(item.evidence_id for item in evidence)),
        "evidence_records": [item.to_dict() for item in evidence],
        "calculations": [],
        "claims": [],
        "errors": [],
        "started_at": 0.0,
    }
    completed = pipeline.finalize(state, "风险很高，无需引用。")

    assert completed["answer_guard_applied"] is True
    assert completed["reward_breakdown"]["hard_gate_passed"] is True
    assert completed["audit"]["validation_errors"] == []
    assert "[E1111111111][E2222222222]" in completed["answer"]
    assert any(event["node"] == "answer_guard" for event in completed["trajectory"])


def test_sft_v2_builder_keeps_groups_disjoint():
    train, evaluation, report = build_dataset(
        [_valid_audit_payload("q1"), _valid_audit_payload("q2")],
        min_reward=0.65,
        eval_ratio=0.5,
        seed=42,
        min_per_task=0,
    )
    assert train and evaluation
    assert report["query_overlap"] == 0
    assert report["source_identity_overlap"] == 0
    assert report["release_gate_passed"] is True
    assert {row["group_id"] for row in train}.isdisjoint({row["group_id"] for row in evaluation})
    assert all("<think>" in row["conversations"][-1]["value"] for row in train + evaluation)


def test_trusted_eval_is_paired_and_detects_improvement():
    cases = [
        {
            "id": "sentiment-1",
            "task_type": "sentiment_analysis",
            "question": "The issuer raised guidance. Label sentiment.",
            "request_as_of": AS_OF,
            "source_group": "issuer-a-event-1",
            "requires_audit": False,
            "scoring": {"mode": "label", "accepted_labels": ["positive"]},
        }
    ]
    report = evaluate(
        cases,
        {"sentiment-1": "negative"},
        {"sentiment-1": "positive"},
        train_question_hashes=set(),
        train_source_groups=set(),
    )
    assert report["paired_comparison"]["improvements"] == 1
    assert report["paired_comparison"]["regressions"] == 0
    assert report["candidate"]["pass_rate"] == 1.0
    assert report["candidate"]["cyrillic_output_count"] == 0
    assert report["release_gate"]["per_task_primary_score_noninferiority"] is True
    assert report["release_gate"]["no_cyrillic_candidate_outputs"] is True


def test_trusted_eval_rejects_cyrillic_output_and_large_per_task_score_regression():
    cases = [
        {
            "id": "sentiment-1",
            "task_type": "sentiment_analysis",
            "question": "The issuer raised guidance. Label sentiment.",
            "request_as_of": AS_OF,
            "source_group": "issuer-a-event-1",
            "requires_audit": False,
            "scoring": {"mode": "label", "accepted_labels": ["positive"]},
        }
    ]
    report = evaluate(
        cases,
        {"sentiment-1": "positive"},
        {"sentiment-1": "негативный"},
        train_question_hashes=set(),
        train_source_groups=set(),
    )
    assert report["candidate"]["cyrillic_output_ids"] == ["sentiment-1"]
    assert report["by_task"]["sentiment_analysis"]["primary_score_delta"] < -0.02
    assert report["release_gate"]["per_task_primary_score_noninferiority"] is False
    assert report["release_gate"]["no_cyrillic_candidate_outputs"] is False


def test_trusted_eval_compares_evidence_source_identity_for_leakage():
    case = {
        "id": "financial-1",
        "task_type": "financial_qa",
        "question": "Explain the ratio.",
        "request_as_of": AS_OF,
        "source_group": "heldout-issuer-a",
        "requires_audit": False,
        "evidence": [
            {
                "source_uri": "https://example.test/report",
                "canonical_url": "https://example.test/report",
                "document_version": "v1",
                "exact_quote": "The reported ratio is 1.5.",
            }
        ],
        "scoring": {"mode": "patterns", "required_patterns": ["ratio"]},
    }
    identity = next(iter(evidence_source_identities(case)))
    report = evaluate(
        [case],
        {"financial-1": "ratio"},
        {"financial-1": "ratio"},
        train_question_hashes=set(),
        train_source_groups={identity},
    )
    assert report["data_leakage"]["count"] == 1
    assert report["release_gate"]["no_detected_leakage"] is False


def test_prediction_protocol_requires_identical_per_case_prompts():
    case = {
        "id": "sentiment-1",
        "task_type": "sentiment_analysis",
        "question": "Label sentiment.",
        "request_as_of": AS_OF,
        "source_group": "issuer-a",
        "requires_audit": False,
        "scoring": {"mode": "label", "accepted_labels": ["positive"]},
    }
    common = {
        "manifest_version": "trusted_predictions.v1",
        "dataset_fingerprint": dataset_fingerprint([case]),
        "expected_samples": 1,
        "completed_samples": 1,
        "model": {"exists": True, "resolved_path": "/models/base"},
        "generation_settings": {"temperature": 0.0},
        "prompt_contract": {
            "fixed_case_evidence_only": True,
            "gold_scoring_hidden": True,
            "system_prompt_sha256": "abc",
            "generator_sha256": "def",
        },
    }
    baseline_manifest = {
        **common,
        "arm": "baseline",
        "adapter": None,
        "predictions": [{"id": "sentiment-1", "prompt_sha256": "prompt-a"}],
    }
    candidate_manifest = {
        **common,
        "arm": "candidate",
        "adapter": {
            "exists": True,
            "resolved_path": "/models/adapter",
            "weight_files": [{"path": "adapter.safetensors", "sha256": "weights"}],
        },
        "predictions": [{"id": "sentiment-1", "prompt_sha256": "prompt-b"}],
    }
    protocol = validate_prediction_protocol(
        [case],
        {"sentiment-1": "positive"},
        {"sentiment-1": "positive"},
        baseline_manifest,
        candidate_manifest,
    )
    assert protocol["passed"] is False
    assert "paired:prompt_hash_mismatch" in protocol["errors"]
