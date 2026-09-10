#!/usr/bin/env python3
"""SFT v2.4 no-think protocol, selection, and E2E decision tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data_processing.build_sft_v2_4_dataset import (
    TARGET_TASKS,
    _quant_errors,
    build_target,
    decision_basis,
    make_hard_negative,
    target_errors,
)
from scripts.evaluation.eval_sft_v2_4_e2e import _score_summary
from scripts.evaluation.select_sft_v2_4_checkpoint import (
    checkpoint_step,
    checkpoint_tag,
    discover_checkpoints,
    rank_key,
    replace_selected_directory,
)
from scripts.evaluation.sft_v2_4_protocol import case_prompt, score_answer
from scripts.rag.audit_schema import content_digest


AS_OF = "2026-09-01T00:00:00+08:00"
FINANCIAL_QUOTE = (
    "示例公司（证券代码600000）报告期2025-09-30的营业收入为100.0000亿元，"
    "归母口径净利润为10.0000亿元，经营活动现金流量净额为8.0000亿元；"
    "同比可比期2024-09-30的营业收入为90.0000亿元，归母口径净利润为9.0000亿元。"
    "营业收入同比变化11.1111%，净利润同比变化11.1111%。"
    "同期资产总计200.0000亿元、负债合计80.0000亿元、流动资产合计60.0000亿元、"
    "流动负债合计30.0000亿元。按负债合计除以资产总计计算，资产负债率为40.0000%；"
    "按流动资产合计除以流动负债合计计算，流动比率为2.0000。"
)
MARKET_QUOTE = (
    "示例公司（证券代码600000）截至2026-03-06收盘价为10.0000元，近20个交易日收益率为5.0000%。"
    "最近60个交易日按日收益率标准差乘以根号252计算的年化波动率为20.0000%，"
    "按收盘价相对历史滚动高点计算的最大回撤为-8.0000%。MA5为10.2000元，"
    "MA20为9.8000元，RSI6为60.0000。"
)


def _evidence(evidence_id: str, quote: str, suffix: str) -> dict:
    source = f"https://example.test/{suffix}"
    return {
        "evidence_id": evidence_id,
        "exact_quote": quote,
        "source": source,
        "source_uri": source,
        "canonical_url": source,
        "content_hash": content_digest(quote),
        "publisher": "example",
        "source_type": "news",
        "reliability_tier": "verified_secondary",
        "published_at": "2026-03-06T15:00:00+08:00",
        "effective_at": "2026-03-06T15:00:00+08:00",
        "fetched_at": "2026-03-07T00:00:00+08:00",
        "document_version": suffix,
        "point_in_time_available": True,
    }


FINANCIAL = _evidence("E1111111111", FINANCIAL_QUOTE, "financial")
MARKET = _evidence("E2222222222", MARKET_QUOTE, "market")


def _prompt(task_type: str, evidence: list[dict] | None = None) -> dict:
    prompt = {
        "query": f"针对示例公司（600000）完成{task_type}任务。",
        "task_type": task_type,
        "request_as_of": AS_OF,
        "evidence": evidence if evidence is not None else [FINANCIAL, MARKET],
    }
    prompt["decision_basis"] = decision_basis(task_type, prompt)
    return prompt


def _case(task_type: str) -> dict:
    evidence = [{key: value for key, value in item.items() if key != "source"} for item in (FINANCIAL, MARKET)]
    return {
        "id": f"case-{task_type}",
        "task_type": task_type,
        "question": f"针对示例公司（600000）完成{task_type}任务。",
        "request_as_of": AS_OF,
        "source_group": f"source-{task_type}",
        "requires_audit": True,
        "evidence": evidence,
        "scoring": {"mode": "patterns", "required_patterns": [], "forbidden_patterns": [], "pass_threshold": 0.0},
    }


@pytest.mark.parametrize("task_type", sorted(TARGET_TASKS))
def test_v2_4_targets_are_nothink_and_pass_strict_protocol(task_type: str):
    prompt = _prompt(task_type)
    target = build_target(task_type, prompt)
    assert "<think>" not in target.lower()
    assert target_errors(task_type, prompt, target) == []


@pytest.mark.parametrize(
    ("task_type", "evidence"),
    [("financial_qa", []), ("quant_strategy", []), ("risk_assessment", [FINANCIAL])],
)
def test_weak_tasks_have_explicit_missing_evidence_boundary(task_type: str, evidence: list[dict]):
    prompt = _prompt(task_type, evidence)
    target = build_target(task_type, prompt)
    assert "证据不足" in target
    assert target_errors(task_type, prompt, target) == []


def test_protocol_rejects_nonempty_think_numeric_drift_and_citation_drop():
    task_type = "financial_qa"
    case = _case(task_type)
    prompt = case_prompt(case)
    target = build_target(task_type, prompt)
    valid = score_answer(case, target)
    assert valid["audit_hard_gate_passed"] is True
    for rejected in (
        f"<think>隐藏推理</think>\n{target}",
        target.replace("40.0000%", "40.00%", 1),
        make_hard_negative(task_type, target, "citation-seed")[0],
    ):
        assert score_answer(case, rejected)["protocol_errors"]


def test_quant_gate_rejects_synthetic_data_and_wrong_schema():
    target = build_target("quant_strategy", _prompt("quant_strategy"))
    bad = target.replace("out = df.loc[:, [\"close\"]].copy()", "out = pd.DataFrame({'close': np.random.randn(20)})")
    assert any(error.startswith("quant_forbidden") for error in _quant_errors(bad))
    assert "quant_artifact_schema" in _quant_errors(target.replace("rolling(5)", "rolling(6)", 1))
    assert "quant_metadata_schema" in _quant_errors(target.replace("quant.v2.4", "quant.v2.3", 1))


def test_checkpoint_discovery_includes_final_adapter(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"final")
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 23}), encoding="utf-8")
    assert discover_checkpoints(tmp_path) == [checkpoint, tmp_path]
    assert checkpoint_step(tmp_path) == 23
    assert checkpoint_tag(tmp_path) == "final-step-23"


def test_selected_directory_overwrite_is_explicit_and_replaces_contents(tmp_path: Path):
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "old").write_text("old", encoding="utf-8")
    temporary = tmp_path / "selected.tmp"
    temporary.mkdir()
    (temporary / "new").write_text("new", encoding="utf-8")
    with pytest.raises(FileExistsError):
        replace_selected_directory(temporary, selected, overwrite=False)
    replace_selected_directory(temporary, selected, overwrite=True)
    assert (selected / "new").read_text(encoding="utf-8") == "new"
    assert not (selected / "old").exists()


def test_checkpoint_ranking_prioritizes_audit_before_primary_score():
    high_audit = {"audit_pass_rate": 0.96, "minimum_task_audit_pass_rate": 0.8, "pass_rate": 0.7, "mean_primary_score": 0.7, "length_finish_count": 0}
    high_primary = {"audit_pass_rate": 0.95, "minimum_task_audit_pass_rate": 0.9, "pass_rate": 0.9, "mean_primary_score": 0.9, "length_finish_count": 0}
    assert rank_key(high_audit, 40) > rank_key(high_primary, 20)


def _summary_row(task: str, *, candidate_at_1: bool, candidate_at_8: bool) -> dict:
    def score(passed: bool) -> dict:
        return {
            "passed": passed,
            "audit_hard_gate_passed": passed,
            "primary_score": float(passed),
            "protocol_errors": [],
        }
    return {
        "task_type": task,
        "source_group": task,
        "baseline_score": score(False),
        "candidate_score_at_1": score(candidate_at_1),
        "candidate_pass_at_8": candidate_at_8,
        "candidate_audit_at_8": candidate_at_8,
        "best_primary_at_8": float(candidate_at_8),
        "baseline_finish_reason": "stop",
        "candidate_finish_reason_at_1": "stop",
        "candidate_finish_reasons_at_8": ["stop"] * 8,
    }


def test_e2e_summary_only_considers_rl_when_pass_at_8_is_high_but_at_1_is_low():
    rows = [_summary_row(task, candidate_at_1=False, candidate_at_8=True) for task in sorted(TARGET_TASKS)]
    report = _score_summary(rows)
    assert report["candidate"]["e2e_at_1"] == 0.0
    assert report["candidate"]["e2e_at_8"] == 1.0
    assert report["rl_decision"]["consider_rl"] is True
    rows[0]["candidate_pass_at_8"] = False
    assert _score_summary(rows)["rl_decision"]["consider_rl"] is False
