"""Strict stock contract used by the v2.8 source-disjoint evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.data_processing.build_sft_v2_6_dataset import SYSTEM_PROMPT
from scripts.evaluation import sft_v2_6_protocol as v26
from scripts.evaluation.task_aware_verifier_v2_7_core import validate_task_answer
from scripts.inference.chat_template import apply_chat_template


CONTRACT_VERSION = "sft_v2.8-stock-contract.v1"


def case_prompt(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("task_type") != "stock_analysis":
        raise ValueError("v2.8 stock contract accepts stock_analysis only")
    prompt = v26.case_prompt(case)
    prompt["output_contract"] = {
        "format": "compact_atomic_lines",
        "required": ["cited price or return", "cited MA5 and MA20 trend", "cited volatility or drawdown risk", "one cited financial fact"],
        "prohibited": ["unsupported price forecast", "return guarantee", "uncited atomic claim"],
        "abstention": "only when evidence for the requested fact is absent",
    }
    return prompt


def render_prompt(tokenizer: Any, case: dict[str, Any]) -> str:
    return apply_chat_template(
        tokenizer,
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": json.dumps(case_prompt(case), ensure_ascii=False, separators=(",", ":"))}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_contract(generator_path: str = "") -> dict[str, Any]:
    contract = dict(v26.prompt_contract())
    contract.update({"contract_version": CONTRACT_VERSION, "core_tasks": ["stock_analysis"], "stock_contract_visible_to_model": True})
    if generator_path:
        contract["generator_sha256"] = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    return contract


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Combine normal audit scoring with the explicit visible-evidence contract."""
    if case.get("task_type") != "stock_analysis":
        raise ValueError("v2.8 stock contract accepts stock_analysis only")
    result = v26.score_answer(case, answer)
    verdict = validate_task_answer(case, answer)
    contract_passed = bool(verdict["hard_gate_passed"])
    result["task_score"] = 1.0 if contract_passed else 0.0
    result["task_details"] = {"mode": "evidence_anchor_stock_contract", **verdict}
    result["primary_score"] = round(0.7 * result["task_score"] + 0.3 * result["audit_score"], 4)
    result["passed"] = contract_passed and bool(result["audit_hard_gate_passed"])
    return result
