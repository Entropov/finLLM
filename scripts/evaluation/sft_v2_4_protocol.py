#!/usr/bin/env python3
"""Shared prompt and scoring protocol for SFT v2.4 audit evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from scripts.data_processing.build_sft_v2_4_dataset import (
    STRICT_CONTRACT,
    SYSTEM_PROMPT,
    TASK_CONTRACTS,
    _quant_errors,
    decision_basis,
    target_errors,
)
from scripts.evaluation.eval_sft_v2_trusted import get_question, score_case
from scripts.inference.chat_template import apply_chat_template


def case_prompt(case: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for raw in case.get("evidence", []):
        evidence.append(
            {
                "evidence_id": raw["evidence_id"],
                "exact_quote": raw["exact_quote"],
                "source": raw.get("canonical_url") or raw.get("source_uri", ""),
                "publisher": raw.get("publisher", ""),
                "reliability_tier": raw.get("reliability_tier", ""),
                "published_at": raw.get("published_at", ""),
                "effective_at": raw.get("effective_at", ""),
                "fetched_at": raw.get("fetched_at", ""),
            }
        )
    prompt = {
        "query": get_question(case),
        "task_type": str(case["task_type"]),
        "request_as_of": str(case["request_as_of"]),
        "evidence": evidence,
        "strict_audit_contract": STRICT_CONTRACT,
        "output_contract": TASK_CONTRACTS[str(case["task_type"])],
    }
    prompt["decision_basis"] = decision_basis(str(case["task_type"]), prompt)
    return prompt


def render_prompt(tokenizer: Any, case: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(case_prompt(case), ensure_ascii=False, separators=(",", ":"))},
    ]
    return apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=False)


def prompt_contract(generator_path: str = "") -> dict[str, Any]:
    payload = {
        "contract_version": "sft_v2.4",
        "fixed_case_evidence_only": True,
        "gold_scoring_hidden": True,
        "enable_thinking": False,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    if generator_path:
        with open(generator_path, "rb") as handle:
            payload["generator_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    return payload


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    result = score_case(case, answer)
    protocol_errors = target_errors(str(case["task_type"]), case_prompt(case), answer)
    nonempty_thinking = any(
        match.group(1).strip()
        for match in re.finditer(r"<think>(.*?)</think>", answer, flags=re.DOTALL | re.IGNORECASE)
    )
    if nonempty_thinking and "nonempty_thinking" not in protocol_errors:
        protocol_errors.append("nonempty_thinking")
    if str(case["task_type"]) == "quant_strategy":
        protocol_errors.extend(item for item in _quant_errors(answer) if item not in protocol_errors)
    protocol_errors = sorted(set(protocol_errors))
    if protocol_errors:
        result["audit_hard_gate_passed"] = False
        result["audit_score"] = 0.0
        result["primary_score"] = round(0.7 * float(result["task_score"]), 4)
        result["passed"] = False
    result["protocol_errors"] = protocol_errors
    return result
