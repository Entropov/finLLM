"""SFT v2.6 compact-claim evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scripts.data_processing import build_sft_v2_4_dataset as v24
from scripts.data_processing.build_sft_v2_5_dataset import rendered_quant_errors
from scripts.data_processing.build_sft_v2_6_dataset import (
    STRICT_CONTRACT,
    SYSTEM_PROMPT,
    TASK_CONTRACTS,
    _target_errors as target_errors,
)
from scripts.evaluation.eval_sft_v2_trusted import get_question, score_case
from scripts.inference.chat_template import apply_chat_template
from scripts.rag.quant_protocol import QUANT_CODE, canonical_quant_action, materialize_quant_output


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
    task_type = str(case["task_type"])
    prompt = {
        "query": get_question(case),
        "task_type": task_type,
        "request_as_of": str(case["request_as_of"]),
        "evidence": evidence,
        "strict_audit_contract": STRICT_CONTRACT,
        "output_contract": TASK_CONTRACTS[task_type],
    }
    prompt["decision_basis"] = v24.decision_basis(task_type, prompt)
    prompt["decision_basis"]["response_budget"] = (
        "one_json_line" if task_type == "quant_strategy" else "compact_atomic_lines"
    )
    return prompt


def render_prompt(tokenizer: Any, case: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(case_prompt(case), ensure_ascii=False, separators=(",", ":"))},
    ]
    return apply_chat_template(tokenizer, messages, add_generation_prompt=True, enable_thinking=False)


def prompt_contract(generator_path: str = "") -> dict[str, Any]:
    payload = {
        "contract_version": "sft_v2.6",
        "fixed_case_evidence_only": True,
        "gold_scoring_hidden": True,
        "enable_thinking": False,
        "quant_model_output": canonical_quant_action(),
        "quant_renderer_sha256": hashlib.sha256(QUANT_CODE.encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    if generator_path:
        payload["generator_sha256"] = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    return payload


def materialize_answer(case: dict[str, Any], raw_answer: str) -> tuple[str, list[str]]:
    if str(case["task_type"]) != "quant_strategy":
        return raw_answer, []
    rendered, errors = materialize_quant_output(raw_answer, case_prompt(case).get("evidence", []))
    return (rendered or raw_answer), errors


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    task_type = str(case["task_type"])
    prompt = case_prompt(case)
    scored_answer, materialization_errors = materialize_answer(case, answer)
    result = score_case(case, scored_answer)
    errors = list(materialization_errors)
    if task_type == "quant_strategy" and not errors:
        errors.extend(rendered_quant_errors(prompt, scored_answer))
    elif task_type != "quant_strategy":
        errors.extend(target_errors(task_type, prompt, answer))
    if any(match.group(1).strip() for match in re.finditer(r"<think>(.*?)</think>", answer, re.S | re.I)):
        errors.append("nonempty_thinking")
    result["protocol_errors"] = sorted(set(errors))
    result["materialized"] = task_type == "quant_strategy" and not materialization_errors
    if result["protocol_errors"]:
        result["audit_hard_gate_passed"] = False
        result["audit_score"] = 0.0
        result["primary_score"] = round(0.7 * float(result["task_score"]), 4)
        result["passed"] = False
    return result
