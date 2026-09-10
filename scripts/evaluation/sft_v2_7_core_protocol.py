"""SFT v2.7 protocol restricted to the three retained core tasks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scripts.evaluation import sft_v2_6_protocol as v26
from scripts.evaluation.sft_v2_7_scope import CORE_TASKS, require_core_task


def case_prompt(case: dict[str, Any]) -> dict[str, Any]:
    require_core_task(str(case.get("task_type")))
    return v26.case_prompt(case)


def render_prompt(tokenizer: Any, case: dict[str, Any]) -> str:
    require_core_task(str(case.get("task_type")))
    return v26.render_prompt(tokenizer, case)


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    require_core_task(str(case.get("task_type")))
    return v26.score_answer(case, answer)


def prompt_contract(generator_path: str = "") -> dict[str, Any]:
    contract = dict(v26.prompt_contract())
    contract["contract_version"] = "sft_v2.7-core"
    contract["core_tasks"] = sorted(CORE_TASKS)
    contract["excluded_from_optimization"] = ["financial_report", "risk_assessment", "sentiment_analysis"]
    if generator_path:
        contract["generator_sha256"] = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    return contract
