"""Task scope for the SFT v2.7 core financial-agent iteration."""

from __future__ import annotations

CORE_TASKS = frozenset({"financial_qa", "quant_strategy", "stock_analysis"})


def require_core_task(task_type: str) -> str:
    """Return a supported core task or fail before scoring/training it."""
    if task_type not in CORE_TASKS:
        raise ValueError(f"task is outside the SFT v2.7 core scope: {task_type}")
    return task_type
