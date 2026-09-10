#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline evaluation for basic RAG vs agentic RAG.

The script can run without loading an LLM. In that mode it evaluates retrieval
and extractive agentic-RAG diagnostics. If prediction files are supplied later,
the same metrics can score generated answers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluation.eval_task_specific import (  # noqa: E402
    ALL_TASKS,
    evaluate_task_predictions,
    load_eval_data,
    run_judge_evaluation,
    _get_question,
    _get_reference,
)
from scripts.rag.agentic_rag import (  # noqa: E402
    AgenticRAGPipeline,
    compute_context_relevance,
    compute_groundedness,
    compute_rag_reward,
    load_agentic_rag_config,
)
from scripts.rag.retriever import RAGRetriever  # noqa: E402

try:
    from scripts.rag.knowledge_agent import WebKnowledgeAgent  # noqa: E402
except Exception:  # pragma: no cover - optional at runtime
    WebKnowledgeAgent = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "saves" / "eval_results"


def _doc_key(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    return f"{metadata.get('source', '')}:{metadata.get('chunk_index', '')}:{doc.get('content', '')[:120]}"


def _retrieval_metrics(query: str, reference: str, docs: list[dict[str, Any]]) -> dict[str, float]:
    if not docs:
        return {
            "retrieval_relevance": 0.0,
            "reference_recall_at_k": 0.0,
            "mrr": 0.0,
            "duplicate_rate": 0.0,
        }
    relevance = compute_context_relevance(query, docs)
    reference_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_.%+-]{2,}", reference.lower()))
    first_hit = 0
    hit_count = 0
    for idx, doc in enumerate(docs, start=1):
        doc_text = doc.get("content", "").lower()
        if reference_terms and any(term in doc_text for term in list(reference_terms)[:80]):
            hit_count += 1
            if not first_hit:
                first_hit = idx
    seen = {_doc_key(doc) for doc in docs}
    duplicate_rate = 1.0 - len(seen) / len(docs)
    return {
        "retrieval_relevance": relevance,
        "reference_recall_at_k": round(hit_count / len(docs), 4),
        "mrr": round(1.0 / first_hit, 4) if first_hit else 0.0,
        "duplicate_rate": round(duplicate_rate, 4),
    }


def _basic_rag_answer(query: str, docs: list[dict[str, Any]], task_type: str) -> str:
    if not docs:
        return "未检索到足够可靠的参考资料，无法给出确定结论。"
    lines = ["基于本地知识库检索结果："]
    for idx, doc in enumerate(docs[:3], start=1):
        source = (doc.get("metadata") or {}).get("source", f"doc{idx}")
        preview = re.sub(r"\s+", " ", doc.get("content", "")).strip()[:180]
        lines.append(f"{idx}. [{source}] {preview}")
    if task_type in {"stock_analysis", "financial_report", "risk_assessment"}:
        lines.append("请结合最新市场信息复核，以上内容仅供参考，不构成投资建议。")
    return "\n".join(lines)


def evaluate_basic_item(retriever: Optional[RAGRetriever], item: dict, top_k: int) -> dict[str, Any]:
    query = _get_question(item)
    task_type = item.get("task_type", "financial_qa")
    docs: list[dict[str, Any]] = []
    if retriever is not None and retriever.is_ready():
        docs = retriever.retrieve(query, top_k=top_k)
    answer = _basic_rag_answer(query, docs, task_type)
    reward = compute_rag_reward(query, answer, docs, task_type)
    retrieval = _retrieval_metrics(query, _get_reference(item), docs)
    return {
        "query": query,
        "task_type": task_type,
        "answer": answer,
        "docs": docs,
        "reward": reward,
        "retrieval": retrieval,
        "path": ["basic_retrieve", "basic_prompt"],
    }


def evaluate_agentic_item(pipeline: AgenticRAGPipeline, item: dict, top_k: int, allow_web: bool) -> dict[str, Any]:
    query = _get_question(item)
    state = pipeline.run(
        query=query,
        system_prompt=item.get("system", ""),
        task_type=item.get("task_type"),
        reference=_get_reference(item),
        runtime_options={"top_k": top_k, "allow_web": allow_web},
    )
    docs = state.get("selected_context", [])
    retrieval = _retrieval_metrics(query, _get_reference(item), docs)
    return {
        "query": query,
        "task_type": state.get("task_type", item.get("task_type", "financial_qa")),
        "answer": state.get("answer", ""),
        "docs": docs,
        "reward": state.get("reward_breakdown", {}),
        "retrieval": retrieval,
        "path": [event.get("node", "") for event in state.get("trajectory", [])],
        "state": state,
    }


def aggregate_results(results_by_task: dict[str, list[dict[str, Any]]], use_judge: bool, max_judge_samples: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"tasks": {}, "overall": {}}
    all_rewards = []
    all_web_rates = []
    for task_name, rows in sorted(results_by_task.items()):
        if not rows:
            continue
        items = [
            {
                "task_type": task_name,
                "conversations": [
                    {"from": "human", "value": row["query"]},
                    {"from": "gpt", "value": row.get("reference", "")},
                ],
            }
            for row in rows
        ]
        predictions = [row.get("answer", "") for row in rows]
        task_metrics = evaluate_task_predictions(task_name, items, predictions)
        rewards = [row.get("reward", {}).get("total_reward", 0.0) for row in rows]
        grounded = [
            row.get("reward", {}).get("claim_support", compute_groundedness(row.get("answer", ""), row.get("docs", [])))
            for row in rows
        ]
        hard_gate_rates = [1.0 if row.get("reward", {}).get("hard_gate_passed", False) else 0.0 for row in rows]
        citation_precision = [row.get("reward", {}).get("citation_precision", 0.0) for row in rows]
        citation_coverage = [row.get("reward", {}).get("citation_coverage", 0.0) for row in rows]
        temporal_validity = [row.get("reward", {}).get("temporal_validity", 0.0) for row in rows]
        hard_failure_counts = Counter(
            failure
            for row in rows
            for failure in row.get("reward", {}).get("hard_failures", [])
        )
        web_rates = []
        for row in rows:
            state = row.get("state") or {}
            trajectory = state.get("trajectory", []) if isinstance(state, dict) else []
            collected = any(
                event.get("node") == "maybe_web_collect" and event.get("action") == "collected"
                for event in trajectory
            )
            web_rates.append(1.0 if collected or state.get("web_docs") else 0.0)
        refine_rates = [1.0 if row.get("path", []).count("plan_queries") > 1 else 0.0 for row in rows]
        retrieval_relevance = [row.get("retrieval", {}).get("retrieval_relevance", 0.0) for row in rows]
        duplicate_rates = [row.get("retrieval", {}).get("duplicate_rate", 0.0) for row in rows]

        task_summary = {
            **task_metrics,
            "samples": len(rows),
            "avg_total_reward": round(mean(rewards), 4) if rewards else 0.0,
            "avg_groundedness": round(mean(grounded), 4) if grounded else 0.0,
            "hard_gate_pass_rate": round(mean(hard_gate_rates), 4) if hard_gate_rates else 0.0,
            "avg_citation_precision": round(mean(citation_precision), 4) if citation_precision else 0.0,
            "avg_citation_coverage": round(mean(citation_coverage), 4) if citation_coverage else 0.0,
            "avg_temporal_validity": round(mean(temporal_validity), 4) if temporal_validity else 0.0,
            "hard_failure_distribution": dict(hard_failure_counts),
            "avg_retrieval_relevance": round(mean(retrieval_relevance), 4) if retrieval_relevance else 0.0,
            "avg_duplicate_rate": round(mean(duplicate_rates), 4) if duplicate_rates else 0.0,
            "web_search_rate": round(mean(web_rates), 4) if web_rates else 0.0,
            "refine_rate": round(mean(refine_rates), 4) if refine_rates else 0.0,
            "path_distribution": dict(Counter("->".join(row.get("path", [])) for row in rows)),
            "judge": None,
        }
        if use_judge:
            task_summary["judge"] = run_judge_evaluation(task_name, items, predictions, max_judge_samples=max_judge_samples)
        else:
            task_summary["judge"] = {"skipped": True, "reason": "judge disabled"}
        summary["tasks"][task_name] = task_summary
        all_rewards.extend(rewards)
        all_web_rates.extend(web_rates)

    summary["overall"] = {
        "samples": sum(len(rows) for rows in results_by_task.values()),
        "avg_total_reward": round(mean(all_rewards), 4) if all_rewards else 0.0,
        "web_search_rate": round(mean(all_web_rates), 4) if all_web_rates else 0.0,
        "hard_gate_pass_rate": round(
            mean(
                1.0 if row.get("reward", {}).get("hard_gate_passed", False) else 0.0
                for rows in results_by_task.values()
                for row in rows
            ),
            4,
        ) if any(results_by_task.values()) else 0.0,
    }
    return summary


def write_markdown_report(summary: dict[str, Any], output_path: Path, mode: str) -> None:
    lines = [
        f"# Agentic RAG Evaluation ({mode})",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Samples: {summary.get('overall', {}).get('samples', 0)}",
        f"- Avg reward: {summary.get('overall', {}).get('avg_total_reward', 0.0)}",
        "",
        f"- Hard-gate pass rate: {summary.get('overall', {}).get('hard_gate_pass_rate', 0.0)}",
        "",
        "| Task | Samples | Reward | Audit pass | Claim support | Citation P/R | Temporal | Retrieval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, result in sorted(summary.get("tasks", {}).items()):
        lines.append(
            f"| {task} | {result.get('samples', 0)} | {result.get('avg_total_reward', 0.0)} | "
            f"{result.get('hard_gate_pass_rate', 0.0)} | {result.get('avg_groundedness', 0.0)} | "
            f"{result.get('avg_citation_precision', 0.0)}/{result.get('avg_citation_coverage', 0.0)} | "
            f"{result.get('avg_temporal_validity', 0.0)} | {result.get('avg_retrieval_relevance', 0.0)} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    task_groups = load_eval_data(args.task)
    for task_name in list(task_groups):
        if args.max_samples:
            task_groups[task_name] = task_groups[task_name][: args.max_samples]

    retriever = None
    if not args.no_retriever:
        try:
            retriever = RAGRetriever(db_dir=args.rag_db_dir, device=args.device)
            if not retriever.is_ready():
                logger.warning("RAG retriever is not ready; evaluation will use empty retrieval results.")
        except Exception as exc:
            logger.warning(f"RAG retriever init failed: {exc}")
            retriever = None

    web_agent = None
    if args.allow_web and WebKnowledgeAgent is not None:
        try:
            web_agent = WebKnowledgeAgent(db_dir=Path(args.rag_db_dir), device=args.device)
        except Exception as exc:
            logger.warning(f"WebKnowledgeAgent init failed; web disabled: {exc}")

    pipeline = None
    config = load_agentic_rag_config(args.config)
    if args.policy:
        config.policy_type = args.policy
    if args.mode != "basic":
        pipeline = AgenticRAGPipeline(retriever=retriever, web_agent=web_agent, config=config)

    results_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_name, items in sorted(task_groups.items()):
        logger.info(f"Evaluating {task_name}: {len(items)} samples")
        for item in items:
            item["task_type"] = item.get("task_type") or task_name
            if args.mode == "basic":
                row = evaluate_basic_item(retriever, item, top_k=args.top_k)
            else:
                row = evaluate_agentic_item(pipeline, item, top_k=args.top_k, allow_web=args.allow_web)
            row["reference"] = _get_reference(item)
            results_by_task[task_name].append(row)

    use_judge = args.judge_mode != "off" and bool(os.environ.get("EVAL_JUDGE_API_KEY"))
    summary = aggregate_results(results_by_task, use_judge=use_judge, max_judge_samples=args.max_judge_samples)
    summary["mode"] = args.mode
    summary["judge_mode"] = args.judge_mode
    if args.judge_mode != "off" and not os.environ.get("EVAL_JUDGE_API_KEY"):
        summary["judge"] = {"skipped": True, "reason": "EVAL_JUDGE_API_KEY not set"}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else RESULTS_DIR / f"rag_{args.mode}_eval.json"
    output_md = Path(args.output) if args.output else RESULTS_DIR / f"rag_{args.mode}_eval.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(summary, output_md, args.mode)
    logger.info(f"JSON report saved: {output_json}")
    logger.info(f"Markdown report saved: {output_md}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate basic/agentic RAG offline")
    parser.add_argument("--mode", choices=["basic", "agentic"], default="agentic")
    parser.add_argument("--task", choices=ALL_TASKS + ["all"], default="all")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag_agentic.yaml"))
    parser.add_argument("--policy", choices=["rules", "bandit", "dpo", "grpo"], default=None)
    parser.add_argument("--rag-db-dir", default=str(PROJECT_ROOT / "saves" / "chroma_db"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--no-retriever", action="store_true", help="Skip Chroma retriever init for dry offline checks")
    parser.add_argument("--judge-mode", choices=["off", "quick", "full"], default="off")
    parser.add_argument("--max-judge-samples", type=int, default=20)
    parser.add_argument("--output", default=None, help="Markdown report path")
    parser.add_argument("--output-json", default=None, help="JSON report path")
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
