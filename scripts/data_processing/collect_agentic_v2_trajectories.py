#!/usr/bin/env python3
"""Collect real model completions through the agentic OpenAI-compatible API."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp


TASK_OUTPUT_CONTRACTS = {
    "stock_analysis": (
        "输出两行证据原句式事实（行情一行、财务一行），再输出一行以‘观点：’开头的趋势与风险判断。"
        "缺失项使用规定的证据不足句，不补行业阈值。"
    ),
    "quant_strategy": (
        "先用一行逐字复述可用行情指标；再给一个Python代码块。代码必须import pandas和numpy，定义接收历史价格DataFrame的函数，"
        "由输入计算MA5、MA20、signal、入场、出场、仓位、止损、Sharpe和drawdown，不得硬编码证据数值或假设收益率；"
        "代码后用一行以‘观点：’开头说明策略仅供研究。"
    ),
    "financial_report": (
        "每行只复述一组证据原值：收入及同比、净利润及同比、经营现金流、资产负债率与流动比率；"
        "最后可用一行‘观点：’概括方向，但不得猜测原因。"
    ),
    "sentiment_analysis": (
        "第一行逐字复述收入、净利润、经营现金流及证据已有的同比；第二行必须以‘观点：基本面情绪标签为"
        "positive/neutral/negative’开头，只根据第一行数据说明，不计算证据未给出的同比。"
    ),
    "financial_qa": (
        "第一行逐字复述负债、资产、资产负债率；第二行逐字复述流动资产、流动负债、流动比率；"
        "第三行以‘观点：’解释两项指标的适用边界，不得引入警戒线或行业标准。"
    ),
    "risk_assessment": (
        "用两行分别逐字复述财务风险指标和市场风险指标；第三行以‘观点：’给出风险等级；"
        "第四行以‘观点：’给出不涉及买卖指令的监测或缓释措施，不得补行业均值、阈值或原因。"
    ),
}

TASK_REPAIR_CONTRACTS = {
    "financial_qa": (
        "不要写除法算式或重新计算比例；逐字复述证据已经给出的负债、资产、资产负债率、"
        "流动资产、流动负债和流动比率。适用边界若无直接证据，只写规定的证据不足句。"
    ),
    "sentiment_analysis": (
        "必须使用发行人的财务证据，不得用行情证据代替。每个财务事实单独一行并引用；"
        "情绪标签单独一行并引用支撑该标签的财务证据。"
    ),
    "risk_assessment": (
        "财务事实与市场事实分行逐字复述；风险等级和缓释措施各自单独一行、以‘观点：’开头并引用。"
        "删除证据没有给出的行业阈值、原因、产品或交易建议。"
    ),
    "stock_analysis": (
        "行情事实与财务事实分行逐字复述；趋势和风险各自单独一行、以‘观点：’开头并引用。"
    ),
    "financial_report": (
        "每行只复述证据中的一组原值，不得自行计算、舍入、解释原因或补充行业比较。"
    ),
    "quant_strategy": (
        "代码外每个事实单独一行并引用；代码必须只从输入DataFrame计算，不得使用证据数值作为策略常量，"
        "不得包含全仓指令。"
    ),
}


def build_retry_system_prompt(
    base_system_prompt: str,
    *,
    task_type: str,
    previous_answer: str,
    failures: list[str],
) -> str:
    failure_text = ", ".join(map(str, failures)) or "missing_audit_summary"
    return (
        base_system_prompt
        + "\n\n【审计失败后的强制修订】\n"
        + f"上一版硬失败：{failure_text}。\n"
        + "只输出修订后的完整答案，不要解释修订过程。逐句检查，不得依赖段末的单个引用覆盖前面的句子。"
        + "每行只允许一个可审计结论，且该结论的引用必须在同一行。"
        + "数字必须从证据逐字复制，禁止重新计算、换算、舍入或改写精度。"
        + "删除无证据的原因、阈值、行业比较、建议和结论；无证据时只写‘证据不足，无法确认该项。’。\n"
        + TASK_REPAIR_CONTRACTS[task_type]
        + "\n【上一版回答，仅用于修订】\n"
        + previous_answer[:6000]
    )


async def wait_until_ready(session: aiohttp.ClientSession, base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            async with session.get(f"{base_url}/health") as response:
                payload = await response.json()
                if response.status == 200 and payload.get("model_loaded"):
                    return payload
                last_error = f"health={response.status}:{payload}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        await asyncio.sleep(2)
    raise TimeoutError(f"agentic API was not ready after {timeout}s: {last_error}")


def load_completed(path: Path) -> set[str]:
    completed = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok") and row.get("id"):
                completed.add(str(row["id"]))
    return completed


def limit_requests_per_task(
    requests: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None:
        return requests
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in requests:
        task_type = str(row["task_type"])
        if counts[task_type] >= limit:
            continue
        selected.append(row)
        counts[task_type] += 1
    return selected


def select_tasks(requests: list[dict[str, Any]], tasks: list[str] | None) -> list[dict[str, Any]]:
    if not tasks:
        return requests
    selected_tasks = set(tasks)
    return [row for row in requests if str(row.get("task_type")) in selected_tasks]


async def collect_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    row: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    max_tokens = 1200 if row["task_type"] == "quant_strategy" else 700
    base_system_prompt = (
        "你是可审计金融分析助手。只使用检索证据；事实、数值和计算结论逐项引用[Evidence ID]。"
        "事实尽量直接摘录，不得四舍五入或引入证据外原因、行业比较和阈值。"
        "趋势、情绪、风险与策略等推断必须以‘观点：’开头并引用证据。"
        "Evidence ID必须从参考知识库逐字符复制并在输出前复核；多个证据分别写成[E...][E...]。"
        "不得缩写、猜测或手工改写ID。不得编造引用，不作收益保证；资料不足或冲突时明确拒答。"
        + TASK_OUTPUT_CONTRACTS[row["task_type"]]
    )
    request = {
        "model": "fin-instruct",
        "messages": [
            {
                "role": "system",
                "content": base_system_prompt,
            },
            {"role": "user", "content": row["question"]},
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.05,
        "stream": False,
        "rag_mode": "agentic",
        "task_type": row["task_type"],
        "rag_top_k": 5,
        "enable_web_knowledge": False,
        "return_audit": True,
        "enable_answer_guard": True,
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.monotonic()
        try:
            async with semaphore:
                async with session.post(f"{base_url}/v1/chat/completions", json=request) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {text[:500]}")
                    payload = json.loads(text)
            answer = str(payload["choices"][0]["message"]["content"])
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
            audit_summary = payload.get("audit_summary") or {}
            accepted = bool(
                not audit_summary.get("validation_errors")
                and audit_summary.get("hard_gate_passed")
                and float(audit_summary.get("total_reward", 0.0)) >= 0.65
            )
            if not accepted and attempt < retries:
                failures = audit_summary.get("hard_failures") or audit_summary.get("validation_errors") or ["missing_audit_summary"]
                request["messages"][0]["content"] = build_retry_system_prompt(
                    base_system_prompt,
                    task_type=str(row["task_type"]),
                    previous_answer=answer,
                    failures=list(map(str, failures)),
                )
                await asyncio.sleep(min(3, attempt))
                continue
            return {
                **row,
                "ok": bool(answer.strip()) and accepted,
                "answer": answer,
                "audit_summary": audit_summary,
                "response_id": payload.get("id", ""),
                "usage": payload.get("usage", {}),
                "attempt": attempt,
                "latency_sec": round(time.monotonic() - started, 3),
            }
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
            last_error = str(exc)
            if attempt < retries:
                await asyncio.sleep(min(10, 2**attempt))
    return {**row, "ok": False, "error": last_error, "attempt": retries}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    requests = json.loads(Path(args.input).read_text(encoding="utf-8"))
    requests = select_tasks(requests, args.tasks)
    requests = limit_requests_per_task(requests, args.limit_per_task)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output.write_text("", encoding="utf-8")
    completed = load_completed(output) if args.resume else set()
    pending = [row for row in requests if str(row.get("id")) not in completed]
    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=30)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 20))
    write_lock = asyncio.Lock()
    counters = Counter()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        health = await wait_until_ready(session, args.base_url.rstrip("/"), args.startup_timeout)
        print(json.dumps({"api_ready": health, "pending": len(pending), "completed": len(completed)}, ensure_ascii=False))
        semaphore = asyncio.Semaphore(args.concurrency)

        async def collect_and_write(row: dict[str, Any]) -> None:
            result = await collect_one(
                session,
                semaphore,
                args.base_url.rstrip("/"),
                row,
                args.retries,
            )
            counters[(result["task_type"], bool(result["ok"]))] += 1
            async with write_lock:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            finished = sum(counters.values())
            if finished % args.progress_every == 0 or finished == len(pending):
                summary = {
                    task: {
                        "ok": counters[(task, True)],
                        "failed": counters[(task, False)],
                    }
                    for task in sorted({item["task_type"] for item in requests})
                }
                print(json.dumps({"finished": finished, "total": len(pending), "tasks": summary}, ensure_ascii=False), flush=True)

        await asyncio.gather(*(collect_and_write(row) for row in pending))

    return {
        "requested": len(requests),
        "already_completed": len(completed),
        "attempted": len(pending),
        "successful": sum(value for (task, ok), value in counters.items() if ok),
        "failed": sum(value for (task, ok), value in counters.items() if not ok),
        "output": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SFT v2 trajectories through the agentic API")
    parser.add_argument("--input", default="data/rag/v2_collection_requests.json")
    parser.add_argument("--output", default="data/rag/v2_collection_responses.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        choices=sorted(TASK_OUTPUT_CONTRACTS),
        help="Collect only the selected task type; repeat for multiple tasks.",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="Collect at most the first N requests of each task type (for probes).",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> int:
    report = asyncio.run(run(parse_args()))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
