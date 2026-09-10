#!/usr/bin/env python3
"""Prepare additional train-only requests from the frozen v2 issuer split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing.prepare_agentic_v2_corpus import (  # noqa: E402
    TASKS,
    build_collection_requests,
    load_snapshot,
)
from scripts.rag.audit_schema import canonical_json  # noqa: E402


def _normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    train_codes = [str(item).zfill(6) for item in split_manifest.get("train_issuers", [])]
    heldout_codes = {str(item).zfill(6) for item in split_manifest.get("heldout_issuers", [])}
    if not train_codes or set(train_codes) & heldout_codes:
        raise ValueError("frozen split manifest is missing or contains issuer overlap")

    report_dir = Path(args.report_dir).resolve()
    price_dir = Path(args.price_dir).resolve()
    names = pd.read_csv(price_dir / "_hs300_stocks.csv", dtype={"code": str})
    name_by_code = {str(row.code).zfill(6): str(row.name) for row in names.itertuples()}
    snapshots = []
    missing_codes = []
    for code in sorted(train_codes):
        name = name_by_code.get(code)
        snapshot = load_snapshot(code, name or code, report_dir, price_dir) if name else None
        if snapshot is None:
            missing_codes.append(code)
        else:
            snapshots.append(snapshot)
    if missing_codes:
        raise RuntimeError(f"cannot reconstruct frozen train snapshots: {missing_codes}")

    end_index = args.start_index + args.per_task - 1
    generated = build_collection_requests(snapshots, end_index)
    requests = [
        row
        for row in generated
        if args.start_index <= int(str(row["id"]).rsplit("-", 1)[-1]) <= end_index
    ]
    old_requests = json.loads(Path(args.existing_requests).read_text(encoding="utf-8"))
    old_ids = {str(row["id"]) for row in old_requests}
    old_questions = {_normalized_hash(str(row["question"])) for row in old_requests}
    request_ids = [str(row["id"]) for row in requests]
    question_hashes = [_normalized_hash(str(row["question"])) for row in requests]
    task_counts = Counter(str(row["task_type"]) for row in requests)
    issuer_codes = {str(row["issuer_code"]).zfill(6) for row in requests}
    release_gate = {
        "expected_size": len(requests) == len(TASKS) * args.per_task,
        "per_task_coverage": all(task_counts.get(task, 0) == args.per_task for task in TASKS),
        "unique_ids": len(request_ids) == len(set(request_ids)),
        "unique_questions": len(question_hashes) == len(set(question_hashes)),
        "no_existing_id_overlap": not (set(request_ids) & old_ids),
        "no_existing_question_overlap": not (set(question_hashes) & old_questions),
        "no_heldout_issuer_overlap": not (issuer_codes & heldout_codes),
        "frozen_train_issuers_only": issuer_codes <= set(train_codes),
    }
    report = {
        "schema_version": "agentic_collection_requests.v2.1",
        "frozen_split_manifest": str(Path(args.split_manifest).resolve()),
        "start_index": args.start_index,
        "end_index": end_index,
        "requests": len(requests),
        "by_task": dict(sorted(task_counts.items())),
        "train_issuers_used": sorted(issuer_codes),
        "heldout_issuer_overlap": sorted(issuer_codes & heldout_codes),
        "request_fingerprint": hashlib.sha256(canonical_json(requests).encode("utf-8")).hexdigest(),
        "release_gate": release_gate,
        "release_gate_passed": all(release_gate.values()),
    }
    _write_json_atomic(Path(args.output), requests)
    _write_json_atomic(Path(args.report), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen-split SFT v2.1 collection requests")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "data/raw/financial_reports"))
    parser.add_argument("--price-dir", default=str(PROJECT_ROOT / "data/raw/stock_prices"))
    parser.add_argument("--split-manifest", default=str(PROJECT_ROOT / "data/evaluation/v2_split_manifest.json"))
    parser.add_argument("--existing-requests", default=str(PROJECT_ROOT / "data/rag/v2_collection_requests.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/rag/v2_1_collection_requests.json"))
    parser.add_argument("--report", default=str(PROJECT_ROOT / "data/rag/v2_1_collection_request_report.json"))
    parser.add_argument("--start-index", type=int, default=121)
    parser.add_argument("--per-task", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    report = prepare(parse_args())
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["release_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
