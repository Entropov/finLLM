#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好数据人工标注工具 (CLI Annotation Tool)
==========================================
终端交互式标注工具，用于人工为模型回答对标注 chosen / rejected。
支持从已有 SFT 数据或自动生成的候选中进行标注，并实时保存进度。

功能：
  - 展示问题和两个候选回答（随机顺序，避免位置偏见）
  - 支持快捷键：A(选A) / B(选B) / T(平局) / S(跳过) / Q(退出)
  - 支持续标（从上次未完成的位置继续）
  - 实时将标注结果追加保存，防止意外丢失

用法：
  # 标注自动生成的候选对
  python scripts/rlhf/annotation_tool.py \\
      --input data/rlhf/fin_preference_train.json \\
      --output data/rlhf/human_annotated.json \\
      --num-samples 200

  # 从空白开始标注（只提供问题列表）
  python scripts/rlhf/annotation_tool.py \\
      --input data/sft/fin_instruct_train.json \\
      --output data/rlhf/human_annotated.json \\
      --mode compare  # 需要同时提供两个模型的回答
"""

import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path
from datetime import datetime

# 终端颜色
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
DIM = "\033[2m"


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def print_header(current: int, total: int, completed: int):
    print(f"{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN} FinLLM 偏好数据标注工具{RESET}  "
          f"{DIM}[{current}/{total}]  已完成: {completed}{RESET}")
    print(f"{CYAN}{'═'*70}{RESET}")
    print(f"{DIM}快捷键: A=选A  B=选B  T=平局(跳过)  S=跳过  Q=退出保存{RESET}\n")


def wrap_text(text: str, width: int = 68, indent: str = "  ") -> str:
    """自动换行显示长文本。"""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(indent + line)
        else:
            wrapped.extend(
                indent + chunk
                for chunk in textwrap.wrap(line, width=width)
            )
    return "\n".join(wrapped)


def display_sample(
    idx: int,
    total: int,
    completed: int,
    question: str,
    system: str,
    resp_a: str,
    resp_b: str,
    task_type: str = "",
):
    """展示一个标注样例。"""
    clear_screen()
    print_header(idx, total, completed)

    if task_type:
        print(f"{DIM}任务类型: {task_type}{RESET}\n")

    if system:
        print(f"{DIM}系统提示: {system[:80]}...{RESET}\n")

    print(f"{BOLD}❓ 问题:{RESET}")
    print(wrap_text(question[:500]))
    print()

    print(f"{BOLD}{GREEN}━━ 回答 A {'━'*50}{RESET}")
    print(wrap_text(resp_a[:800]))
    print()

    print(f"{BOLD}{YELLOW}━━ 回答 B {'━'*50}{RESET}")
    print(wrap_text(resp_b[:800]))
    print()

    print(f"{CYAN}{'─'*70}{RESET}")
    choice = input(f"  你的选择 [A/B/T/S/Q]: ").strip().upper()
    return choice


def load_existing_results(output_path: str) -> tuple[list[dict], set]:
    """加载已有标注结果，用于续标。"""
    path = Path(output_path)
    if not path.exists():
        return [], set()

    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        # 用 conversations[0].value 作为已标注标识
        done_keys = {
            item["conversations"][0]["value"][:80]
            for item in existing
            if item.get("conversations")
        }
        return existing, done_keys
    except Exception:
        return [], set()


def save_results(results: list[dict], output_path: str):
    """保存标注结果到文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="人工偏好数据标注工具")
    parser.add_argument(
        "--input",
        default="data/rlhf/fin_preference_train.json",
        help="输入数据（DPO 格式，含 chosen/rejected）",
    )
    parser.add_argument(
        "--output",
        default="data/rlhf/human_annotated.json",
        help="标注结果输出路径",
    )
    parser.add_argument("--num-samples", type=int, default=200, help="本次标注数量")
    parser.add_argument(
        "--task-filter",
        nargs="+",
        default=None,
        help="只标注特定 task_type",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="随机打乱顺序",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载输入数据
    if not Path(args.input).exists():
        print(f"{RED}错误: 输入文件不存在: {args.input}{RESET}")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        raw_data = json.load(f)

    # 任务过滤
    if args.task_filter:
        raw_data = [d for d in raw_data if d.get("task_type") in args.task_filter]

    if args.shuffle:
        random.shuffle(raw_data)

    # 加载已有标注（续标）
    existing_results, done_keys = load_existing_results(args.output)
    results = list(existing_results)

    # 过滤掉已标注的
    pending = [
        item for item in raw_data
        if item.get("conversations") and
        item["conversations"][0]["value"][:80] not in done_keys
    ]

    if len(pending) > args.num_samples:
        pending = pending[:args.num_samples]

    total = len(pending)
    if total == 0:
        print(f"{GREEN}所有样本已标注完成！共 {len(results)} 条。{RESET}")
        return

    print(f"\n{CYAN}准备标注 {total} 条（已完成 {len(existing_results)} 条）{RESET}")
    print(f"输出路径: {args.output}")
    input("按 Enter 开始标注...")

    stats = {"A_chosen": 0, "B_chosen": 0, "tie": 0, "skip": 0}
    save_every = 5  # 每标注5条自动保存

    for i, item in enumerate(pending):
        convs = item.get("conversations", [])
        if not convs:
            continue

        question = convs[0].get("value", "")
        system = item.get("system", "")
        task_type = item.get("task_type", "")

        # 取 chosen/rejected（DPO 格式）
        raw_chosen = item.get("chosen", {})
        raw_rejected = item.get("rejected", {})
        if isinstance(raw_chosen, dict):
            chosen_text = raw_chosen.get("value", "")
        else:
            chosen_text = str(raw_chosen)
        if isinstance(raw_rejected, dict):
            rejected_text = raw_rejected.get("value", "")
        else:
            rejected_text = str(raw_rejected)

        if not chosen_text or not rejected_text:
            continue

        # 随机化 A/B 顺序
        swap = random.random() < 0.5
        resp_a = rejected_text if swap else chosen_text
        resp_b = chosen_text if swap else rejected_text

        choice = display_sample(
            idx=i + 1 + len(existing_results),
            total=total + len(existing_results),
            completed=len(results),
            question=question,
            system=system,
            resp_a=resp_a,
            resp_b=resp_b,
            task_type=task_type,
        )

        if choice == "Q":
            print(f"\n{YELLOW}已退出，正在保存...{RESET}")
            break
        elif choice == "S":
            stats["skip"] += 1
            continue
        elif choice == "T":
            stats["tie"] += 1
            continue
        elif choice in ("A", "B"):
            if choice == "A":
                human_chosen = resp_a
                human_rejected = resp_b
                stats["A_chosen"] += 1
            else:
                human_chosen = resp_b
                human_rejected = resp_a
                stats["B_chosen"] += 1

            results.append({
                "conversations": [{"from": "human", "value": question}],
                "chosen": {"from": "gpt", "value": human_chosen},
                "rejected": {"from": "gpt", "value": human_rejected},
                "system": system,
                "task_type": task_type,
                "_source": "human_annotated",
                "_annotated_at": datetime.now().isoformat(),
            })

            # 自动保存
            if len(results) % save_every == 0:
                save_results(results, args.output)
                print(f"  {DIM}💾 已自动保存 ({len(results)} 条){RESET}")
        else:
            # 无效输入，重新提示
            continue

    # 最终保存
    save_results(results, args.output)

    clear_screen()
    print(f"\n{GREEN}{'='*50}{RESET}")
    print(f"{GREEN}✅ 标注完成！{RESET}")
    print(f"   总计标注: {len(results)} 条")
    print(f"   选A: {stats['A_chosen']}  选B: {stats['B_chosen']}")
    print(f"   平局: {stats['tie']}  跳过: {stats['skip']}")
    print(f"   保存路径: {args.output}")
    print(f"{GREEN}{'='*50}{RESET}\n")

    # 提示下一步
    print(f"{CYAN}下一步: 将人工标注数据合并到训练集{RESET}")
    print(f"  python scripts/data_processing/convert_to_preference.py \\")
    print(f"      --input {args.output} data/rlhf/fin_preference_train.json \\")
    print(f"      --output data/rlhf/fin_preference_train_merged.json \\")
    print(f"      --shuffle\n")


if __name__ == "__main__":
    main()
