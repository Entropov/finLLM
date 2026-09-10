#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train RLVF adapter with TRL GRPOTrainer."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.rlhf.alignment_policy import compute_reward


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def import_training_deps():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "无法导入 PyTorch，当前环境的 torch/CUDA/NCCL 运行库不匹配。"
            "请先重装同一 CUDA 版本的 torch 与 nvidia-* 依赖后再运行 GRPO。"
            f" 原始错误: {exc}"
        ) from exc

    try:
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "无法导入 TRL-GRPO 训练依赖。请先修复/安装依赖："
            "pip install -r requirements.txt，并确认 torch、trl、vllm 的 CUDA 版本一致。"
            "如果 use_vllm=false 仍在导入 vLLM 时报错，请卸载或重装与 torch 匹配的 vLLM；"
            "TRL 的 GRPOTrainer 会在导入阶段加载 VLLMClient。"
            f" 原始错误: {exc}"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML mapping: {path}")
    return data


def get_question(item: dict) -> str:
    for turn in item.get("conversations", []):
        if turn.get("from") == "human":
            return turn.get("value", "")
    return item.get("prompt") or item.get("question") or ""


def build_prompt(item: dict, tokenizer=None) -> str:
    question = get_question(item)
    system = item.get("system") or "你是一个严谨的金融 AI 助手。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def load_grpo_dataset(path: Path, tokenizer=None):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"GRPO 数据必须是 JSON list: {path}")

    rows = []
    for item in raw:
        prompt = build_prompt(item, tokenizer)
        reference = item.get("reference", "")
        if not prompt or not reference:
            continue
        rows.append(
            {
                "prompt": prompt,
                "task_type": item.get("task_type", "financial_qa"),
                "reference": reference,
                "answer_key": item.get("answer_key", ""),
                "reward_spec": item.get("reward_spec", {}),
            }
        )
    return rows


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion or "")


def task_reward(completions, task_type, reference, answer_key=None, **kwargs):
    rewards = []
    answer_key = answer_key or [""] * len(completions)
    for completion, task, ref, key in zip(completions, task_type, reference, answer_key):
        text = _completion_to_text(completion)
        rewards.append(float(compute_reward(task, text, ref, key).get("reward", 0.0)))
    return rewards


def format_reward(completions, task_type, **kwargs):
    rewards = []
    for completion, task in zip(completions, task_type):
        text = _completion_to_text(completion).strip()
        if not text:
            rewards.append(0.0)
        elif task == "sentiment_analysis":
            rewards.append(1.0 if len(text) <= 120 else 0.4)
        elif task == "quant_strategy":
            rewards.append(1.0 if "```" in text and "def " in text else 0.4)
        else:
            rewards.append(1.0 if ("\n" in text or "风险" in text) else 0.5)
    return rewards


def length_reward(completions, task_type, **kwargs):
    rewards = []
    for completion, task in zip(completions, task_type):
        length = len(_completion_to_text(completion).strip())
        if task == "sentiment_analysis":
            rewards.append(1.0 if 1 <= length <= 120 else 0.2)
        elif task == "quant_strategy":
            rewards.append(1.0 if 120 <= length <= 3000 else 0.5)
        else:
            rewards.append(1.0 if 80 <= length <= 2000 else 0.5)
    return rewards


def build_training_args(config: dict, GRPOConfig):
    reward_weights = config.get("reward_weights") or {}
    weights = [
        float(reward_weights.get("task_reward", 1.0)),
        float(reward_weights.get("format_reward", 0.15)),
        float(reward_weights.get("length_reward", 0.05)),
    ]
    return GRPOConfig(
        output_dir=config["output_dir"],
        max_prompt_length=int(config.get("max_prompt_length", 768)),
        max_completion_length=int(config.get("max_completion_length", 384)),
        num_generations=int(config.get("num_generations", 4)),
        temperature=float(config.get("temperature", 0.7)),
        top_p=float(config.get("top_p", 0.95)),
        top_k=int(config.get("top_k", 50)) if config.get("top_k") is not None else None,
        beta=float(config.get("beta", 0.02)),
        epsilon=float(config.get("epsilon", 0.2)),
        loss_type=config.get("loss_type", "dapo"),
        scale_rewards=config.get("scale_rewards", "group"),
        num_iterations=int(config.get("num_iterations", 1)),
        reward_weights=weights,
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        learning_rate=float(config.get("learning_rate", 2e-6)),
        num_train_epochs=float(config.get("num_train_epochs", 1.0)),
        max_steps=int(config.get("max_steps", -1)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        weight_decay=float(config.get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        bf16=bool(config.get("bf16", True)),
        seed=int(config.get("seed", 42)),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=int(config.get("eval_steps", 100)),
        save_steps=int(config.get("save_steps", 100)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        report_to=config.get("report_to", "tensorboard"),
        log_completions=bool(config.get("log_completions", True)),
        num_completions_to_print=int(config.get("num_completions_to_print", 2)),
        use_vllm=bool(config.get("use_vllm", False)),
        # use_transformers_paged=bool(config.get("use_transformers_paged", False)),
        remove_unused_columns=bool(config.get("remove_unused_columns", False)),
    )


def load_model(config: dict, deps: dict):
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    PeftModel = deps["PeftModel"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]

    model_kwargs = {
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        "device_map": "auto",
    }
    if config.get("load_in_4bit", True):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16", True) else torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if config.get("bf16", True) else torch.float16

    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **model_kwargs)
    if config.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))

    adapter_path = config.get("adapter_name_or_path")
    if adapter_path:
        logger.info(f"Loading trainable adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRL GRPO for FinLLM RLVF")
    parser.add_argument("--config", default="configs/qwen3_8b_qlora_grpo_trl.yaml")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only validate config, data, and imports")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    if args.train_file:
        config["train_file"] = args.train_file
    if args.eval_file:
        config["eval_file"] = args.eval_file
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    deps = import_training_deps()
    AutoTokenizer = deps["AutoTokenizer"]
    Dataset = deps["Dataset"]
    LoraConfig = deps["LoraConfig"]
    GRPOTrainer = deps["GRPOTrainer"]
    GRPOConfig = deps["GRPOConfig"]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_grpo_dataset(Path(config["train_file"]), tokenizer)
    eval_rows = load_grpo_dataset(Path(config["eval_file"]), tokenizer) if config.get("eval_file") and Path(config["eval_file"]).exists() else []
    if not train_rows:
        raise ValueError(f"GRPO 训练数据为空: {config['train_file']}")
    logger.info(f"Train rows: {len(train_rows)}")
    logger.info(f"Eval rows: {len(eval_rows)}")

    if args.dry_run:
        logger.info("Dry run passed: config, imports, tokenizer, and data are valid")
        return

    model = load_model(config, deps)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows and config.get("eval_strategy", "steps") != "no" else None

    peft_config = None
    if not config.get("adapter_name_or_path"):
        peft_config = LoraConfig(
            r=int(config.get("lora_rank", 16)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=config.get("lora_target_modules", "all-linear"),
            task_type="CAUSAL_LM",
        )

    training_args = build_training_args(config, GRPOConfig)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[task_reward, format_reward, length_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    logger.info(f"TRL GRPO adapter saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
