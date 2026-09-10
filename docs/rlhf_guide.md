# FinLLM RLVF 使用指南

本项目的对齐阶段从单一 DPO 改为任务感知 RLVF：`SFT -> DPO -> 可选 GRPO`。DPO 负责开放生成任务的偏好对齐；GRPO 负责标签、代码、风险等级等可规则验证任务的奖励优化。

## 方法选择

| 任务 | 主方法 | 辅助方法 | 依据 |
|------|--------|----------|------|
| `stock_analysis` | DPO | GRPO | 股票分析偏好完整论证，GRPO 只奖励技术结构覆盖 |
| `financial_report` | DPO | - | 财报解读开放性强，偏好对齐更合适 |
| `financial_qa` | DPO | GRPO | 开放问答用 DPO，选择题/可抽取答案可用 GRPO |
| `sentiment_analysis` | GRPO | DPO | 情感标签正确性可直接奖励 |
| `quant_strategy` | GRPO | DPO | Python 语法、函数、回测指标可验证 |
| `risk_assessment` | DPO | GRPO | 风险论证用 DPO，风险等级和量化指标用 GRPO |

本地 LLaMA-Factory 源码当前支持 `dpo/rm/ppo/kto`，不支持 `stage: grpo`。因此本项目使用 LLaMA-Factory 执行 DPO，并使用 TRL `GRPOTrainer` 执行 GRPO，最终输出 `saves/qwen3-8b/lora/rlvf`。

## 快速开始

```bash
# 完整默认流程：DPO 数据、GRPO 数据、DPO 训练、TRL-GRPO 训练
bash scripts/training/train_rlhf.sh

# 只准备数据和校验入口，不启动训练
bash scripts/training/train_rlhf.sh --method all --skip-train

# 仅 DPO
bash scripts/training/train_rlhf.sh --method dpo

# 仅 GRPO（需要已有 data/rlhf/fin_grpo_prompts_train.json）
bash scripts/training/train_rlhf.sh --method grpo --skip-data

# 指定任务子集
bash scripts/training/train_rlhf.sh --method all --tasks sentiment_analysis quant_strategy risk_assessment
```

## 数据准备

```bash
# DPO 偏好数据，输出 train/eval 并保留 fin_preference_* 兼容别名
python scripts/data_processing/synthesize_preference_data.py \
  --input data/sft/fin_instruct_train.json \
  --output data/rlhf/fin_dpo_preference_train.json \
  --mode rules \
  --max-samples 5000

# GRPO prompt 和离线奖励数据
python scripts/data_processing/prepare_grpo_data.py \
  --input data/sft/fin_instruct_train.json \
  --output data/rlhf/fin_grpo_prompts_train.json \
  --max-per-task 5000
```

关键输出：
- `data/rlhf/fin_dpo_preference_train.json`
- `data/rlhf/fin_dpo_preference_eval.json`
- `data/rlhf/fin_grpo_prompts_train.json`
- `data/rlhf/fin_grpo_prompts_eval.json`
- `data/rlhf/alignment_policy_report.json`

## 训练配置

DPO 配置仍使用 `configs/qwen3_8b_qlora_dpo.yaml`，但数据集名改为 `fin_dpo_preference_train`。训练从 `saves/qwen3-8b/lora/sft` 继续，输出到 `saves/qwen3-8b/lora/dpo`。

GRPO 使用 `configs/qwen3_8b_qlora_grpo_trl.yaml` 和 `scripts/training/train_grpo_trl.py`。训练脚本会：
- 读取 `data/rlhf/fin_grpo_prompts_train.json` / `eval.json`。
- 从 `Qwen/Qwen3-8B` 加载基座模型，并加载 `saves/qwen3-8b/lora/dpo` 作为可训练 LoRA adapter。
- 使用任务奖励、格式奖励、长度奖励三个 reward function。
- 输出 RLVF adapter 到 `saves/qwen3-8b/lora/rlvf`。

单独校验 TRL-GRPO 配置：

```bash
python scripts/training/train_grpo_trl.py \
  --config configs/qwen3_8b_qlora_grpo_trl.yaml \
  --dry-run
```

若 dry-run 报 `numpy>=1.17 found=None`，说明当前 Python 环境的 numpy 包元数据损坏，需要重新安装 numpy；这不是 GRPO 脚本逻辑问题。

## 多维评估

```bash
# 离线数据与奖励校验，不加载模型
python scripts/evaluation/eval_rlhf_comparison.py \
  --task all \
  --judge-mode quick \
  --output logs/rlvf_comparison.md

# 加载模型比较 SFT、DPO、RLVF
python scripts/evaluation/eval_rlhf_comparison.py \
  --model-path Qwen/Qwen3-8B \
  --sft-adapter saves/qwen3-8b/lora/sft \
  --dpo-adapter saves/qwen3-8b/lora/dpo \
  --rlvf-adapter saves/qwen3-8b/lora/rlvf \
  --task all \
  --num-samples 100 \
  --judge-mode quick \
  --output logs/rlvf_comparison.md
```

评估包含：
- 任务专项指标：情感 accuracy/macro-F1、QA ROUGE-L/关键词、股票技术结构、财报指标覆盖、量化代码语法、风险等级和缓释建议。
- 对齐专项指标：DPO chosen 奖励胜率、GRPO reward 均值/分位数、格式合规、风险提示、各任务相对提升。
- 可选 LLM Judge：默认关闭，仅评估开放生成任务；使用 `EVAL_JUDGE_API_KEY`、`EVAL_JUDGE_API_BASE`、`EVAL_JUDGE_MODEL`。

## 验证命令

```bash
python scripts/data_processing/synthesize_preference_data.py --max-samples 500 --mode rules
bash scripts/training/train_rlhf.sh --method all --skip-train
python scripts/training/train_grpo_trl.py --config configs/qwen3_8b_qlora_grpo_trl.yaml --dry-run
python scripts/evaluation/eval_rlhf_comparison.py --task all --judge-mode quick
pytest tests/ -v
```
