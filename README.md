# finLLM

基于 **Qwen3-8B**、**LLaMA-Factory** 与 **Agentic RAG** 的可审计金融大模型研究项目。项目覆盖金融指令数据构建、QLoRA/LoRA SFT、DPO/GRPO 实验入口、证据检索、Claim-to-Evidence 审计、可信评测，以及 OpenAI 兼容 API 和 Gradio 推理界面。

> **免责声明：** 本项目仍处于研究阶段，模型输出仅供学习和实验参考，不构成投资建议，也不应直接用于生产决策。

## 核心能力

- 六类金融任务数据管线：股票分析、量化策略、财报解读、情感分析、金融问答与风险评估。
- 面向三个核心任务的 SFT v2.x 数据构建、audit-weighted loss、checkpoint 选择与 paired regression。
- 基础 RAG 与 LangGraph Agentic RAG，支持查询规划、证据归一化、Claim-to-Evidence 映射、验证、重试和轨迹落盘。
- 非补偿式审计 hard gate，覆盖伪造引用、证据不相关、数值漂移、未来信息、合规违规和无效量化产物。
- DPO 偏好数据、hard negative、TRL-GRPO 数据和训练脚手架。
- Transformers/vLLM 推理、OpenAI 兼容 API、Gradio WebUI 和批量推理。

## 项目结构

```text
finLLM/
|-- configs/                 # Qwen2.5/Qwen3 训练、推理与 Agentic RAG 配置
|-- data/
|   |-- dataset_info.json    # LLaMA-Factory 数据集注册
|   |-- evaluation/          # trusted、adversarial 与 preflight 评测集
|   |-- knowledge/           # 本地金融知识库
|   |-- rag/                 # audit schema、轨迹、偏好与评测集
|   |-- rlhf/                # DPO/GRPO 数据
|   `-- sft_v2*/             # 版本化 SFT 数据和构建报告
|-- docs/                    # 研究报告、审计报告和实验指南
|-- prompts/                 # 六类任务与通用系统提示词
|-- scripts/
|   |-- data_collection/     # 行情、财报、新闻与开源数据采集
|   |-- data_processing/     # 清洗、合成、SFT/DPO/GRPO 数据构建
|   |-- evaluation/          # 可信评测、selector、validator 与 reward 测试
|   |-- inference/           # API、WebUI、批量推理和 vLLM 后端
|   |-- rag/                 # 检索、Agent 编排、审计 schema 与 reward
|   |-- rlhf/                # 对齐策略、自动标注和人工标注工具
|   `-- training/            # SFT、DPO、GRPO、合并与导出入口
|-- tests/                   # 数据、RAG、审计、推理和训练协议测试
|-- requirements.txt
`-- README.md
```

模型权重、运行日志、缓存及大部分运行时生成数据不会作为源码提交。仓库中的研究数据和评测制品用于复现实验边界，不代表生产数据集。

## 环境准备


```bash
git clone https://github.com/Entropov/finLLM.git
cd finLLM

conda create -n finllm python=3.11 -y
conda activate finllm
pip install -r requirements.txt

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

可选安装 FlashAttention：

```bash
pip install flash-attn --no-build-isolation
```

## 快速开始

### 1. 基础 QLoRA SFT

```bash
llamafactory-cli train configs/qwen3_8b_qlora_sft.yaml

# 评估与合并
llamafactory-cli train configs/qwen3_8b_qlora_sft_eval.yaml
bash scripts/training/merge_lora.sh
```

基础配置使用 Qwen3-8B、4-bit NF4、LoRA rank 16、4096 token 上下文和 bf16。版本化 SFT v2.x 属于研究流程，运行前应先阅读对应构建报告与 release gate。

### 2. 当前 SFT 基准复现

```bash
# 构建三项核心任务数据并检查数据 gate
python scripts/data_processing/build_sft_v2_7_core_dataset.py

# 仅检查训练配置
bash scripts/training/train_sft_v2_7_core.sh --skip-build --dry-run

# 启动训练
bash scripts/training/train_sft_v2_7_core.sh --skip-build
```

v2.7-core 的 checkpoint 选择、trusted/adversarial paired regression 和 selector replay 命令见 [Qwen3-8B 可审计金融 Agent 技术研究报告](docs/qwen3_8b_auditable_financial_agent_research.md)。v2.8.2 仅保留作失败实验与诊断记录，不是推荐训练目标。

### 3. 构建本地知识库

```bash
python scripts/rag/build_vector_db.py \
  --data-dir data/knowledge/v2_train \
  --db-dir saves/chroma_v2_train \
  --chunk-strategy semantic
```

### 4. 启动 Agentic RAG API

```bash
python scripts/inference/api_server.py \
  --backend vllm \
  --model-path Qwen/Qwen3-8B \
  --adapter-path saves/qwen3-8b/lora/sft-v2.7-core-selected \
  --enable-rag \
  --rag-mode agentic \
  --rag-agentic-config configs/rag_agentic.yaml \
  --rag-db-dir saves/chroma_v2_train \
  --port 8000
```

使用 Transformers 后端时，将 `--backend vllm` 改为 `--backend transformers`。联网知识补充必须显式增加 `--enable-web-knowledge`；默认配置关闭 Web 检索。

OpenAI SDK 调用示例：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="fin-instruct",
    messages=[
        {"role": "system", "content": "你是一位严谨的金融研究助手。"},
        {"role": "user", "content": "基于可用证据分析贵州茅台的主要风险。"},
    ],
    temperature=0.2,
)
print(response.choices[0].message.content)
```

### 5. 启动 Gradio WebUI

```bash
python scripts/inference/chat_demo.py \
  --model-path Qwen/Qwen3-8B \
  --adapter-path saves/qwen3-8b/lora/sft-v2.7-core-selected \
  --rag-mode agentic \
  --rag-agentic-config configs/rag_agentic.yaml \
  --rag-db-dir saves/chroma_v2_train
```

### 6. 审计与离线 RAG 评测

```bash
# 确定性 reward attack suite
python scripts/evaluation/validate_agentic_rewards.py

# 不初始化检索器的离线 Agent 流程检查
python scripts/rag/eval_agentic_rag.py \
  --mode agentic \
  --task all \
  --no-retriever \
  --max-samples 20

# 项目自有测试（排除 vendored LLaMA-Factory 与联网 FinEval 脚本）
pytest tests/ -q
```

### 7. DPO/GRPO 实验入口

以下入口仅用于后续受控实验。当前 Release Gate 为 FAIL，请勿把它们作为推荐训练流程直接启动。

```bash
# DPO 配置示例
llamafactory-cli train configs/qwen3_8b_qlora_dpo.yaml

# GRPO 仅校验配置、数据和依赖
python scripts/training/train_grpo_trl.py \
  --config configs/qwen3_8b_qlora_grpo_trl.yaml \
  --dry-run
```

当前 GRPO reward 仍是 task/format/length 的脚手架，尚未完整消费 `reward_v2` trajectory。方法边界和准入条件见 [RLVF 使用指南](docs/rlhf_guide.md) 与 [v2.9 阶段策略复盘](docs/finllm_v2_9_strategy_review.md)。

## 审计契约

核心实现位于：

- `scripts/rag/audit_schema.py`：Evidence、Claim、Calculation 与轨迹数据结构。
- `data/rag/audit_v2.schema.json`：可交换 JSON Schema。
- `scripts/rag/reward_v2.py`：先 hard gate、后软分数的非补偿式奖励。
- `scripts/rag/quant_protocol.py`：受限 quant action 与确定性 renderer。
- `scripts/evaluation/task_aware_verifier_v2_7_core.py`：三项核心任务 verifier。

Evidence ID、来源、发布时间、生效时间、抓取时间、原文引用和内容哈希均参与审计。任何 hard gate 失败的轨迹奖励为 0，市场反馈上限为 `0.05`，不能补偿事实、引用或合规失败。

## 关键研究结果

在三任务 v2.7-core seen regression 中，选定 checkpoint 相对 Qwen3-8B 基座取得以下结果：

| 指标 | Trusted | Adversarial |
|---|---:|---:|
| E2E@1 | 0.5733 | 0.5933 |
| E2E@8 | 0.9000 | 0.8600 |
| Audit@1 | 0.7400 | 0.8600 |
| Audit@8 | 0.9600 | 0.9733 |
| Mean primary delta | +0.0804 | +0.0814 |

这些结果证明 SFT 改善了结构、引用与量化任务稳定性，但 Audit@1 尚未达到 95% 目标，QA 和 stock_analysis 仍存在任务级回退。v2.8.2 的全量 stock target rewrite 又使 seen preflight E2E@8 从 `0.68` 降至 `0.28`，因此当前总体结论仍是 FAIL / HOLD。

## 文档索引

- [Qwen3-8B 可审计金融 Agent 技术研究报告](docs/qwen3_8b_auditable_financial_agent_research.md)：系统设计、实验台账、结果与复现命令。
- [v2.9 阶段策略复盘](docs/finllm_v2_9_strategy_review.md)：当前结论、数据谱系、SFT 边界与 GRPO 准入条件。
- [Financial Agent Repository Audit](docs/financial_agent_audit.md)：仓库能力、缺口和 Financial Agent Harness 扩展边界。
- [Agentic SFT v2](docs/agentic_sft_v2.md)：轨迹采集、数据 gate 与 trusted evaluation。
- [Validator Calibration Report](docs/finllm_v2_8_validator_calibration_report.md)：validator 混淆矩阵与 stock blocker。
- [Stock Contract Repair Report](docs/finllm_v2_8_2_stock_contract_repair_report.md)：v2.8.2 失败实验与 adapter 拒绝依据。
- [RLVF 使用指南](docs/rlhf_guide.md)：DPO/GRPO 数据、配置与验证入口。

## 模型导出

```bash
# Hugging Face 格式
bash scripts/training/export_model.sh --format hf

# GGUF（用于 llama.cpp / Ollama）
bash scripts/training/export_model.sh --format gguf --quant q4_k_m
```

## License

本项目仅供学习和研究使用。使用模型与训练框架时，请同时遵守 [Qwen License](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE) 和 [LLaMA-Factory License](https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE)。
