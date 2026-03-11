# Fin-Instruct 🏦

基于 **Qwen2.5-7B-Instruct** + **LLaMA-Factory** 框架的金融大语言模型 QLoRA 微调项目，旨在辅助股票市场分析、量化策略研发与财报解读。

> ⚠️ **免责声明**：本项目所有模型输出仅供学习研究参考，不构成任何投资建议。投资有风险，决策需谨慎。

---

## 📋 项目概述

| 项目 | 详情 |
|------|------|
| 基座模型 | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| 微调框架 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| 微调方法 | QLoRA (4-bit NF4 量化 + LoRA) |
| 目标 GPU | NVIDIA RTX 5090 32GB (单卡) |
| 预估显存 | ~12 GB (QLoRA 4-bit) |
| 数据格式 | ShareGPT |

### 支持的 6 大金融任务

1. **📈 股票分析** — 个股基本面与技术面分析
2. **🤖 量化策略** — 量化交易策略设计与代码实现
3. **📊 财报解读** — 三大报表分析与财务指标解读
4. **💬 情感分析** — 金融新闻与舆情情感判断
5. **❓ 金融问答** — 金融知识百科问答
6. **⚠️ 风险评估** — 信用风险与市场风险评估

---

## 🗂️ 项目结构

```
finLLM/
├── configs/                          # 训练/推理配置
│   ├── qwen2.5_7b_qlora_sft.yaml    # QLoRA SFT 训练配置
│   ├── qwen2.5_7b_qlora_sft_eval.yaml # 评估配置
│   ├── qwen2.5_7b_merge_lora.yaml   # LoRA 合并配置
│   └── qwen2.5_7b_inference.yaml    # 推理配置
├── data/
│   ├── raw/                          # 原始数据（git 忽略）
│   ├── processed/                    # 处理后数据（git 忽略）
│   └── dataset_info.json            # LLaMA-Factory 数据集注册
├── prompts/
│   └── system_prompts.json          # 6 类任务 + 通用系统提示词
├── scripts/
│   ├── data_collection/             # 数据采集
│   │   ├── download_open_datasets.py # 下载开源数据集
│   │   ├── fetch_stock_data.py      # 获取股票行情数据
│   │   ├── fetch_financial_reports.py # 获取上市公司财报
│   │   └── fetch_news.py           # 获取金融新闻
│   ├── data_processing/             # 数据处理
│   │   ├── clean_data.py           # 数据清洗
│   │   ├── synthesize_instructions.py # LLM 指令合成
│   │   ├── convert_to_sharegpt.py  # 格式转换
│   │   ├── quality_filter.py       # 质量过滤与去重
│   │   └── merge_datasets.py       # 数据集合并与拆分
│   ├── training/                    # 训练管理
│   │   ├── train.sh                # 一键训练脚本
│   │   ├── merge_lora.sh           # LoRA 权重合并
│   │   └── export_model.sh         # 模型导出 (HF/GGUF)
│   ├── evaluation/                  # 评估
│   │   ├── eval_finance_bench.py   # FinEval 基准测试
│   │   ├── eval_task_specific.py   # 任务级评估
│   │   └── eval_metrics.py         # 评估指标工具库
│   └── inference/                   # 推理与演示
│       ├── api_server.py           # OpenAI 兼容 API 服务
│       ├── chat_demo.py            # Gradio WebUI 交互
│       └── batch_inference.py      # 批量推理
├── saves/                           # 模型存储（git 忽略）
├── tests/                           # 单元测试
│   ├── test_data_pipeline.py       # 数据管道测试
│   └── test_inference.py           # 推理模块测试
├── requirements.txt                 # Python 依赖
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url> finLLM && cd finLLM

# 创建 conda 环境（推荐）
conda create -n fin-instruct python=3.11 -y
conda activate fin-instruct

# 安装依赖
pip install -r requirements.txt

# 安装 LLaMA-Factory（开发模式）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..
```

### 2. 数据准备

```bash
# 步骤 1: 下载开源数据集
python scripts/data_collection/download_open_datasets.py

# 步骤 2: 采集 A 股行情数据
python scripts/data_collection/fetch_stock_data.py

# 步骤 3: 采集上市公司财报
python scripts/data_collection/fetch_financial_reports.py

# 步骤 4: 采集金融新闻
python scripts/data_collection/fetch_news.py

# 步骤 5: 数据清洗
python scripts/data_processing/clean_data.py

# 步骤 6: 指令合成（需设置 API 密钥环境变量）
export SYNTH_API_KEY="your-api-key"
export SYNTH_API_BASE="https://api.openai.com/v1"
export SYNTH_API_MODEL="gpt-4o-mini"
python scripts/data_processing/synthesize_instructions.py

# 步骤 7: 格式转换
python scripts/data_processing/convert_to_sharegpt.py

# 步骤 8: 质量过滤
python scripts/data_processing/quality_filter.py

# 步骤 9: 合并数据集并拆分训练/验证集
python scripts/data_processing/merge_datasets.py
```

### 3. 模型训练

```bash
# 方式一: 使用封装脚本（推荐）
bash scripts/training/train.sh

# 方式二: 后台训练
bash scripts/training/train.sh --background

# 方式三: 直接使用 LLaMA-Factory CLI
llamafactory-cli train configs/qwen2.5_7b_qlora_sft.yaml

# 使用 TensorBoard 监控训练
tensorboard --logdir saves/qwen2.5-7b/qlora-sft/runs --port 6006
```

### 4. LoRA 权重合并

```bash
# 合并 LoRA adapter 到基座模型
bash scripts/training/merge_lora.sh
```

### 5. 评估

```bash
# FinEval 基准测试
python scripts/evaluation/eval_finance_bench.py

# 任务级评估
python scripts/evaluation/eval_task_specific.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --adapter-path saves/qwen2.5-7b/qlora-sft

# 运行单元测试
pytest tests/ -v
```

### 6. 推理与部署

```bash
# Gradio 交互演示
python scripts/inference/chat_demo.py \
  --model-path saves/qwen2.5-7b/merged

# OpenAI 兼容 API 服务
python scripts/inference/api_server.py \
  --model-path saves/qwen2.5-7b/merged \
  --port 8000

# 批量推理
python scripts/inference/batch_inference.py \
  --input questions.json \
  --task stock_analysis \
  --model-path saves/qwen2.5-7b/merged
```

---

## ⚙️ 训练配置详情

| 参数 | 值 |
|------|-----|
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Target | all (全部线性层) |
| LoRA Dropout | 0.05 |
| 量化方式 | 4-bit NF4 (bitsandbytes) |
| 序列长度 | 4096 |
| Batch Size | 4 (per GPU) |
| 梯度累积 | 4 (有效 batch = 16) |
| 学习率 | 1e-4 |
| 调度器 | cosine |
| Warmup | 10% |
| 训练轮次 | 3 |
| 精度 | bf16 |

---

## 📊 数据来源

| 数据源 | 类型 | 说明 |
|--------|------|------|
| [FinGPT](https://huggingface.co/FinGPT) | 开源数据集 | 金融情感、QA、关系抽取、标题分类 |
| [DISC-FinLLM](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) | 开源数据集 | 约 246K 金融 SFT 数据 |
| [FinEval](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval) | 基准数据 | 4,661 条金融考试题 |
| [AKShare](https://akshare.akfamily.xyz/) | API 数据 | A 股行情、财报、新闻（免费） |
| [Tushare](https://tushare.pro/) | API 数据 | 金融数据接口（需注册） |
| LLM 合成 | 合成数据 | 基于大模型的指令合成 |

---

## 🧪 API 使用示例

启动 API 服务后，可直接使用 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="fin-instruct",
    messages=[
        {"role": "system", "content": "你是一位专业的金融分析师。"},
        {"role": "user", "content": "请分析贵州茅台（600519）的投资价值。"},
    ],
    temperature=0.7,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 🔧 模型导出

```bash
# 导出为 HuggingFace 格式
bash scripts/training/export_model.sh --format hf

# 导出为 GGUF 格式（用于 llama.cpp / Ollama）
bash scripts/training/export_model.sh --format gguf --quant q4_k_m
```

---

## 🖥️ 硬件要求

| 阶段 | 最低要求 | 推荐配置 |
|------|---------|---------|
| QLoRA 训练 | 16GB VRAM | RTX 5090 32GB |
| 4-bit 推理 | 8GB VRAM  | 16GB+ VRAM |
| 全精度推理 | 16GB VRAM | 24GB+ VRAM |
| 数据处理 | 16GB RAM  | 32GB+ RAM |

---

## 📝 开发路线

- [x] 项目框架搭建
- [ ] 数据采集与清洗 Pipeline
- [ ] 指令合成与质量过滤
- [ ] QLoRA 训练与超参调优
- [ ] 基准评估 (FinEval)
- [ ] 任务级评估
- [ ] Gradio Demo 上线
- [ ] API 服务部署
- [ ] GGUF 导出 + Ollama 集成

---

## 📄 License

本项目仅供学习研究使用。模型基于 Qwen2.5 系列，请遵守 [Qwen License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE)。
