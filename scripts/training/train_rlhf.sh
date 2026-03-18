#!/bin/bash
# ============================================================
# Fin-Instruct RLHF (DPO) 一键训练脚本
#
# 训练流程:
#   Step 1: 检查 SFT checkpoint
#   Step 2: 生成偏好数据 (可跳过，如数据已存在)
#   Step 3: 执行 DPO 训练
#   Step 4: 输出训练摘要
#
# 用法:
#   bash scripts/training/train_rlhf.sh                     # 完整流程
#   bash scripts/training/train_rlhf.sh --skip-data         # 跳过数据生成
#   bash scripts/training/train_rlhf.sh --background        # 后台运行
#   bash scripts/training/train_rlhf.sh --mode rules        # 规则模式生成数据
#   bash scripts/training/train_rlhf.sh -h                  # 帮助
# ============================================================

set -e

# ==================== 默认参数 ====================
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DPO_CONFIG="${PROJECT_ROOT}/configs/qwen2.5_7b_qlora_dpo.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
PREF_DATA="${PROJECT_ROOT}/data/rlhf/fin_preference_train.json"
SFT_CHECKPOINT="${PROJECT_ROOT}/saves/qwen2.5-7b/lora/sft/checkpoint-2000"
SYNTH_SCRIPT="${PROJECT_ROOT}/scripts/data_processing/synthesize_preference_data.py"
SFT_TRAIN_DATA="${PROJECT_ROOT}/data/sft/fin_instruct_train.json"

BACKGROUND=false
SKIP_DATA=false
DATA_MODE="both"
MAX_PREF_SAMPLES=5000
CONFIG_FILE="${DPO_CONFIG}"

# ==================== 解析参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --background|-bg)       BACKGROUND=true;    shift ;;
        --skip-data)            SKIP_DATA=true;     shift ;;
        --config|-c)            CONFIG_FILE="$2";   shift 2 ;;
        --mode|-m)              DATA_MODE="$2";     shift 2 ;;
        --max-samples|-n)       MAX_PREF_SAMPLES="$2"; shift 2 ;;
        --help|-h)
            echo "用法: bash train_rlhf.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --background, -bg       后台运行"
            echo "  --skip-data             跳过偏好数据生成 (若数据已存在)"
            echo "  --config, -c <path>     指定 DPO 配置文件"
            echo "  --mode, -m <mode>       数据生成模式: rules|llm|both (默认 both)"
            echo "  --max-samples, -n <N>   最大偏好样本数 (默认 5000)"
            echo "  --help, -h              显示帮助"
            exit 0 ;;
        *)  echo "未知参数: $1"; exit 1 ;;
    esac
done

# ==================== 颜色输出 ====================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
section() { echo -e "\n${CYAN}══════════════════════════════════════════${NC}"; \
            echo -e "${CYAN}  $1${NC}"; \
            echo -e "${CYAN}══════════════════════════════════════════${NC}\n"; }

# ==================== Step 0: 环境检查 ====================
section "🔍 环境检查"

# CUDA
if ! command -v nvidia-smi &> /dev/null; then
    error "未找到 nvidia-smi，请确保已安装 NVIDIA 驱动"; exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
info "GPU: ${GPU_INFO}"

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${GPU_MEM}" -lt 20000 ]; then
    warn "GPU 显存 ${GPU_MEM}MB 可能不足，DPO 建议 ≥24GB"
fi

# LLaMA-Factory
if ! command -v llamafactory-cli &> /dev/null; then
    error "未找到 llamafactory-cli，请先安装 LLaMA-Factory"; exit 1
fi
info "LLaMA-Factory: $(llamafactory-cli version 2>/dev/null || echo '已安装')"

# Python
PYTHON_VER=$(python3 --version 2>&1)
info "Python: ${PYTHON_VER}"

# 配置文件
if [ ! -f "${CONFIG_FILE}" ]; then
    error "DPO 配置文件不存在: ${CONFIG_FILE}"; exit 1
fi
info "DPO 配置: ${CONFIG_FILE}"

# ==================== Step 1: 检查 SFT Checkpoint ====================
section "📦 检查 SFT Checkpoint"

if [ -d "${SFT_CHECKPOINT}" ]; then
    ADAPTER_FILES=$(find "${SFT_CHECKPOINT}" -name "adapter_model.safetensors" -o -name "adapter_model.bin" 2>/dev/null | wc -l)
    if [ "${ADAPTER_FILES}" -gt 0 ]; then
        info "✅ SFT checkpoint 存在: ${SFT_CHECKPOINT}"
    else
        warn "SFT 目录存在但未找到 adapter 权重文件"
        warn "请确认 SFT 训练已完成，或检查路径是否正确"
    fi
else
    warn "⚠️  SFT checkpoint 不存在: ${SFT_CHECKPOINT}"
    warn "DPO 将从基座模型开始（效果可能不如从 SFT 继续）"
    warn "建议先运行: bash scripts/training/train.sh"
fi

# ==================== Step 2: 生成偏好数据 ====================
section "📊 偏好数据准备"

mkdir -p "${PROJECT_ROOT}/data/rlhf"

if [ "${SKIP_DATA}" = true ]; then
    info "跳过数据生成 (--skip-data)"
    if [ ! -f "${PREF_DATA}" ]; then
        error "偏好数据不存在: ${PREF_DATA}"
        error "请先生成数据或移除 --skip-data 选项"
        exit 1
    fi
else
    if [ -f "${PREF_DATA}" ]; then
        EXISTING_LINES=$(python3 -c "import json; print(len(json.load(open('${PREF_DATA}'))))" 2>/dev/null || echo "0")
        if [ "${EXISTING_LINES}" -gt 0 ]; then
            warn "偏好数据已存在 (${EXISTING_LINES} 条): ${PREF_DATA}"
            warn "如需重新生成，请手动删除该文件后重新运行"
            info "使用已有数据继续..."
        else
            SKIP_DATA=false
        fi
    fi

    if [ ! -f "${PREF_DATA}" ]; then
        info "开始生成偏好数据 (模式: ${DATA_MODE}, 最大样本: ${MAX_PREF_SAMPLES})"

        if [ ! -f "${SFT_TRAIN_DATA}" ]; then
            error "SFT 训练数据不存在: ${SFT_TRAIN_DATA}"
            error "请先运行 SFT 数据处理流程"
            exit 1
        fi

        # 临时数据从规则生成（即使没有 API key 也能运行）
        ACTUAL_MODE="${DATA_MODE}"
        if [ "${DATA_MODE}" = "llm" ] && [ -z "${SYNTH_API_KEY}" ]; then
            warn "未设置 SYNTH_API_KEY，自动切换到 rules 模式"
            ACTUAL_MODE="rules"
        fi

        python3 "${SYNTH_SCRIPT}" \
            --input "${SFT_TRAIN_DATA}" \
            --output "${PREF_DATA}" \
            --mode "${ACTUAL_MODE}" \
            --max-samples "${MAX_PREF_SAMPLES}" \
            --seed 42

        GENERATED=$(python3 -c "import json; print(len(json.load(open('${PREF_DATA}'))))" 2>/dev/null || echo "0")
        if [ "${GENERATED}" -lt 100 ]; then
            error "生成的偏好样本数量过少 (${GENERATED})，请检查数据生成脚本"
            exit 1
        fi
        info "✅ 偏好数据生成完成: ${GENERATED} 条"

        # 划分训练/验证集 (95/5)
        info "划分训练/验证集 (95/5)..."
        python3 - <<'PYEOF'
import json, random
random.seed(42)
PREF_DATA = "${PREF_DATA}".replace("${PREF_DATA}", "${PREF_DATA}")
data = json.load(open("${PREF_DATA}"))
random.shuffle(data)
split = int(len(data) * 0.95)
train_data, eval_data = data[:split], data[split:]
import os
base = os.path.dirname("${PREF_DATA}")
with open(os.path.join(base, "fin_preference_train.json"), "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open(os.path.join(base, "fin_preference_eval.json"), "w") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)
print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
PYEOF
    fi
fi

# 显示数据统计
PREF_COUNT=$(python3 -c "import json; print(len(json.load(open('${PREF_DATA}'))))" 2>/dev/null || echo "未知")
info "偏好训练集: ${PREF_COUNT} 条样本"

# ==================== Step 3: 启动 DPO 训练 ====================
section "🚀 启动 DPO 训练"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/dpo_${TIMESTAMP}.log"

info "日志文件: ${LOG_FILE}"
info "配置文件: ${CONFIG_FILE}"
info "输出目录: saves/qwen2.5-7b/lora/dpo"

cd "${PROJECT_ROOT}"

if [ "${BACKGROUND}" = true ]; then
    info "后台运行，使用 'tail -f ${LOG_FILE}' 查看日志"
    nohup llamafactory-cli train "${CONFIG_FILE}" > "${LOG_FILE}" 2>&1 &
    DPO_PID=$!
    info "DPO 训练进程 PID: ${DPO_PID}"
    echo "${DPO_PID}" > "${LOG_DIR}/dpo_${TIMESTAMP}.pid"
    info "PID 已保存: ${LOG_DIR}/dpo_${TIMESTAMP}.pid"
else
    llamafactory-cli train "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

    # ==================== Step 4: 训练摘要 ====================
    section "📈 训练完成摘要"
    info "DPO 模型保存路径: saves/qwen2.5-7b/lora/dpo"
    info "日志文件: ${LOG_FILE}"
    info ""
    info "下一步操作："
    info "  1. 合并 LoRA 权重:"
    info "     bash scripts/training/merge_lora.sh"
    info "  2. 对比评估 SFT vs DPO:"
    info "     python scripts/evaluation/eval_rlhf_comparison.py \\"
    info "       --sft-adapter saves/qwen2.5-7b/lora/sft \\"
    info "       --dpo-adapter saves/qwen2.5-7b/lora/dpo \\"
    info "       --output logs/rlhf_comparison.md"
    info "  3. 查看 TensorBoard:"
    info "     tensorboard --logdir saves/qwen2.5-7b/lora/dpo/runs --port 6006"
fi
