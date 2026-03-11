#!/bin/bash
# ============================================================
# Fin-Instruct 一键训练启动脚本
#
# 功能:
#   1. 环境检查（CUDA、GPU显存、Python依赖）
#   2. 启动 QLoRA SFT 训练
#   3. 支持后台运行和日志记录
#
# 用法:
#   bash scripts/training/train.sh                  # 前台运行
#   bash scripts/training/train.sh --background     # 后台运行
#   bash scripts/training/train.sh --config <path>  # 指定配置文件
# ============================================================

set -e  # 遇到错误立即退出

# ==================== 默认参数 ====================
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/qwen2.5_7b_qlora_sft.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
BACKGROUND=false

# ==================== 解析命令行参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --background|-bg)
            BACKGROUND=true
            shift
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash train.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --background, -bg     后台运行训练"
            echo "  --config, -c <path>   指定训练配置文件 (默认: configs/qwen2.5_7b_qlora_sft.yaml)"
            echo "  --help, -h            显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==================== 环境检查 ====================
info "=============================================="
info "Fin-Instruct QLoRA 训练启动"
info "=============================================="

# 检查 CUDA
info "检查 CUDA 环境..."
if ! command -v nvidia-smi &> /dev/null; then
    error "未找到 nvidia-smi，请确保已安装 NVIDIA 驱动"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
info "GPU: ${GPU_INFO}"

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${GPU_MEM}" -lt 20000 ]; then
    warn "GPU显存 ${GPU_MEM}MB 可能不足，建议 ≥24GB"
fi

# 检查 Python
info "检查 Python 环境..."
PYTHON_VERSION=$(python3 --version 2>&1)
info "Python: ${PYTHON_VERSION}"

# 检查 LLaMA-Factory
if ! command -v llamafactory-cli &> /dev/null; then
    error "未找到 llamafactory-cli"
    error "请先安装 LLaMA-Factory:"
    error "  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git"
    error "  cd LLaMA-Factory && pip install -e ."
    exit 1
fi
info "LLaMA-Factory: $(llamafactory-cli version 2>/dev/null || echo '已安装')"

# 检查配置文件
if [ ! -f "${CONFIG_FILE}" ]; then
    error "配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi
info "配置文件: ${CONFIG_FILE}"

# 检查数据集
DATASET_DIR="${PROJECT_ROOT}/data"
TRAIN_DATA="${DATASET_DIR}/sft/fin_instruct_train.json"
if [ ! -f "${TRAIN_DATA}" ]; then
    warn "训练数据不存在: ${TRAIN_DATA}"
    warn "请先运行数据处理管道生成训练数据"
fi

# ==================== 创建日志目录 ====================
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ==================== 启动训练 ====================
info ""
info "开始训练..."
info "日志文件: ${LOG_FILE}"
info "=============================================="

cd "${PROJECT_ROOT}"

if [ "${BACKGROUND}" = true ]; then
    info "后台运行模式, 使用 'tail -f ${LOG_FILE}' 查看日志"
    nohup llamafactory-cli train "${CONFIG_FILE}" > "${LOG_FILE}" 2>&1 &
    TRAIN_PID=$!
    info "训练进程 PID: ${TRAIN_PID}"
    echo "${TRAIN_PID}" > "${LOG_DIR}/train_${TIMESTAMP}.pid"
    info "PID 已保存到: ${LOG_DIR}/train_${TIMESTAMP}.pid"
else
    llamafactory-cli train "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"
    info ""
    info "=============================================="
    info "训练完成！"
    info "模型保存路径: saves/qwen2.5-7b/lora/sft"
    info "日志文件: ${LOG_FILE}"
    info "=============================================="
fi
