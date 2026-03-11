#!/bin/bash
# ============================================================
# 模型导出脚本
#
# 支持多种导出格式:
#   1. HuggingFace 格式（默认，合并 LoRA 后的完整模型）
#   2. GGUF 格式（用于 llama.cpp 推理）
#
# 用法:
#   bash scripts/training/export_model.sh
#   bash scripts/training/export_model.sh --format gguf
# ============================================================

set -e

# ==================== 默认参数 ====================
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXPORT_FORMAT="hf"  # hf 或 gguf
MERGED_MODEL_DIR="${PROJECT_ROOT}/saves/qwen2.5-7b/merged"
EXPORT_DIR="${PROJECT_ROOT}/saves/qwen2.5-7b/exported"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --format|-f)
            EXPORT_FORMAT="$2"
            shift 2
            ;;
        --input|-i)
            MERGED_MODEL_DIR="$2"
            shift 2
            ;;
        --output|-o)
            EXPORT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash export_model.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --format, -f <hf|gguf>  导出格式 (默认: hf)"
            echo "  --input, -i <path>      输入模型路径 (默认: saves/qwen2.5-7b/merged)"
            echo "  --output, -o <path>     导出输出路径"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# ==================== 颜色输出 ====================
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==================== 检查 ====================
info "=============================================="
info "模型导出 (格式: ${EXPORT_FORMAT})"
info "=============================================="

# 检查合并后的模型是否存在
if [ ! -d "${MERGED_MODEL_DIR}" ]; then
    warn "合并模型不存在，先执行 LoRA 合并..."
    bash "${PROJECT_ROOT}/scripts/training/merge_lora.sh"
fi

if [ ! -f "${MERGED_MODEL_DIR}/config.json" ]; then
    error "合并模型目录中未找到 config.json: ${MERGED_MODEL_DIR}"
    exit 1
fi

info "输入模型: ${MERGED_MODEL_DIR}"
info "输出目录: ${EXPORT_DIR}"

# ==================== 执行导出 ====================
mkdir -p "${EXPORT_DIR}"

case "${EXPORT_FORMAT}" in
    hf)
        info "导出 HuggingFace 格式..."
        # 直接复制合并后的模型（已经是 HF 格式）
        cp -r "${MERGED_MODEL_DIR}"/* "${EXPORT_DIR}/"
        info "HuggingFace 格式导出完成"
        ;;
    gguf)
        info "导出 GGUF 格式..."
        if ! command -v python3 &> /dev/null; then
            error "需要 Python3 环境"
            exit 1
        fi
        # 使用 llama.cpp 的转换脚本
        if [ ! -d "${PROJECT_ROOT}/llama.cpp" ]; then
            warn "未找到 llama.cpp，正在克隆..."
            git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${PROJECT_ROOT}/llama.cpp"
        fi
        python3 "${PROJECT_ROOT}/llama.cpp/convert_hf_to_gguf.py" \
            "${MERGED_MODEL_DIR}" \
            --outfile "${EXPORT_DIR}/fin-instruct-qwen2.5-7b.gguf" \
            --outtype f16
        info "GGUF 格式导出完成"
        ;;
    *)
        error "未知格式: ${EXPORT_FORMAT}"
        error "支持的格式: hf, gguf"
        exit 1
        ;;
esac

info ""
info "=============================================="
info "导出完成！"
info "输出路径: ${EXPORT_DIR}"
info "=============================================="
