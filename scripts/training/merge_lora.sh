#!/bin/bash
# ============================================================
# LoRA 权重合并脚本
#
# 将训练好的 LoRA adapter 合并回基座模型，
# 生成可独立部署的完整模型权重（safetensors 格式）。
#
# 用法:
#   bash scripts/training/merge_lora.sh
#   bash scripts/training/merge_lora.sh --config <path>
# ============================================================

set -e

# ==================== 默认参数 ====================
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/qwen2.5_7b_merge_lora.yaml"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash merge_lora.sh [--config <path>]"
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
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==================== 检查 ====================
info "=============================================="
info "LoRA 权重合并"
info "=============================================="

# 检查 adapter 是否存在
ADAPTER_DIR="${PROJECT_ROOT}/saves/qwen2.5-7b/lora/sft"
if [ ! -d "${ADAPTER_DIR}" ]; then
    error "LoRA adapter 目录不存在: ${ADAPTER_DIR}"
    error "请先完成训练"
    exit 1
fi

if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ] && [ ! -f "${ADAPTER_DIR}/adapter_model.bin" ]; then
    error "未找到 adapter 权重文件"
    error "请确保训练已完成并保存了 checkpoint"
    exit 1
fi

info "Adapter 路径: ${ADAPTER_DIR}"
info "配置文件: ${CONFIG_FILE}"

# ==================== 执行合并 ====================
info "开始合并 LoRA 权重..."

cd "${PROJECT_ROOT}"
llamafactory-cli export "${CONFIG_FILE}"

OUTPUT_DIR="${PROJECT_ROOT}/saves/qwen2.5-7b/merged"
info ""
info "=============================================="
info "合并完成！"
info "合并模型路径: ${OUTPUT_DIR}"
info ""
info "下一步可以:"
info "  1. 启动推理: llamafactory-cli chat configs/qwen2.5_7b_inference.yaml"
info "  2. 部署 API: python scripts/inference/api_server.py"
info "  3. 启动 Demo: python scripts/inference/chat_demo.py"
info "=============================================="
