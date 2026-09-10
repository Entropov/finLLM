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
CONFIG_FILE="${PROJECT_ROOT}/configs/qwen3_8b_merge_lora.yaml"

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

if [ ! -f "${CONFIG_FILE}" ]; then
    error "配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

CONFIG_DIR="$(dirname "${CONFIG_FILE}")"
read_yaml_value() {
    local key="$1"
    python3 - "${CONFIG_FILE}" "${key}" <<'PY'
import sys

config_file, key = sys.argv[1], sys.argv[2]
with open(config_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip() == key:
            print(v.strip().strip("'\""))
            break
PY
}

# 检查 adapter 是否存在
ADAPTER_DIR="$(read_yaml_value adapter_name_or_path)"
OUTPUT_DIR="$(read_yaml_value export_dir)"

if [ -z "${ADAPTER_DIR}" ]; then
    error "配置文件中未找到 adapter_name_or_path: ${CONFIG_FILE}"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    error "配置文件中未找到 export_dir: ${CONFIG_FILE}"
    exit 1
fi

case "${ADAPTER_DIR}" in
    /*) ;;
    *) ADAPTER_DIR="${PROJECT_ROOT}/${ADAPTER_DIR}" ;;
esac

case "${OUTPUT_DIR}" in
    /*) ;;
    *) OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}" ;;
esac

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

info ""
info "=============================================="
info "合并完成！"
info "合并模型路径: ${OUTPUT_DIR}"
info ""
info "下一步可以:"
info "  1. 启动推理: llamafactory-cli chat configs/qwen3_8b_inference.yaml"
info "  2. 部署 API: python scripts/inference/api_server.py"
info "  3. 启动 Demo: python scripts/inference/chat_demo.py"
info "=============================================="
