#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/super/.conda/envs/finllm/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/super/.conda/envs/finllm/bin/llamafactory-cli}"
SELECTED_ADAPTER="$PROJECT_ROOT/saves/qwen3-8b/lora/sft-v2.4-selected/adapter_model.safetensors"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/training/train_dpo_v2_4.sh [--dry-run]" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
[[ -f "$SELECTED_ADAPTER" || "$DRY_RUN" == true ]] || {
  echo "Dev-Audit-selected SFT adapter is missing: $SELECTED_ADAPTER" >&2
  exit 2
}
[[ -f data/rlhf/sft_v2_4_hard_negative_preference.json ]] || {
  echo "Hard-negative preference data is missing; run build_sft_v2_4_dataset.py" >&2
  exit 2
}

CONFIG="$PROJECT_ROOT/configs/qwen3_8b_qlora_dpo_v2_4_hard_negative.yaml"
if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" -c '
from omegaconf import OmegaConf
from llamafactory.hparams import get_train_args
import sys
config = OmegaConf.to_container(OmegaConf.load(sys.argv[1]))
_, data_args, _, finetuning_args, _ = get_train_args(config)
assert data_args.enable_thinking is False
assert finetuning_args.stage == "dpo"
' "$CONFIG"
  echo "Dry run passed: config=$CONFIG"
  exit 0
fi
"$LLAMAFACTORY_CLI" train "$CONFIG"
