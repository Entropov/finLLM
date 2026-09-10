#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/super/.conda/envs/finllm/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/super/.conda/envs/finllm/bin/llamafactory-cli}"
SKIP_BUILD=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) echo "Usage: bash scripts/training/train_sft_v2_7_core.sh [--skip-build] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ "$SKIP_BUILD" != true ]]; then
  "$PYTHON_BIN" scripts/data_processing/build_sft_v2_7_core_dataset.py
fi
"$PYTHON_BIN" -c 'import json; r=json.load(open("data/sft_v2_7_core/build_report.json")); assert r["release_gate_passed"], r["release_gate"]'
[[ -x "$LLAMAFACTORY_CLI" ]] || { echo "llamafactory-cli is not executable: $LLAMAFACTORY_CLI" >&2; exit 127; }
if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" -c 'from omegaconf import OmegaConf; from llamafactory.hparams import get_train_args; c=OmegaConf.to_container(OmegaConf.load("configs/qwen3_8b_qlora_sft_v2_7_core_answer.yaml")); _,d,_,f,_=get_train_args(c); assert d.enable_thinking is False and f.use_audit_weighted_loss is True; print("SFT v2.7 core dry run passed")'
  exit 0
fi
exec "$LLAMAFACTORY_CLI" train configs/qwen3_8b_qlora_sft_v2_7_core_answer.yaml
