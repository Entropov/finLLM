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
    -h|--help) echo "Usage: bash scripts/training/train_sft_v2_5.sh [--skip-build] [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "$SKIP_BUILD" != true ]]; then
  "$PYTHON_BIN" scripts/data_processing/build_sft_v2_5_dataset.py
fi

"$PYTHON_BIN" -c '
import json
from pathlib import Path
report = json.loads(Path("data/sft_v2_5/build_report.json").read_text(encoding="utf-8"))
if not report.get("release_gate_passed"):
    raise SystemExit("SFT v2.5 data gate failed: {}".format(report.get("release_gate")))
print("SFT v2.5 gate passed: {} train / {} loss-eval / {} Dev-Audit / {} hard negatives".format(
    report["components"]["train"]["samples"], report["components"]["loss_eval"]["samples"],
    report["dev_audit"]["samples"], report["preference"]["samples"],
))
'
"$PYTHON_BIN" -c 'import json; json.load(open("data/dataset_info.json", encoding="utf-8"))'
[[ -x "$LLAMAFACTORY_CLI" ]] || { echo "llamafactory-cli is not executable: $LLAMAFACTORY_CLI" >&2; exit 127; }

CONFIG="$PROJECT_ROOT/configs/qwen3_8b_qlora_sft_v2_5_answer.yaml"
if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" -c '
from omegaconf import OmegaConf
from llamafactory.hparams import get_train_args
import sys
config = OmegaConf.to_container(OmegaConf.load(sys.argv[1]))
_, data_args, _, finetuning_args, _ = get_train_args(config)
assert data_args.enable_thinking is False
assert finetuning_args.use_audit_weighted_loss is True
assert finetuning_args.audit_eos_weight == 12.0
' "$CONFIG"
  echo "Dry run passed: config=$CONFIG"
  exit 0
fi
"$LLAMAFACTORY_CLI" train "$CONFIG"
