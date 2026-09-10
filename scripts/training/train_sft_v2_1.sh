#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/super/.conda/envs/finllm/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/super/.conda/envs/finllm/bin/llamafactory-cli}"
COMPONENT="answer"
SKIP_BUILD=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: bash scripts/training/train_sft_v2_1.sh [options]

Options:
  --component NAME  Train answer, policy, or both (default: answer).
  --skip-build      Reuse an existing verified SFT v2.1 dataset.
  --dry-run         Validate data/config without launching training.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --component) COMPONENT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$COMPONENT" in
  answer|policy|both) ;;
  *) echo "Invalid component: $COMPONENT" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "$SKIP_BUILD" != true ]]; then
  "$PYTHON_BIN" scripts/data_processing/build_sft_v2_1_dataset.py
fi

"$PYTHON_BIN" -c '
import json
from pathlib import Path
report = json.loads(Path("data/sft_v2_1/build_report.json").read_text(encoding="utf-8"))
if not report.get("release_gate_passed", False):
    raise SystemExit("SFT v2.1 data gate failed: {}".format(report.get("release_gate", {})))
components = report["components"]
print("SFT v2.1 data gate passed: {} answer train / {} answer eval / {} policy train / {} policy eval".format(
    components["answer_train"]["samples"], components["answer_eval"]["samples"],
    components["policy_train"]["samples"], components["policy_eval"]["samples"],
))
'

"$PYTHON_BIN" -c 'import json; json.load(open("data/dataset_info.json", encoding="utf-8"))'
[[ -x "$LLAMAFACTORY_CLI" ]] || {
  echo "llamafactory-cli is not executable: $LLAMAFACTORY_CLI" >&2
  exit 127
}

train_component() {
  local name="$1"
  local config="$PROJECT_ROOT/configs/qwen3_8b_qlora_sft_v2_1_${name}.yaml"
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
  if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run passed: component=$name config=$config"
    return
  fi
  "$LLAMAFACTORY_CLI" train "$config"
}

if [[ "$COMPONENT" == "both" ]]; then
  train_component answer
  train_component policy
else
  train_component "$COMPONENT"
fi
