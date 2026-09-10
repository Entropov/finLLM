#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/super/.conda/envs/finllm/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/home/super/.conda/envs/finllm/bin/llamafactory-cli}"
SKIP_BUILD=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: bash scripts/training/train_sft_v2_3.sh [options]

Options:
  --skip-build  Reuse an existing verified SFT v2.3 dataset.
  --dry-run     Validate data/config without launching training.
  -h, --help    Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "$SKIP_BUILD" != true ]]; then
  "$PYTHON_BIN" scripts/data_processing/build_sft_v2_3_dataset.py
fi

"$PYTHON_BIN" -c '
import json
from pathlib import Path
report = json.loads(Path("data/sft_v2_3/build_report.json").read_text(encoding="utf-8"))
if not report.get("release_gate_passed", False):
    raise SystemExit("SFT v2.3 data gate failed: {}".format(report.get("release_gate", {})))
train = report["components"]["train"]
evaluation = report["components"]["eval"]
adversarial = report["adversarial"]
print("SFT v2.3 data gate passed: {} train / {} eval / {} independent adversarial; audit={}/{}".format(
    train["samples"], evaluation["samples"], adversarial["samples"],
    train["audit_hard_pass_rate"], evaluation["audit_hard_pass_rate"],
))
'

"$PYTHON_BIN" -c 'import json; json.load(open("data/dataset_info.json", encoding="utf-8"))'
[[ -x "$LLAMAFACTORY_CLI" ]] || {
  echo "llamafactory-cli is not executable: $LLAMAFACTORY_CLI" >&2
  exit 127
}

CONFIG="$PROJECT_ROOT/configs/qwen3_8b_qlora_sft_v2_3_answer.yaml"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run passed: config=$CONFIG"
  exit 0
fi
"$LLAMAFACTORY_CLI" train "$CONFIG"
