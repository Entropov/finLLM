#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$PROJECT_ROOT/configs/qwen3_8b_qlora_sft_v2.yaml"
SKIP_BUILD=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage: bash scripts/training/train_sft_v2.sh [options]

Options:
  --config PATH   Override the LLaMA-Factory config.
  --skip-build    Reuse an existing verified SFT v2 dataset.
  --dry-run       Validate data and config without launching training.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"

if [[ "$SKIP_BUILD" != true ]]; then
  python scripts/data_processing/build_sft_v2_dataset.py
fi

python -c '
import json
from pathlib import Path
report = json.loads(Path("data/sft_v2/build_report.json").read_text(encoding="utf-8"))
if not report.get("release_gate_passed", False):
    raise SystemExit(f"SFT v2 data gate failed: {report.get('task_shortfalls', {})}")
print("SFT v2 data gate passed: {} train / {} eval samples".format(report["train_samples"], report["eval_samples"]))
'

python -c 'import json, sys; json.load(open("data/dataset_info.json", encoding="utf-8")); sys.exit(0)'

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run passed: $CONFIG"
  exit 0
fi

command -v llamafactory-cli >/dev/null 2>&1 || {
  echo "llamafactory-cli is not installed or not on PATH" >&2
  exit 127
}
llamafactory-cli train "$CONFIG"
