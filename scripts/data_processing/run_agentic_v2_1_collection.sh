#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/super/.conda/envs/finllm/bin/python}"
PORT="${AGENTIC_COLLECTION_PORT:-8001}"
BASE_URL="http://127.0.0.1:${PORT}"
SERVER_LOG="${AGENTIC_COLLECTION_SERVER_LOG:-$PROJECT_ROOT/saves/eval_results/agentic_v2_1_api.log}"
INPUT="${AGENTIC_COLLECTION_INPUT:-data/rag/v2_1_collection_requests.json}"
OUTPUT="${AGENTIC_COLLECTION_OUTPUT:-$PROJECT_ROOT/data/rag/v2_1_collection_responses.jsonl}"
TASKS="${AGENTIC_COLLECTION_TASKS:-}"
CONCURRENCY="${AGENTIC_COLLECTION_CONCURRENCY:-12}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$SERVER_LOG")"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export VLLM_USE_FLASHINFER_SAMPLER="0"
export TOKENIZERS_PARALLELISM="false"

"$PYTHON_BIN" scripts/inference/api_server.py \
  --backend vllm \
  --model-path saves/qwen3-8b/merged \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 8192 \
  --enable-rag \
  --rag-mode agentic \
  --rag-agentic-config configs/rag_agentic.yaml \
  --rag-db-dir saves/chroma_v2_train \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

COLLECT_ARGS=(
  --input "$INPUT"
  --output "$OUTPUT"
  --base-url "$BASE_URL"
  --concurrency "$CONCURRENCY"
  --retries 3
  --progress-every 20
)
for task in $TASKS; do
  COLLECT_ARGS+=(--task "$task")
done

"$PYTHON_BIN" scripts/data_processing/collect_agentic_v2_trajectories.py "${COLLECT_ARGS[@]}"
