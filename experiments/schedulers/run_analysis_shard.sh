#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# Number of array jobs/shards. Must match the scheduler array range length.
NUM_SHARDS="${NUM_SHARDS:-32}"
SHARD_INDEX="${SHARD_INDEX:-${PBS_ARRAYID:-}}"
if [ -z "$SHARD_INDEX" ]; then
  echo "SHARD_INDEX is required (set SHARD_INDEX or submit with qsub -t start-end)."
  exit 1
fi

# Runtime controls (override via qsub -v if needed)
BATCH_SIZE="${BATCH_SIZE:-}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-16}"
INPUT_MODE="${INPUT_MODE:-control}"
CONFIDENCE_NUM_SAMPLES="${CONFIDENCE_NUM_SAMPLES:-5000}"
CONFIDENCE_BETA="${CONFIDENCE_BETA:-1e-5}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/data/analysis}"
MAX_SETTINGS="${MAX_SETTINGS:-}"

echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} started at $(date)"
echo "Python command: python"
echo "Output dir: ${OUTPUT_DIR}"

CMD=(
  python experiments/paper/analysis_trajflow.py
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --sample-batch-size "$SAMPLE_BATCH_SIZE"
  --input-mode "$INPUT_MODE"
  --confidence-num-samples "$CONFIDENCE_NUM_SAMPLES"
  --confidence-beta "$CONFIDENCE_BETA"
  --num-random-seeds "$NUM_RANDOM_SEEDS"
  --output-dir "$OUTPUT_DIR"
)

if [ -n "$BATCH_SIZE" ]; then
  CMD+=(--batch-size "$BATCH_SIZE")
fi

if [ -n "$MAX_SETTINGS" ]; then
  CMD+=(--max-settings "$MAX_SETTINGS")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Shard ${SHARD_INDEX} finished at $(date)"
