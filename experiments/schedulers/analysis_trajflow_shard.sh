#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

PROJECT_DIR="${PROJECT_DIR:-$HOME/Projects/RandomizedSmoothing}"
cd "$PROJECT_DIR"

APPTAINER_IMAGE="${APPTAINER_IMAGE:-}"

if [ -z "$APPTAINER_IMAGE" ]; then
  echo "APPTAINER_IMAGE is required (example: $PROJECT_DIR/randomized-smoothing.sif)"
  exit 1
fi

PYTHON_CMD=(
  apptainer exec
  --bind "$PROJECT_DIR:$PROJECT_DIR"
  --pwd "$PROJECT_DIR"
  "$APPTAINER_IMAGE"
  python
)

# Number of array jobs/shards. Must match the scheduler array range length.
NUM_SHARDS="${NUM_SHARDS:-32}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [ -z "$SHARD_INDEX" ]; then
  echo "SHARD_INDEX is required (set SHARD_INDEX or use SLURM_ARRAY_TASK_ID)."
  exit 1
fi

# Runtime controls (override via sbatch --export if needed)
BATCH_SIZE="${BATCH_SIZE:-}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-16}"
INPUT_MODE="${INPUT_MODE:-control}"
CONFIDENCE_NUM_SAMPLES="${CONFIDENCE_NUM_SAMPLES:-5000}"
CONFIDENCE_BETA="${CONFIDENCE_BETA:-1e-5}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/data/analysis}"
MAX_SETTINGS="${MAX_SETTINGS:-}"

echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} started at $(date)"
echo "Python command: ${PYTHON_CMD[*]}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Apptainer image: ${APPTAINER_IMAGE}"

CMD=(
  "${PYTHON_CMD[@]}" experiments/paper/analysis_trajflow.py
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
