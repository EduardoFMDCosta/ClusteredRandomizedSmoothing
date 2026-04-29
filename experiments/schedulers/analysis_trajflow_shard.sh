#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Projects/RandomizedSmoothing}"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-randomized_smoothing}"

if [ -n "${CONDA_ENV:-}" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "No Python environment found. Set CONDA_ENV or create .venv in $PROJECT_DIR"
  exit 1
fi

PYTHON_BIN="python"

# Number of array jobs/shards. Must match the qsub array range length.
NUM_SHARDS="${NUM_SHARDS:-32}"
SHARD_INDEX="${PBS_ARRAYID}"

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
echo "Python: ${PYTHON_BIN}"
echo "Output dir: ${OUTPUT_DIR}"

CMD=(
  "$PYTHON_BIN" experiments/paper/analysis_trajflow.py
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

"${CMD[@]}"

echo "Shard ${SHARD_INDEX} finished at $(date)"
