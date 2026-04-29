#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

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
INPUT_DIR="${OUTPUT_DIR:-experiments/data/analysis}"
EXPECTED_NUM_SHARDS="${NUM_SHARDS:-}"
OUTPUT_FILE="${MERGED_OUTPUT_FILE:-results.json}"

echo "Merge started at $(date)"
echo "Input dir: ${INPUT_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Conda env: ${CONDA_ENV}"

CMD=(
  "$PYTHON_BIN" experiments/paper/merge_analysis_results.py
  --input-dir "$INPUT_DIR"
  --output-file "$OUTPUT_FILE"
)

if [ -n "$EXPECTED_NUM_SHARDS" ]; then
  CMD+=(--expected-num-shards "$EXPECTED_NUM_SHARDS")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Merge finished at $(date)"
