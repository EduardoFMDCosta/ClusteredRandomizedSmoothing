#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Projects/RandomizedSmoothing}"
cd "$PROJECT_DIR"

source .venv/bin/activate
PYTHON_BIN="python"
INPUT_DIR="${OUTPUT_DIR:-experiments/data/analysis_parallel}"
EXPECTED_NUM_SHARDS="${NUM_SHARDS:-}"
OUTPUT_FILE="${MERGED_OUTPUT_FILE:-results.json}"

echo "Merge started at $(date)"
echo "Input dir: ${INPUT_DIR}"
echo "Output file: ${OUTPUT_FILE}"

CMD=(
  "$PYTHON_BIN" experiments/paper/merge_analysis_results.py
  --input-dir "$INPUT_DIR"
  --output-file "$OUTPUT_FILE"
)

if [ -n "$EXPECTED_NUM_SHARDS" ]; then
  CMD+=(--expected-num-shards "$EXPECTED_NUM_SHARDS")
fi

"${CMD[@]}"

echo "Merge finished at $(date)"
