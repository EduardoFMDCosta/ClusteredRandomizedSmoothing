#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
INPUT_DIR="${OUTPUT_DIR:-experiments/data/analysis}"
EXPECTED_NUM_SHARDS="${NUM_SHARDS:-}"
OUTPUT_FILE="${MERGED_OUTPUT_FILE:-results.json}"

echo "Merge started at $(date)"
echo "Input dir: ${INPUT_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Python command: python"

CMD=(
  python experiments/paper/merge_analysis_results.py
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
