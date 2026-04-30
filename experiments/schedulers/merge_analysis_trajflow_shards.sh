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
INPUT_DIR="${OUTPUT_DIR:-experiments/data/analysis}"
EXPECTED_NUM_SHARDS="${NUM_SHARDS:-}"
OUTPUT_FILE="${MERGED_OUTPUT_FILE:-results.json}"

echo "Merge started at $(date)"
echo "Input dir: ${INPUT_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Apptainer image: ${APPTAINER_IMAGE}"

CMD=(
  "${PYTHON_CMD[@]}" experiments/paper/merge_analysis_results.py
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
