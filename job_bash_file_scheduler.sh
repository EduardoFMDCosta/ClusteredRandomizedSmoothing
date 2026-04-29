#!/bin/bash
#PBS -N trajflow_analysis
#PBS -l nodes=1:ppn=8,mem=32gb
#PBS -j oe
#PBS -o logging/${EXP_SCRIPT%.sh}_${PBS_ARRAYID}.pbs.txt

set -Eeuo pipefail

trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

PROJECT_DIR="$HOME/Projects/RandomizedSmoothing"

cd "$PROJECT_DIR"
mkdir -p logging

ARRAY_ID="${PBS_ARRAYID:-noarray}"
RUNTIME_LOG="logging/${EXP_SCRIPT%.sh}_${ARRAY_ID}_runtime.log"
exec > >(tee -a "$RUNTIME_LOG") 2>&1

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Experiment script: ${EXP_SCRIPT}"
echo "Array index (used as shard): ${PBS_ARRAYID}"
echo "Runtime log: ${RUNTIME_LOG}"

auto_script="$PROJECT_DIR/experiments/schedulers/${EXP_SCRIPT}"
if [ ! -f "$auto_script" ]; then
    echo "Missing scheduler script: $auto_script"
    exit 1
fi

bash "$auto_script"

echo "Job finished at $(date)"
