#!/bin/bash
#PBS -N trajflow_analysis
#PBS -l nodes=1:ppn=8,mem=32gb
#PBS -j oe
#PBS -o logging/${EXP_SCRIPT%.sh}_${PBS_ARRAYID}.pbs.txt

set -euo pipefail

PROJECT_DIR="$HOME/Projects/RandomizedSmoothing"

cd "$PROJECT_DIR"
mkdir -p logging

echo "Job started at $(date)"
echo "Experiment script: ${EXP_SCRIPT}"
echo "Array index (used as shard): ${PBS_ARRAYID}"

auto_script="$PROJECT_DIR/experiments/schedulers/${EXP_SCRIPT}"
if [ ! -f "$auto_script" ]; then
    echo "Missing scheduler script: $auto_script"
    exit 1
fi

bash "$auto_script"

echo "Job finished at $(date)"
