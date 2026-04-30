#!/usr/bin/env bash
#SBATCH --job-name=trajflow_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-31
#SBATCH --output=logging/analysis_trajflow_%A_%a.out

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_DIR/randomized-smoothing.sif}"
NUM_SHARDS="${NUM_SHARDS:-32}"

cd "$PROJECT_DIR"
mkdir -p logging

export PROJECT_DIR
export APPTAINER_IMAGE
export NUM_SHARDS
export SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"

bash experiments/schedulers/analysis_trajflow_shard.sh
