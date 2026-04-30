#!/usr/bin/env bash
#SBATCH --job-name=trajflow_merge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logging/merge_trajflow_%j.out

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Projects/RandomizedSmoothing}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_DIR/randomized-smoothing.sif}"

cd "$PROJECT_DIR"
mkdir -p logging

export PROJECT_DIR
export APPTAINER_IMAGE

bash experiments/schedulers/merge_analysis_trajflow_shards.sh
