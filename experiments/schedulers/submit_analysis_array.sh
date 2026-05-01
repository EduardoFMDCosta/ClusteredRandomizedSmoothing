#!/usr/bin/env bash
#PBS -N trajflow_analysis
#PBS -l nodes=1:ppn=8
#PBS -o logging/analysis_trajflow_$PBS_JOBID.out
#PBS -e logging/analysis_trajflow_$PBS_JOBID.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:-$(pwd)}}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_DIR/randomized-smoothing.sif}"
NUM_SHARDS="${NUM_SHARDS:-32}"

cd "$PROJECT_DIR"
mkdir -p logging

export PROJECT_DIR
export APPTAINER_IMAGE
export NUM_SHARDS
export SHARD_INDEX="${SHARD_INDEX:-${PBS_ARRAYID:-}}"

if [ -z "${SHARD_INDEX}" ]; then
  echo "SHARD_INDEX is required (set SHARD_INDEX or submit with qsub -t start-end)."
  exit 1
fi

apptainer exec "$APPTAINER_IMAGE" experiments/schedulers/run_analysis_shard.sh
