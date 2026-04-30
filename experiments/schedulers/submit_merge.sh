#!/usr/bin/env bash
#PBS -N trajflow_merge
#PBS -l nodes=1:ppn=2,walltime=00:30:00
#PBS -o logging/merge_trajflow_$PBS_JOBID.out
#PBS -e logging/merge_trajflow_$PBS_JOBID.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${PBS_O_WORKDIR:-$(pwd)}}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_DIR/randomized-smoothing.sif}"

cd "$PROJECT_DIR"
mkdir -p logging

export PROJECT_DIR
export APPTAINER_IMAGE

apptainer exec "$APPTAINER_IMAGE" experiments/schedulers/run_merge_shards.sh
