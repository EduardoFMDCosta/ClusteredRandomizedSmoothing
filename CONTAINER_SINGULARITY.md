# Running This Project On TU Delft HPC With Apptainer/Singularity

This guide assumes you build the container image directly on the HPC server from
`Apptainer.def` and submit jobs with `qsub` (PBS/Torque).

## Files used in this workflow

- `Apptainer.def`
- `experiments/schedulers/analysis_trajflow_shard.sh`
- `experiments/schedulers/merge_analysis_trajflow_shards.sh`
- `experiments/schedulers/scheduler_analysis_trajflow_array.sh`
- `experiments/schedulers/scheduler_merge_trajflow_results.sh`

## Step 1: Build image on HPC from Apptainer.def

```bash
cd "$HOME/Projects/RandomizedSmoothing"
apptainer build randomized-smoothing.sif Apptainer.def
```

If your cluster exposes `singularity` instead of `apptainer`, use:

```bash
singularity build randomized-smoothing.sif Apptainer.def
```

## Step 2: Quick runtime check

```bash
cd "$HOME/Projects/RandomizedSmoothing"
apptainer exec randomized-smoothing.sif python -c "import torch; print('torch', torch.__version__)"
```

## Step 3: Submit shard analysis jobs with qsub array

Make scripts executable once:

```bash
cd "$HOME/Projects/RandomizedSmoothing"
chmod +x experiments/schedulers/*.sh
```

Submit 32 shards:

```bash
cd "$HOME/Projects/RandomizedSmoothing"
qsub -t 0-31 -v PROJECT_DIR="$PWD",APPTAINER_IMAGE="$PWD/randomized-smoothing.sif",NUM_SHARDS=32 experiments/schedulers/scheduler_analysis_trajflow_array.sh
```

The scheduler wrapper launches each shard with:

```bash
apptainer exec randomized-smoothing.sif experiments/schedulers/analysis_trajflow_shard.sh
```

## Step 4: Merge shard outputs

After shard jobs finish:

```bash
cd "$HOME/Projects/RandomizedSmoothing"
qsub -v PROJECT_DIR="$PWD",APPTAINER_IMAGE="$PWD/randomized-smoothing.sif",NUM_SHARDS=32,OUTPUT_DIR=experiments/data/analysis experiments/schedulers/scheduler_merge_trajflow_results.sh
```

## Where outputs are written

- shard files: `experiments/data/analysis/results_shard_*_of_*.json`
- merged file: `experiments/data/analysis/results.json`
- scheduler logs: `logging/`

## Useful queue commands

- submit: `qsub <job_script>`
- status: `qstat -a`
- delete job: `qdel <jobid>`

## Notes

- `qsub -t` sets `PBS_ARRAYID`; the shard script reads this automatically.
- You can override runtime variables with `qsub -v`, for example:
  `BATCH_SIZE`, `SAMPLE_BATCH_SIZE`, `CONFIDENCE_NUM_SAMPLES`.
