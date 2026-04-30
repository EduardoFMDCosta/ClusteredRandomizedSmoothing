# RandomizedSmoothing

## Running On TU Delft HPC

Build the container image directly on the server from `Apptainer.def`, then
submit jobs with `qsub` (PBS/Torque).

### Scheduler scripts

| Script | Purpose |
|--------|---------|
| `experiments/schedulers/submit_analysis_array.sh` | Submit PBS array job (one task per shard) |
| `experiments/schedulers/submit_merge.sh` | Submit PBS merge job after all shards finish |
| `experiments/schedulers/run_analysis_shard.sh` | Worker: runs one analysis shard inside container |
| `experiments/schedulers/run_merge_shards.sh` | Worker: merges shard outputs inside container |

### Step 1 — Build image on HPC

```bash
cd "$HOME/Projects/RandomizedSmoothing"
apptainer build randomized-smoothing.sif Apptainer.def
```

### Step 2 — Quick sanity check

```bash
apptainer exec randomized-smoothing.sif python -c "import torch; print(torch.__version__)"
```

### Step 3 — Submit analysis array

```bash
cd "$HOME/Projects/RandomizedSmoothing"
chmod +x experiments/schedulers/*.sh
qsub -t 0-31 \
  -v PROJECT_DIR="$PWD",APPTAINER_IMAGE="$PWD/randomized-smoothing.sif",NUM_SHARDS=32 \
  experiments/schedulers/submit_analysis_array.sh
```

Check job status: `qstat -a`

### Step 4 — Merge results

```bash
qsub \
  -v PROJECT_DIR="$PWD",APPTAINER_IMAGE="$PWD/randomized-smoothing.sif",NUM_SHARDS=32,OUTPUT_DIR=experiments/data/analysis \
  experiments/schedulers/submit_merge.sh
```

### Output locations

| Path | Content |
|------|---------|
| `experiments/data/analysis/results_shard_*_of_*.json` | Per-shard results |
| `experiments/data/analysis/results.json` | Merged results |
| `logging/` | PBS stdout/stderr logs |

### Overriding parameters

Pass extra variables via `qsub -v`, e.g. `BATCH_SIZE=64`, `SAMPLE_BATCH_SIZE=32`,
`CONFIDENCE_NUM_SAMPLES=10000`. See `run_analysis_shard.sh` for all options.