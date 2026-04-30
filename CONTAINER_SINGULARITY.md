# Running This Project With Apptainer (Singularity)

This is a first-time-user guide.

## Minimal workflow overview

You will run the project with Apptainer on DelftBlue.

Apptainer needs a container image file (`.sif`).

There are two practical ways to get that image:

1. Pull from a public Docker Hub image (easy to pull; no sign-in needed on server).
2. Build your own image locally and push it once.

This guide keeps the server side Apptainer-only.

## What "registry image name" means

Format:

`<namespace>/<image>:<tag>`

Example:

`alice/randomized-smoothing:latest`

This is the name used in:

`apptainer pull ... docker://alice/randomized-smoothing:latest`

## Files already prepared in this repo

- `Dockerfile`
- `experiments/schedulers/analysis_trajflow_shard.sh` (container-only)
- `experiments/schedulers/merge_analysis_trajflow_shards.sh` (container-only)
- `experiments/schedulers/scheduler_analysis_trajflow_array.sh`
- `experiments/schedulers/scheduler_merge_trajflow_results.sh`

The scheduler scripts now always run with Apptainer and require `APPTAINER_IMAGE`.

## Step 0: Prerequisites on DelftBlue

- your project folder cloned (for example `~/Projects/RandomizedSmoothing`)
- `apptainer` available (DelftBlue docs say no module needed)

## Step 1: Configure Apptainer cache on DelftBlue

This follows DelftBlue recommendations:

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer/cache
mkdir -p "$APPTAINER_CACHEDIR"
```

Optional (recommended): add this export to `~/.bashrc`.

## Step 2: Get image as `.sif` on DelftBlue

Option A (recommended): pull from a public Docker Hub image

If your image is public, DelftBlue can pull it without signing in:

```bash
cd $HOME/Projects/RandomizedSmoothing
apptainer pull randomized-smoothing.sif docker://<namespace>/randomized-smoothing:latest
```

Option B: if you need to create your own image first

Run these on your local machine (not on DelftBlue):

```bash
docker build -t randomized-smoothing:latest .
docker login
docker tag randomized-smoothing:latest <your_dockerhub_username>/randomized-smoothing:latest
docker push <your_dockerhub_username>/randomized-smoothing:latest
```

Then on DelftBlue pull it with Apptainer using the same `apptainer pull` command.

## Step 3: Quick test before submitting jobs

```bash
cd $HOME/Projects/RandomizedSmoothing
export PROJECT_DIR=$PWD
export APPTAINER_IMAGE=$PROJECT_DIR/randomized-smoothing.sif

apptainer exec --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$PROJECT_DIR" "$APPTAINER_IMAGE" \
  python -c "import torch; print('torch', torch.__version__)"
```

If this prints a torch version, the image is ready.

## Step 4: Submit array analysis job

```bash
cd $HOME/Projects/RandomizedSmoothing
chmod +x experiments/schedulers/*.sh

export PROJECT_DIR=$PWD
export APPTAINER_IMAGE=$PROJECT_DIR/randomized-smoothing.sif
export NUM_SHARDS=32

sbatch --array=0-31 experiments/schedulers/scheduler_analysis_trajflow_array.sh
```

Check status:

```bash
squeue -u $USER
```

## Step 5: Merge shard outputs

When all array tasks are done:

```bash
cd $HOME/Projects/RandomizedSmoothing
export PROJECT_DIR=$PWD
export APPTAINER_IMAGE=$PROJECT_DIR/randomized-smoothing.sif
export OUTPUT_DIR=experiments/data/analysis
export NUM_SHARDS=32

sbatch experiments/schedulers/scheduler_merge_trajflow_results.sh
```

## Where files appear

- shard outputs: `experiments/data/analysis/results_shard_*_of_*.json`
- merged output: `experiments/data/analysis/results.json`
- logs: `logging/`

## Common issues and fixes

- `apptainer pull` fails due quota: ensure `APPTAINER_CACHEDIR` is in `/scratch`.
- pull fails for private image: make image public, or run `apptainer registry login docker.io`.
- command not found `apptainer`: verify DelftBlue environment/session.

## Optional: build `.sif` directly from definition file

If your DelftBlue account supports it, you can build from an Apptainer definition (`.def`) instead of Docker Hub. That is not required for this project path, but I can add it if you want.
