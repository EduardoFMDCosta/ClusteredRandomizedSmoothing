# Running This Project On TU Delft HPC With Apptainer/Singularity

This is a first-time-user guide.

## Minimal workflow overview

You will run the project on TU Delft HPC with Apptainer/Singularity.

Apptainer needs a container image file (`.sif`).

There are three practical ways to get that image:

1. Pull from a public Docker Hub image (easy to pull; no sign-in needed on server).
2. Build your own image locally and push it once.
3. Build automatically in GitHub Actions and publish to GHCR (no local Docker install).

This guide keeps the server side Apptainer-only.

## What "registry image name" means

Format:

`NAMESPACE/IMAGE:TAG`

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

## Step 0: Prerequisites on HPC

- your project folder cloned (for example `~/Projects/RandomizedSmoothing`)
- `apptainer` or `singularity` available

## Step 1: Configure container cache on HPC

This follows common TU Delft HPC practice:

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer/cache
mkdir -p "$APPTAINER_CACHEDIR"
```

Optional (recommended): add this export to `~/.bashrc`.

If `/scratch/$USER` is not accessible, use this robust fallback:

```bash
if mkdir -p /scratch/$USER/.apptainer/cache 2>/dev/null; then
  export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer/cache
elif [ -n "${TMPDIR:-}" ] && mkdir -p "$TMPDIR/.apptainer/cache" 2>/dev/null; then
  export APPTAINER_CACHEDIR="$TMPDIR/.apptainer/cache"
else
  export APPTAINER_CACHEDIR=$HOME/.apptainer/cache
  mkdir -p "$APPTAINER_CACHEDIR"
fi
echo "APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"
```

## Step 2: Build/publish image without local Docker (recommended)

Option A (best if you cannot install Docker locally): GitHub Actions + GHCR

This repository already includes workflow [build-container.yml](.github/workflows/build-container.yml).
It builds the image in GitHub and publishes to:

ghcr.io/YOUR_GITHUB_USERNAME/randomized-smoothing:latest

How to use it:

1. Push your current branch to GitHub.
2. In GitHub, open Actions and run "Build and Publish Container" (or merge to main).
3. Wait until workflow success.
4. In GitHub Packages, open randomized-smoothing package and set visibility to public.

Then, on HPC, pull with Apptainer:

```bash
cd $HOME/Projects/RandomizedSmoothing
apptainer pull randomized-smoothing.sif docker://ghcr.io/YOUR_GITHUB_USERNAME/randomized-smoothing:latest
```

If your GHCR package is private, log in first:

```bash
apptainer registry login ghcr.io
```

Private repository + public container is supported on GitHub.
In GitHub Packages, set package visibility for `randomized-smoothing` to public.
That keeps code private while allowing public image pulls.

Option B (recommended if you already have public Docker Hub image): pull from Docker Hub

If your image is public, HPC can pull it without signing in:

```bash
cd $HOME/Projects/RandomizedSmoothing
apptainer pull randomized-smoothing.sif docker://YOUR_DOCKERHUB_USERNAME/randomized-smoothing:latest
```

Important: do not type angle brackets (`<` `>`). They are shell redirection symbols.
Use a real value, for example:

```bash
apptainer pull randomized-smoothing.sif docker://alice/randomized-smoothing:latest
```

Option C: build locally and push to Docker Hub

Run these on your local machine (not on HPC):

```bash
docker build -t randomized-smoothing:latest .
docker login
docker tag randomized-smoothing:latest YOUR_DOCKERHUB_USERNAME/randomized-smoothing:latest
docker push YOUR_DOCKERHUB_USERNAME/randomized-smoothing:latest
```

Then on HPC pull it with Apptainer using the same `apptainer pull` command.

## Step 3: Pull image on HPC

Use whichever runtime exists on your cluster:

```bash
if command -v apptainer >/dev/null 2>&1; then
  RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME=singularity
else
  echo "No apptainer/singularity in PATH"; exit 1
fi

cd $HOME/Projects/RandomizedSmoothing
$RUNTIME pull randomized-smoothing.sif docker://ghcr.io/YOUR_GITHUB_USERNAME/randomized-smoothing:latest
```

## Step 4: Quick test before submitting jobs

```bash
cd $HOME/Projects/RandomizedSmoothing
export PROJECT_DIR=$PWD
export APPTAINER_IMAGE=$PROJECT_DIR/randomized-smoothing.sif

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME=apptainer
else
  RUNTIME=singularity
fi

$RUNTIME exec --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$PROJECT_DIR" "$APPTAINER_IMAGE" \
  python -c "import torch; print('torch', torch.__version__)"
```

If this prints a torch version, the image is ready.

## Step 5: Submit array analysis job (Slurm)

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

If your cluster does not use Slurm (`sbatch` not found), run shard jobs manually:

```bash
cd $HOME/Projects/RandomizedSmoothing
export PROJECT_DIR=$PWD
export APPTAINER_IMAGE=$PROJECT_DIR/randomized-smoothing.sif
export NUM_SHARDS=32

for SHARD_INDEX in $(seq 0 31); do
  SHARD_INDEX=$SHARD_INDEX bash experiments/schedulers/analysis_trajflow_shard.sh
done
```

## Step 6: Merge shard outputs

When all array tasks are done:

```bash
cd $HOME/Projects/RandomizedSmoothing
export PROJECT_DIR=$PWD
export APPTAINER_IMAGE=$PROJECT_DIR/randomized-smoothing.sif
export OUTPUT_DIR=experiments/data/analysis
export NUM_SHARDS=32

sbatch experiments/schedulers/scheduler_merge_trajflow_results.sh
```

Manual merge alternative:

```bash
cd $HOME/Projects/RandomizedSmoothing
export PROJECT_DIR=$PWD
export APPTAINER_IMAGE=$PROJECT_DIR/randomized-smoothing.sif
bash experiments/schedulers/merge_analysis_trajflow_shards.sh
```

## Where files appear

- shard outputs: `experiments/data/analysis/results_shard_*_of_*.json`
- merged output: `experiments/data/analysis/results.json`
- logs: `logging/`

## Common issues and fixes

- image pull fails due quota: ensure cache path has enough space.
- pull fails for private image: make image public, or run `apptainer registry login ghcr.io`.
- command not found `apptainer`: try `singularity`.

## Optional: build `.sif` directly from definition file

If your HPC account supports it, you can build from an Apptainer definition (`.def`) instead of Docker Hub. That is not required for this project path, but I can add it if you want.
