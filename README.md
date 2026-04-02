# RandomizedSmoothing

Scientific Python prototype for stochastic-function smoothing with robust
alpha-trimming strategies.

## Project Structure

```text
RandomizedSmoothing/
├─ main.py
├─ bounds.py
├─ plotting.py
├─ pyproject.toml
└─ src/
	 └─ randomized_smoothing/
			├─ __init__.py
			├─ stochastic.py
			├─ clusters.py
			├─ kmeans.py
			└─ smoothing.py
```

## Core Concepts

- `StochasticFunction`:
	interface for stochastic mappings. The `stochastic.py` module also contains
	concrete example implementations for experiments and tests.
- `ThreeMode1DGaussianMixture`:
	ready-to-use example stochastic function for experiments and tests.
- `AlphaTrimmingSmoothing`:
	robust smoother that returns alpha-trimmed averages `(m, q)`.
- `Cluster`:
	holds per-input cluster `centroids (m, k, q)` and centroid weights `(m, k)`.
- `ClusterParameters`:
	configuration object that fixes the number of k-means clusters and solver tolerances.
- `ClusterizedAlphaTrimmingSmoothing`:
	trims inside nearest-centroid clusters, computes cluster means, then samples
	from weighted cluster means.
- `ClopperPearsonLowerBound` and `lower_bound_clopper_pearson(...)`:
	utilities for the confidence bounds used in certification.

## Demonstration

`main.py` includes a full experiment:

1. Defines a 1D three-mode Gaussian-mixture stochastic function.
2. Observes many samples for each `x` and fits one 3-centroid `Cluster` per input.
3. Runs both smoothing methods for a given alpha and prints results.
4. Rebuilds the clusterized smoother from fresh stochastic samples to visualize
	its resulting distribution.
5. Repeats full smoothing pipelines many times (Monte Carlo):
	sample from stochastic function, compute trimming, sample from trimmed smoother.
	For clusterized Monte Carlo, clusters are fitted again on each fresh sample batch.

## Run

```bash
python main.py
```

The script also saves charts in `outputs/`:

- `distribution_comparison.png`

## Install As Project

From the repository root:

```bash
pip install -e .
```