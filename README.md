# Clustered Randomized Smoothing for Stochastic Prediction Functions

Official implementation of the NeurIPS 2026 paper **"Clustered Randomized Smoothing for Stochastic Prediction Functions"**.

**Eduardo Figueiredo\*¹, Frederik Mathiesen\*², Julian Schumann¹, Jens Kober³, Arkady Zgonnikov¹, Luca Laurenti¹˒⁴**

¹ Delft University of Technology · ² University of Oxford · ³ University of Stuttgart · ⁴ AI4I · \* equal contribution

## Overview

Randomized smoothing certifies the robustness of a predictor by averaging its outputs over noisy inputs. For *stochastic* predictors (normalizing flows, VAEs, mixture-density policies, ...) whose outputs are multi-modal, this average collapses the modes into one uninformative value. **Clustered α-smoothing** instead

1. clusters the noisy output samples (any clustering algorithm),
2. applies α-trimmed smoothing *within* each cluster, and
3. recombines the clusters as a mixture, weighted by cluster size.

It comes with a high-confidence lower bound on the probability that the smoothed prediction stays inside a union of compact regions (one per mode) for every input perturbation within an ℓ₂ radius `r`. The ordinary α-smoothing certificate of Rekavandi et al. (NeurIPS 2024) is recovered with a single cluster, and RS-Reg (Rekavandi et al., TMLR 2025) with a single cluster and `α = 0`. Both are used as baselines.

This repository contains

- the **library** (`src/randomized_smoothing/`): the clustered α-smoother, coverage-region construction, Clopper–Pearson confidence bounds, and the certificate of Theorem 3.1 / Proposition 3.2;
- the **experiments** of the paper: stochastic trajectory prediction with TrajFlow on L-GAP, and a multi-modal RL quadrotor controller (UAV3D) with and without an adversarial attack;
- **pre-trained models and cached results**, so every table and figure can be regenerated in seconds to minutes without re-running the expensive Monte-Carlo experiments;
- a pinned environment (`uv.lock`, `Dockerfile`) and unit tests.

---

## Installation

**Requirements.** Python ≥ 3.10 (the Docker image uses 3.12; development used 3.13). No GPU is needed: all experiments in the paper were run on a laptop CPU (Intel Core i7-1365U, 16 GB RAM), and scripts use CUDA automatically when it is available.

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/EduardoFMDCosta/ClusteredRandomizedSmoothing.git
cd ClusteredRandomizedSmoothing

# install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync --extra plots        # creates .venv/ from uv.lock; --extra plots adds matplotlib + pyvista
uv run pytest                # 16 unit tests, a few seconds
```

`--extra plots` is only needed to regenerate figures; the library itself does not use it. Prefix commands with `uv run`, or activate `.venv`. On Linux, PyPI's PyTorch wheel pulls in the CUDA libraries, so the first install is a large download even though no GPU is required.

The scripts under `experiments/rl_benchmarks/` import helpers from `experiments/`, so put that folder on the Python path once per shell:

```bash
export PYTHONPATH="$PWD/experiments"          # PowerShell: $env:PYTHONPATH = "$PWD\experiments"
```

**Extra system requirements for figures**

- **LaTeX.** The figure scripts under `experiments/paper/` share `plotting.py`, which sets `text.usetex`, so a LaTeX installation with `cm-super` and `dvipng` is required (TeX Live on Linux, MiKTeX on Windows).
- **OpenGL.** The 3-D quadrotor renders use PyVista. On a headless Linux machine install OSMesa (`apt install libosmesa6 libegl1 libgl1-mesa-dri`) and set `PYVISTA_OFF_SCREEN=true`, or prefix the command with `xvfb-run -a`.

### Docker

The `Dockerfile` pins Python, uv and all system libraries (LaTeX, OSMesa/EGL, Xvfb), and sets `PYTHONPATH` and `PYVISTA_OFF_SCREEN` for you.

```bash
docker build -t clustered-rs .
docker run --rm -it \
    -v "$PWD/experiments/data:/app/experiments/data" \
    -v "$PWD/experiments/uav_3d:/app/experiments/uav_3d" \
    clustered-rs
```

The mounts supply the cached TrajFlow results (excluded from the image by `.dockerignore`) and bring generated files back to the host. Inside the container, run the commands below without `uv run`.

---

## Using the library

The only thing the library needs from your model is a `StochasticFunction`: a callable that maps an input batch of shape `(..., d)` to one random output sample of shape `(..., q)` per input.

```python
import torch
from randomized_smoothing import (
    ClusteringParameters, ConfidenceParameters, GaussianAlphaTrimmingSmoothing,
    Parameters, SmoothingParameters, StochasticFunction,
)

class BimodalPredictor(StochasticFunction):
    """Toy stochastic predictor: outputs x - 3 with probability 0.4 and x + 3 otherwise."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sign = torch.where(torch.rand(x.shape[:-1]) < 0.4, -1.0, 1.0).unsqueeze(-1)
        return x[..., :1] + 3.0 * sign + 0.1 * torch.randn(*x.shape[:-1], 1)

x = torch.zeros(1, 1)                                   # a batch of inputs, shape (batch, d)
parameters = Parameters(
    smoothing=SmoothingParameters(N=30, alpha=0.2),     # N noisy samples per prediction, trimming level alpha
    clustering=ClusteringParameters(max_num_clusters=3, coverage=0.9, strategy="dbscan",
                                    dbscan_eps=0.5, dbscan_min_samples=50),
    confidence=ConfidenceParameters(num_samples=4000, beta=1e-2),   # samples / confidence for the certificate
)
smoother = GaussianAlphaTrimmingSmoothing(
    stochastic_function=BimodalPredictor(), x=x, sigma=0.05, parameters=parameters,
)

smoother.smooth().sample(size=())              # one draw from the clustered alpha-smoothed prediction
smoother.monte_carlo(num_simulations=1000)     # many draws, shape (batch, 1000, q)
smoother.regions                               # coverage regions R_m (axis-aligned hyperrectangles)

for radius in (0.0, 0.005, 0.01):              # certify all perturbations with ||delta||_2 <= radius
    cert = smoother.robust_certification(radius=torch.tensor([[radius]]))
    print(radius, cert.probs, cert.total_probs)
```

`cert.probs` holds the per-region lower bounds (Eq. 6 of the paper) and `cert.total_probs` the bound for the union of all regions (Eq. 5). The certified probability decays with `radius / sigma`.

Setting `max_num_clusters=1` gives α-smoothing, and additionally `alpha=0` gives RS-Reg. Clustering strategies: `dbscan`, `hdbscan`, `endpoint_dbscan`, and the problem-specific `behavior_split` used for TrajFlow (see `ClusteringParameters`).

| Paper | Code |
|---|---|
| `N`, `α` | `SmoothingParameters.N`, `.alpha` (fraction trimmed from *each* tail, `0 ≤ α < 0.5`) |
| `σ` | `sigma` of `GaussianAlphaTrimmingSmoothing` |
| `p` (coverage), `M_max` | `ClusteringParameters.coverage`, `.max_num_clusters` |
| `N̄`, `β` | `ConfidenceParameters.num_samples`, `.beta` |
| Algorithm 1, coverage sets `R_m` and generalized Voronoi partition `V_m` | `clusters.Cluster`, `regions.AxisAlignedHyperRectangle` |
| Algorithm 2, certification | `GaussianAlphaTrimmingSmoothing.robust_certification` |
| Theorem 3.1 per-region bound; Proposition 3.2 union bound | `optimization.robustness_bound`; `optimization.compute_total_prob_lower_bound` |
| Proposition 4.1 (Clopper–Pearson) | `clopper_pearson.ClopperPearsonInterval` |

---

## Reproducing the paper

All commands are run from the repository root. Each experiment script ships with its **cached results** (`experiments/data/analysis/`, `experiments/uav_3d/analysis/`), so there are two ways to run it:

- **Fast path (seconds to minutes):** re-render the table or figure from the cached results (`--no-regenerate`, `--plot`, or the plotting-only script).
- **Full re-run (hours on a CPU):** recompute the Monte-Carlo data from the pre-trained models.

### Overview

| Paper item | What it shows | Script | Output |
|---|---|---|---|
| Fig. 2 | α- vs clustered α-smoothing on a 1-D mixture | `experiments/paper/compare_distributions.py` | `experiments/data/figures/smoothing_distributions_comparison.pdf` |
| Fig. 3 | Certificate lower bound vs `N`, `α`, `p_V`, `p_R/p_V` | `experiments/paper/bound_comparison.py` | `experiments/data/figures/bound_clustered_grid.pdf` |
| Fig. 4 | TrajFlow safety analysis (risky ego decisions) | `experiments/paper/trajflow_safety.py` | `experiments/data/figures/trajflow_safety_analysis.pdf` |
| Table 1 | TrajFlow 2-Wasserstein distance to `h_w(x+ε)` | `experiments/paper/trajflow_wasserstein.py` | `aggregate` in `experiments/data/analysis/wasserstein/trajflow_wasserstein.json` |
| Fig. 5 | TrajFlow certified lower bounds vs radius | `experiments/paper/trajflow_robustness.py` | `experiments/data/figures/trajflow_robustness_bounds.pdf` |
| Fig. 6, Table 2 | Quadrotor rollouts, action distributions, kinematics | `experiments/rl_benchmarks/collect_uav3d_rollouts.py`, `visualize_uav3d_robustified_pyvista.py`, `visualize_bimodal_robustified.py` | `experiments/uav_3d/bimodal_figures/`, `experiments/uav_3d/bimodal_plots/` |
| Fig. 8 | TrajFlow predictive distributions, one scene | `experiments/paper/trajflow_examples.py` | `experiments/data/figures/final_timestep_examples_2.pdf` |
| Fig. 9 | Quadrotor: Wasserstein distortion over the state space | `experiments/rl_benchmarks/uav3d_wasserstein.py` | `experiments/uav_3d/wasserstein_figures/uav3d_wasserstein.png` |
| Fig. 10 | Quadrotor under adversarial attack | `collect_uav3d_attacked_rollouts.py`, `visualize_uav3d_robustified_pyvista.py`, `visualize_bimodal_attacked.py` | `experiments/uav_3d/adversarial_figures/`, `experiments/uav_3d/bimodal_plots/*_attacked.png` |
| Fig. 11 | Outcome and return vs attack radius | `experiments/rl_benchmarks/uav3d_attack_sweep.py` | `experiments/uav_3d/adversarial_figures/uav3d_attack_sweep.png` |
| Fig. 12 | Generalized Voronoi partition | `experiments/paper/generalized_voronoi_illustration.py` | `experiments/data/figures/generalized_voronoi_illustration.pdf` |
| Fig. 13 | Quantization artifact of α-trimming | `experiments/paper/quantization_illustration.py` | `experiments/data/figures/quantization_illustration.pdf` |
| Tables 3–7 | Experiment parameters | defaults of the scripts (`experiments/paper/experiment_configs.py` for TrajFlow) | |

### Theory and illustrations

```bash
uv run python experiments/paper/compare_distributions.py            # Fig. 2
uv run python experiments/paper/bound_comparison.py                 # Fig. 3
uv run python experiments/paper/generalized_voronoi_illustration.py # Fig. 12
uv run python experiments/paper/quantization_illustration.py        # Fig. 13
```

### Experiment 1: trajectory prediction (TrajFlow on L-GAP)

The TrajFlow scripts prompt for anything not given on the command line (number of inputs, whether to regenerate), so pass the flags below to run non-interactively. `trajflow_safety.py` also asks for the decision threshold: press Enter for the default, or pass `--decision-threshold`. The evaluation inputs are the first `--num-inputs` entries of the fixed random order in `experiments/data/fixed_random_index_order.json`; `--selected-indices 1,5,9` overrides it.

```bash
# Fast path: re-plot from the cached Monte-Carlo results
uv run python experiments/paper/trajflow_safety.py      --no-regenerate   # Fig. 4
uv run python experiments/paper/trajflow_wasserstein.py --no-regenerate   # Table 1
uv run python experiments/paper/trajflow_robustness.py  --no-regenerate   # Fig. 5

# Full re-run from the pre-trained TrajFlow model
uv run python experiments/paper/trajflow_safety.py      --regenerate --num-inputs 200
uv run python experiments/paper/trajflow_wasserstein.py --regenerate --num-inputs 200
uv run python experiments/paper/trajflow_robustness.py  --regenerate --num-inputs 20

# Fig. 8: always runs the model. Writes final_timestep_examples_{0..4}.pdf; the paper shows the one ending in _2.
uv run python experiments/paper/trajflow_examples.py
```

- **Table 1** is not printed; read it from the `aggregate` field of the cached JSON. Keys containing `_traj_` are the full-trajectory distances, the others the last-step distances.
  ```bash
  uv run python -c "import json; print(json.dumps(json.load(open('experiments/data/analysis/wasserstein/trajflow_wasserstein.json'))['aggregate'], indent=1))"
  ```
  The script also writes a few diagnostic plots (histograms, per-timestep distances) that are not in the paper.
- The scripts take all model and smoothing parameters from `experiments/paper/experiment_configs.py`; edit the corresponding dataclass to change them.

### Experiment 2: multi-modal quadrotor control (UAV3D)

The pre-trained PPO policy `experiments/uav_3d/policy.pt` is the one used for all quadrotor results. The experiments start from the bimodal state `BIMODAL_START_STATE` in `experiments/rl_benchmarks/rollout_io.py`.

**Fig. 6 and Table 2** (clean rollouts)

```bash
# Prints the statistics of Table 2 and writes experiments/uav_3d/bimodal_rollouts.pkl.
# Compute-heavy (hours on a CPU); use e.g. --num-seeds 2 for a quick check.
uv run python experiments/rl_benchmarks/collect_uav3d_rollouts.py --num-trajectories 100 --num-seeds 100 --seed 42

# top row: 3-D renders (reads the .pkl above, does not re-simulate)
uv run python experiments/rl_benchmarks/visualize_uav3d_robustified_pyvista.py

# bottom row: action distributions and certified-probability insets
uv run python experiments/rl_benchmarks/visualize_bimodal_robustified.py
```

The `.pkl` files are Python pickles. Only load files you trust (the shipped ones are safe to use).

**Fig. 9** (Wasserstein distortion over the state space)

```bash
uv run python experiments/rl_benchmarks/uav3d_wasserstein.py --plot   # re-plot from the cached JSON
uv run python experiments/rl_benchmarks/uav3d_wasserstein.py          # recompute from scratch (slow)
```

Distances use the entropic (Sinkhorn) solver, which is the default and what the paper uses. `--ot-method emd` switches to exact optimal transport, which gives different values.

**Fig. 10 and Fig. 11** (adversarial attack: white-box PGD on the observation; see Appendix C.2)

```bash
# Fig. 10, top row (PGD at every environment step: the most expensive experiment)
uv run python experiments/rl_benchmarks/collect_uav3d_attacked_rollouts.py \
    --attack-radius 0.03 --num-trajectories 100 --num-seeds 100 --seed 42
uv run python experiments/rl_benchmarks/visualize_uav3d_robustified_pyvista.py \
    --data experiments/uav_3d/attacked_rollouts.pkl --out-dir experiments/uav_3d/adversarial_figures

# Fig. 10, bottom row: attacked action distributions (the certification boxes are the clean ones)
uv run python experiments/rl_benchmarks/visualize_bimodal_attacked.py

# Fig. 11: one shard per radius is written to experiments/uav_3d/analysis/adversarial/, so radii can run as independent jobs
uv run python experiments/rl_benchmarks/uav3d_attack_sweep.py --plot                  # re-plot from the cached shards
uv run python experiments/rl_benchmarks/uav3d_attack_sweep.py                         # all radii in one process
uv run python experiments/rl_benchmarks/uav3d_attack_sweep.py --radius-values 0.025   # a single shard, then --plot
```

The attack settings (`--pgd-steps`, `--pgd-restarts`, `--eot-samples`) default to the paper's values; lowering them gives a cheaper but weaker attack.

**Re-training the policy (optional).** The defaults of `train_rl.py` are the environment, network and PPO settings of the paper (Tables 4 to 6):

```bash
uv run python experiments/rl_benchmarks/train_rl.py          # long; overwrites policy.pt and training_metrics.json
```

A retrained policy is a different policy: the bimodal start state is tied to one checkpoint. Re-validate it with `experiments/rl_benchmarks/find_bimodal_state.py` before running the experiments above.

### Notes on reproducibility

- **Seeds.** Every script seeds PyTorch/NumPy (see the `--seed` options and `experiment_configs.py`). The cached results were produced with these seeds on CPU; results can differ in the last digits across platforms and PyTorch versions.
- **Cost.** The illustrations, cached re-plots and action-distribution figures take minutes at most on a laptop CPU. The Monte-Carlo re-runs (TrajFlow, quadrotor rollouts, attacked rollouts, attack sweep) take hours.
- **Dataset.** TrajFlow inputs are L-GAP scenario indices. The shipped `Inputs.npy` has 885 scenarios, 813 of them with a valid ground-truth future.
- **Not in the paper.** `trajflow_adversarial.py`, `analysis_trajflow.py` and `trajflow_clustering_benchmark.py` in `experiments/paper/`, and `uav3d_attack_mechanism.py` in `experiments/rl_benchmarks/`, are exploratory and not needed for any reported result.

---

## Repository layout

```
src/randomized_smoothing/        Library: smoother, clustering, regions, certificate, Clopper-Pearson bounds.
tests/                           Unit tests for the optimization and clustering primitives.
experiments/
├── paper/                       Scripts for the theory figures and the TrajFlow experiments
│                                (experiment_configs.py holds their parameters, plotting.py the figure code).
├── rl_benchmarks/               UAV3D environment, PPO training, rollouts, attacks, and quadrotor figures.
├── stochastic_functions.py      Wrappers exposing TrajFlow / UAV3D / mixtures as StochasticFunction.
├── control_action.py            Unicycle model: positions -> (acceleration, curvature) control inputs.
├── TrajFlowExperiment/          TrajFlow model code, pre-trained weights, and the L-GAP evaluation inputs.
├── uav_3d/                      Pre-trained PPO policy, training curves, cached rollouts, figures, analysis JSON.
└── data/                        Cached TrajFlow results (analysis/), figures (figures/), and the fixed index order.
pyproject.toml, uv.lock          Dependencies (resolved lock file).
Dockerfile                       Pinned environment, including LaTeX and headless OpenGL.
```

---

## Third-party code, data and models

- **TrajFlow** (Mészáros, Schumann, Alonso-Mora, Zgonnikov, Kober, IV 2024) is the stochastic predictor in `experiments/TrajFlowExperiment/`. The model was trained for this paper with the [STEP](https://arxiv.org/abs/2509.14801) framework (MIT license) on **nuScenes** (CC BY-NC-SA) and **L-GAP** (CC BY). The training and data-conversion code is in [General-Framework](https://github.com/DAI-Lab-HEReALD/General-Framework/blob/main/Framework/simulations_randomized_smoothing.py); this repository ships only the resulting weights and the L-GAP evaluation inputs (`Inputs.npy`, `Outputs_gt.npy`). The weights were trained partly on nuScenes, so please use them in accordance with its terms (non-commercial research).
- `experiments/TrajFlowExperiment/TrajFlow/rqs.py` follows the rational-quadratic spline flows of [bayesiains/nsf](https://github.com/bayesiains/nsf) (MIT), and the message-passing layer in `SocialEncodingModule.py` is adapted from the Geometric Deep Learning course practical cited in its docstring.
- The PyTorch Geometric version is pinned to 2.4: the pickled TrajFlow checkpoints reference its internals. `torch-scatter` is not required (a pure-PyTorch stand-in is used when it is missing).
- **UAV3D** is a Python re-implementation of the quadrotor benchmark of Badings et al. (JAIR 2023, [LAVA-LAB/DynAbs](https://github.com/LAVA-LAB/DynAbs)), extended with a reward and termination criteria (Appendix B.2). The mixture policy follows Ren et al. (2021).
- **α-smoothing** (Rekavandi et al., NeurIPS 2024) and **RS-Reg** (Rekavandi et al., TMLR 2025) are reproduced as the special cases described above, using this repository's own code.

---

## Citation

If you use this code or build on the method, please cite

```bibtex
@inproceedings{figueiredo2026clustered,
  title     = {Clustered Randomized Smoothing for Stochastic Prediction Functions},
  author    = {Figueiredo, Eduardo and Mathiesen, Frederik and Schumann, Julian and Kober, Jens and Zgonnikov, Arkady and Laurenti, Luca},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). Bundled third-party components and the data and models listed [above](#third-party-code-data-and-models) keep their own licenses.

## Acknowledgements

This research is partially supported by the NWO (grant OCENW.M.22.056) and by the EPSRC grant EP/Y028872/1, *Mathematical Foundations of Intelligence: An "Erlangen Programme" for AI*.
