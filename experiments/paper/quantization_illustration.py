"""Illustrate the alpha-trimmed mean's quantization artifact on a bimodal
output.

When the clusterized smoothed predictor is collapsed to ``max_clusters=1`` on
a bimodal output, every Monte Carlo draw becomes a near-deterministic
function of how many of the ``N`` noisy samples landed in each mode.  The
smoothed predictor therefore concentrates on a small, equally spaced set of
"quantized" output values rather than a continuous distribution.

This appendix figure visualises the artifact on a 1-D 2-component Gaussian
mixture (40/60 weights, well-separated means).  The bottom panel shows the
empirical histogram of alpha-trimmed means over many simulations, overlaid
with the theoretical quantization levels and their binomial weights.
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SRC = EXPERIMENTS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plotting import TEXTWIDTH  # noqa: E402  -- applies shared rcParams

from randomized_smoothing.dataclasses import (  # noqa: E402
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing  # noqa: E402
from randomized_smoothing.stochastic import StochasticFunction  # noqa: E402


class BimodalGMMStochasticFunction(StochasticFunction):
    """Output one i.i.d. sample from a 1-D bimodal Gaussian mixture, ignoring
    the input.  Used to feed the standard smoothing pipeline a synthetic
    multi-modal predictor whose distribution is independent of the smoothed
    state."""

    def __init__(self, mu_a: float, mu_b: float, sigma: float, p_a: float):
        self.mu_a = float(mu_a)
        self.mu_b = float(mu_b)
        self.sigma = float(sigma)
        self.p_a = float(p_a)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = x.shape[:-1] + (1,)
        mode_a = torch.rand(target_shape, device=x.device, dtype=x.dtype) < self.p_a
        means = torch.where(mode_a, self.mu_a, self.mu_b)
        noise = self.sigma * torch.randn(target_shape, device=x.device, dtype=x.dtype)
        return means + noise


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _build_smoother(
    mu_a: float, mu_b: float, sigma: float, p_a: float,
    n_samples: int, alpha: float,
) -> GaussianAlphaTrimmingSmoothing:
    stochastic_fn = BimodalGMMStochasticFunction(
        mu_a=mu_a, mu_b=mu_b, sigma=sigma, p_a=p_a,
    )
    parameters = Parameters(
        smoothing=SmoothingParameters(N=n_samples, alpha=alpha),
        clustering=ClusteringParameters(max_num_clusters=1, coverage=0.9),
        confidence=ConfidenceParameters(num_samples=1000, beta=0.1),
    )
    x = torch.zeros((1, 1), dtype=torch.float64)
    return GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=0.1,  # input perturbation; ignored since stochastic_fn is input-independent
        parameters=parameters,
    )


def _expected_order_stats(
    n: int, num_mc: int, rng: np.random.Generator
) -> np.ndarray:
    """E[X_(j)] for j = 1..n where X_i ~ N(0, 1), via Monte Carlo."""
    samples = rng.standard_normal(size=(num_mc, n))
    samples.sort(axis=1)
    return samples.mean(axis=0)


def _quantization_levels(
    mu_a: float, mu_b: float, sigma: float, n: int, alpha: float, p_a: float,
    seed: int = 0, num_mc: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected alpha-trimmed mean per binomial-split bin and its probability.

    With ``k`` of the ``n`` noisy samples drawn from mode A (sorted to the
    lower end of the merged sample), the trimmed core ``[left, right)``
    contains the **top** ``c_a = clip(k - left, 0, core)`` order statistics of
    the ``k`` mode-A samples and the **bottom** ``c_b = core - c_a`` order
    statistics of the ``n - k`` mode-B samples.  The expected trimmed mean is
    therefore not simply ``(c_a mu_a + c_b mu_b) / core``: each contribution
    carries a Gaussian order-statistic bias of magnitude ``sigma * E[X_(j:m)]``
    that pulls both ends *toward* the centre.

    Bins where ``c_a`` saturates (``k <= left`` or ``k >= right``) absorb
    multiple ``k`` values whose level positions differ slightly; we report the
    binomial-weighted average for those bins.
    """
    left = int(np.floor(alpha * n))
    right = n - left
    core = right - left

    pmf = np.array([comb(n, k) * p_a**k * (1 - p_a) ** (n - k) for k in range(n + 1)])
    os_rng = np.random.default_rng(seed)
    os_cache: dict[int, np.ndarray] = {}

    def order_stats(m: int) -> np.ndarray:
        if m not in os_cache:
            os_cache[m] = _expected_order_stats(m, num_mc=num_mc, rng=os_rng)
        return os_cache[m]

    def expected_trimmed_mean(k: int) -> float:
        c_a = max(0, min(k, right) - left)
        c_b = max(0, right - max(left, k))
        total = 0.0
        if c_a > 0:
            os_a = order_stats(k)[left:min(k, right)]
            total += mu_a * c_a + sigma * float(os_a.sum())
        if c_b > 0:
            os_b = order_stats(n - k)[max(left, k) - k:right - k]
            total += mu_b * c_b + sigma * float(os_b.sum())
        return total / core

    levels = np.zeros(core + 1)
    weights = np.zeros(core + 1)

    p_low = float(pmf[: left + 1].sum())
    weights[0] = p_low
    if p_low > 0:
        levels[0] = sum(pmf[k] * expected_trimmed_mean(k) for k in range(left + 1)) / p_low

    for i in range(1, core):
        k = left + i
        weights[i] = float(pmf[k])
        levels[i] = expected_trimmed_mean(k)

    p_high = float(pmf[right:].sum())
    weights[-1] = p_high
    if p_high > 0:
        levels[-1] = sum(pmf[k] * expected_trimmed_mean(k) for k in range(right, n + 1)) / p_high

    return levels, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENTS / "data" / "figures"))
    parser.add_argument("--mu-a", type=float, default=-3.0)
    parser.add_argument("--mu-b", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--p-a", type=float, default=0.4,
                        help="Mixture weight on mode A (mode B gets 1 - p_a).")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Noisy samples N per smoothing call.")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    smoother = _build_smoother(
        mu_a=args.mu_a, mu_b=args.mu_b, sigma=args.sigma, p_a=args.p_a,
        n_samples=args.n_samples, alpha=args.alpha,
    )
    trimmed = (
        smoother.monte_carlo(num_simulations=args.simulations)
        .squeeze(0).squeeze(-1).detach().cpu().numpy()
    )
    levels, weights = _quantization_levels(
        args.mu_a, args.mu_b, args.sigma, args.n_samples, args.alpha, args.p_a,
        seed=args.seed,
    )

    pad = 4 * args.sigma
    grid = np.linspace(args.mu_a - pad, args.mu_b + pad, 800)
    gmm = (
        args.p_a * _gaussian_pdf(grid, args.mu_a, args.sigma)
        + (1 - args.p_a) * _gaussian_pdf(grid, args.mu_b, args.sigma)
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(TEXTWIDTH, 4.4), sharex=True,
        constrained_layout=True,
    )
    ax_top, ax_bot = axes

    ax_top.plot(grid, gmm, color="#0173B2", linewidth=1.6)
    ax_top.fill_between(grid, gmm, alpha=0.2, color="#0173B2")
    ax_top.axvline(args.mu_a, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_top.axvline(args.mu_b, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ymax_top = float(gmm.max())
    ax_top.text(args.mu_a, 0.92 * ymax_top, rf"$\pi_A = {args.p_a:.2f}$",
                ha="center", va="top", fontsize=9)
    ax_top.text(args.mu_b, 0.92 * ymax_top, rf"$\pi_B = {1 - args.p_a:.2f}$",
                ha="center", va="top", fontsize=9)
    ax_top.set_ylabel("density")
    ax_top.set_title("Underlying 2-component Gaussian mixture")

    bins = np.linspace(args.mu_a - pad, args.mu_b + pad, 250)
    counts_hist, edges_hist, _ = ax_bot.hist(
        trimmed, bins=bins, color="#0173B2", alpha=0.55, density=True,
        edgecolor="white", linewidth=0.3,
        label=rf"empirical ({args.simulations} simulations)",
    )
    centers_hist = 0.5 * (edges_hist[:-1] + edges_hist[1:])
    spacing = (args.mu_b - args.mu_a) / max(len(levels) - 1, 1)
    ymax_bot = ax_bot.get_ylim()[1]
    label_offset = 0.03 * ymax_bot
    for i, (level, weight) in enumerate(zip(levels, weights)):
        ax_bot.axvline(
            level, color="#D55E00", linestyle="--", linewidth=0.9, alpha=0.85,
            label="quantization level" if i == 0 else None,
        )
        local = counts_hist[np.abs(centers_hist - level) < 0.45 * spacing]
        peak_height = float(local.max()) if local.size else 0.0
        ax_bot.text(level, peak_height + label_offset, f"{weight:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#D55E00")

    left = int(np.floor(args.alpha * args.n_samples))
    core = args.n_samples - 2 * left
    ax_bot.set_ylabel("density")
    ax_bot.set_title(
        rf"Empirical $\alpha$-trimmed mean ($N = {args.n_samples}$, "
        rf"$\alpha = {args.alpha}$, core size = {core})"
    )
    ax_bot.legend(loc="upper left", fontsize=8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_pdf = out_dir / "quantization_illustration.pdf"
    fig.savefig(out_path_pdf)
    plt.close(fig)
    print(f"Saved {out_path_pdf}")


if __name__ == "__main__":
    main()
