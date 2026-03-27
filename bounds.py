import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product


def binomial_lower_tail(N: int, p: float, alpha: float) -> torch.Tensor:
    J = int(alpha * N)
    if J < 0:
        return torch.tensor(0.0, dtype=torch.float64)

    dtype = torch.float64
    j = torch.arange(J + 1, dtype=dtype)

    log_binom = (
        torch.lgamma(torch.tensor(N + 1, dtype=dtype))
        - torch.lgamma(j + 1)
        - torch.lgamma(torch.tensor(N, dtype=dtype) - j + 1)
    )
    log_terms = (
        log_binom
        + j * torch.log(torch.tensor(1 - p, dtype=dtype))
        + (N - j) * torch.log(torch.tensor(p, dtype=dtype))
    )
    return torch.exp(torch.logsumexp(log_terms, dim=0))


def double_binomial_sum(N: int, p: float, p_check: float, alpha: float) -> torch.Tensor:
    dtype = torch.float64

    log1mp    = torch.log(torch.tensor(1.0 - p,       dtype=dtype))
    logp      = torch.log(torch.tensor(p,             dtype=dtype))
    log1mpchk = torch.log(torch.tensor(1.0 - p_check, dtype=dtype))
    logpchk   = torch.log(torch.tensor(p_check,       dtype=dtype))

    log_N_fact = torch.lgamma(torch.tensor(N + 1, dtype=dtype))

    ell = torch.arange(1, N + 1, dtype=dtype)
    J   = (alpha * ell).long()

    log_outer = (
        log_N_fact
        - torch.lgamma(ell + 1)
        - torch.lgamma(torch.tensor(N, dtype=dtype) - ell + 1)
        + ell       * log1mpchk
        + (N - ell) * logpchk
    )

    j_grid = torch.arange(N, dtype=dtype).unsqueeze(0)
    ell_2d = ell.unsqueeze(1)
    J_2d   = J.unsqueeze(1).to(dtype)

    log_inner_binom = (
        torch.lgamma(ell_2d + 1)
        - torch.lgamma(j_grid + 1)
        - torch.lgamma(ell_2d - j_grid + 1)
    )
    log_inner_terms = (
        log_inner_binom
        + j_grid            * log1mp
        + (ell_2d - j_grid) * logp
    )
    valid = (j_grid <= J_2d) & (j_grid <= ell_2d)
    log_inner_terms = log_inner_terms.masked_fill(~valid, -torch.inf)

    log_inner_cdf = torch.logsumexp(log_inner_terms, dim=1)
    log_summands  = log_outer + log_inner_cdf
    return torch.exp(torch.logsumexp(log_summands, dim=0))


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# (p, p_check) pairs — always p < p_check
PP_PAIRS = [
    (0.3, 0.4),
    (0.5, 0.6),
    (0.7, 0.8),
]

ALPHAS  = [0.1, 0.2, 0.49]           # inner-sum fraction
N_VALUES = list(range(5, 101, 5))     # N from 5 to 100

# Color per alpha level (colorblind-friendly)
ALPHA_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# ---------------------------------------------------------------------------
# Compute results
# ---------------------------------------------------------------------------

def compute_grid(pp_pairs, alphas, n_values):
    """Returns dict: results[(p, p_check, alpha)] = (b1_list, b2_list)"""
    results = {}
    total = len(pp_pairs) * len(alphas) * len(n_values)
    done  = 0
    for (p, p_check), alpha in product(pp_pairs, alphas):
        b1_list, b2_list = [], []
        for N in n_values:
            b1 = binomial_lower_tail(N, p, alpha).item()
            b2 = double_binomial_sum(N, p, p_check, alpha).item()
            b1_list.append(b1)
            b2_list.append(b2)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} done...")
        results[(p, p_check, alpha)] = (b1_list, b2_list)
    return results

print("Computing bounds...")
results = compute_grid(PP_PAIRS, ALPHAS, N_VALUES)
print("Done.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

n_cols = len(PP_PAIRS)
n_rows = 1

fig = plt.figure(figsize=(6 * n_cols, 5))
gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.35)

for col, (p, p_check) in enumerate(PP_PAIRS):
    ax = fig.add_subplot(gs[0, col])

    for k, alpha in enumerate(ALPHAS):
        color = ALPHA_COLORS[k]
        b1_list, b2_list = results[(p, p_check, alpha)]

        ax.plot(
            N_VALUES, b1_list,
            color=color, linestyle="-", linewidth=1.8,
            label=f"B1  α={alpha}" if col == 0 else f"α={alpha}",
        )
        ax.plot(
            N_VALUES, b2_list,
            color=color, linestyle="--", linewidth=1.8,
            label=f"B2  α={alpha}" if col == 0 else "_nolegend_",
        )

    ax.set_title(f"p={p},  p̌={p_check}", fontsize=12)
    ax.set_xlabel("N", fontsize=11)
    if col == 0:
        ax.set_ylabel("Bound value", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=9)

# --- shared legend ---
# Build two-part legend: line style encodes bound, color encodes alpha
from matplotlib.lines import Line2D

style_handles = [
    Line2D([0], [0], color="gray", linestyle="-",  linewidth=2, label="Bound 1 (single sum)"),
    Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Bound 2 (double sum)"),
]
alpha_handles = [
    Line2D([0], [0], color=ALPHA_COLORS[k], linestyle="-", linewidth=2, label=f"α = {a}")
    for k, a in enumerate(ALPHAS)
]

fig.legend(
    handles=style_handles + alpha_handles,
    loc="lower center",
    ncol=len(style_handles) + len(alpha_handles),
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.08),
)

fig.suptitle("Binomial bounds comparison across N and α", fontsize=13, y=1.02)

#plt.savefig("/mnt/user-data/outputs/bounds_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: bounds_comparison.png")
plt.show()