# Record-length benchmark against sub-state rate separation.
#
# 300 K(3,3) correlation blind spots, built with topology.zero_correlation_partner
# (deviation rank 2, margin 0.9) from targets whose stationary probabilities are
# drawn with a spread widened in five steps of 60:
#     pi_k ~ exp(U(-s/2, s/2)),  s in SPREADS,   W ~ U(0.4, 2.0)  (as in generate_figures)
# For each: 5-bin noncentrality, 90%-power / 5-sigma benchmark (independent-pair
# asymptotic, pairs_needed_chi2), joint rank, and the rate separation
#     sep = sqrt( (max/min exit rate on E) * (max/min exit rate on F) ).
# The four Fig. 3 constructions are recomputed and overlaid.
#
# Writes data/n2_separation.csv, data/n2_fig3_constructions.csv and
# figures/fig3c_record_length_vs_separation.{pdf,png} (3.5 in column width).
#
# Run: python scripts/record_length_vs_separation.py
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stochastic_ctmc.topology import (bipartite_generator_from_coupling,
                                      zero_correlation_partner)
from stochastic_ctmc.dwell import joint_density_rank
from stochastic_ctmc.correlation import analytic_adjacent_moments
from stochastic_ctmc.independence import chi2_noncentrality, pairs_needed_chi2
from stochastic_ctmc.export import export_columns_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEV6 = [0, 0, 0, 1, 1, 1]
SPREADS = (0.5, 1.5, 2.5, 3.5, 4.5)
PER_STEP = 60
N_BINS = 5
BASE_SEED = 20261019
BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, np.inf))


def separation(Q):
    r = -np.diag(Q)
    return float(np.sqrt((r[:3].max() / r[:3].min()) * (r[3:].max() / r[3:].min())))


def sweep():
    rows = []
    for k, s in enumerate(SPREADS):
        for j in range(PER_STEP):
            rng = np.random.default_rng(BASE_SEED + 1000 * k + j)
            pi = np.exp(rng.uniform(-s / 2, s / 2, 6))
            pi /= pi.sum()
            W = rng.uniform(0.4, 2.0, (3, 3))
            target = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
            Q = zero_correlation_partner(target, LEV6, deviation_rank=2, seed=j, margin=0.9)
            rows.append((k, j, s, separation(Q),
                         chi2_noncentrality(Q, LEV6, n_bins=N_BINS),
                         pairs_needed_chi2(Q, LEV6, n_bins=N_BINS),
                         joint_density_rank(Q, LEV6),
                         analytic_adjacent_moments(Q, LEV6, 1).rho))
    return np.array(rows, dtype=float)


def paper_constructions():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from generate_figures import _k33_degenerate_triple
    out = []
    for seed, dev in ((19, 1), (7, 1), (7, 2), (11, 2)):
        _, _, Q = _k33_degenerate_triple(seed=seed, deviation_rank=dev, margin=0.9)
        out.append((seed, dev, separation(Q), pairs_needed_chi2(Q, LEV6, n_bins=N_BINS)))
    return np.array(out, dtype=float)


def main():
    R = sweep()
    P4 = paper_constructions()
    step, idx, spread, sep, lam, pairs, rank, rho = R.T
    assert np.all(rank == 3) and np.max(np.abs(rho)) < 1e-12

    export_columns_csv(os.path.join(ROOT, "data", "n2_separation.csv"),
                       {"spread_step": step, "index": idx, "pi_log_spread": spread,
                        "separation": sep, "noncentrality_5bin": lam,
                        "pairs_needed": pairs, "joint_rank": rank, "rho": rho})
    export_columns_csv(os.path.join(ROOT, "data", "n2_fig3_constructions.csv"),
                       {"target_seed": P4[:, 0], "deviation_rank": P4[:, 1],
                        "separation": P4[:, 2], "pairs_needed": P4[:, 3]})

    rs = stats.spearmanr(np.log(sep), np.log(pairs))[0]
    print(f"{len(R)} constructions, all joint rank 3, max |rho| {np.max(np.abs(rho)):.1e}")
    print(f"Spearman(log separation, log pairs) = {rs:+.3f}")
    print(f"{'band':>8} {'n':>4} {'median lambda':>14} {'median pairs':>13} {'min':>10} {'max':>10}")
    for lo, hi in BANDS:
        m = (sep >= lo) & (sep < hi)
        lab = f"{lo}-{hi}x" if np.isfinite(hi) else f">{lo}x"
        if m.any():
            print(f"{lab:>8} {m.sum():4d} {np.median(lam[m]):14.2e} {np.median(pairs[m]):13.2e} "
                  f"{pairs[m].min():10.2e} {pairs[m].max():10.2e}")
    for seed, dev, s, n in P4:
        print(f"Fig. 3 construction seed {int(seed)}, deviation rank {int(dev)}: "
              f"separation {s:.2f}x, {int(n):,} pairs")

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.loglog(sep, pairs, "o", ms=3, mfc="none", mec="0.55", mew=0.6,
              label="300 constructions")
    ax.loglog(P4[:, 2], P4[:, 3], "s", ms=5, mfc="black", mec="black",
              label="Fig. 3(a) constructions")
    ax.set_xlabel("sub-state rate separation (max/min, geometric mean)")
    ax.set_ylabel("pairs for 90% power at 5$\\sigma$")
    ax.grid(which="major", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3)
    ax.text(0.03, 0.04, f"Spearman $r_s$ = {rs:.2f}", transform=ax.transAxes)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"fig3c_record_length_vs_separation.{ext}"),
                    dpi=600)
    print("wrote figures/fig3c_record_length_vs_separation.{pdf,png}")


if __name__ == "__main__":
    main()
