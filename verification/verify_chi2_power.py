# Replicated check of chi-squared blind-spot detection power.
#
# Replaces the earlier single-seed check, which drew one 4,000,000-pair sample
# and asserted only p < 1e-8. One realisation says nothing about power. Here the
# test is repeated and the rejection rate is compared with the noncentral
# chi-squared prediction that the paper's record-length benchmarks rest on.
#
# Part 1 (independent recomputation). The per-pair noncentrality lambda_1 of the
# Fig. 2 / Fig. 3 seed-11, deviation-rank-2 K(3,3) construction is recomputed
# from the exact bipartite cycle-level mixture
#
#     P(component E=i, F=j) = W_ij / C,  t_f ~ Exp(K_j / pi_Fj),  t_e ~ Exp(D_i / pi_Ei)
#
# without calling independence.py, and compared with the library value and with
# the value recorded in data/fig3_chi2_power.csv.
#
# Part 2 (replicated power). For the strongest K(3,3) blind spot found by
# scripts/reproduce_blind_spot_k33.py (rng seed 2024; 6 quantile bins), R
# replicate records of n pairs are drawn, n being the 90%-power benchmark at
# 5 sigma, and each is put through stochastic_ctmc.independence.independence_test
# (bin edges estimated from the sample). Two sampling schemes:
#   independent  -- iid pairs from the exact mixture above;
#   continuous   -- adjacent pairs from one Gillespie trajectory per replicate
#                   (stochastic_ctmc.gillespie.gillespie_ssa + adjacent_pairs),
#                   which carries the hidden-chain serial dependence.
# Asserted: each empirical power lies within 4 Monte Carlo standard errors plus
# ASYMPTOTIC_ALLOWANCE of the prediction, and the two schemes agree within 4
# standard errors of their difference.
#
# Part 3 (optional, slow: --fig2 R). The same comparison for the seed-11
# construction at its own benchmark n (~3.4e6 pairs). 1,000 replicates take
# roughly an hour per scheme; nothing is asserted unless it is run.
#
# Usage:  python verification/verify_chi2_power.py [--reps R] [--fig2 R2]
from pathlib import Path
import argparse
import csv
import sys

import numpy as np
from scipy import stats, optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from stochastic_ctmc.topology import (bipartite_generator_from_coupling,
                                      zero_correlation_partner, coupling_from_generator)
from stochastic_ctmc.independence import (chi2_noncentrality, pairs_needed_chi2,
                                          independence_test)
from stochastic_ctmc.gillespie import gillespie_ssa
from stochastic_ctmc.correlation import adjacent_pairs, analytic_adjacent_moments

LEV6 = [0, 0, 0, 1, 1, 1]
N_SIGMA = 5.0
POWER = 0.9
# The noncentral-chi2 law is a large-sample approximation. At n ~ 2.6e3 pairs it
# overstates power by about one percentage point even for exactly independent
# pairs (scripts/continuous_trajectory_power.py); this allowance keeps that known bias from
# failing the check while still catching a gross error in the benchmark.
ASYMPTOTIC_ALLOWANCE = 0.03


def fig2_construction():
    # Same construction as scripts/generate_figures.py, fig2/fig3 seed 11.
    rng = np.random.default_rng(11)
    pi = rng.uniform(0.5, 1.5, 6)
    pi /= pi.sum()
    W = rng.uniform(0.4, 2.0, (3, 3))
    target = bipartite_generator_from_coupling(pi[:3], pi[3:], W)
    return zero_correlation_partner(target, np.array(LEV6), deviation_rank=2,
                                    seed=3, margin=0.9)


def mixture_noncentrality(Q, n_bins):
    # lambda_1 from the cycle-level mixture, independent of independence.py.
    pi_e, pi_f, W = coupling_from_generator(Q, LEV6)
    C = W.sum()
    rF, rE = W.sum(0) / pi_f, W.sum(1) / pi_e
    B = (W / C) * rE[:, None] * rF[None, :]          # density coefficient of exp(-rE_i te - rF_j tf)
    aF, aE = (B / rE[:, None]).sum(0), (B / rF[None, :]).sum(1)

    def edges(r, a):
        cdf = lambda t: float(np.sum(a * (1.0 - np.exp(-r * t)) / r))
        hi = 1.0
        while cdf(hi) < 1.0 - 1e-13:
            hi *= 2.0
        inner = [optimize.brentq(lambda z, q=q: cdf(z) - q, 0.0, hi, xtol=1e-15)
                 for q in np.arange(1, n_bins) / n_bins]
        return np.array([0.0, *inner, np.inf])

    def masses(r, e):
        return (np.exp(-r[:, None] * e[:-1])
                - np.where(np.isinf(e[1:]), 0.0, np.exp(-r[:, None] * e[1:]))) / r[:, None]

    P = masses(rF, edges(rF, aF)).T @ B.T @ masses(rE, edges(rE, aE))
    P /= P.sum()
    ex = np.outer(P.sum(1), P.sum(0))
    return float(np.sum((P - ex) ** 2 / ex))


def iid_pairs(Q, n, rng):
    pi_e, pi_f, W = coupling_from_generator(Q, LEV6)
    rF, rE = W.sum(0) / pi_f, W.sum(1) / pi_e
    k = rng.choice(W.size, size=n, p=(W / W.sum()).ravel())
    i, j = k // W.shape[1], k % W.shape[1]
    return rng.exponential(1.0 / rF[j]), rng.exponential(1.0 / rE[i])


def trajectory_pairs(Q, n, seed, cycle):
    # One continuous trajectory; extend until it yields n adjacent pairs.
    t_max = (n + 50) * cycle * 1.2
    while True:
        tf, te = adjacent_pairs(gillespie_ssa(Q, LEV6.index(1), t_max, rng=seed,
                                              validate=False), LEV6, 1)
        if tf.size >= n:
            return tf[:n], te[:n]
        t_max *= 1.5


def replicated_power(Q, n_bins, reps, base_seed, label):
    dof = (n_bins - 1) ** 2
    crit = stats.chi2.isf(2.0 * stats.norm.sf(N_SIGMA), dof)
    lam = chi2_noncentrality(Q, LEV6, n_bins=n_bins)
    n = pairs_needed_chi2(Q, LEV6, n_bins=n_bins, n_sigma=N_SIGMA, power=POWER)
    pred = float(stats.ncx2.sf(crit, dof, n * lam))
    m = analytic_adjacent_moments(Q, LEV6, 1)
    cycle = m.mean_f + m.mean_e
    print(f"\n[{label}] {n_bins} bins, lambda_1 = {lam:.4e}, n = {n:,} pairs, "
          f"{reps} replicates per scheme")
    print(f"  prediction: mean chi2 {dof + n * lam:.2f}, "
          f"sd {np.sqrt(2 * (dof + 2 * n * lam)):.2f}, power {pred:.3f}")
    out = {}
    for scheme in ("independent", "continuous"):
        chis = np.empty(reps)
        for r in range(reps):
            seed = base_seed + (0 if scheme == "independent" else 1_000_000) + r
            if scheme == "independent":
                tf, te = iid_pairs(Q, n, np.random.default_rng(seed))
            else:
                tf, te = trajectory_pairs(Q, n, seed, cycle)
            chis[r] = independence_test(tf, te, n_bins=n_bins).statistic
        pw = float(np.mean(chis >= crit))
        se = float(np.sqrt(max(pw * (1 - pw), 1e-12) / reps))
        out[scheme] = (pw, se)
        print(f"  {scheme:<12}: mean chi2 {chis.mean():.2f}, sd {chis.std(ddof=1):.2f}, "
              f"power {pw:.3f} +/- {se:.3f}")
    return pred, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--fig2", type=int, default=0, metavar="R2")
    args = ap.parse_args()

    # Part 1
    Q11 = fig2_construction()
    lam_mix = mixture_noncentrality(Q11, 5)
    lam_lib = chi2_noncentrality(Q11, LEV6, n_bins=5)
    with open(ROOT / "data" / "fig3_chi2_power.csv") as f:
        rec = [r for r in csv.DictReader(f)
               if int(float(r["target_seed"])) == 11 and int(float(r["deviation_rank"])) == 2]
    lam_rec = float(rec[0]["noncentrality"])
    print(f"seed-11 lambda_1: mixture {lam_mix:.10e}, library {lam_lib:.10e}, "
          f"fig3 data {lam_rec:.10e}")
    assert abs(lam_mix - lam_lib) < 1e-9 * lam_lib
    assert abs(lam_lib - lam_rec) < 1e-8 * lam_lib    # CSV stores 10 significant figures

    # Part 2
    from reproduce_blind_spot_k33 import strongest_blind_spot
    _, Q_strong, _ = strongest_blind_spot(np.random.default_rng(2024))
    pred, out = replicated_power(Q_strong, 6, args.reps, 20261019, "strongest K(3,3)")
    for scheme, (pw, se) in out.items():
        assert abs(pw - pred) <= 4 * se + ASYMPTOTIC_ALLOWANCE, (scheme, pw, pred)
    (p1, s1), (p2, s2) = out["independent"], out["continuous"]
    assert abs(p1 - p2) <= 4 * np.hypot(s1, s2), (p1, p2)

    # Part 3
    if args.fig2:
        pred, out = replicated_power(Q11, 5, args.fig2, 20261119, "seed-11 construction")
        for scheme, (pw, se) in out.items():
            assert abs(pw - pred) <= 4 * se + ASYMPTOTIC_ALLOWANCE, (scheme, pw, pred)

    print("PASS")


if __name__ == "__main__":
    main()
