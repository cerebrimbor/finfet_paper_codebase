# Power of the quantile-binned chi-squared test on continuous hidden-state
# trajectories, compared with independent pairs and with the noncentral
# chi-squared prediction behind the record-length benchmarks.
#
# Two K(3,3) blind spots, each at its own 90%-power / 5-sigma benchmark n:
#   strong  - the strongest blind spot found by reproduce_blind_spot_k33.py
#             (rng seed 2024), 6 quantile bins
#   fig2    - the Fig. 2 construction (target seed 11, deviation rank 2), 5 bins
# For each replicate, n adjacent (filled, empty) pairs are drawn either from one
# continuous trajectory or independently from the same joint law, and put
# through stochastic_ctmc.independence.independence_test (bin edges estimated
# from the record).
#
# The trajectory sampler is exact for bipartite generators (no transitions inside
# a level): entering sub-state s, the dwell is Exp(-Q_ss) and the next sub-state
# is drawn from Q_s,:/(-Q_ss). `--validate` compares it with the library's
# Gillespie SSA. It needs numba (pip install numba).
#
# Run:  python scripts/continuous_trajectory_power.py strong 5000   (~10 s)
#       python scripts/continuous_trajectory_power.py fig2 1000     (~20 min)
#       python scripts/continuous_trajectory_power.py --validate
from __future__ import annotations
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from scipy import stats
from numba import njit
from stochastic_ctmc.topology import coupling_from_generator
from stochastic_ctmc.independence import chi2_noncentrality, pairs_needed_chi2, independence_test

LEV = np.array([0, 0, 0, 1, 1, 1])


@njit(cache=True)
def trajectory_pairs(cumP, rates, start, n, seed):
    np.random.seed(seed)
    tf = np.empty(n); te = np.empty(n); s = start
    for k in range(n):
        tf[k] = np.random.exponential(1.0 / rates[s])
        u = np.random.random(); s2 = 0
        while cumP[s, s2] < u: s2 += 1
        s = s2
        te[k] = np.random.exponential(1.0 / rates[s])
        u = np.random.random(); s2 = 0
        while cumP[s, s2] < u: s2 += 1
        s = s2
    return tf, te


def setup(Q):
    rates = -np.diag(Q).copy()
    P = Q / rates[:, None]; np.fill_diagonal(P, 0.0)
    cumP = np.cumsum(P, axis=1); cumP[:, -1] = 1.0
    _, _, W = coupling_from_generator(Q, list(LEV))
    return cumP, rates, W


def iid_pairs(W, rates, n, rng):
    idx = rng.choice(9, size=n, p=(W / W.sum()).ravel()); k, j = idx // 3, idx % 3
    return rng.exponential(1.0 / rates[3 + j]), rng.exponential(1.0 / rates[k])


def construction(which):
    if which == "strong":
        from reproduce_blind_spot_k33 import strongest_blind_spot
        return strongest_blind_spot(np.random.default_rng(2024))[1], 6, 31_000_000
    from generate_figures import _k33_degenerate_triple
    return _k33_degenerate_triple(seed=11, deviation_rank=2, margin=0.9)[2], 5, 41_000_000


def run(which, reps):
    Q, n_bins, base = construction(which)
    dof = (n_bins - 1) ** 2
    crit = stats.chi2.isf(2.0 * stats.norm.sf(5.0), dof)
    lam = chi2_noncentrality(Q, list(LEV), n_bins); n = pairs_needed_chi2(Q, list(LEV), n_bins)
    out = {"case": which, "n_bins": n_bins, "lambda1": lam, "n": int(n),
           "prediction": {"mean": dof + n * lam, "sd": float(np.sqrt(2 * (dof + 2 * n * lam))),
                          "power": float(stats.ncx2.sf(crit, dof, n * lam))}}
    cumP, rates, W = setup(Q)
    K = W.sum(0); entryF = K / K.sum()
    for scheme in ("independent", "continuous"):
        chis = np.empty(reps); t0 = time.time()
        for r in range(reps):
            seed = base + (0 if scheme == "independent" else 10_000_000) + r
            if scheme == "continuous":
                start = 3 + np.random.default_rng(seed).choice(3, p=entryF)
                tf, te = trajectory_pairs(cumP, rates, start, n, seed)
            else:
                tf, te = iid_pairs(W, rates, n, np.random.default_rng(seed))
            chis[r] = independence_test(tf, te, n_bins=n_bins).statistic
        pw = float(np.mean(chis >= crit))
        out[scheme] = {"reps": reps, "mean": float(chis.mean()), "sd": float(chis.std(ddof=1)),
                       "power": pw, "power_se": float(np.sqrt(pw * (1 - pw) / reps)),
                       "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))


def validate():
    from stochastic_ctmc.gillespie import gillespie_ssa
    from stochastic_ctmc.correlation import adjacent_pairs, analytic_adjacent_moments
    Q = construction("fig2")[0]
    cumP, rates, _ = setup(Q)
    m = analytic_adjacent_moments(Q, list(LEV), 1)
    a_tf, a_te = trajectory_pairs(cumP, rates, 3, 400_000, 7)
    g_tf, g_te = adjacent_pairs(gillespie_ssa(Q, 3, 400_000 * (m.mean_f + m.mean_e) * 1.02, rng=8),
                                list(LEV), 1)
    print(f"analytic  mean_f {m.mean_f:.5f} mean_e {m.mean_e:.5f}")
    for lab, tf, te in (("sampler", a_tf, a_te), ("SSA", g_tf, g_te)):
        print(f"{lab:8s} n {tf.size:,} mean_f {tf.mean():.5f} mean_e {te.mean():.5f} "
              f"corr {np.corrcoef(tf, te)[0, 1]:+.4f}")
    print("KS p-values: filled", stats.ks_2samp(a_tf, g_tf).pvalue, " empty", stats.ks_2samp(a_te, g_te).pvalue)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--validate"]:
        validate()
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "strong", int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
