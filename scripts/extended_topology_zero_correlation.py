# Zero-correlation, higher-rank generators on the six-state extended defect
# topology of Wimmer et al., Proc. R. Soc. A 472, 20160009 (2016), Fig. 12:
#   neutral level: 1, 1', 0n        positive level: 2, 2', 0+
#   charge exchange: 1-2', 1'-2, 0n-0+     inside a level: 1-1', 2-2', 2'-0+
# Reversible by construction (q_ij = c_ij / pi_i). Stationary weights and five
# conductances ~ exp(U(0,3)); the 0n-0+ conductance is swept over 1e-3..1e3 and
# each sign change of the covariance is solved for exact zero.
#
# Writes data/extended_topology_zero_correlation.csv.
# Run: python scripts/extended_topology_zero_correlation.py   (~10 s)
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy import optimize
from stochastic_ctmc.dwell import joint_density_rank
from stochastic_ctmc.correlation import analytic_adjacent_moments
from stochastic_ctmc.generator import is_detailed_balance
from stochastic_ctmc.independence import chi2_noncentrality, pairs_needed_chi2
from stochastic_ctmc.export import export_columns_csv

S = {'1': 0, "1'": 1, '0n': 2, '2': 3, "2'": 4, '0+': 5}
LEV = [0, 0, 0, 1, 1, 1]
EDGES = [('1', "2'"), ("1'", '2'), ('0n', '0+'), ('1', "1'"), ('2', "2'"), ("2'", '0+')]
SWEEP = 2  # index of the 0n-0+ edge


def generator(pi, c):
    Q = np.zeros((6, 6))
    for (a, b), cab in zip(EDGES, c):
        i, j = S[a], S[b]
        Q[i, j] = cab / pi[i]; Q[j, i] = cab / pi[j]
    Q[np.diag_indices(6)] = -Q.sum(1)
    return Q


def rho(pi, c, log10c):
    cc = c.copy(); cc[SWEEP] = 10.0 ** log10c
    return analytic_adjacent_moments(generator(pi, cc), LEV, 1).rho


def main():
    grid = np.linspace(-3, 3, 121)
    rows, with_root = [], 0
    for d in range(400):
        rng = np.random.default_rng(16_000 + d)
        pi = np.exp(rng.uniform(0, 3, 6)); pi /= pi.sum()
        c = np.exp(rng.uniform(0, 3, 6))
        r = np.array([rho(pi, c, g) for g in grid])
        roots = [optimize.brentq(lambda x: rho(pi, c, x), grid[k], grid[k + 1], xtol=1e-15, rtol=1e-15)
                 for k in np.flatnonzero(np.sign(r[:-1]) * np.sign(r[1:]) < 0)]
        with_root += bool(roots)
        for x in roots:
            cc = c.copy(); cc[SWEEP] = 10.0 ** x
            Q = generator(pi, cc)
            try:
                lam, n = chi2_noncentrality(Q, LEV, 5), pairs_needed_chi2(Q, LEV, 5)
            except ValueError:
                lam, n = np.nan, -1
            rows.append((d, x, analytic_adjacent_moments(Q, LEV, 1).rho, joint_density_rank(Q, LEV),
                         is_detailed_balance(Q, atol=1e-10), lam, n))
    R = np.array(rows, dtype=float)
    export_columns_csv(os.path.join(ROOT, "data", "extended_topology_zero_correlation.csv"),
                       {"draw": R[:, 0], "log10_c_0n_0plus": R[:, 1], "rho": R[:, 2],
                        "joint_rank": R[:, 3], "reversible": R[:, 4],
                        "noncentrality_5bin": R[:, 5], "pairs_needed": R[:, 6]})
    P = R[R[:, 6] > 0, 6]
    ranks, counts = np.unique(R[:, 3].astype(int), return_counts=True)
    print(f"draws with a sign change: {with_root}/400; zero-correlation generators: {len(R)}")
    print(f"max |rho| {np.max(np.abs(R[:, 2])):.1e}; reversible {int(R[:, 4].sum())}/{len(R)}; "
          f"joint ranks {dict(zip(ranks.tolist(), counts.tolist()))}")
    print(f"5-bin benchmark pairs: min {P.min():,.0f}  median {np.median(P):.3g}  max {P.max():.3g}")


if __name__ == "__main__":
    main()
