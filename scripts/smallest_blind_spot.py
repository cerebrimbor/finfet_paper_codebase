# The smallest correlation blind spots: K(2,3) and K(3,2).
#
# For 40 random targets of each shape, zero_correlation_partner (deviation
# rank 1, margin 0.9) builds a reversible partner with the target's exact
# one-dimensional dwell laws and zero adjacent-dwell correlation. Reports how
# many are reversible, have |rho| < 1e-12 and joint rank 2, how closely their
# dwell-mixture parameters match the rank-one partner, and the 5-bin chi-squared
# record-length benchmark (90% power at 5 sigma).
#
# Run: python scripts/smallest_blind_spot.py
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stochastic_ctmc.topology import (bipartite_generator_from_coupling, independence_partner,
                                      zero_correlation_partner, mixture_params)
from stochastic_ctmc.dwell import joint_density_rank
from stochastic_ctmc.correlation import analytic_adjacent_moments
from stochastic_ctmc.generator import is_detailed_balance
from stochastic_ctmc.independence import pairs_needed_chi2


def signature(Q, lev):
    return np.concatenate([np.concatenate(mixture_params(Q, lev, l)) for l in (0, 1)])


def main():
    for nE, nF in ((2, 3), (3, 2)):
        lev = [0] * nE + [1] * nF
        rev = zero = rank2 = 0
        sig_err, rho_max, pairs = 0.0, 0.0, []
        for s in range(40):
            rng = np.random.default_rng(5000 + 100 * nE + s)
            pi = rng.uniform(0.5, 1.5, nE + nF); pi /= pi.sum()
            W = rng.uniform(0.4, 2.0, (nE, nF))
            target = bipartite_generator_from_coupling(pi[:nE], pi[nE:], W)
            Qb = zero_correlation_partner(target, lev, deviation_rank=1, seed=s, margin=0.9)
            Q0 = independence_partner(target, lev)
            rho = analytic_adjacent_moments(Qb, lev, 1).rho
            rev += is_detailed_balance(Qb, atol=1e-10)
            zero += abs(rho) < 1e-12
            rank2 += joint_density_rank(Qb, lev) == 2
            rho_max = max(rho_max, abs(rho))
            sig_err = max(sig_err, float(np.max(np.abs(signature(Qb, lev) - signature(Q0, lev)))))
            pairs.append(pairs_needed_chi2(Qb, lev, n_bins=5))
        P = np.array(pairs, float)
        print(f"K({nE},{nF}): reversible {rev}/40, |rho|<1e-12 {zero}/40, joint rank 2 {rank2}/40, "
              f"max |rho| {rho_max:.1e}, max dwell-mixture difference {sig_err:.1e}")
        print(f"   5-bin benchmark pairs: min {P.min():,.0f}  median {np.median(P):.3g}  max {P.max():.3g}")


if __name__ == "__main__":
    main()
