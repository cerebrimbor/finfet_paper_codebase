# Independence testing on the binned 2-D dwell histogram.
#
# Pearson correlation detects linear dependence only. A rank-2 joint density can
# have rho = 0 exactly (topology.zero_correlation_partner constructs one), so
# the correlation test has a blind spot.
#
# A binned 2-D histogram is a contingency table, and a contingency table has rank
# 1 exactly when its rows and columns are independent. So the Pearson chi-squared
# test of independence is a rank-1 test on the joint, with a known null
# distribution and sensitivity to any departure from factorisation.
#
# Bins are marginal quantiles rather than fixed widths: dwell times span decades,
# so fixed-width bins starve the peak and empty the tail, and chi-squared needs
# expected cell counts of roughly 5+. Equal-frequency marginal bins make every
# expected count n/(nf*ne), so the asymptotics hold without pooling. (Log-spaced
# bins in the style of Sigworth & Sine 1987 suit plotting the histogram, not
# testing it.)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats, optimize

from .generator import validate_generator
from .dwell import _blocks, entry_distribution

__all__ = [
    "quantile_bin_edges",
    "binned_joint",
    "IndependenceResult",
    "independence_test",
    "joint_mixture_form",
    "analytic_binned_joint",
    "chi2_noncentrality",
    "pairs_needed_chi2",
]


def quantile_bin_edges(t, n_bins: int) -> np.ndarray:
    """Equal-frequency bin edges: each bin holds ~1/n_bins of the sample.

    Edges are made strictly increasing and the outer ones pushed to +-inf so
    every sample lands in a bin regardless of ties.
    """
    t = np.asarray(t, dtype=float)
    if n_bins < 2:
        raise ValueError("need at least 2 bins")
    e = np.quantile(t, np.linspace(0.0, 1.0, n_bins + 1))
    e = np.unique(e)
    if e.size < 3:
        raise ValueError("dwell times too degenerate to bin")
    e[0], e[-1] = -np.inf, np.inf
    return e


def binned_joint(t_f, t_e, edges_f, edges_e) -> np.ndarray:
    """Contingency table of adjacent dwell pairs. Shape (len(edges_f)-1, ...)."""
    table, _, _ = np.histogram2d(np.asarray(t_f, dtype=float),
                                 np.asarray(t_e, dtype=float),
                                 bins=[edges_f, edges_e])
    return table


@dataclass(frozen=True)
class IndependenceResult:
    """Chi-squared test of independence on the binned adjacent-dwell joint."""

    statistic: float
    p_value: float
    dof: int
    n_pairs: int
    n_bins: tuple[int, int]
    min_expected: float
    cramers_v: float        # effect size in [0, 1]; scale-free, unlike chi2

    @property
    def rejects_independence(self) -> bool:
        """True at the 5% level. Independence <=> the joint has rank 1."""
        return self.p_value < 0.05

    @property
    def n_sigma(self) -> float:
        """Two-sided normal-equivalent sigmas, for comparison with a z-score.

        Uses the survival function in log space so that p-values far below
        double precision (which underflow to 0.0) still give a finite number.
        """
        logp = stats.chi2.logsf(self.statistic, self.dof)
        if not np.isfinite(logp):
            return np.inf
        # Solve log(2 * Phi(-z)) = logp for z.
        return float(-stats.norm.ppf(0.5 * np.exp(min(logp, 0.0))))


def independence_test(t_f, t_e, n_bins: int = 5) -> IndependenceResult:
    """Test whether adjacent dwells are independent, i.e. whether the joint is
    rank 1.

    Rejecting independence proves the joint has rank > 1 and therefore that the
    aggregate has multiple gateways with internal structure -- a topological
    statement the 1-D marginals cannot make.

    Not rejecting proves nothing: absence of evidence. Report the effect size
    (Cramer's V) and the sample size alongside the p-value.
    """
    t_f = np.asarray(t_f, dtype=float)
    t_e = np.asarray(t_e, dtype=float)
    if t_f.shape != t_e.shape:
        raise ValueError("t_f and t_e must be paired")
    n = t_f.size
    if n < 20 * n_bins ** 2:
        raise ValueError(f"need >= {20 * n_bins ** 2} pairs for {n_bins}x{n_bins} "
                         f"bins, got {n}")

    table = binned_joint(t_f, t_e,
                         quantile_bin_edges(t_f, n_bins),
                         quantile_bin_edges(t_e, n_bins))

    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)

    r, c = table.shape
    v = float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if min(r, c) > 1 else 0.0

    return IndependenceResult(
        statistic=float(chi2),
        p_value=float(p),
        dof=int(dof),
        n_pairs=int(n),
        n_bins=(int(r), int(c)),
        min_expected=float(expected.min()),
        cramers_v=v,
    )


# Sizing an experiment without simulation
#
# independence_test above needs a sample. The functions below need only Q:
# they diagonalise the joint dwell density in closed form (the 2-D analogue of
# topology.mixture_params) and get the chi-squared noncentrality from it.


def joint_mixture_form(Q, level_map, level: int = 1):
    """Exact exponential-mixture form of the 2-D adjacent-dwell joint.

        f(t_f, t_e) = sum_mn B[m, n] exp(-rF[m] t_f) exp(-rE[n] t_e)

    Diagonalising Q_FF and Q_EE turns the matrix-exponential joint density
    used in correlation.analytic_adjacent_moments into a finite sum of
    products -- the 2-D analogue of topology.mixture_params, which is
    recovered by marginalising either variable out:

        sum_n B[m, n] / rE[n]  ==  mixture_params(Q, level_map, level)[1]
        sum_m B[m, n] / rF[m]  ==  mixture_params(Q, level_map, 1 - level)[1]

    (checked in tests/test_blind_spot_k33.py).
    """
    Q = validate_generator(Q)
    phi = entry_distribution(Q, level_map, level)
    F, E, Q_FF, Q_FE, Q_EE, Q_EF = _blocks(Q, level_map, level)

    lamF, VF = np.linalg.eig(Q_FF)
    lamE, VE = np.linalg.eig(Q_EE)
    if max(np.abs(lamF.imag).max(), np.abs(lamE.imag).max()) > 1e-9:
        raise ValueError("complex dwell spectrum; mixture form does not apply")
    lamF, VF, lamE, VE = lamF.real, VF.real, lamE.real, VE.real
    if max(np.linalg.cond(VF), np.linalg.cond(VE)) > 1e10:
        raise ValueError("a level's generator block is defective (repeated "
                         "eigenvalues); mixture form degenerates -- perturb "
                         "the rates")

    a = phi @ VF
    G = np.linalg.solve(VF, Q_FE @ VE)
    d = np.linalg.solve(VE, Q_EF @ np.ones(F.size))
    return -lamF, -lamE, a[:, None] * G * d[None, :]


def _exponential_mixture_quantiles(rates, coeffs, n_bins):
    """Exact equal-frequency bin edges of sum_k coeffs[k] exp(-rates[k] t).

    Mirrors quantile_bin_edges, but from the analytic CDF via root-finding
    instead of empirical order statistics. coeffs/rates
    are a valid dwell density's mixture form (mixture_params or the marginal
    of joint_mixture_form), so the CDF is sum_k (coeffs[k]/rates[k])
    (1 - exp(-rates[k] t)) and is monotone, making the root well posed.
    """
    def cdf(t):
        return float(np.sum(coeffs * (1.0 - np.exp(-rates * t)) / rates))

    hi = 1.0
    while cdf(hi) < 1.0 - 1e-13:
        hi *= 2.0
    quantiles = np.arange(1, n_bins) / n_bins
    inner = [optimize.brentq(lambda t, q=q: cdf(t) - q, 0.0, hi,
                             xtol=1e-14, rtol=1e-14)
             for q in quantiles]
    return np.array([0.0, *inner, np.inf])


def _exponential_mass(rates, edges):
    """M[k, I] = integral of exp(-rates[k] t) over bin I of `edges`."""
    lo, hi = edges[:-1], edges[1:]
    e_lo = np.exp(-np.outer(rates, lo))
    e_hi = np.where(np.isinf(hi), 0.0, np.exp(-np.outer(rates, hi)))
    return (e_lo - e_hi) / rates[:, None]


def analytic_binned_joint(Q, level_map, n_bins: int = 5, level: int = 1):
    """Exact cell probabilities of the quantile-binned adjacent-dwell joint.

    Uses joint_mixture_form, so no trajectory is simulated: this is what
    independence_test's binned_joint converges to as the sample size grows,
    computed directly from the exact marginal CDFs instead of empirical
    quantiles. Same equal-frequency binning convention as quantile_bin_edges.
    """
    rF, rE, B = joint_mixture_form(Q, level_map, level)
    aF = (B / rE[None, :]).sum(axis=1)      # marginal-F mixture coefficients
    aE = (B / rF[:, None]).sum(axis=0)      # marginal-E mixture coefficients
    edges_f = _exponential_mixture_quantiles(rF, aF, n_bins)
    edges_e = _exponential_mixture_quantiles(rE, aE, n_bins)
    P = _exponential_mass(rF, edges_f).T @ B @ _exponential_mass(rE, edges_e)
    return P / P.sum()


def chi2_noncentrality(Q, level_map, n_bins: int = 5, level: int = 1) -> float:
    """
    Population Pearson chi-squared effect size per adjacent-dwell pair.

    For n independent pairs, the usual large-sample approximation gives a
    noncentral chi-squared statistic with degrees of freedom
    (n_bins - 1)^2 and noncentrality n * lambda.

    Adjacent dwell pairs extracted from a single hidden-state CTMC trajectory
    need not be independent. Therefore this noncentral-chi-squared result is
    used as an asymptotic independent-pair benchmark; empirical power for
    continuous CTMC records must be calibrated separately.
    """
    P = analytic_binned_joint(Q, level_map, n_bins, level)
    pf, pe = P.sum(axis=1), P.sum(axis=0)
    expected = np.outer(pf, pe)
    return float(np.sum((P - expected) ** 2 / expected))


def pairs_needed_chi2(Q, level_map, n_bins: int = 5, level: int = 1,
                      n_sigma: float = 5.0, power: float = 0.9) -> int:
    """Adjacent-dwell pairs needed to reject independence at n_sigma, at the
    stated statistical power -- sized from chi2_noncentrality, no trial run.

    The n-pair statistic is noncentral chi-squared with noncentrality
    n * lambda; this solves stats.ncx2.sf(crit, dof, n * lambda) = power for n
    directly, not via a Gaussian power approximation. ncx2 is right-skewed at
    these noncentralities, so E[chi2] reaching the n_sigma threshold
    corresponds to power ~ 0.5 only.

    Counterpart of correlation.pairs_needed. Where rho = 0 makes pairs_needed
    return "never" while the joint is still rank > 1 (the construction in
    topology.zero_correlation_partner), this returns a finite sample size.
    """
    dof = (n_bins - 1) ** 2
    lam = chi2_noncentrality(Q, level_map, n_bins, level)
    if lam <= 0.0:
        return int(np.iinfo(np.int64).max)

    crit = stats.chi2.isf(2.0 * stats.norm.sf(n_sigma), dof)

    def power_deficit(n):
        return stats.ncx2.sf(crit, dof, lam * n) - power

    lo, hi = 1.0, max(2.0, (crit - dof) / lam)
    while power_deficit(hi) < 0.0:
        hi *= 2.0
    n = optimize.brentq(power_deficit, lo, hi, xtol=1e-6, rtol=1e-10)
    return int(np.ceil(n))