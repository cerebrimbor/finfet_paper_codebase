# Is the chain uniquely identified, or is some degenerate partner invisible?
#
# max |rho| says how little data you need in the best case. The infimum of
# |rho| over the rank-2 reversible partners decides how strong a claim
# correlation can support. For CHAIN that infimum is zero (see
# test_no_positive_lower_bound_on_rho_among_rank_two_partners): the partners
# form a segment through the rank-1 product coupling, so correlation separates
# rank 1 from rank 2 only at exact zero and offers no finite-record guarantee.

import numpy as np
import pytest

from stochastic_ctmc.generator import n_state_generator, is_detailed_balance
from stochastic_ctmc.dwell import joint_density_rank, gateway_block_rank
from stochastic_ctmc.topology import (
    extremal_correlation_reversible, LEVELS_4, coupling_from_generator,
    bipartite_generator_from_coupling, _marginal_signature,
)
from stochastic_ctmc.correlation import analytic_adjacent_moments
from stochastic_ctmc.correlation import pairs_needed

CHAIN = n_state_generator(
    {(0, 1): 5.0, (1, 0): 5.0,
     (1, 2): 1.0, (2, 1): 1.0,
     (2, 3): 4.0, (3, 2): 4.0},
    n_states=4,
)
LEVELS = list(LEVELS_4)


def test_sense_is_validated():
    with pytest.raises(ValueError):
        extremal_correlation_reversible(CHAIN, LEVELS, sense="sideways")


@pytest.mark.slow
def test_matched_exit_rates_equal_the_chains_eigenvalues():
    """The bipartite partner has diagonal Q_FF, so its mixture rates are its
    bare exit rates, and they must equal the eigenvalues of the chain's coupled
    Q_FF block: same observable time constants, different mechanism.
    """
    res = extremal_correlation_reversible(CHAIN, LEVELS, sense="max", n_samples=80)
    assert res.success

    for idx, label in (([2, 3], "filled"), ([0, 1], "empty")):
        blk = CHAIN[np.ix_(idx, idx)]
        eig = np.sort(-np.linalg.eigvals(blk).real)
        exits = np.sort(-np.diag(res.Q)[idx])
        assert np.allclose(eig, exits, rtol=1e-6), f"{label}: {eig} vs {exits}"


@pytest.mark.slow
def test_report_correlation_range_over_the_manifold():
    hi = extremal_correlation_reversible(CHAIN, LEVELS, sense="max", n_samples=200)
    lo = extremal_correlation_reversible(CHAIN, LEVELS, sense="min", n_samples=200)
    assert hi.success and lo.success

    print(f"\n--- reversible degenerate-partner manifold ---")
    print(f"max |rho| = {abs(hi.rho):.6f}  -> {pairs_needed(hi.rho):>14,} pairs @ 5 sigma")
    print(f"min |rho| = {abs(lo.rho):.6f}  -> {pairs_needed(lo.rho):>14,} pairs @ 5 sigma")

    for r in (hi, lo):
        assert is_detailed_balance(r.Q, atol=1e-8)
        assert r.signature_error < 1e-7
        assert gateway_block_rank(r.Q, LEVELS) == 2

    assert abs(lo.rho) <= abs(hi.rho) + 1e-9


@pytest.mark.slow
def test_no_positive_lower_bound_on_rho_among_rank_two_partners():
    """inf |rho| over the rank-2 reversible partners of CHAIN is zero.

    Replaces test_worst_case_partner_is_still_detectable, which asserted
    |rho| > 1e-4 at the optimiser's minimum. That bound held only because the
    SLSQP search stops at |rho| ~ 1.4e-3; it is not a property of the family.
    In the coupling chart the bipartite partners of CHAIN are the segment
    W(t) = W0 + t (W - W0) of the K(2,2) transportation polytope, which passes
    through the product coupling W0 at t = 0. The 1-D marginals depend on W only
    through pi and the margins D, K, so every W(t) is marginal-equivalent;
    Cov = det(W)(mE0-mE1)(mF0-mF1)/C^2 and det W(t) = t det W, so rho is linear
    in t and the joint stays rank 2 for every t != 0. This is the manuscript's
    statement (Sec. III.A) that no positive universal lower bound on |rho|
    exists: correlation separates rank 1 from rank 2 only at exact zero.
    """
    lo = extremal_correlation_reversible(CHAIN, LEVELS, sense="min", n_samples=200)
    assert lo.success
    assert joint_density_rank(CHAIN, LEVELS) == 1
    assert joint_density_rank(lo.Q, LEVELS) == 2
    print(f"\noptimiser minimum |rho| = {abs(lo.rho):.3e} "
          f"({pairs_needed(lo.rho, n_sigma=5.0):,} pairs @ 5 sigma)")

    pi_E, pi_F, W = coupling_from_generator(lo.Q, LEVELS)
    W0 = np.outer(W.sum(axis=1), W.sum(axis=0)) / W.sum()
    sig = _marginal_signature(lo.Q, LEVELS)
    for t in (1e-1, 1e-3, 1e-5):
        Q_t = bipartite_generator_from_coupling(pi_E, pi_F, W0 + t * (W - W0))
        rho_t = analytic_adjacent_moments(Q_t, LEVELS, 1).rho
        assert is_detailed_balance(Q_t, atol=1e-10)
        assert np.max(np.abs(_marginal_signature(Q_t, LEVELS) - sig)) < 1e-9
        assert joint_density_rank(Q_t, LEVELS) == 2
        assert rho_t == pytest.approx(t * lo.rho, rel=1e-6)
    # Smallest t checked: |rho| is two orders below the old 1e-4 "bound".
    assert abs(rho_t) < 1e-6
