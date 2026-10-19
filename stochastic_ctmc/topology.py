from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize

from .generator import (
    n_state_generator, validate_generator, stationary_distribution,
)
from .dwell import _blocks, entry_distribution
from .correlation import analytic_adjacent_moments

__all__ = [
    "bipartite_generator",
    "bipartite_reversible_generator",
    "mixture_params",
    "match_marginals",
    "match_marginals_reversible",
    "explore_reversible_manifold",
    "extremal_correlation_reversible",
    "max_correlation_reversible",
    "gateway_rank_ratio",
    "gateway_singular_ratios",
    "min_gateway_rank",
    "bipartite_generator_from_coupling",
    "coupling_from_generator",
    "independence_partner",
    "zero_correlation_partner",
    "MatchResult",
    "LEVELS_4",
]

LEVELS_4 = (0, 0, 1, 1)  # states 0,1 = E1,E2 (empty); 2,3 = F1,F2 (filled)

BIPARTITE_EDGES = (
    (0, 2), (0, 3),   # E1 -> F1, F2
    (1, 2), (1, 3),   # E2 -> F1, F2
    (2, 0), (2, 1),   # F1 -> E1, E2
    (3, 0), (3, 1),   # F2 -> E1, E2
)

BIPARTITE_UNDIRECTED = ((0, 2), (0, 3), (1, 2), (1, 3))

# Search box in log space. Wide enough to hold any physically sensible rate
# ratio, tight enough that exp() cannot overflow.
_LOG_LO, _LOG_HI = -9.0, 9.0


def bipartite_generator(rates) -> np.ndarray:
    """4-state generator, every empty state connected to every filled state.

    No intra-level transitions, so Q_FF and Q_EE are diagonal and each
    aggregated dwell is a plain 2-exponential mixture. rank(Q_FE) = 2
    generically: two independent gateways out of the filled level.

    ``rates`` follows BIPARTITE_EDGES.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (8,):
        raise ValueError(f"expected 8 rates, got {rates.shape}")
    return n_state_generator(dict(zip(BIPARTITE_EDGES, rates)), n_states=4)


def bipartite_reversible_generator(log_pi, log_c) -> np.ndarray:
    """Reversible K(2,2) generator, with detailed balance STRUCTURAL.

    Detailed balance says pi_i q_ij = pi_j q_ji. Name that common value the edge
    conductance c_ij and read the rates back off it:

        q_ij = c_ij / pi_i,      q_ji = c_ij / pi_j

    Any (pi > 0, c > 0) gives a reversible chain with stationary distribution
    pi, and every reversible chain on this graph arises this way.

    GAUGE: pi is normalised after exponentiation, so adding a constant to every
    entry of log_pi leaves Q unchanged. That flat direction lets an optimiser
    drift to overflow. Callers doing optimisation must fix the gauge (e.g.
    log_pi[0] = 0, as _pack does); this constructor only guards the range.
    """
    log_pi = np.asarray(log_pi, dtype=float)
    log_c = np.asarray(log_c, dtype=float)
    if log_pi.shape != (4,) or log_c.shape != (4,):
        raise ValueError("log_pi must have 4 entries and log_c 4 entries")
    if not (np.all(np.isfinite(log_pi)) and np.all(np.isfinite(log_c))):
        raise ValueError("non-finite log-parameters")
    if np.ptp(log_pi) > 40.0 or np.max(np.abs(log_c)) > 40.0:
        raise ValueError("log-parameters out of representable range")

    # Subtract the max before exp: removes the gauge freedom's numerical bite
    # without changing the normalised pi.
    pi = np.exp(log_pi - log_pi.max())
    pi = pi / pi.sum()
    if np.any(pi <= 0) or not np.all(np.isfinite(pi)):
        raise ValueError("degenerate stationary distribution")
    c = np.exp(log_c)

    rates: dict[tuple[int, int], float] = {}
    for (i, j), cij in zip(BIPARTITE_UNDIRECTED, c):
        rates[(i, j)] = cij / pi[i]
        rates[(j, i)] = cij / pi[j]
    return n_state_generator(rates, n_states=4)


def _pack(x):
    """(7,) -> (log_pi, log_c) with the gauge log_pi[0] = 0 fixed.

    pi has 4 entries but only 3 degrees of freedom after normalisation. Pinning
    the first entry removes the flat direction that otherwise sends an optimiser
    to infinity.
    """
    x = np.asarray(x, dtype=float)
    return np.concatenate([[0.0], x[:3]]), x[3:]


def _bounds7():
    return (np.full(7, _LOG_LO), np.full(7, _LOG_HI))


def gateway_rank_ratio(Q, level_map=LEVELS_4) -> float:
    # Smallest singular value over the largest, minimised over the two gateway blocks.
    Q = np.asarray(Q, dtype=float)
    lv = np.asarray(level_map)
    A = np.flatnonzero(lv == 0)
    B = np.flatnonzero(lv == 1)
    out = []
    for blk in (Q[np.ix_(A, B)], Q[np.ix_(B, A)]):
        s = np.linalg.svd(blk, compute_uv=False)
        if s[0] <= 0.0:
            return 0.0
        out.append(s[-1] / s[0])
    return float(min(out))


def gateway_singular_ratios(Q, level_map=LEVELS_4):
    # Normalized singular values for both gateway blocks.
    Q = np.asarray(Q, dtype=float)
    lv = np.asarray(level_map)
    A = np.flatnonzero(lv == 0)
    B = np.flatnonzero(lv == 1)
    out = []
    for blk in (Q[np.ix_(A, B)], Q[np.ix_(B, A)]):
        s = np.linalg.svd(blk, compute_uv=False)
        if s[0] <= 0.0:
            raise ValueError("gateway block is identically zero")
        out.append(s / s[0])
    return tuple(out)


def min_gateway_rank(Q, level_map=LEVELS_4, tol: float = 1e-6) -> int:
    """Numerical rank of the level-crossing gateway, on the scale-free spectrum.

    Minimum over both crossing blocks (dwell.gateway_block_rank takes one).

    tol is a ratio to sigma_1, so it is invariant under a global change of time
    units -- an absolute singular-value cut would move when the rates are
    rescaled.
    """
    return min(int(np.sum(r > tol)) for r in gateway_singular_ratios(Q, level_map))


def bipartite_generator_from_coupling(pi_E, pi_F, W) -> np.ndarray:
    #Build a reversible bipartite generator from stationary weights and fluxes
    W = np.asarray(W, dtype=float)
    pi_E = np.asarray(pi_E, dtype=float)
    pi_F = np.asarray(pi_F, dtype=float)
    n_e, n_f = W.shape
    if pi_E.shape != (n_e,) or pi_F.shape != (n_f,):
        raise ValueError("pi_E / pi_F shapes do not match W")
    if np.any(pi_E <= 0) or np.any(pi_F <= 0):
        raise ValueError("stationary weights must be strictly positive")
    if np.any(W < 0):
        raise ValueError("fluxes must be non-negative")

    n = n_e + n_f
    Q = np.zeros((n, n))
    Q[:n_e, n_e:] = W / pi_E[:, None]
    Q[n_e:, :n_e] = W.T / pi_F[:, None]
    Q[np.diag_indices(n)] = -Q.sum(axis=1)
    return validate_generator(Q)


def coupling_from_generator(Q, level_map=LEVELS_4, *, tol: float = 1e-12):
    """Inverse chart: recover (pi_E, pi_F, W) from a reversible bipartite Q.

    W[k, j] = pi_E[k] q(E_k -> F_j) is the stationary flux across the edge, and
    equals pi_F[j] q(F_j -> E_k) exactly when Q satisfies detailed balance.

    Rejects generators with intra-level transitions: this chart describes the
    bipartite family only, where a level's dwell is a single exponential set by
    the sub-state entered. A generator with internal edges (a linear chain, say)
    has genuinely different dwell structure and is not a point of this family.
    """
    Q = validate_generator(Q)
    lv = np.asarray(level_map)
    E, F = np.flatnonzero(lv == 0), np.flatnonzero(lv == 1)
    for idx in (E, F):
        blk = Q[np.ix_(idx, idx)]
        if np.max(np.abs(blk - np.diag(np.diag(blk)))) > tol:
            raise ValueError("target has intra-level transitions; the coupling "
                             "chart covers bipartite generators only")
    pi = stationary_distribution(Q)
    return pi[E], pi[F], pi[E][:, None] * Q[np.ix_(E, F)]


def _centring_basis(n: int) -> np.ndarray:
    """Columns e_i - e_{n-1}: a basis for the zero-sum vectors in R^n."""
    U = np.zeros((n, n - 1))
    U[:n - 1, :] = np.eye(n - 1)
    U[n - 1, :] = -1.0
    return U


def _zero_covariance_perturbation(a, b, rank: int, rng) -> np.ndarray:
    """T with a^T T b = 0 exactly and rank(T) == rank."""
    m_e, m_f = a.size, b.size
    if m_e == 1 and m_f == 1:
        raise ValueError(
            "two sub-states per level: the perturbation space is one "
            "dimensional, so a^T T b = 0 forces T = 0 and every rho = 0 partner "
            "is the product coupling. No blind spot exists at this size -- that "
            "is a theorem, not a search failure")
    if not 1 <= rank <= min(m_e, m_f):
        raise ValueError(f"deviation_rank must lie in 1..{min(m_e, m_f)}")
    if rank < m_e:
        # Draw every column of T from the orthogonal complement of a, so the
        # covariance vanishes term by term rather than by cancellation.
        P = np.linalg.svd(a[None, :])[2][1:].T          # a^T P = 0
        return P[:, :rank] @ rng.normal(size=(rank, m_f))
    # Full row rank: project a generic T onto the hyperplane <T, a b^T> = 0.
    G = np.outer(a, b)
    T = rng.normal(size=(m_e, m_f))
    return T - (np.vdot(G, T) / np.vdot(G, G)) * G


def independence_partner(Q_target, level_map=LEVELS_4) -> np.ndarray:
    """The rank-1 member of Q_target's degenerate family: W = D K^T / C.

    This is the product coupling, so adjacent sub-states are independent, the
    2-D joint factorises and rho = 0 for structural reasons.
    """
    pi_E, pi_F, W = coupling_from_generator(Q_target, level_map)
    return bipartite_generator_from_coupling(
        pi_E, pi_F, np.outer(W.sum(axis=1), W.sum(axis=0)) / W.sum())


def zero_correlation_partner(Q_target, level_map=LEVELS_4, *,
                             deviation_rank: int = 1, seed: int = 0,
                             margin: float = 0.5) -> np.ndarray:
    """Partner with Q_target's exact marginals, rho = 0 exactly and joint rank > 1.

    Bipartite levels only (no intra-level edges), with distinct sub-state exit
    rates within each level. There the 1-D marginals fix pi and both margins
    of the flux matrix W, so the partners of Q_target are exactly the
    transportation polytope {W >= 0, W 1 = D, W^T 1 = K}, of dimension
    (n_E - 1)(n_F - 1). On it

        Cov(t_f, t_e) = (1/C) x^T (W - D K^T / C) y

    with x = pi_E / D and y = pi_F / K the sub-state mean dwells. This is
    linear in W, so rho = 0 is a single hyperplane through the product
    coupling. Writing the deviation as W~ = U_E T U_F^T in the centring bases
    reduces the condition to a^T T b = 0 with a = x - x_last, b = y - y_last,
    which has nonzero solutions whenever (n_E - 1)(n_F - 1) >= 2, with T of
    any rank up to min(n_E, n_F) - 1.

    At two sub-states per level T is a scalar and a^T T b = 0 forces T = 0
    unless a or b vanishes, which recovers the three degenerate branches and
    leaves no blind spot.

    deviation_rank is the rank of W~. The gateway W then has rank
    deviation_rank + 1, and the joint has the same rank when the sub-state
    exit rates within each level are distinct. seed picks the random
    direction T. margin is the fraction of the largest step that keeps every
    entry of W non-negative.
    """
    pi_E, pi_F, W = coupling_from_generator(Q_target, level_map)
    D, K = W.sum(axis=1), W.sum(axis=0)
    C = W.sum()
    x, y = pi_E / D, pi_F / K

    T = _zero_covariance_perturbation(x[:-1] - x[-1], y[:-1] - y[-1],
                                      deviation_rank,
                                      np.random.default_rng(seed))
    M = _centring_basis(pi_E.size) @ T @ _centring_basis(pi_F.size).T

    W0 = np.outer(D, K) / C
    if not np.any(M < 0):
        raise ValueError("perturbation has no negative entry to bound the step")
    step = margin * np.min(-W0[M < 0] / M[M < 0])
    return bipartite_generator_from_coupling(pi_E, pi_F, W0 + step * M)


def mixture_params(Q, level_map, level: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Exponential-mixture form of the aggregated dwell density at `level`.

    Returns ``(rates, coeffs)`` with ``f(t) = sum_k coeffs[k] exp(-rates[k] t)``,
    sorted by rate. Derived by diagonalising Q_FF:

        f(t) = phi exp(Q_FF t) u = sum_k (phi V)_k e^{lam_k t} (V^-1 u)_k

    with u = (-Q_FF) 1. For a 2x2 sub-block of a generator the discriminant is
    (a-d)^2 + 4bc >= 0 (off-diagonals are non-negative), so the eigenvalues are
    always real.

    These 2|F| numbers ARE the 1-D histogram's entire information content. Two
    generators agreeing on them are indistinguishable to 1-D analysis.
    """
    Q = validate_generator(Q)
    phi = entry_distribution(Q, level_map, level)
    F, _, Q_FF, _, _, _ = _blocks(Q, level_map, level)
    u = (-Q_FF) @ np.ones(F.size)

    lam, V = np.linalg.eig(Q_FF)
    if np.max(np.abs(lam.imag)) > 1e-9:
        raise ValueError("complex eigenvalues in Q_FF; mixture form does not apply")
    lam = lam.real
    V = V.real

    if np.linalg.cond(V) > 1e10:
        raise ValueError("Q_FF is defective (repeated eigenvalues); mixture form "
                         "degenerates -- perturb the rates")

    coeffs = (phi @ V) * np.linalg.solve(V, u)
    order = np.argsort(-lam)
    return -lam[order], coeffs[order]


def _marginal_signature(Q, level_map) -> np.ndarray:
    """Both levels' mixture params, flattened. The 1-D-observable content of Q."""
    r1, c1 = mixture_params(Q, level_map, level=1)
    r0, c0 = mixture_params(Q, level_map, level=0)
    return np.concatenate([r1, c1, r0, c0])


class MatchResult:
    """Outcome of a marginal-matching fit.

    ``x`` is the raw parameter vector, retained so a constrained optimiser can
    warm-start from a known-feasible point instead of diverging from a cold one.
    """

    def __init__(self, Q, cost, signature_error, success, n_restarts_used,
                 rho=None, x=None, rank_ratio=None, best_rank_ratio=None):
        self.Q = Q
        self.cost = cost
        self.signature_error = signature_error
        self.success = success
        self.n_restarts_used = n_restarts_used
        self.rho = rho
        self.x = x
        self.rank_ratio = rank_ratio
        self.best_rank_ratio = best_rank_ratio

    def __repr__(self):
        rho = "None" if self.rho is None else f"{self.rho:.6f}"
        rr = "None" if self.rank_ratio is None else f"{self.rank_ratio:.3e}"
        return (f"MatchResult(success={self.success}, "
                f"max_signature_error={self.signature_error:.3e}, "
                f"restarts={self.n_restarts_used}, rho={rho}, "
                f"rank_ratio={rr})")


def match_marginals(
    Q_target,
    level_map=LEVELS_4,
    *,
    seed: int = 0,
    n_restarts: int = 40,
    atol: float = 1e-9,
) -> MatchResult:
    #Fit a bipartite generator to the target's dwell-time marginals.
    Q_target = validate_generator(Q_target)
    target = _marginal_signature(Q_target, level_map)

    def residual(log_rates):
        try:
            return _marginal_signature(bipartite_generator(np.exp(log_rates)),
                                       LEVELS_4) - target
        except (ValueError, np.linalg.LinAlgError):
            return np.full(target.size, 1e3)

    rng = np.random.default_rng(seed)
    best = None                          # lowest-cost candidate seen, any cost
    best_exact, best_exact_rho = None, -1.0   # largest |rho| among cost < atol**2
    for _ in range(n_restarts):
        x0 = np.log(rng.uniform(0.05, 12.0, size=8))
        sol = least_squares(residual, x0, method="trf",
                            bounds=(np.full(8, _LOG_LO), np.full(8, _LOG_HI)),
                            max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
        if sol.cost < atol ** 2:
            try:
                rho = abs(analytic_adjacent_moments(
                    bipartite_generator(np.exp(sol.x)), LEVELS_4, 1).rho)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if rho > best_exact_rho:
                best_exact, best_exact_rho = sol, rho

    chosen = best_exact if best_exact is not None else best
    Q = bipartite_generator(np.exp(chosen.x))
    err = float(np.max(np.abs(_marginal_signature(Q, LEVELS_4) - target)))
    ok = err < 1e-7
    rho = float(analytic_adjacent_moments(Q, LEVELS_4, 1).rho) if ok else None
    return MatchResult(Q, float(chosen.cost), err, ok, n_restarts, rho, x=chosen.x)


def _reversible_residual_factory(target):
    def residual(x):
        try:
            return _marginal_signature(
                bipartite_reversible_generator(*_pack(x)), LEVELS_4) - target
        except (ValueError, np.linalg.LinAlgError):
            return np.full(target.size, 1e3)
    return residual


def match_marginals_reversible(
    Q_target,
    level_map=LEVELS_4,
    *,
    seed: int = 0,
    n_restarts: int = 60,
    atol: float = 1e-9,
) -> MatchResult:
    #Fit a reversible bipartite generator to the target's dwell marginals.
    Q_target = validate_generator(Q_target)
    target = _marginal_signature(Q_target, level_map)
    residual = _reversible_residual_factory(target)

    rng = np.random.default_rng(seed)
    best, i = None, 0
    for i in range(n_restarts):
        x0 = np.concatenate([rng.normal(0.0, 1.5, 3),
                             np.log(rng.uniform(0.05, 8.0, 4))])
        sol = least_squares(residual, x0, method="trf", bounds=_bounds7(),
                            max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < atol ** 2:
            break

    Q = bipartite_reversible_generator(*_pack(best.x))
    err = float(np.max(np.abs(_marginal_signature(Q, LEVELS_4) - target)))
    ok = err < 1e-7
    rho = float(analytic_adjacent_moments(Q, LEVELS_4, 1).rho) if ok else None
    return MatchResult(Q, float(best.cost), err, ok, i + 1, rho, x=best.x)


def explore_reversible_manifold(
    Q_target,
    level_map=LEVELS_4,
    *,
    seed: int = 0,
    n_samples: int = 200,
    tol: float = 1e-7,
) -> list[MatchResult]:
    """Sample the reversible degenerate-partner manifold from many restarts.

    Each successful fit is a valid degenerate partner: same 1-D marginals as
    Q_target, reversible, two gateways. They differ in where they sit on the
    1-D solution manifold and therefore in rho. Reporting the SPREAD of rho is
    more honest than reporting whichever point one fit happened to reach.

    Returns successful MatchResults sorted by |rho| descending, each carrying
    its parameter vector ``x`` so a constrained optimiser can warm-start there.
    """
    Q_target = validate_generator(Q_target)
    target = _marginal_signature(Q_target, level_map)
    residual = _reversible_residual_factory(target)

    rng = np.random.default_rng(seed)
    out: list[MatchResult] = []
    for _ in range(n_samples):
        x0 = np.concatenate([rng.normal(0.0, 2.0, 3),
                             np.log(rng.uniform(0.02, 10.0, 4))])
        sol = least_squares(residual, x0, method="trf", bounds=_bounds7(),
                            max_nfev=8000)
        try:
            Q = bipartite_reversible_generator(*_pack(sol.x))
            err = float(np.max(np.abs(_marginal_signature(Q, LEVELS_4) - target)))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if err < tol:
            rho = float(analytic_adjacent_moments(Q, LEVELS_4, 1).rho)
            out.append(MatchResult(Q, float(sol.cost), err, True, 1, rho, x=sol.x))

    out.sort(key=lambda r: -abs(r.rho))
    return out


def extremal_correlation_reversible(
    Q_target,
    level_map=LEVELS_4,
    *,
    sense: str = "max",
    seed: int = 0,
    n_samples: int = 200,
) -> MatchResult:
    #Find a reversible marginal-matched partner with extreme |rho|.
    if sense not in ("max", "min"):
        raise ValueError("sense must be 'max' or 'min'")

    samples = explore_reversible_manifold(Q_target, level_map,
                                          seed=seed, n_samples=n_samples)
    if not samples:
        return MatchResult(None, np.inf, np.inf, False, n_samples)

    sign = -1.0 if sense == "max" else 1.0
    best = samples[0] if sense == "max" else samples[-1]

    target = _marginal_signature(validate_generator(Q_target), level_map)
    residual = _reversible_residual_factory(target)

    def objective(x):
        try:
            Q = bipartite_reversible_generator(*_pack(x))
            return sign * abs(analytic_adjacent_moments(Q, LEVELS_4, 1).rho)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0 if sense == "max" else 1e3

    lo, hi = _bounds7()
    starts = samples[: min(12, len(samples))] if sense == "max" \
        else samples[-min(12, len(samples)):]
    for start in starts:
        sol = minimize(objective, start.x, method="SLSQP",
                       bounds=list(zip(lo, hi)),
                       constraints=[{"type": "eq", "fun": residual}],
                       options={"maxiter": 500, "ftol": 1e-14})
        try:
            Q = bipartite_reversible_generator(*_pack(sol.x))
            err = float(np.max(np.abs(_marginal_signature(Q, LEVELS_4) - target)))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if err >= 1e-7:
            continue
        rho = float(analytic_adjacent_moments(Q, LEVELS_4, 1).rho)
        better = abs(rho) > abs(best.rho) if sense == "max" \
            else abs(rho) < abs(best.rho)
        if better:
            best = MatchResult(Q, float(sol.fun), err, True, n_samples, rho, x=sol.x)

    return best


def partner_with_rho(
    Q_target,
    rho_target: float,
    level_map=LEVELS_4,
    *,
    seed: int = 0,
    n_restarts: int = 80,
    atol: float = 1e-9,
    min_rank_ratio: float = 1e-6,
) -> MatchResult:
    #Find a reversible partner with matching marginals and prescribed rho.
    Q_target = validate_generator(Q_target)
    target = _marginal_signature(Q_target, level_map)

    def residual(x):
        try:
            Q = bipartite_reversible_generator(*_pack(x))
            marg = _marginal_signature(Q, LEVELS_4) - target
            rho = analytic_adjacent_moments(Q, LEVELS_4, 1).rho
            return np.append(marg, rho - rho_target)
        except (ValueError, np.linalg.LinAlgError):
            return np.full(target.size + 1, 1e3)

    rng = np.random.default_rng(seed)
    best, fallback, i = None, None, 0
    best_ratio = 0.0
    for i in range(n_restarts):
        x0 = np.concatenate([rng.normal(0.0, 2.0, 3),
                             np.log(rng.uniform(0.02, 10.0, 4))])
        sol = least_squares(residual, x0, method="trf", bounds=_bounds7(),
                            max_nfev=20000)
        if fallback is None or sol.cost < fallback.cost:
            fallback = sol
        try:
            ratio = gateway_rank_ratio(
                bipartite_reversible_generator(*_pack(sol.x)), LEVELS_4)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if sol.cost < atol ** 2:
            best_ratio = max(best_ratio, ratio)
        if ratio < min_rank_ratio:
            continue                      # rank-1 branch: rho = 0 structurally
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < atol ** 2:
            break

    sol = best if best is not None else fallback
    Q = bipartite_reversible_generator(*_pack(sol.x))
    err = float(np.max(np.abs(_marginal_signature(Q, LEVELS_4) - target)))
    rho = float(analytic_adjacent_moments(Q, LEVELS_4, 1).rho)
    ratio = gateway_rank_ratio(Q, LEVELS_4)
    ok = (best is not None and err < 1e-7
          and abs(rho - rho_target) < 1e-6 and ratio >= min_rank_ratio)
    return MatchResult(Q, float(sol.cost), err, ok, i + 1, rho, x=sol.x,
                       rank_ratio=ratio, best_rank_ratio=best_ratio)


def max_correlation_reversible(Q_target, level_map=LEVELS_4, **kw) -> MatchResult:
    """Best reversible degenerate partner: marginals matched, |rho| maximised."""
    return extremal_correlation_reversible(Q_target, level_map, sense="max", **kw)