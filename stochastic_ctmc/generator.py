# CTMC generator matrices.
#
# Q[i, j] >= 0 for i != j is the transition rate from state i to state j
# (1/time) and Q[i, i] = -sum_{j != i} Q[i, j], so every row sums to zero.
# The total exit rate of state i is -Q[i, i].

from __future__ import annotations

import numpy as np

__all__ = [
    "GeneratorError",
    "two_state_generator",
    "n_state_generator",
    "validate_generator",
    "stationary_distribution",
    "is_detailed_balance",
]


class GeneratorError(ValueError):
    """Raised when a matrix is not a valid CTMC generator."""


def two_state_generator(k01: float, k10: float) -> np.ndarray:
    # k01 is the 0 -> 1 rate (closed -> open, empty -> filled), k10 the 1 -> 0 rate.
    if k01 < 0 or k10 < 0:
        raise GeneratorError(f"rates must be non-negative, got k01={k01}, k10={k10}")
    Q = np.array([[-k01, k01], [k10, -k10]], dtype=float)
    return Q


def n_state_generator(rates: dict[tuple[int, int], float], n_states: int) -> np.ndarray:
    # rates[(i, j)] is the i -> j rate for i != j. Missing pairs are 0 and the
    # diagonal is filled in so each row sums to zero.
    if n_states < 1:
        raise GeneratorError(f"n_states must be >= 1, got {n_states}")

    Q = np.zeros((n_states, n_states), dtype=float)
    for (i, j), rate in rates.items():
        if not (0 <= i < n_states and 0 <= j < n_states):
            raise GeneratorError(
                f"transition ({i}, {j}) is out of bounds for {n_states} states"
            )
        if i == j:
            raise GeneratorError(
                f"diagonal entry ({i}, {j}) may not be set directly; "
                "it is derived from the off-diagonal rates"
            )
        if rate < 0:
            raise GeneratorError(f"rate for ({i}, {j}) must be non-negative, got {rate}")
        Q[i, j] = rate

    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n_states)] = -Q.sum(axis=1)
    return Q


def validate_generator(Q: np.ndarray, *, atol: float = 1e-9) -> np.ndarray:
    """Check that Q is square, finite, has non-negative off-diagonals and zero row sums.

    Returns Q as a float array, raises GeneratorError otherwise.

    atol is relative to the size of the rates, not an absolute cutoff. Q -> kQ is
    the same system in a different unit of time, and a numerically assembled
    generator carries row-sum residuals of order eps * max|Q|, so a fixed
    threshold would reject valid generators at fast rates and accept bad ones at
    slow rates. Each row sum is compared against atol * max|row| and the sign
    check against atol * max|Q|. An identically zero row (absorbing state) still
    has to sum to exactly zero.
    """
    Q = np.asarray(Q, dtype=float)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise GeneratorError(f"generator must be a square 2-D matrix, got shape {Q.shape}")

    if not np.all(np.isfinite(Q)):
        raise GeneratorError("generator contains non-finite entries")

    scale = float(np.abs(Q).max()) if Q.size else 0.0

    off_diagonal = Q[~np.eye(Q.shape[0], dtype=bool)]
    if np.any(off_diagonal < -atol * scale):
        raise GeneratorError("off-diagonal rates must be non-negative")

    row_sums = Q.sum(axis=1)
    limit = atol * np.abs(Q).max(axis=1)
    if np.any(np.abs(row_sums) > limit):
        bad = int(np.argmax(np.abs(row_sums) - limit))
        raise GeneratorError(
            f"row {bad} sums to {row_sums[bad]:.3e}, expected 0 within "
            f"atol={atol:.1e} relative to its largest rate "
            f"{np.abs(Q).max(axis=1)[bad]:.3e}"
        )

    return Q


def stationary_distribution(Q: np.ndarray, *, atol: float = 1e-9) -> np.ndarray:
    """Stationary distribution ``pi`` of the CTMC, i.e. ``pi @ Q = 0``, ``sum(pi) = 1``.

    Solved as the null space of ``Q^T`` (the left null vector of ``Q``). For a
    generator with a single recurrent class this is unique. With absorbing
    states or multiple classes the null space has dimension > 1 and the result
    is one (not necessarily unique) valid stationary vector.
    """
    Q = validate_generator(Q, atol=atol)
    n = Q.shape[0]

    # Q^T pi = 0 with the normalisation sum(pi) = 1 appended, solved by least squares.
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Clean up tiny negatives from least-squares round-off and renormalise.
    pi = np.where(np.abs(pi) < atol, 0.0, pi)
    if np.any(pi < 0):
        raise GeneratorError(
            "computed stationary vector has negative entries; the chain may have "
            "multiple recurrent classes (stationary distribution is not unique)"
        )
    total = pi.sum()
    if total <= 0:
        raise GeneratorError("degenerate stationary distribution (sums to <= 0)")
    return pi / total


def is_detailed_balance(Q: np.ndarray, pi: np.ndarray | None = None, *, atol: float = 1e-9) -> bool:
    """Test the detailed-balance (reversibility) condition ``pi_i Q_ij = pi_j Q_ji``.

    If ``pi`` is not supplied it is computed via :func:`stationary_distribution`.
    Any 2-state chain with both rates positive satisfies detailed balance; the
    check matters for ``N >= 3`` where cyclic flux can break reversibility.
    """
    Q = validate_generator(Q, atol=atol)
    if pi is None:
        pi = stationary_distribution(Q, atol=atol)
    pi = np.asarray(pi, dtype=float)

    flux = pi[:, None] * Q  # flux[i, j] = pi_i * Q_ij
    return bool(np.allclose(flux, flux.T, atol=atol))
