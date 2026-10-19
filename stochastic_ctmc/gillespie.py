# Gillespie direct-method simulation of a CTMC with generator Q.
#
# In state i the total exit rate is a0 = -Q[i, i]. The waiting time is
# tau = -ln(u)/a0 and the next state j != i is chosen with probability
# Q[i, j] / a0. A state with a0 == 0 is absorbing and ends the trajectory.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generator import validate_generator

__all__ = ["SSAResult", "gillespie_ssa"]


@dataclass
class SSAResult:
    # times[k] is the time at which the chain entered states[k]; it stays there on
    # [times[k], times[k+1]) and past times[-1] up to t_max. absorbed is True if the
    # run stopped early in an absorbing state.

    times: np.ndarray
    states: np.ndarray
    t_max: float
    absorbed: bool

    def state_at(self, t: float) -> int:
        """State occupied at continuous time ``t`` (right-continuous step function)."""
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        idx = max(0, idx)
        return int(self.states[idx])

    def occupancy_fractions(self, n_states: int) -> np.ndarray:
        """Time-weighted fraction of [0, t_max] spent in each state."""
        dwell = np.zeros(n_states, dtype=float)
        edges = np.concatenate([self.times, [self.t_max]])
        durations = np.diff(edges)
        for s, d in zip(self.states, durations):
            dwell[int(s)] += d
        total = dwell.sum()
        if total <= 0:
            raise ValueError("trajectory has zero total duration")
        return dwell / total


def gillespie_ssa(
    Q: np.ndarray,
    initial_state: int,
    t_max: float,
    *,
    rng: np.random.Generator | int | None = None,
    max_steps: int | None = None,
    validate: bool = True,
) -> SSAResult:
    """One CTMC trajectory by Gillespie's direct method.

    rng may be a Generator, an int seed or None. max_steps caps the number of
    transitions. validate=False skips validate_generator for callers that have
    already checked Q.
    """
    if validate:
        Q = validate_generator(Q)
    else:
        Q = np.asarray(Q, dtype=float)

    n_states = Q.shape[0]
    if not (0 <= initial_state < n_states):
        raise ValueError(f"initial_state {initial_state} out of range for {n_states} states")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    if isinstance(rng, np.random.Generator):
        generator = rng
    else:
        generator = np.random.default_rng(rng)

    exit_rates = -np.diag(Q)  # a0 for each state
    off = Q.copy()
    np.fill_diagonal(off, 0.0)

    times = [0.0]
    states = [int(initial_state)]
    t = 0.0
    state = int(initial_state)
    absorbed = False
    step = 0

    while t < t_max:
        a0 = exit_rates[state]
        if a0 <= 0.0:
            # absorbing state
            absorbed = True
            break

        tau = generator.exponential(1.0 / a0)
        t += tau
        if t >= t_max:
            break

        probs = off[state] / a0
        state = int(generator.choice(n_states, p=probs))

        times.append(t)
        states.append(state)

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    return SSAResult(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=int),
        t_max=float(t_max),
        absorbed=absorbed,
    )
