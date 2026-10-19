# Single-trap random-telegraph noise (RTN) as a 2-state CTMC.
#
# State 0 = empty, state 1 = filled. Carriers are captured (empty -> filled) at
# rate k_c and emitted (filled -> empty) at rate k_e, so
#
#     Q_RTN = [[-k_c,  k_c],
#              [ k_e, -k_e]]
#
# which is two_state_generator(k_c, k_e). A filled trap shifts the threshold
# voltage by delta_Vth.
#
# No circuit simulator is called here. DVTSHIFT in the BSIM-CMG card acts as a
# rigid gate-voltage offset: re-extracting the shift between the baseline and
# trap Id-Vg sweeps at matched drain current gives 5.000000 mV on 200 of the 201
# swept points, spread 2 nV. The excluded point is Vg = 0, where the trap-curve
# current lies below every current in the baseline sweep and could only be
# matched by extrapolation. See device_provenance/ (ngspice 46 + OSDI BSIMCMG).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RTNConfig
from .generator import two_state_generator, stationary_distribution
from .gillespie import gillespie_ssa, SSAResult

__all__ = [
    "rtn_generator",
    "DomainMap",
    "STRUCTURAL_IDENTITY",
    "simulate_rtn",
    "RTNTrace",
    "rtn_stationary_filled",
]


def rtn_generator(capture_rate: float, emission_rate: float) -> np.ndarray:
    # State 0 = empty, state 1 = filled.
    return two_state_generator(capture_rate, emission_rate)


@dataclass(frozen=True)
class DomainMap:
    # How one physical system maps onto the 2-state CTMC: a is the 0->1 rate,
    # b the 1->0 rate.

    domain: str
    state0: str
    state1: str
    a_meaning: str   # what the 0->1 rate is called in this domain
    b_meaning: str   # what the 1->0 rate is called in this domain
    observable: str  # what the state modulates in the measured signal


STRUCTURAL_IDENTITY: tuple[DomainMap, ...] = (
    DomainMap(
        domain="Generic CTMC",
        state0="state 0", state1="state 1",
        a_meaning="rate k01", b_meaning="rate k10",
        observable="occupancy",
    ),
    DomainMap(
        domain="Ion channel gate (HH)",
        state0="closed", state1="open",
        a_meaning="alpha(V)", b_meaning="beta(V)",
        observable="gating variable / conductance",
    ),
    DomainMap(
        domain="Channel population",
        state0="closed fraction", state1="open fraction",
        a_meaning="opening rate a", b_meaning="closing rate b",
        observable="open fraction x",
    ),
    DomainMap(
        domain="FinFET oxide trap (RTN)",
        state0="empty", state1="filled",
        a_meaning="capture rate k_c", b_meaning="emission rate k_e",
        observable="threshold-voltage shift delta_Vth",
    ),
)


@dataclass
class RTNTrace:
    ssa: SSAResult
    delta_vth: float

    def vth_signal(self, t: np.ndarray) -> np.ndarray:
        # 0 when the trap is empty, delta_vth when filled
        return np.array([self.ssa.state_at(float(ti)) * self.delta_vth for ti in t])

    def stationary_filled_fraction(self) -> float:
        return float(self.ssa.occupancy_fractions(2)[1])


def simulate_rtn(
    cfg: RTNConfig,
    t_max: float,
    *,
    rng: np.random.Generator | int | None = None,
    initial_state: int = 0,
) -> RTNTrace:
    Q = rtn_generator(cfg.capture_rate, cfg.emission_rate)
    ssa = gillespie_ssa(Q, initial_state=initial_state, t_max=t_max, rng=rng)
    return RTNTrace(ssa=ssa, delta_vth=cfg.delta_vth)


def rtn_stationary_filled(cfg: RTNConfig) -> float:
    # k_c / (k_c + k_e)
    Q = rtn_generator(cfg.capture_rate, cfg.emission_rate)
    return float(stationary_distribution(Q)[1])
