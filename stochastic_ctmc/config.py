# Shared parameters for the CTMC, SDE, HH and RTN models.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CTMCConfig:
    # rates maps (i, j) -> rate in 1/s and is passed to n_state_generator.

    name: str
    n_states: int
    rates: dict[tuple[int, int], float]
    initial_state: int = 0


@dataclass(frozen=True)
class SSAConfig:
    t_max: float = 10.0
    seed: int = 0


@dataclass(frozen=True)
class EMConfig:
    dt: float = 1e-4
    t_max: float = 10.0
    n_seeds: int = 100
    confidence: float = 0.95


@dataclass(frozen=True)
class HHConfig:
    # Standard squid-axon values (mS/cm^2, mV, uF/cm^2). Channel noise scales as
    # 1/sqrt(N), so the deterministic model is recovered as n_channels -> inf.

    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_leak: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_leak: float = -54.4
    n_na_channels: int = 6000
    n_k_channels: int = 1800
    i_ext: float = 10.0


@dataclass(frozen=True)
class RTNConfig:
    # Single trap as a 2-state CTMC (empty <-> filled).

    capture_rate: float = 1.0e3   # empty -> filled  (1/s)
    emission_rate: float = 2.0e3  # filled -> empty  (1/s)
    delta_vth: float = 5.0e-3     # threshold-voltage shift when trap filled (V)


# Generators shared by the Gillespie, EM, HH and RTN code

TWO_STATE = CTMCConfig(
    name="two_state_channel",
    n_states=2,
    rates={(0, 1): 1.0, (1, 0): 2.0},
    initial_state=0,
)

THREE_STATE = CTMCConfig(
    name="three_state_reversible",
    n_states=3,
    # reversible 3-cycle, so detailed balance holds
    rates={
        (0, 1): 1.0, (1, 0): 1.0,
        (1, 2): 2.0, (2, 1): 2.0,
        (0, 2): 0.5, (2, 0): 0.5,
    },
    initial_state=0,
)


@dataclass(frozen=True)
class Config:
    ctmc: CTMCConfig = TWO_STATE
    ssa: SSAConfig = field(default_factory=SSAConfig)
    em: EMConfig = field(default_factory=EMConfig)
    hh: HHConfig = field(default_factory=HHConfig)
    rtn: RTNConfig = field(default_factory=RTNConfig)


DEFAULT = Config()
