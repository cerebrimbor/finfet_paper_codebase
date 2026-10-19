# Stochastic Hodgkin-Huxley with channel noise: subunit Langevin form of
# Fox & Lu (1994), Phys. Rev. E 49, 3421.
#
# The equations are the standard subunit form as it is usually reproduced. They
# have not been checked line by line against the original paper (FOX_LU_STATUS).
#
#     C dV/dt = I_ext - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L)
#
# Every gating variable y in {m, h, n} is a 2-state closed <-> open CTMC with
# rates alpha_y(V) (closed -> open) and beta_y(V) (open -> closed). Its diffusion
# approximation is
#
#     dy = [alpha_y (1 - y) - beta_y y] dt
#          + sqrt( (alpha_y (1 - y) + beta_y y) / N ) dW_y
#
# with N = N_Na for m, h and N = N_K for n. This is channel_langevin_terms from
# sde.py evaluated at the instantaneous voltage. As N -> inf the noise vanishes
# and the deterministic HH equations are recovered.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HHConfig

__all__ = [
    "FOX_LU_STATUS",
    "HHRates",
    "gate_rates",
    "deterministic_hh_step",
    "simulate_hh",
    "HHResult",
    "steady_state_gate",
]

FOX_LU_STATUS = (
    "BLOCKED: using the standard Fox & Lu 1994 subunit Langevin form; "
    "line-by-line transcription from the sourced PDF is still outstanding."
)


# Voltage-dependent rate functions (classic HH, modern mV convention)
# V in mV; rates in 1/ms. Singularities at V=-40 (alpha_m) and V=-55 (alpha_n)
# are removable and handled by their analytic limits.

def _alpha_m(V):
    x = np.asarray(V, dtype=float) + 40.0
    small = np.abs(x) < 1e-7
    xs = np.where(small, 1.0, x)  # avoid 0/0 in the masked-out branch
    return np.where(small, 1.0, 0.1 * xs / (1.0 - np.exp(-xs / 10.0)))


def _beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def _alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def _beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def _alpha_n(V):
    x = np.asarray(V, dtype=float) + 55.0
    small = np.abs(x) < 1e-7
    xs = np.where(small, 1.0, x)  # avoid 0/0 in the masked-out branch
    return np.where(small, 0.1, 0.01 * xs / (1.0 - np.exp(-xs / 10.0)))


def _beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


@dataclass
class HHRates:
    # rates in 1/ms

    alpha_m: float
    beta_m: float
    alpha_h: float
    beta_h: float
    alpha_n: float
    beta_n: float


def gate_rates(V: float) -> HHRates:
    return HHRates(
        alpha_m=float(_alpha_m(V)), beta_m=float(_beta_m(V)),
        alpha_h=float(_alpha_h(V)), beta_h=float(_beta_h(V)),
        alpha_n=float(_alpha_n(V)), beta_n=float(_beta_n(V)),
    )


def steady_state_gate(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


@dataclass
class HHResult:
    t: np.ndarray       # ms
    V: np.ndarray       # mV
    m: np.ndarray
    h: np.ndarray
    n: np.ndarray


def _ionic_currents(V, m, h, n, cfg: HHConfig):
    i_na = cfg.g_na * m**3 * h * (V - cfg.e_na)
    i_k = cfg.g_k * n**4 * (V - cfg.e_k)
    i_leak = cfg.g_leak * (V - cfg.e_leak)
    return i_na, i_k, i_leak


def deterministic_hh_step(V, m, h, n, dt, cfg: HHConfig):
    """One explicit Euler step of the deterministic HH equations (dt in ms)."""
    r = gate_rates(V)
    i_na, i_k, i_leak = _ionic_currents(V, m, h, n, cfg)
    dV = (cfg.i_ext - i_na - i_k - i_leak) / cfg.c_m
    dm = r.alpha_m * (1 - m) - r.beta_m * m
    dh = r.alpha_h * (1 - h) - r.beta_h * h
    dn = r.alpha_n * (1 - n) - r.beta_n * n
    return V + dt * dV, m + dt * dm, h + dt * dh, n + dt * dn


def simulate_hh(
    cfg: HHConfig,
    t_max: float,
    dt: float = 0.01,
    *,
    V0: float = -65.0,
    stochastic: bool = True,
    rng: np.random.Generator | int | None = None,
    clamp_V: float | None = None,
) -> HHResult:
    """Integrate the (stochastic) Hodgkin-Huxley model with Euler-Maruyama.

    t_max and dt are in ms, V0 in mV. Gates start at their steady state for V0.
    stochastic=False integrates the deterministic equations (the N -> inf limit).
    clamp_V holds the voltage fixed so a single gate can be compared with the
    2-state CTMC at a known rate pair.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n_steps = int(np.ceil(t_max / dt))
    sqrt_dt = np.sqrt(dt)

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    V = np.empty(n_steps + 1)
    m = np.empty(n_steps + 1)
    h = np.empty(n_steps + 1)
    n = np.empty(n_steps + 1)

    V[0] = clamp_V if clamp_V is not None else V0
    r0 = gate_rates(V[0])
    m[0] = steady_state_gate(r0.alpha_m, r0.beta_m)
    h[0] = steady_state_gate(r0.alpha_h, r0.beta_h)
    n[0] = steady_state_gate(r0.alpha_n, r0.beta_n)

    def gate_step(y, alpha, beta, N):
        drift = alpha * (1 - y) - beta * y
        if stochastic:
            var = max((alpha * (1 - y) + beta * y) / N, 0.0)
            noise = np.sqrt(var) * sqrt_dt * generator.standard_normal()
        else:
            noise = 0.0
        return min(max(y + dt * drift + noise, 0.0), 1.0)

    for i in range(n_steps):
        r = gate_rates(V[i])
        m[i + 1] = gate_step(m[i], r.alpha_m, r.beta_m, cfg.n_na_channels)
        h[i + 1] = gate_step(h[i], r.alpha_h, r.beta_h, cfg.n_na_channels)
        n[i + 1] = gate_step(n[i], r.alpha_n, r.beta_n, cfg.n_k_channels)

        if clamp_V is not None:
            V[i + 1] = clamp_V
        else:
            i_na, i_k, i_leak = _ionic_currents(V[i], m[i], h[i], n[i], cfg)
            dV = (cfg.i_ext - i_na - i_k - i_leak) / cfg.c_m
            V[i + 1] = V[i] + dt * dV

    return HHResult(t=t, V=V, m=m, h=h, n=n)
