from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt

from .ensemble import EnsembleSummary

__all__ = [
    "plot_trajectory_comparison",
    "plot_error_band",
    "plot_convergence",
    "save_figure",
]


def plot_trajectory_comparison(
    series: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str = "Trajectory comparison",
    xlabel: str = "time",
    ylabel: str = "state / occupancy",
):

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, t, y in series:
        ax.plot(np.asarray(t), np.asarray(y), lw=1.2, alpha=0.85, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_error_band(
    summary: EnsembleSummary,
    *,
    reference: np.ndarray | None = None,
    reference_label: str = "deterministic",
    title: str = "Ensemble mean +/- CI",
    xlabel: str = "time",
    ylabel: str = "value",
):

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        summary.times, summary.ci_low, summary.ci_high,
        alpha=0.25, label=f"{int(summary.confidence * 100)}% CI (n={summary.n_seeds})",
    )
    ax.plot(summary.times, summary.mean, lw=1.5, label="ensemble mean")
    if reference is not None:
        ax.plot(summary.times, reference, "k--", lw=1.2, label=reference_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_convergence(
    param: np.ndarray,
    error: np.ndarray,
    *,
    reference_slope: float | None = 0.5,
    param_label: str = r"$\Delta t$",
    title: str = "Convergence",
):

    param = np.asarray(param, dtype=float)
    error = np.asarray(error, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(param, error, "o-", lw=1.4, label="observed error")
    if reference_slope is not None and np.all(error > 0):
        anchor = error[np.argmax(param)] / (param.max() ** reference_slope)
        guide = anchor * param ** reference_slope
        ax.loglog(param, guide, "k--", lw=1.0, label=f"slope {reference_slope:g}")
    ax.set_xlabel(param_label)
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str, *, dpi: int = 150) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
