from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.module.regime_storage import RegimeStorage


def viz_seq_with_band(
    raw_data: np.ndarray,
    pred_data: np.ndarray,
    assignment: dict,
    interval: Tuple[int, int],
    n_rgm: int,
    title: str,
    hlines_alpha: float = 0.6,
    raw_alpha: float = 0.6,
    vspan_alpha: float = 0.8,
    dark_mode: bool = False,
    show: bool = True,
) -> Figure:
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1])
    axes = {"seq": fig.add_subplot(gs[0]), "band": fig.add_subplot(gs[1])}
    t_arange = np.arange(*interval)
    raw_clr = "lightgray" if dark_mode else "gray"
    COLORS = [i["color"] for i in plt.rcParams["axes.prop_cycle"]]

    axes["seq"].set_ylabel("Value")
    axes["seq"].set_xlim(*interval)
    axes["seq"].set_ylim(raw_data.min() * 1.1, raw_data.max() * 1.1)
    axes["band"].set_ylabel("Regime")
    axes["band"].set_xlim(*interval)
    axes["band"].set_ylim(0, n_rgm)
    axes["band"].set_yticks([i + 0.5 for i in range(n_rgm)])
    axes["band"].set_yticklabels([f"#{i+1}" for i in range(n_rgm)])
    fig.suptitle(title)

    axes["band"].hlines([i for i in range(n_rgm + 1)], *interval, color="gray", alpha=hlines_alpha)
    axes["seq"].plot(
        t_arange,
        raw_data[:, interval[0] : interval[1]].T,
        linestyle="dashed",
        color=raw_clr,
        alpha=raw_alpha,
    )
    axes["seq"].plot(t_arange, pred_data[:, : len(t_arange)].T)
    for i, (_interval, color) in enumerate(zip(assignment.values(), COLORS)):
        for st, en in _interval:
            axes["band"].axvspan(st, en, i / n_rgm, (i + 1) / n_rgm, color=color, alpha=vspan_alpha)
    fig.tight_layout()
    fig.align_labels()
    if show:
        plt.show()

    return fig


def viz_eigenvalues_and_dynamics(
    X: np.ndarray,
    regime_storage: RegimeStorage,
    d: int,
    length: int = 200,
    dark_mode: bool = False,
    show: bool = True,
) -> Figure:
    n_rgm = len(regime_storage)
    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(2, n_rgm, i + 1) for i in range(2 * n_rgm)]
    circle_clr = "lightgray" if dark_mode else "gray"

    for regime in regime_storage:
        _ax = axes[regime.idx]
        _ax.set_title(f"Regime {regime.idx + 1}")

        _ax.plot(*circle(), c=circle_clr, linestyle="dashed")
        Lamb, W = np.linalg.eig(regime.A_tilde)
        Phi = regime.U @ W
        for lamb in Lamb:
            _ax.scatter(
                np.real(lamb),
                np.imag(lamb),
            )
            _ax.set_aspect("equal")
            _ax.set_xlabel(r"$\it{Re}\,\lambda$")
            _ax.set_ylabel(r"$\it{Im}\,\lambda$")

    h = X.shape[0] // d
    t_arange = np.arange(length)
    for regime in regime_storage:
        _ax = axes[regime.idx + n_rgm]
        _ax.set_title(f"Model {regime.idx + 1}")

        st = regime.intervals[0][0]
        X0 = X[:, regime.intervals[0][0]]
        Lamb, W = np.linalg.eig(regime.A_tilde)
        Phi = regime.U @ W
        b = np.linalg.pinv(Phi) @ X0
        pred = np.zeros([X0.shape[0], len(t_arange)], dtype="complex")
        for t in t_arange:
            pred[:, t] = Phi @ np.diag(Lamb**t) @ b
        _ax.plot(t_arange + st, np.real(pred[[i * h - 1 for i in range(1, d + 1)]].T))
    fig.tight_layout()
    if show:
        plt.show()

    return fig


def viz_modes(
    X: np.ndarray,
    regime_storage: RegimeStorage,
    length: int = 100,
    show: bool = True,
) -> Figure:
    n_rgm = len(regime_storage)
    n_modes = [len(np.linalg.eig(regime.A_tilde)[0]) for regime in regime_storage]
    t_arange = np.arange(length)

    fig = plt.figure(figsize=(15, 15))
    axes = [fig.add_subplot(max(n_modes), n_rgm, i + 1) for i in range(max(n_modes) * n_rgm)]
    for i, regime in enumerate(regime_storage):
        st = regime.intervals[0][0]
        X0 = X[:, regime.intervals[0][0]]
        modes = regime.get_modes(X0, length)
        for j in range(n_modes[i]):
            axes[n_rgm * j + regime.idx].plot(t_arange + st, np.real(modes[j].T))
            if not np.allclose(np.imag(modes[j].T), 0):
                axes[n_rgm * j + regime.idx].plot(t_arange + st, np.imag(modes[j].T))
    fig.tight_layout()
    if show:
        plt.show()

    return fig


def viz_forecasting(
    raw_data: np.ndarray,
    pred_data: np.ndarray,
    pred_st: tuple,
    show: bool = True,
) -> Figure:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(raw_data.T, color="darkgray", linestyle="dashed")
    ax.plot(np.arange(len(pred_data.T)) + pred_st, pred_data.T)
    ax.axvline(x=pred_st, color="gray")
    fig.tight_layout()
    fig.align_labels()
    if show:
        plt.show()

    return fig


def viz_snapshot(
    raw_data: np.ndarray,
    pred: np.ndarray,
    h: int,
    tm: int,
    tc: int,
    tf: int,
    te: int,
    err: float,
    show: bool = True,
) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(tm, tc, color="skyblue", alpha=0.4)
    ax.plot(
        np.arange(tm, te),
        raw_data[:, tm:te].T,
        alpha=0.4,
        linestyle="dashed",
        linewidth=7,
    )
    ax.set_prop_cycle(None)
    ax.axvspan(tf, te, color="lightcoral", alpha=0.4)
    ax.plot(
        np.arange(tm + h - 1, te),
        pred,
        linewidth=2.5,
    )
    ax.set_ylim(-4, 4)
    ax.set_title(f"RMSE: {err}")

    plt.rcParams["mathtext.fontset"] = "stix"
    ax.text(0.05, 0.92, r"$\mathcal{X}_c$", color="skyblue", fontsize=25, fontweight="bold", transform=ax.transAxes)
    ax.text(0.845, 0.92, r"$\mathcal{X}_f$", color="lightcoral", fontsize=25, fontweight="bold", transform=ax.transAxes)
    fig.tight_layout()
    if show:
        plt.show()

    return fig


def circle(radius: float = 1.0) -> Tuple[list, list]:
    x, y = [], []
    for angle in np.linspace(-180, 180, 360):
        x.append(radius * np.sin(np.radians(angle)))
        y.append(radius * np.cos(np.radians(angle)))
    return x, y
