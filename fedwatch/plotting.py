"""Visualisation layer: probability heatmap, curve chart, percentile fan."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_probability_heatmap(mat: pd.DataFrame, ax=None, title: str | None = None,
                             vmin: float = 0.0, vmax: float | None = None):
    """Heatmap of a probability matrix (rows = horizons, cols = 25bp buckets)."""
    ax = ax or plt.gca()
    data = mat * 100                            # rows = horizons, cols = buckets
    vmax = vmax or max(1.0, float(np.nanmax(data.values)))
    step = max(1, data.shape[1] // 12)
    im = ax.imshow(data.values, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                   origin="lower")
    ax.set_xticks(range(data.shape[1])[::step])
    ax.set_xticklabels([f"{float(c):.0f}" for c in data.columns[::step]],
                       rotation=45, fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(list(data.index), fontsize=8)
    ax.set_xlabel("target / rate bucket (bps, midpoints)")
    ax.set_title(title or "Implied rate probabilities (%)")
    return im


def plot_curve(res: dict, ax=None):
    """Zero curve + instantaneous forwards + node marks."""
    ax = ax or plt.gca()
    curve, asof = res["curve"], res["asof"]
    ts = np.linspace(0.05, 30, 400)
    zeros = [curve.zero(res["asof"] + pd.Timedelta(days=round(365 * t))) * 100 for t in ts]
    fwd = curve.inst_fwd_grid(n=400) * 100
    ax.plot(ts, zeros, label="zero curve", lw=1.8)
    ax.plot(fwd.index, fwd.values, label="inst. forward", lw=1.0, alpha=0.8)
    node_t = [(d - asof).days / 365.0 for d in curve.dates[1:]]
    node_z = [curve.zero(d) * 100 for d in curve.dates[1:]]
    ax.scatter(node_t, node_z, s=12, color="k", zorder=5, label="bootstrap nodes")
    eris = res.get("eris_eval")
    if eris is not None and len(eris):
        ax.scatter(eris.tenor_years, eris.par_rate_bps / 100, marker="x",
                   color="crimson", zorder=6, label="Eris par (implied)")
    ax.set_xlabel("tenor (years)")
    ax.set_ylabel("rate (%)")
    ax.set_title(f"Unified 0-30Y SOFR curve — as of {asof.date()}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_hw_fan(fan: pd.DataFrame, curve, asof, ax=None):
    """Hull-White percentile fan vs today's zero curve."""
    ax = ax or plt.gca()
    ts = fan.index.values.astype(float)
    ax.fill_between(ts, fan.p5, fan.p95, alpha=0.18, label="5-95%")
    ax.fill_between(ts, fan.p25, fan.p75, alpha=0.30, label="25-75%")
    ax.plot(ts, fan.p50, lw=1.8, label="HW median")
    zc = [curve.zero(asof + pd.Timedelta(days=round(365.25 * t))) * 1e4 for t in ts]
    ax.plot(ts, zc, "k--", lw=1.0, label="today's zero")
    ax.set_xlabel("horizon (years)")
    ax.set_ylabel("zero rate (bps)")
    ax.set_title("Hull-White simulated zero-rate distribution (model-based)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_option_pdf(pdf: pd.Series, fit: dict, ax=None):
    """BL density + fitted smile context for one option expiry."""
    ax = ax or plt.gca()
    from .sabr_bl import sabr_normal_vol
    ax.plot(pdf.index, pdf.values * 100, lw=1.8,
            label=f"BL density (T={fit['T']:.2f}Y)")
    ax.axvline(fit["F_bps"], color="k", ls="--", lw=0.8, label="forward")
    ax.set_xlabel("rate strike (bps)")
    ax.set_ylabel("density per bp (%)")
    ax.set_title("Option-implied PDF (Breeden-Litzenberger)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
