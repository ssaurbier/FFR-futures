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


# ---------------------------------------------------------------- P4 surface
def plot_surface_pane(surface, asof, ax=None):
    """One pane of the 3D surface: horizon x bucket heatmap at an as-of."""
    from .surface import slice_asof
    ax = ax or plt.gca()
    pane = slice_asof(surface, asof)                  # bucket x horizon
    data = (pane.T * 100)
    im = ax.imshow(data.values, aspect="auto", cmap="magma", origin="lower",
                   vmin=0, vmax=max(1.0, float(np.nanmax(data.values))))
    n = data.shape[1]                              # buckets (columns)
    step = max(1, n // 10)
    ax.set_xticks(range(n)[::step])
    ax.set_xticklabels([f"{float(c):.0f}" for c in data.columns[::step]],
                       rotation=45, fontsize=7)
    m = data.shape[0]                              # horizons (rows)
    astep = max(1, m // 18)
    ax.set_yticks(range(m)[::astep])
    ax.set_yticklabels([pd.Timestamp(c).strftime("%Y-%m") for c in data.index[::astep]],
                       fontsize=7)
    ax.set_xlabel("rate bucket (bps mid)")
    ax.set_ylabel("horizon (meeting / option expiry)")
    ax.set_title(f"Probability surface — {pd.Timestamp(asof).date()}")
    return im


def plot_horizon_evolution(mat, title="Distribution evolution", ax=None):
    """THE 3D slice: as-of time (y) x rate buckets (x), colour = probability."""
    ax = ax or plt.gca()
    data = (mat * 100)
    im = ax.imshow(data.values, aspect="auto", cmap="magma", origin="lower",
                   vmin=0, vmax=max(1.0, float(np.nanmax(data.values))))
    n = data.shape[1]
    step = max(1, n // 14)
    ax.set_xticks(range(n)[::step])
    ax.set_xticklabels(list(data.columns[::step]), rotation=45, fontsize=7)
    m = data.shape[0]
    astep = max(1, m // 10)
    idx = mat.index
    ax.set_yticks(range(m)[::astep])
    ax.set_yticklabels([pd.Timestamp(t).strftime("%Y-%m") for t in idx[::astep]],
                       fontsize=7)
    ax.set_xlabel("rate bucket (bps mid)")
    ax.set_ylabel("as-of date")
    ax.set_title(title)
    return im


def animate_density(mat, path=None, title="Density evolution", fps=8):
    """GIF animation of one horizon's distribution morphing through time."""
    from .config import REPO_ROOT
    path = path or str(REPO_ROOT / "notebooks" / "density_evolution.gif")
    from matplotlib.animation import FuncAnimation, PillowWriter
    cols = mat.columns.values.astype(float)
    vals = mat.values
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def draw(i):
        ax.clear()
        ax.fill_between(cols, vals[i] * 100, color="crimson", alpha=0.35)
        ax.plot(cols, vals[i] * 100, color="crimson", lw=1.8)
        m = float(np.dot(cols, vals[i]))
        ax.axvline(m, color="k", ls="--", lw=0.8)
        ax.set_xlim(cols.min(), cols.max())
        ax.set_ylim(0, max(1.0, float(np.nanmax(vals)) * 100) * 1.05)
        ax.set_xlabel("rate bucket (bps mid)")
        ax.set_ylabel("probability (%)")
        ax.set_title(f"{title} — {pd.Timestamp(mat.index[i]).date()}")

    anim = FuncAnimation(fig, draw, frames=len(mat), interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_mean_band(surface, layer, tenor=None, ax=None, nsd=2.0):
    """Mean +- nsd*sd of one horizon family through history."""
    from .surface import horizon_series
    ax = ax or plt.gca()
    mat = horizon_series(surface, layer, tenor)
    cols = mat.columns.values.astype(float)
    p = mat.values
    mean = p @ cols
    sd = np.sqrt(((cols[None, :] - mean[:, None]) ** 2 * p).sum(axis=1))
    t = mat.index
    ax.fill_between(t, mean - nsd * sd, mean + nsd * sd, alpha=0.25,
                    label=f"mean +- {nsd:g}sd")
    ax.plot(t, mean, lw=1.6, color="crimson", label="mean")
    ax.set_ylabel("rate (bps)")
    lab = f"{layer}" + (f" {tenor:.0f}Y" if tenor else "")
    ax.set_title(f"Distribution centre through time — {lab}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------- plotly 3D
def plotly_forward_surface(curve_hist, value: str = "zero_bp", title: str = None):
    """THE forward expected-rate surface: horizon (x) x as-of date (y) x rate (z).

    The 30-year-deep rebirth of the old 'FFR Curves' chart.
    """
    import plotly.graph_objects as go
    from .curve_history import forward_surface

    mat = forward_surface(curve_hist, value=value)
    fig = go.Figure(data=[go.Surface(
        z=mat.values, x=mat.columns.values, y=mat.index,
        colorscale="RdBu_r", reversescale=True, connectgaps=True)])
    span = f"{mat.index.min().year}–{mat.index.max().year}" if len(mat) else "-"
    fig.update_layout(
        title=title or (f"Expected future interest-rate surface, 0-30Y "
                        f"({span}; eras: FRED CMT → hybrid → full futures stack)"),
        width=1250, height=900,
        scene=dict(xaxis_title="horizon (years)",
                   yaxis_title="as-of date",
                   zaxis_title="rate (bps)",
                   aspectratio=dict(x=1, y=2.2, z=0.6)))
    return fig


def plotly_pdf_surface(surface, layer: str, tenor: float | None = None,
                       pick: str = "nearest", title: str = None):
    """3D probability surface for one horizon family:
    rate bucket (x) x as-of date (y) x probability% (z)."""
    import plotly.graph_objects as go
    from .surface import horizon_series

    mat = horizon_series(surface, layer, tenor, pick=pick)
    cols = mat.columns.values.astype(float)
    keep = (cols >= 0)
    mat, cols = mat.loc[:, keep], cols[keep]
    fig = go.Figure(data=[go.Surface(
        z=mat.values * 100, x=cols, y=mat.index,
        colorscale="Magma", connectgaps=True)])
    lab = layer + (f" {tenor:.0f}Y" if tenor else "")
    fig.update_layout(
        title=title or f"Distribution evolution — {lab} (probability %)",
        width=1150, height=800,
        scene=dict(xaxis_title="rate bucket (bps mid)",
                   yaxis_title="as-of date",
                   zaxis_title="probability (%)",
                   aspectratio=dict(x=1, y=1.6, z=0.6)))
    return fig


def plotly_forward_pane(surface, as_of, title: str = None):
    """FedWatch-style forward probability view at one as-of date, 0-30Y:
    horizon_years (x) x rate bucket (y) x probability% (z)."""
    import plotly.graph_objects as go
    from .surface import slice_asof

    pane = slice_asof(surface, as_of)                     # bucket x horizon
    day = surface[surface["as_of"] == pd.Timestamp(as_of)]
    hy = day.groupby("horizon")["horizon_years"].first()
    hy = hy.reindex(pane.columns)
    fig = go.Figure(data=[go.Surface(
        z=(pane.T * 100).values, x=hy.values, y=pane.index.values,
        colorscale="Magma", connectgaps=True)])
    fig.update_layout(
        title=title or f"Forward probability view — {pd.Timestamp(as_of).date()} (0-30Y)",
        width=1150, height=800,
        scene=dict(xaxis_title="horizon (years)",
                   yaxis_title="rate bucket (bps mid)",
                   zaxis_title="probability (%)",
                   aspectratio=dict(x=1.4, y=1, z=0.6)))
    return fig


def plotly_future_dated_surface(mat, ztitle: str = "expected rate (bps)",
                                title: str = None):
    """3D surface: absolute future date (x) x as-of date (y) x value (z)."""
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Surface(
        z=mat.values, x=mat.columns, y=mat.index,
        colorscale="RdBu_r" if "rate" in ztitle else "Viridis",
        reversescale="rate" in ztitle, connectgaps=True)])
    span = f"{mat.index.min().year}-{mat.index.max().year}" if len(mat) else "-"
    fig.update_layout(
        title=title or f"{ztitle} — as-of {span}, by future contract date",
        width=1250, height=950,
        scene=dict(xaxis_title="future contract date",
                   yaxis_title="current date (as-of)",
                   zaxis_title=ztitle,
                   aspectratio=dict(x=1, y=2.2, z=0.6)))
    return fig
