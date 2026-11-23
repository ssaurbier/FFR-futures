"""Phase-1 probability densities beyond the FOMC horizon (model-based).

Two complementary methods (both replaced by option-implied PDFs in Phase 2):
  * historical-vol densities — normal density centred on the bootstrapped
    forward with sigma scaled from realised changes of the relevant product
    family (SR3 for <=3Y horizons, Eris pars beyond);
  * Hull-White 1F simulation — exact discretisation off the bootstrapped
    curve, giving a percentile fan of future short/zero rates.

Output schema: 25bp buckets (midpoints), same as the FOMC matrix, flagged
`method="model"` until Phase 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .regime1_fomc import bucket_midpoints


# ---------------------------------------------------------------- helpers
def bucket_density(mean_bps: float, sigma_bps: float,
                   bucket_bp: float = 25.0, span: float = 4.0) -> pd.Series:
    """Normal density discretised into FedWatch-style 25bp buckets."""
    if sigma_bps <= 0:
        mids = bucket_midpoints(np.array([mean_bps]))
        return pd.Series([1.0], index=mids)
    grid = np.arange(mean_bps - span * sigma_bps,
                     mean_bps + span * sigma_bps + bucket_bp, bucket_bp)
    # integrate the normal pdf over each bucket [k - 12.5, k + 12.5]
    from scipy.stats import norm
    edges = grid - bucket_bp / 2.0
    cdf = norm.cdf(edges, loc=mean_bps, scale=sigma_bps)
    probs = np.diff(cdf)
    mids = bucket_midpoints((edges[:-1] + edges[1:]) / 2.0)
    s = pd.Series(probs, index=mids)
    return s / s.sum()


# ---------------------------------------------------------------- method 1
def _realized_sigma(fut: pd.DataFrame, horizon_years: float, asof,
                    lookback: int = 750) -> float:
    """Ann-vol (bps) of daily changes of the relevant forward, sqrt-scaled."""
    asof = pd.Timestamp(asof)
    if horizon_years <= 3.0:
        df = fut[fut.root == "SR3"].copy()
        df["rate"] = (100.0 - df["close"]) * 100.0
        idx = df.groupby("date")["contract_month"].idxmin()
        s = df.loc[idx].set_index("date")["rate"]
    else:
        # Eris par marks from the published par curve history if available
        from .ingest import eris_io
        from pathlib import Path
        if eris_io.PARCOUPON_PARQUET.exists():
            pc = pd.read_parquet(eris_io.PARCOUPON_PARQUET)
            col = {5: "SOFR5Y", 7: "SOFR7Y", 10: "SOFR10Y",
                   15: "SOFR15Y", 20: "SOFR20Y", 30: "SOFR30Y"}.get(
                       int(horizon_years), "SOFR10Y")
            s = pd.to_numeric(pc[col], errors="coerce").dropna()
        else:
            df = fut[fut.root == "SR3"].copy()
            df["rate"] = (100.0 - df["close"]) * 100.0
            idx = df.groupby("date")["contract_month"].idxmin()
            s = df.loc[idx].set_index("date")["rate"]
    s = s[s.index <= asof].tail(lookback)
    daily = s.diff().dropna().std()
    return float(daily * np.sqrt(252))


def historical_vol_densities(curve, tenors=(2, 3, 5, 7, 10, 15, 20, 30),
                             fut: pd.DataFrame | None = None, asof=None
                             ) -> pd.DataFrame:
    """Rows = 25bp bucket mids; cols = tenors; values = probability."""
    asof = pd.Timestamp(asof or curve.asof)
    cols = {}
    for T in tenors:
        d = asof + pd.Timedelta(days=round(365.25 * T))
        fwd = curve.zero(d) * 1e4
        sig = _realized_sigma(fut, T, asof) if fut is not None else 90.0
        sig_h = sig * np.sqrt(T)                    # sqrt-time scaling
        cols[T] = bucket_density(fwd, sig_h).rename(T)
    mat = pd.concat(cols.values(), axis=1).fillna(0.0)
    return mat / mat.sum(axis=0)


# ---------------------------------------------------------------- method 2
def hw_theta(curve, t: float, a: float, sigma: float, h: float = 1e-4) -> float:
    """HW drift theta(t) fitted to the curve's instantaneous forwards."""
    fp = curve._fwd_inst(t + h)
    fm = curve._fwd_inst(max(t - h, 0.0))
    dfdt = (fp - fm) / (h + h if t - h > 0 else h)
    f = curve._fwd_inst(t)
    return dfdt + a * f + sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a * t))


def hw_simulate(curve, tenors=(2, 3, 5, 7, 10, 15, 20, 30), a: float = 0.03,
                sigma: float | None = None, n_paths: int = 20_000,
                dt: float = 1 / 48, seed: int = 42) -> pd.DataFrame:
    """Exact HW 1F discretisation; returns percentile fan per tenor.

    Rows = percentiles (5/25/50/75/95) of the zero rate at each tenor.
    """
    rng = np.random.default_rng(seed)
    sigma = sigma if sigma is not None else 0.0095   # Phase-1 proxy (~95bp)
    t_max = max(tenors) + dt
    n_steps = int(np.ceil(t_max / dt))
    r = np.full(n_paths, curve._fwd_inst(1e-6))
    zeros = {T: np.zeros(n_paths) for T in tenors}
    tenor_set = set(tenors)
    acc, acc_t = np.zeros(n_paths), 0.0

    for i in range(1, n_steps + 1):
        t_prev = (i - 1) * dt
        th = hw_theta(curve, max(t_prev, 1e-6), a, sigma)
        z = rng.standard_normal(n_paths)
        e = np.exp(-a * dt)
        r = r * e + th * (1 - e) / a + sigma * np.sqrt((1 - e * e) / (2 * a)) * z
        acc += r * dt
        acc_t += dt
        if round(acc_t, 6) in {round(float(T), 6) for T in tenor_set}:
            T = round(acc_t, 6)
            zeros[int(T) if T == int(T) else T] = acc / acc_t * 1e4  # zero bps

    rows = {}
    for T in tenors:
        z = zeros[T]
        rows[T] = {"p5": np.percentile(z, 5), "p25": np.percentile(z, 25),
                   "p50": np.percentile(z, 50), "p75": np.percentile(z, 75),
                   "p95": np.percentile(z, 95)}
    return pd.DataFrame(rows).T


def hw_bucket_densities(curve, tenors=(2, 3, 5, 10, 20, 30), a: float = 0.03,
                        sigma: float | None = None, n_paths: int = 20_000
                        ) -> pd.DataFrame:
    """Empirical HW zero-rate distributions bucketed into 25bp buckets."""
    rng = np.random.default_rng(7)
    sigma = sigma if sigma is not None else 0.0095
    out = {}
    for T in tenors:
        n_steps = int(round(T * 48))
        dt = T / n_steps
        r = np.full(n_paths, curve._fwd_inst(1e-6))
        acc = np.zeros(n_paths)
        for i in range(n_steps):
            t_prev = i * dt
            th = hw_theta(curve, max(t_prev, 1e-6), a, sigma)
            z = rng.standard_normal(n_paths)
            e = np.exp(-a * dt)
            r = r * e + th * (1 - e) / a + sigma * np.sqrt((1 - e * e) / (2 * a)) * z
            acc += r * dt
        zero_bps = acc / T * 1e4
        mids = bucket_midpoints(zero_bps)
        s = pd.Series(mids).value_counts(normalize=True).sort_index()
        out[T] = s.rename(T)
    return pd.concat(out.values(), axis=1).fillna(0.0)
