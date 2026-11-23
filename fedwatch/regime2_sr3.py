"""Regime 2 (18M -> ~10Y): SR3 quarterly strip with convexity adjustment.

3M SOFR futures (SR3) settle to compounded daily SOFR over the IMM quarter
[3rd Wed of contract month, 3rd Wed of month+3). The futures rate embeds a
convexity bias versus the true forward OIS rate (futures = daily-settled,
no discounting of margin flows). We remove the Hull-White 1-factor bias:

    f_ois = f_fut - (sigma^2 / (2 a^2)) * (1 - e^{-a dt})
            * [ (1 - e^{-a dt})/2 * (1 - e^{-2 a t1}) + (1 - e^{-a t1})^2 ]

with t1 = accrual start (years), dt = accrual length (years), a = mean
reversion (~0.03 in Phase 1) and sigma the (normal) short-rate vol.

Phase 1 sigma proxy: annualised stdev of daily changes of the front SR3
contract's implied rate. Phase 2 replaces this with ATM option-implied vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import sr3_accrual

DAYS_YEAR = 365.0


# ---------------------------------------------------------------- strip
def sr3_strip(fut: pd.DataFrame, asof: pd.Timestamp | None = None,
              ffill_limit: int = 90) -> pd.DataFrame:
    """Latest (or as-of) SR3 strip: contract, accrual window, futures rate.

    Returns frame indexed by contract_month with columns
    [accrual_start, accrual_end, t1_years, dt_years, rate_fut_bps, price].
    """
    df = fut[fut.root == "SR3"].copy()
    df["rate_fut_bps"] = (100.0 - df["close"]) * 100.0
    px = (df.pivot_table(index="date", columns="contract_month", values="rate_fut_bps")
          .sort_index().ffill(limit=ffill_limit))
    px.columns = pd.DatetimeIndex(px.columns)
    asof = asof or px.index.max()
    row = px.loc[px.index <= asof].iloc[-1].dropna()

    out = pd.DataFrame({"rate_fut_bps": row})
    out.index.name = "contract_month"
    acc = [sr3_accrual(m) for m in out.index]
    out["accrual_start"] = [a for a, _ in acc]
    out["accrual_end"] = [b for _, b in acc]
    dt = [(b - a).days for a, b in acc]
    t1 = [(a - asof).days for a, _ in acc]
    out["t1_years"] = np.array(t1, dtype=float) / DAYS_YEAR
    out["dt_years"] = np.array(dt, dtype=float) / DAYS_YEAR
    out = out[out["t1_years"] >= -0.02]                    # drop expired
    return out


# ---------------------------------------------------------------- convexity
def hw_convexity_bps(t1_years: np.ndarray | float, dt_years: np.ndarray | float,
                      sigma: float, a: float = 0.03) -> np.ndarray | float:
    """Hull-White 1F futures->forward convexity adjustment, in bps.

    sigma: normal (bps/year^1.5-free) short-rate vol in RATE units/year^0.5
           (e.g. 0.01 = 100bp normal vol); returns bps.
    """
    t1, dt = np.asarray(t1_years, dtype=float), np.asarray(dt_years, dtype=float)
    ea = np.exp(-a * dt)
    e1 = np.exp(-a * t1)
    conv = (sigma ** 2) / (2 * a ** 2) * (1 - ea) * (
        (1 - ea) / 2 * (1 - np.exp(-2 * a * t1)) + (1 - e1) ** 2)
    return conv * 1e4


def estimate_hw_sigma(fut: pd.DataFrame, lookback: int = 500,
                      asof: pd.Timestamp | None = None) -> float:
    """Annualised stdev of front SR3 daily implied-rate changes (rate units).

    A normal-vol proxy for sigma_hw; document as Phase 1 approximation.
    """
    df = fut[fut.root == "SR3"].copy()
    df["rate"] = (100.0 - df["close"])          # percent
    idx = df.groupby("date")["contract_month"].idxmin()   # front contract per date
    front = df.loc[idx].set_index("date")["rate"].sort_index()
    if asof is not None:
        front = front[front.index <= pd.Timestamp(asof)]
    front = front.tail(lookback)
    daily = front.diff().dropna()
    return float(daily.std() / 100.0 * np.sqrt(252))   # -> rate units, annualised


def estimate_hw_sigma_from_options(fut: pd.DataFrame, asof=None,
                                   opt: pd.DataFrame | None = None) -> float | None:
    """ATM Bachelier normal vol (rate units) of the front quarterly chains.

    Preferred Phase-2 source for the convexity sigma; returns None when no
    usable option fit exists (caller falls back to the historical proxy).
    """
    from .phase2 import option_snapshots
    _, fits = option_snapshots(fut, pd.Timestamp(asof), opt=opt)
    if fits is None or fits.empty:
        return None
    front = fits[(fits.horizon_years > 0.2) & (fits.n_quotes >= 6)].head(3)
    if front.empty:
        return None
    return float(front["atm_vol_bp"].mean()) / 1e4


def adjusted_strip(fut: pd.DataFrame, asof: pd.Timestamp | None = None,
                   sigma: float | None = None, a: float = 0.03) -> pd.DataFrame:
    """SR3 strip + forward OIS rates (futures rate minus convexity, bps)."""
    strip = sr3_strip(fut, asof=asof)
    if sigma is None:
        sigma = estimate_hw_sigma(fut, asof=asof)
    strip["convexity_bps"] = hw_convexity_bps(strip["t1_years"].values,
                                              strip["dt_years"].values, sigma, a)
    strip["rate_ois_bps"] = strip["rate_fut_bps"] - strip["convexity_bps"]
    strip.attrs["hw_sigma"] = sigma
    strip.attrs["hw_a"] = a
    return strip
