"""Market-calibrated volatility term structure -> PDFs at EVERY horizon to 30Y.

The expansion of FedWatch beyond the 2-year listed-option horizon:
  * option-implied ATM vols exist for horizons ~3 weeks to ~2.1 years
    (SR3 options) and per-tenor at ~1-3 months (Treasury options);
  * fit sigma(T) to those REAL market vols with a mean-reverting
    functional form, sigma(T)^2 = s_inf^2 + (s_0^2 - s_inf^2) * (1-e^{-kT})/(kT)
    scaled so variance V(T) = sigma(T)^2 * T;
  * extrapolate to 30Y: sigma(T) -> s_inf (damped, no sqrt-T explosion);
  * every horizon 0-30Y then has a PDF whose centre is the futures strip
    (Eris/SR3/SR1 — real market forwards) and whose width is anchored to
    real option vols where they exist and calibrated beyond.

This replaces the realized-vol sqrt-T hack for every date with options.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def fit_sigma_term(horizons: np.ndarray, atm_vols: np.ndarray
                   ) -> dict | None:
    """Fit mean-reverting normal-vol term structure to ATM vols.

    sigma(T)^2 = s_inf^2 + (s_0^2 - s_inf^2) * (1 - e^{-k T}) / (k T)
    (sigma(0) -> s_0; sigma(inf) -> s_inf; k in [0.02, 3].)
    Returns {'s0', 's_inf', 'k'} or None if too few points.
    """
    h, v = np.asarray(horizons, float), np.asarray(atm_vols, float)
    keep = (h > 0.01) & np.isfinite(v) & (v > 1.0)
    h, v = h[keep], v[keep]
    if len(h) < 3:
        return None

    def resid(p):
        s0, si, k = np.exp(p[0]), np.exp(p[1]), np.exp(p[2])
        kT = np.clip(k * h, 1e-8, 50.0)
        model = np.sqrt(si ** 2 + (s0 ** 2 - si ** 2) * (1 - np.exp(-kT)) / kT)
        return model - v

    try:
        sol = least_squares(resid, x0=[np.log(np.median(v)),
                                       np.log(np.median(v)), np.log(0.3)],
                            bounds=([np.log(1.0), np.log(1.0), np.log(0.02)],
                                    [np.log(2000.0), np.log(2000.0), np.log(3.0)]))
    except Exception:
        return None
    s0, si, k = np.exp(sol.x)
    model = np.sqrt(si ** 2 + (s0 ** 2 - si ** 2) *
                    (1 - np.exp(-k * h)) / np.clip(k * h, 1e-8, 50.0))
    rmse = float(np.sqrt(np.mean((model - v) ** 2)))
    if rmse > 0.35 * np.median(v):          # fit failed to capture term shape
        return None
    return {"s0": float(s0), "s_inf": float(si), "k": float(k),
            "rmse_bp": rmse, "n": len(h)}


def sigma_of_T(fit: dict, T: np.ndarray | float) -> np.ndarray:
    """sigma(T) in bps from a fitted term structure."""
    T = np.asarray(T, float)
    kT = np.clip(fit["k"] * T, 1e-8, 50.0)
    return np.sqrt(fit["s_inf"] ** 2 +
                   (fit["s0"] ** 2 - fit["s_inf"] ** 2) *
                   (1 - np.exp(-kT)) / kT)


def horizon_sigma(fit: dict, T: float) -> float:
    """Cumulative sd to horizon T: sqrt(integral sigma(t)^2 dt), in bps."""
    if T <= 0:
        return 0.0
    grid = np.linspace(max(T / 200.0, 1e-4), T, 200)
    s2 = sigma_of_T(fit, grid) ** 2
    return float(np.sqrt(np.trapezoid(s2, grid)))


def daily_option_fits(res: dict, asof) -> pd.DataFrame:
    """SR3 + Treasury option ATM fits at `asof` as (horizon_years, atm_vol_bp)."""
    rows = []
    try:
        from .phase2 import option_snapshots
        _, fits = option_snapshots(res["bundle"]["fut"], pd.Timestamp(asof))
        for r in fits.itertuples():
            rows.append((r.horizon_years, r.atm_vol_bp))
    except Exception:
        pass
    try:
        from .treasury_options import treasury_chains, fit_sabr_price, PRODUCTS
        chains = treasury_chains(res["bundle"]["fut"], pd.Timestamp(asof))
        for key, ch in chains.items():
            f = fit_sabr_price(ch, min_quotes=6)
            if f is None:
                continue
            oz = ch["oz_root"].iloc[0]
            _, tenor, dv, _ = PRODUCTS[oz]
            rows.append((f["T"], f["atm_vol_pts"] / dv * 100.0))  # yield bp
    except Exception:
        pass
    return pd.DataFrame(rows, columns=["horizon_years", "atm_vol_bp"])
