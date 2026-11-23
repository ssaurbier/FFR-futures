"""Option-implied densities: Bachelier pricing, Normal SABR (beta=0), and the
Breeden-Litzenberger PDF engine.

Conventions:
  * SR3 options are premium-paid in futures points, margined futures-style —
    no discounting: Bachelier C(K) = (F-K)N(d) + sigma_N sqrt(T) phi(d).
  * All rate quantities in bps: strikes/forwards via (100 - price)*100,
    normal vols in bps*sqrt(year).
  * SABR: leading-order Hagan normal (beta=0):
        zeta = nu (F-K) / alpha
        chi(zeta) = ln( (sqrt(1 - 2 rho zeta + zeta^2) + zeta - rho) / (1-rho) )
        sigma_N(K) = nu (F-K) / chi(zeta),   sigma_N(F) = alpha
    calibrated (alpha, rho, nu) by least_squares on implied vols.
  * BL: q(K) = d2C/dK2 on a dense strike grid via central differences on the
    SABR-fitted call surface; clipped at zero, renormalised, bucketed to 25bp
    in rate space (rates are mirrored vs price strikes: high price = low rate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm


# ---------------------------------------------------------------- Bachelier
def bachelier_price(F: float, K: np.ndarray | float, sigma, T: float,
                    cp: str = "C") -> np.ndarray | float:
    """Undiscounted Bachelier premium; sigma scalar or per-strike array."""
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sst = sigma * np.sqrt(T)
    sst_safe = np.where(sst > 0.0, sst, 1e-12)
    d = (F - K) / sst_safe
    zero_vol = sst <= 0.0
    if cp == "C":
        px = (F - K) * norm.cdf(d) + sst * norm.pdf(d)
        intrinsic = np.maximum(F - K, 0.0)
    else:
        px = (K - F) * norm.cdf(-d) + sst * norm.pdf(d)
        intrinsic = np.maximum(K - F, 0.0)
    return np.where(zero_vol, intrinsic, px)


def bachelier_implied_vol(price: float, F: float, K: float, T: float,
                          cp: str = "C") -> float | None:
    """Bracketed root solve (Bachelier premium is monotone in sigma)."""
    from scipy.optimize import brentq
    intrinsic = max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    if price <= intrinsic + 1e-12:
        return None
    hi = 200.0                                   # far beyond any rate vol
    if bachelier_price(F, K, hi, T, cp) < price:
        return None
    try:
        return brentq(lambda s: float(bachelier_price(F, K, s, T, cp)) - price,
                      1e-8, hi, xtol=1e-12, rtol=1e-12)
    except ValueError:
        return None


# ---------------------------------------------------------------- Normal SABR
def sabr_normal_vol(K: np.ndarray | float, F: float, T: float,
                    alpha: float, rho: float, nu: float) -> np.ndarray:
    """Leading-order Hagan normal SABR (beta=0), same units as F/K."""
    K = np.asarray(K, dtype=float)
    z = (nu * (F - K)) / alpha
    # chi(z) with rho; numerically safe form
    sq = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    chi = np.log((sq + z - rho) / (1.0 - rho))
    out = np.where(np.abs(chi) > 1e-10, nu * (F - K) / np.where(np.abs(chi) > 1e-10, chi, 1.0),
                   alpha)
    # at/near ATM the limit is alpha (with mild smile slope)
    atm = np.abs(z) < 1e-10
    if np.any(atm):
        out = np.where(atm, alpha, out)
    return out


def fit_sabr(chain: pd.DataFrame, min_strikes: int = 5) -> dict | None:
    """Calibrate (alpha, rho, nu) to one chain snapshot.

    Uses calls and puts (puts via parity-mirrored strikes); vols in BPS.
    Returns dict with params + fit diagnostics, or None if too few quotes.
    """
    F = float(chain["fut_price"].iloc[0])
    T = float(chain["T_years"].iloc[0])
    F_bps = float(chain["fut_rate_bps"].iloc[0])

    # implied vols in bps: price pts -> bps scaling of 100
    Ks, vols, w = [], [], []
    for r in chain.itertuples():
        K_bps = (100.0 - r.strike_price) * 100.0
        v = bachelier_implied_vol(r.price, F, r.strike_price, T, r.cp)
        if v is None or not np.isfinite(v):
            continue
        Ks.append(K_bps)
        vols.append(v * 100.0)                      # price-pts/sqrt(y) -> bps
        w.append(1.0 + (r.volume if np.isfinite(r.volume) else 0.0) * 0.01)
    if len(Ks) < min_strikes:
        return None
    Ks, vols, w = map(np.asarray, (Ks, vols, w))
    atm_vol = float(np.median(vols[np.abs(Ks - F_bps) < 150])) if np.any(np.abs(Ks - F_bps) < 150) else float(np.median(vols))

    def resid(p):
        alpha, rho, nu = p
        model = sabr_normal_vol(Ks, F_bps, T, alpha, np.clip(rho, -0.999, 0.999), nu)
        return (model - vols) / w

    try:
        sol = least_squares(resid, x0=[atm_vol, 0.0, 0.3],
                            bounds=([1e-4, -0.999, 1e-4], [500.0, 0.999, 10.0]))
    except Exception:
        return None
    alpha, rho, nu = sol.x
    model = sabr_normal_vol(Ks, F_bps, T, alpha, rho, nu)
    rmse = float(np.sqrt(np.mean((model - vols) ** 2)))
    if rmse > 50.0:                                  # >50bp vol RMSE = bad fit
        return None
    return {"alpha": float(alpha), "rho": float(rho), "nu": float(nu),
            "T": T, "F_bps": F_bps, "rmse_bp": rmse, "n_quotes": len(Ks),
            "atm_vol_bp": float(sabr_normal_vol(F_bps, F_bps, T, alpha, rho, nu))}


# ---------------------------------------------------------------- BL PDF
def bl_pdf(fit: dict, k_lo_bps: float | None = None, k_hi_bps: float | None = None,
           step_bp: float = 2.5, wing_bp: float = 400.0) -> pd.Series:
    """Breeden-Litzenberger PDF in RATE bps from a fitted SABR surface.

    Returns Series indexed by 25bp-compatible grid of rate strikes (bps),
    values = density per bp (integrates to ~1 before bucketing).
    """
    F, T = fit["F_bps"], fit["T"]
    lo = k_lo_bps if k_lo_bps is not None else F - wing_bp
    hi = k_hi_bps if k_hi_bps is not None else F + wing_bp
    grid = np.arange(lo, hi + step_bp, step_bp)
    alpha, rho, nu = fit["alpha"], fit["rho"], fit["nu"]
    vols = sabr_normal_vol(grid, F, T, alpha, rho, nu)
    # calls on the rate (mirrored): C is Bachelier on F vs K in bps directly
    calls = bachelier_price(F, grid, vols, T, cp="C")
    q = np.gradient(np.gradient(calls, grid), grid)   # d2C/dK2, uniform grid
    q = np.clip(q, 0.0, None)
    s = pd.Series(q, index=grid)
    area = np.trapezoid(s.values, s.index.values)
    if area <= 0:
        raise ValueError("BL density has no positive mass")
    return s / area


def bucket_pdf(pdf: pd.Series, bucket_bp: float = 25.0) -> pd.Series:
    """Integrate a fine-grained density into FedWatch 25bp bucket midpoints."""
    from .regime1_fomc import bucket_midpoints
    idx = pdf.index.values
    mids = bucket_midpoints(idx)
    df = pd.DataFrame({"mid": mids, "p": pdf.values})
    out = df.groupby("mid")["p"].sum() * bucket_bp     # avg density * width
    total = out.sum()
    return out / total if total > 0 else out


def pdf_mean_bps(pdf: pd.Series) -> float:
    return float(np.trapezoid(pdf.index.values * pdf.values, pdf.index.values))


def option_density(chain: pd.DataFrame) -> tuple[str, pd.Series, dict] | None:
    """Full pipeline for one chain: fit SABR -> BL PDF -> 25bp buckets.

    Returns (underlying_label, bucketed_density, fit_diagnostics) or None.
    """
    fit = fit_sabr(chain)
    if fit is None:
        return None
    pdf = bl_pdf(fit)
    return "chain", bucket_pdf(pdf), fit
