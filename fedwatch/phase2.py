"""Phase 2: option-implied densities replacing model-based columns.

Pipeline per as-of date:
  SR3 option chains (GLBX ohlcv-1d) -> Bachelier implied vols -> Normal SABR
  fit per expiry -> dense call surface -> Breeden-Litzenberger PDF -> 25bp
  FedWatch buckets, labelled by option expiry date with an '(option)' flag.

Also provides the ATM normal-vol term structure used to calibrate the
Hull-White sigma for the Regime-2 convexity adjustment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .options_chain import chain_snapshot, option_expiry, underlying_month
from .sabr_bl import bl_pdf, bucket_pdf, fit_sabr, pdf_mean_bps

OPTIONS_PARQUET = config.DATA_DIR / "options_sr3_ohlcv1d.parquet"


def load_options() -> pd.DataFrame:
    if not OPTIONS_PARQUET.exists():
        raise FileNotFoundError(
            f"{OPTIONS_PARQUET.name} missing — run the SR3.OPT monthly pull")
    df = pd.read_parquet(OPTIONS_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


def option_snapshots(fut: pd.DataFrame, asof: pd.Timestamp,
                     opt: pd.DataFrame | None = None,
                     min_quotes: int = 5) -> tuple[dict, pd.DataFrame]:
    """Chains + SABR fits at `asof`. Returns (chains, fits_df)."""
    opt = load_options() if opt is None else opt
    chains = chain_snapshot(opt, fut, asof)
    rows = []
    for und, chain in chains.items():
        fit = fit_sabr(chain, min_strikes=min_quotes)
        if fit is None:
            continue
        um = underlying_month(und, asof)
        fit["underlying"] = und
        fit["expiry"] = option_expiry(um)
        fit["horizon_years"] = (fit["expiry"] - asof).days / 365.0
        rows.append(fit)
    fits = pd.DataFrame(rows).sort_values("horizon_years").reset_index(drop=True)
    return chains, fits


def option_density_columns(fits: pd.DataFrame, fut: pd.DataFrame,
                           asof: pd.Timestamp, opt: pd.DataFrame | None = None,
                           martingale_tol_bp: float = 10.0,
                           rmse_tol_bp: float = 30.0
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """25bp-bucketed option-implied densities, one column per expiry.

    Fits are gated twice: SABR rmse (thin serial-month chains fit noisily)
    and the BL martingale condition |PDF mean - forward|.
    Returns (density_matrix mids x expiries, validation_df).
    """
    if opt is None:
        opt = load_options()
    chains = chain_snapshot(opt, fut, asof)

    cols, checks = {}, []
    for r in fits.itertuples():
        chain = chains.get(r.underlying)
        if chain is None:
            continue
        if r.rmse_bp > rmse_tol_bp:
            continue                       # reject noisy surface
        pdf = bl_pdf({"alpha": r.alpha, "rho": r.rho, "nu": r.nu,
                      "T": r.T, "F_bps": r.F_bps})
        mean_gap = pdf_mean_bps(pdf) - r.F_bps
        if abs(mean_gap) > martingale_tol_bp:
            continue                       # reject unreliable fit
        label = f"{r.expiry.strftime('%Y-%m-%d')} (option)"
        cols[label] = bucket_pdf(pdf)
        checks.append({"underlying": r.underlying, "expiry": r.expiry,
                       "horizon_years": r.horizon_years,
                       "atm_vol_bp": r.atm_vol_bp, "rmse_bp": r.rmse_bp,
                       "mean_gap_bp": mean_gap})
    mat = (pd.DataFrame(cols).fillna(0.0) if cols else pd.DataFrame())
    if len(mat):
        mat = mat / mat.sum(axis=0)
    return mat, pd.DataFrame(checks)


def atm_vol_term_structure(fits: pd.DataFrame) -> pd.Series:
    """ATM normal vol (bps) by horizon (years) — HW sigma calibration input."""
    return pd.Series(fits["atm_vol_bp"].values, index=fits["horizon_years"]).sort_index()


def merge_into_master(master: pd.DataFrame, option_cols: pd.DataFrame
                      ) -> pd.DataFrame:
    """Union option columns (flagged) with the Phase-1 master matrix."""
    if option_cols is None or option_cols.empty:
        return master
    both = [c for c in master.columns if c in option_cols.columns]
    merged = master.drop(columns=both).join(option_cols, how="outer").fillna(0.0)
    return merged.sort_index()
