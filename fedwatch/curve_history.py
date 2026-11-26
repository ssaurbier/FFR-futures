"""Historical forward curves: the expected-rate surface 0-30Y through time.

At each as-of date, `pipeline.build_curve` produces the unified SOFR
discounting curve; this module evaluates it on a standard horizon grid and
stores the daily/weekly history:

    data/curve_history.parquet  [as_of, horizon_years, zero_bp, par_bp]

This IS the "entire forward curve from historical time to now, itself going
30 years into the future" — the object the whole project points at, and the
mean/centre of the 3D probability surface.

Three eras, assembled into one surface with an `era` label per row:
  fred        1990 -> Jun 2010   FRED CMT spot par curves (DGS1MO-DGS30)
  hybrid      Jun 2010 -> Oct 2020  ZQ futures front end (expected EFFR
              path) + CMT spot long end — per-point `source` column
  sofr_full   Oct 2020 -> now    the full SR1/SR3/Eris futures stack

Seam semantics: fred/hybrid long horizons are SPOT par yields; sofr_full
horizons are true forwards. The era column keeps the distinction visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CURVE_HISTORY_PARQUET = config.DATA_DIR / "curve_history.parquet"

# standard horizon grid (years)
HORIZONS = np.array([1 / 12, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30])


# CMT series by horizon (years); grid points without a series interpolate
_CMT = {1 / 12: "DGS1MO", 0.25: "DGS3MO", 0.5: "DGS6MO", 1.0: "DGS1",
        2.0: "DGS2", 3.0: "DGS3", 5.0: "DGS5", 7.0: "DGS7",
        10.0: "DGS10", 20.0: "DGS20", 30.0: "DGS30"}


def _cmt_block(fred_bps: pd.DataFrame, grid, verbose=True) -> pd.DataFrame:
    """Era A: FRED CMT spot curves on the horizon grid."""
    weekly = fred_bps[list(_CMT.values())].resample("W-FRI").last().ffill()
    # map each grid horizon to the nearest CMT series, interpolate between
    tenor_pts = np.array(sorted(_CMT))
    rows = []
    for t, r in weekly.iterrows():
        pars = np.array([r[_CMT[x]] for x in tenor_pts])
        if np.isnan(pars).any():
            pars = pd.Series(pars).ffill().bfill().values
        for h in HORIZONS:
            v = float(np.interp(h, tenor_pts, pars))
            rows.append((t, h, v, v, "fred", "cmt"))
    df = pd.DataFrame(rows, columns=["as_of", "horizon_years", "zero_bp",
                                     "par_bp", "era", "source"])
    return df[df["as_of"] >= "1990-01-01"]


def _hybrid_block(bundle: dict, grid, verbose=True) -> pd.DataFrame:
    """Era B: ZQ futures front (expected EFFR path) + CMT spot long end."""
    fut, fred = bundle["fut"], bundle["fred"]
    zq = fut[fut.root == "ZQ"].copy()
    zq["imp"] = (100.0 - zq["close"]) * 100.0
    strip = (zq.pivot_table(index="date", columns="contract_month", values="imp")
             .sort_index().ffill(limit=60))
    strip.columns = pd.DatetimeIndex(strip.columns)

    weekly = fred[list(_CMT.values())].resample("W-FRI").last().ffill()
    tenor_pts = np.array(sorted(_CMT))
    rows = []
    for t in grid:
        if t not in strip.index:
            srow = strip.loc[:t]
            if srow.empty:
                continue
            srow = srow.iloc[-1]
        else:
            srow = strip.loc[t]
        srow = srow.dropna()
        if srow.empty:
            continue
        cmt = weekly.loc[:t]
        if cmt.empty:
            continue
        cmt = cmt.iloc[-1]
        pars = np.array([cmt[_CMT[x]] for x in tenor_pts])
        if np.isnan(pars).any():
            pars = pd.Series(pars).ffill().bfill().values
        for h in HORIZONS:
            # front horizons: mean of ZQ month-averages spanning (t, t+h]
            months = srow[(srow.index > t) & (srow.index <= t + pd.DateOffset(months=round(h * 12)))]
            if len(months) >= 1 and h <= 2.0:
                v = float(months.mean())
                rows.append((t, h, v, v, "hybrid", "zq_futures"))
            else:
                v = float(np.interp(h, tenor_pts, pars))
                rows.append((t, h, v, v, "hybrid", "cmt"))
    return pd.DataFrame(rows, columns=["as_of", "horizon_years", "zero_bp",
                                       "par_bp", "era", "source"])


def build_curve_history(freq: str = "W-FRI", start=None, end=None,
                        out=CURVE_HISTORY_PARQUET, verbose=True) -> pd.DataFrame:
    """Evaluate the unified curve on the horizon grid at each grid date."""
    from .pipeline import build_curve, load_bundle

    bundle = load_bundle()
    px = bundle["eris_prices"]
    sofr_start = px[px["zcode"].str.startswith("YI")]["date"].min()
    sr1_start = bundle["fut"][bundle["fut"]["root"] == "SR1"]["date"].min()
    eris_start = max(sofr_start, sr1_start)       # SOFR-era build requires both
    fut_end = bundle["fut"]["date"].max()
    grid = pd.date_range(max(pd.Timestamp(start or eris_start), eris_start),
                         min(pd.Timestamp(end or fut_end), fut_end), freq=freq)

    rows = []
    for i, t in enumerate(grid):
        try:
            res = build_curve(bundle, asof=t)
        except Exception as e:                        # noqa: BLE001
            if verbose:
                print(f"  {t.date()}: FAILED {type(e).__name__}", flush=True)
            continue
        curve = res["curve"]
        last_t = (curve.dates[-1] - t).days / 365.25
        for h in HORIZONS:
            if h > last_t + 1 / 12:                   # beyond bootstrapped span
                rows.append((t, h, np.nan, np.nan))
                continue
            d = t + pd.Timedelta(days=round(365.25 * h))
            zero = curve.zero(d) * 1e4
            if h >= 0.25:                    # par needs >= 1 accrual period
                par, _ = curve.par_swap(t + pd.Timedelta(days=2), d)
                par *= 1e4
            else:
                par = np.nan
            rows.append((t, h, zero, par))
        if verbose and i % 26 == 0:
            print(f"  {t.date()}: ok (span {last_t:.1f}Y)", flush=True)

    sofr = pd.DataFrame(rows, columns=["as_of", "horizon_years", "zero_bp", "par_bp"])
    sofr["era"], sofr["source"] = "sofr_full", "futures_stack"

    from .pipeline import load_bundle
    bundle2 = load_bundle()
    era_a = _cmt_block(bundle2["fred"], None)
    grid_b = pd.date_range("2010-06-11", sofr_start - pd.Timedelta(days=1), freq=freq)
    era_b = _hybrid_block(bundle2, grid_b)
    era_c = sofr

    df = pd.concat([era_a[era_a["as_of"] < grid_b.min()], era_b, era_c],
                   ignore_index=True)
    df.to_parquet(out, index=False)
    if verbose:
        print(f"curve history: {len(df):,} rows, {df['as_of'].nunique()} dates "
              f"({df['as_of'].min().date()} -> {df['as_of'].max().date()}); "
              f"eras: {df.groupby('era')['as_of'].nunique().to_dict()}", flush=True)
    return df


def load_curve_history() -> pd.DataFrame:
    df = pd.read_parquet(CURVE_HISTORY_PARQUET)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df


def forward_surface(curve_hist: pd.DataFrame, value: str = "zero_bp"
                    ) -> pd.DataFrame:
    """The chart-ready matrix: rows = as-of dates, columns = horizon years."""
    mat = curve_hist.pivot_table(index="as_of", columns="horizon_years",
                                 values=value)
    return mat.sort_index()


# ---------------------------------------------------------------- future-dated views
def _future_bucket(curve_hist: pd.DataFrame, bucket: str) -> pd.Series:
    days = (curve_hist["horizon_years"] * 365.25).round().astype(int)
    future = curve_hist["as_of"] + pd.to_timedelta(days, unit="D")
    return future.dt.to_period(bucket).dt.to_timestamp()


def future_dated_matrix(curve_hist: pd.DataFrame, value: str = "zero_bp",
                        bucket: str = "Y") -> pd.DataFrame:
    """Pivot to as-of rows x ABSOLUTE FUTURE DATES columns.

    Each (as_of, horizon) point lands in the calendar bucket containing
    as_of + horizon — a 1995 row spans 1995..2025, a 2026 row spans
    2026..2056: the full forward-curve fan, 30Y forward from every date.
    """
    df = curve_hist.copy()
    df["future_date"] = _future_bucket(df, bucket)
    return df.pivot_table(index="as_of", columns="future_date", values=value,
                          aggfunc="mean").sort_index()


def horizon_detail_matrix(curve_hist: pd.DataFrame, value: str = "zero_bp",
                          max_horizon_years: float = 2.0) -> pd.DataFrame:
    """Short-end detail: as-of x future QUARTER, horizons <= 2Y."""
    df = curve_hist[curve_hist["horizon_years"] <= max_horizon_years].copy()
    df["future_date"] = _future_bucket(df, "Q")
    return df.pivot_table(index="as_of", columns="future_date", values=value,
                          aggfunc="mean").sort_index()
