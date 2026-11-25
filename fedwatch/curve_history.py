"""Historical forward curves: the expected-rate surface 0-30Y through time.

At each as-of date, `pipeline.build_curve` produces the unified SOFR
discounting curve; this module evaluates it on a standard horizon grid and
stores the daily/weekly history:

    data/curve_history.parquet  [as_of, horizon_years, zero_bp, par_bp]

This IS the "entire forward curve from historical time to now, itself going
30 years into the future" — the object the whole project points at, and the
mean/centre of the 3D probability surface.

Coverage note: the full 0-30Y span requires Eris SOFR listings (2020-10 ->).
Before that, curves are built futures-only (front end) and long horizons are
NaN — honest, visible, and extendable later via Eris LIBOR-era contracts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CURVE_HISTORY_PARQUET = config.DATA_DIR / "curve_history.parquet"

# standard horizon grid (years)
HORIZONS = np.array([1 / 12, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30])


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

    if not rows:
        raise RuntimeError("no curve-history rows built — check data coverage")
    df = pd.DataFrame(rows, columns=["as_of", "horizon_years", "zero_bp", "par_bp"])
    df.to_parquet(out, index=False)
    if verbose:
        ok = df.dropna(subset=["zero_bp"])
        print(f"curve history: {len(df):,} rows, {df['as_of'].nunique()} dates "
              f"({df['as_of'].min().date()} -> {df['as_of'].max().date()}); "
              f"{len(ok):,} valid points", flush=True)
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
