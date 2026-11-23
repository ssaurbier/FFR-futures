"""FRED ingestion: realized anchors (EFFR/SOFR/targets) + treasury yields.

Increments the cached parquet forward; stores values in raw FRED units
(percent). Callers convert: bps = value * 100.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..config import FRED_PARQUET, LEGACY_FRED_CSV


def _fetch(series: str, start: str) -> pd.Series:
    import requests
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series, "api_key": config.get_fred_key(),
                "file_type": "json", "observation_start": start},
        timeout=30,
    )
    obs = [o for o in r.json().get("observations", []) if o["value"] != "."]
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs}, name=series)
    return s.sort_index()


def load_fred(refresh: bool = False) -> pd.DataFrame:
    """DataFrame indexed by date, one column per series, values in PERCENT."""
    if not refresh and FRED_PARQUET.exists():
        return pd.read_parquet(FRED_PARQUET)

    if FRED_PARQUET.exists():
        base = pd.read_parquet(FRED_PARQUET)
        start = (base.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    elif LEGACY_FRED_CSV.exists():
        base = pd.read_csv(LEGACY_FRED_CSV, index_col="Date", parse_dates=True)
        start = (base.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    else:
        base = pd.DataFrame()
        start = "1990-01-01"

    fetched = {s: _fetch(s, start) for s in config.FRED_SERIES}
    fresh = pd.DataFrame(fetched)
    fresh.index.name = "Date"
    if not base.empty:
        fresh = pd.concat([base.loc[base.index < fresh.index.min()], fresh])
    fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
    fresh.to_parquet(FRED_PARQUET)
    return fresh


def fred_bps(fred: pd.DataFrame) -> pd.DataFrame:
    """Percent (FRED raw) -> bps."""
    return fred * 100.0
