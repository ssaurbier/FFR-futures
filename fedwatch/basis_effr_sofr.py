"""EFFR-SOFR basis: bridge ZQ (EFFR) expectations onto the SOFR curve.

ZQ prices average EFFR; SR1 prices average SOFR over the same calendar
months. Where both trade (SR1 listed since 2019), their implied-rate spread
per contract month is the market's forward EFFR-SOFR basis for that month.
We use median basis by months-ahead bucket so ZQ forwards can be mapped onto
SOFR discounting before joining the SR3/Eris segments.
"""
from __future__ import annotations

import pandas as pd


def forward_basis(fut: pd.DataFrame, months_ahead_bins: list[int] | None = None) -> pd.DataFrame:
    """Historical (date, contract_month) EFFR-SOFR implied spread, bps.

    Returns frame with columns [date, contract_month, months_ahead, basis_bps].
    """
    months_ahead_bins = months_ahead_bins or [1, 3, 6, 12, 24]
    zq = fut[fut.root == "ZQ"].copy()
    sr1 = fut[fut.root == "SR1"].copy()
    zq["imp"] = (100.0 - zq["close"]) * 100.0
    sr1["imp"] = (100.0 - sr1["close"]) * 100.0
    zq = zq.set_index(["date", "contract_month"])[["imp"]]
    sr1 = sr1.set_index(["date", "contract_month"])[["imp"]]

    joined = zq.join(sr1, lsuffix="_zq", rsuffix="_sr1", how="inner").dropna()
    joined["basis_bps"] = joined["imp_zq"] - joined["imp_sr1"]
    joined = joined.reset_index()
    joined["months_ahead"] = ((joined["contract_month"].dt.year - joined["date"].dt.year) * 12
                              + joined["contract_month"].dt.month - joined["date"].dt.month)
    return joined


def basis_lookup(fut: pd.DataFrame, asof: pd.Timestamp | None = None,
                 lookback_days: int = 500) -> pd.Series:
    """Median basis (bps) by months-ahead bucket, as of `asof` (default: last)."""
    b = forward_basis(fut)
    if asof is not None:
        b = b[b["date"] <= pd.Timestamp(asof)]
    b = b[b["date"] >= b["date"].max() - pd.Timedelta(days=lookback_days)]
    bucket = b["months_ahead"].clip(upper=36)
    return b.groupby(bucket)["basis_bps"].median()


def adjust_effr_to_sofr(strip: pd.DataFrame, lookup: pd.Series,
                        date_col: str = "date") -> pd.Series:
    """Subtract median basis from EFFR-implied forwards (bps), row-wise."""
    ma = ((strip["contract_month"].dt.year - strip[date_col].dt.year) * 12
          + strip["contract_month"].dt.month - strip[date_col].dt.month).clip(upper=36)
    adj = ma.map(lookup).fillna(0.0)
    return strip["implied"] - adj
