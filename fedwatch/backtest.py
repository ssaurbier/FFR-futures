"""Historical validation: did the probability engine predict realized outcomes?

Compares the most-likely post-meeting target bucket against the realized
target (DFEDTARU) bucket, bucketed by days-to-meeting — the same harness as
the original research notebook (cell 65), now on the package's record schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .regime1_fomc import bucket_midpoints


def backtest(records: pd.DataFrame, fred_bps: pd.DataFrame,
             end: str = "2026-11-01") -> pd.DataFrame:
    """Hit-rate table by days-to-meeting bins."""
    target = fred_bps["DFEDTARU"].ffill()

    bt = records[records["meeting_date"] < end].copy()
    if bt.empty:
        return pd.DataFrame()

    bt["realized_bucket"] = bucket_midpoints(
        target.reindex(bt["meeting_date"], method="ffill").values)
    bt["dtm"] = (bt["meeting_date"] - bt["asof"]).dt.days

    # most likely bucket + probability of realized outcome per (asof, meeting)
    rows = []
    for rec in bt.itertuples():
        grid = rec.grid
        if not grid:
            continue
        levels = rec.effr + 25.0 * np.array(list(grid.keys()), dtype=float)
        probs = np.array(list(grid.values()))
        mids = bucket_midpoints(levels)
        df = pd.DataFrame({"mid": mids, "p": probs}).groupby("mid")["p"].sum()
        rows.append({"dtm": rec.dtm, "realized": rec.realized_bucket,
                     "top": df.idxmax(), "p_realized": float(df.get(rec.realized_bucket, 0.0))})
    hits = pd.DataFrame(rows)
    if hits.empty:
        return pd.DataFrame()

    hits["hit"] = hits["top"] == hits["realized"]
    bins = pd.cut(hits["dtm"], [0, 7, 14, 30, 90, 180, 400])
    return (hits.groupby(bins, observed=True)
                 .agg(hit_rate=("hit", "mean"),
                      avg_p_realized=("p_realized", "mean"),
                      n=("hit", "size"))
                 .round(3))
