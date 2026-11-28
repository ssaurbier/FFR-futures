"""THE flagship product: the 30-year FedWatch.

FedWatch gives, for today: a ladder of future FOMC dates with the
probability distribution of the target rate at each — out to ~2 years
(the listed horizon of Fed funds options).

`forward30()` gives the same thing out to 30 YEARS: one ladder covering
every future date the market can price, each rung a 25bp-bucketed
probability distribution of the expected future interest rate:

  horizon 0-18M   FOMC meeting ladder ......... ZQ futures (FedWatch parity)
  horizon <=2.1Y  option expiries ............. SR3 option PDFs (SABR/BL)
  tenor rungs     5/10/30Y UST yields ......... Treasury option PDFs
  ALL horizons    0-30Y, every date ........... futures strip mean (SR1/
                   SR3/Eris — real market forwards) with market-calibrated
                   vol term structure (option-anchored, mean-reverting
                   extrapolation)

Each rung carries `source` so market-implied vs calibrated sections are
explicit. This is the single table the whole project exists to produce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LADDER_YEARS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0,
                10.0, 15.0, 20.0, 25.0, 30.0)


def forward30(asof: pd.Timestamp | None = None,
              curve_hist: pd.DataFrame | None = None,
              surface: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 30-year forward probability ladder at `asof`.

    Returns rows = rungs: [future_date, horizon_years, source, mean_bp,
    sd_bp, p25, p50, p75, top_bucket] with the full 25bp distribution in
    wide columns bucket_<mid>.
    """
    from .curve_history import load_curve_history
    from .surface import load_surface

    if curve_hist is None:
        curve_hist = load_curve_history()
    if surface is None:
        surface = load_surface()

    # --- pick the data date at/before asof (None -> latest)
    if asof is None:
        t = curve_hist["as_of"].max()
    else:
        t = curve_hist[curve_hist["as_of"] <= pd.Timestamp(asof)]["as_of"].iloc[-1]
    zero = curve_hist[curve_hist["as_of"] == t].set_index("horizon_years")["zero_bp"]

    day = surface[surface["as_of"] == t]
    best = {"sr3_opt": 0, "ust_opt": 1, "model": 2, "fomc": 3}

    rows = []
    for h in LADDER_YEARS:
        # nearest available rung value on the futures strip
        fwd = float(zero.index[np.argmin(np.abs(zero.index.values - h))])
        rung_zero = zero.loc[fwd]
        future_date = t + pd.Timedelta(days=round(365.25 * h))

        # probability rung: most market-implied layer covering this horizon
        cand = day[(day["horizon_years"] - h).abs() <= max(0.35, 0.25 * h)]
        source = "futures+calibrated_vol"
        w = None
        if len(cand):
            cand = cand.assign(pref=cand["layer"].map(best)).sort_values("pref")
            g = cand[cand["pref"] == cand["pref"].iloc[0]]
            w = g.groupby("bucket_mid")["p"].sum()
            w = w / w.sum()
            source = {"sr3_opt": "sr3_options", "ust_opt": "ust_options",
                      "model": "futures+calibrated_vol",
                      "fomc": "zq_fomc"}[g["layer"].iloc[0]]
        if w is None:
            rows.append({"future_date": future_date, "horizon_years": h,
                         "source": "futures_strip", "mean_bp": rung_zero})
            continue
        mids, p = w.index.values, w.values
        mean = float((mids * p).sum())
        sd = float((p * (mids - mean) ** 2).sum() ** 0.5)
        cdf = np.cumsum(p)
        row = {"future_date": future_date, "horizon_years": h,
               "source": source, "mean_bp": mean, "sd_bp": sd,
               "p25": float(mids[np.searchsorted(cdf, 0.25)]),
               "p50": float(mids[np.searchsorted(cdf, 0.50)]),
               "p75": float(mids[np.searchsorted(cdf, 0.75)]),
               "top_bucket": float(mids[np.argmax(p)])}
        for mid, prob in zip(mids, p):
            row[f"bucket_{mid:.1f}"] = prob
        rows.append(row)
    return pd.DataFrame(rows)


def forward30_matrix(ladder: pd.DataFrame) -> pd.DataFrame:
    """Bucket x rung probability matrix for heatmapming."""
    cols = [c for c in ladder.columns if c.startswith("bucket_")]
    mat = ladder.set_index("horizon_years")[cols].T
    mat.index = (mat.index.str.replace("bucket_", "").astype(float))
    return mat.sort_index()
