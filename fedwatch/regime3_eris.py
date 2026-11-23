"""Regime 3 (1Y -> 30Y): Eris SOFR swap futures -> par swap anchors.

Eris mechanics (per erisfutures.com specs):
  * Each contract replicates an annual-fixed vs SOFR-float swap with fixed
    coupon K (set at listing to the rounded forward rate).
  * Price(t) = 100 + A + B - C where A = NPV of remaining flows, B = past
    fixed+floating payments, C = PAI. For FORWARD-STARTING vintages
    (effective > today) B = 0 and C ~ 0, so the price isolates the pure
    forward swap value:  P - 100 = A x 100 with
        A = K x Ann - (P(0,eff) - P(0,mat))     [per unit notional]
  * Implied forward par rate:   S = K - (P - 100) / (100 x Ann).
  * Bootstrapping the terminal DF is then LINEAR in P(0,mat):
        P(0,mat) = [P(0,eff) - K x Ann_partial + (P-100)/100] / (1 + K x tau_n)

We use the nearest forward-starting vintage per tenor, with official Eris
FTP settlements (Eris_Historical_Prices + Instruments Settles for coupons).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_strip(prices: pd.DataFrame, ref: pd.DataFrame,
                  asof: pd.Timestamp) -> pd.DataFrame:
    """Nearest forward-starting Eris vintage per tenor, priced at `asof`.

    Returns [tenor_years, zcode, coupon, effective, maturity, settle, A, B, C].
    """
    asof = pd.Timestamp(asof)
    px = prices[prices["date"] <= asof].sort_values("date").groupby("zcode").tail(1)
    px = px.set_index("zcode")

    fwd = ref[ref["effective"] > asof].sort_values(["tenor_years", "effective"])
    nearest = fwd[fwd.groupby("tenor_years")["effective"].rank(method="first") == 1]

    rows = []
    for r in nearest.itertuples():
        if r.zcode not in px.index:
            continue
        p = px.loc[r.zcode]
        rows.append({"tenor_years": float(r.tenor_years), "zcode": r.zcode,
                     "coupon": float(r.coupon_pct) / 100.0,
                     "effective": r.effective, "maturity": r.maturity,
                     "settle": float(p["settle"]), "A": float(p["A"]),
                     "B": float(p["B"]), "C": float(p["C"])})
    return pd.DataFrame(rows).sort_values("tenor_years").reset_index(drop=True)


def implied_par_rates(strip: pd.DataFrame, curve) -> pd.DataFrame:
    """Add market-implied forward par swap rate per Eris vintage.

    S = K - (settle - 100) / (100 x Ann), Ann from the current curve.
    """
    out = strip.copy()
    anns, pars = [], []
    for r in out.itertuples():
        _, ann = curve.par_swap(r.effective, r.maturity)
        s = r.coupon - (r.settle - 100.0) / (100.0 * ann)
        anns.append(ann)
        pars.append(s)
    out["annuity_years"] = anns
    out["par_rate"] = pars
    out["par_rate_bps"] = out["par_rate"] * 1e4
    return out


def extend_with_eris(curve, strip: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Sequentially extend a DiscountCurve with Eris terminal nodes.

    For each tenor ascending (only maturities beyond the current last node):
        P(0,mat) = [P(0,eff) - K Ann_partial + (P-100)/100] / (1 + K tau_n)
    Returns (new_nodes {date: df}, evaluated strip with par rates).
    """
    from .curve_bootstrap import DiscountCurve

    dates = list(curve.dates)
    dfs = list(np.exp(curve._logdf))
    new_nodes: dict[pd.Timestamp, float] = {}
    evals = []

    for r in strip.sort_values("tenor_years").itertuples():
        if pd.Timestamp(r.maturity) <= curve.dates[-1]:
            continue
        c = DiscountCurve(curve.asof, dates, dfs)
        tau, pay, _ = c.swap_schedule(r.effective, r.maturity)
        # annuity excluding the final payment's DF (unknown terminal node)
        a_partial = sum(t * c.df(d) for t, d in zip(tau[:-1], pay[:-1]))
        tau_n = tau[-1]
        p_eff = c.df(r.effective)
        delta = (r.settle - 100.0) / 100.0
        p_mat = (p_eff - r.coupon * a_partial + delta) / (1.0 + r.coupon * tau_n)
        dates.append(pd.Timestamp(pay[-1]))
        dfs.append(p_mat)
        new_nodes[pd.Timestamp(r.maturity)] = p_mat

        c2 = DiscountCurve(curve.asof, dates, dfs)
        par, ann = c2.par_swap(r.effective, r.maturity)
        evals.append({"tenor_years": r.tenor_years, "zcode": r.zcode,
                      "coupon": r.coupon, "settle": r.settle,
                      "par_rate_bps": par * 1e4, "annuity_years": ann,
                      "df_mat": p_mat})

    return new_nodes, pd.DataFrame(evals)
