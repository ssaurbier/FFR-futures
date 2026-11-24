"""End-to-end pipeline: assembles the unified 0-30Y SOFR discount curve.

Node schedule (sequential bootstrap, cheapest-to-liquid preference):
  1. anchor        : as-of date, DF = 1
  2. SR1 (1M SOFR) : calendar-month forwards, chained; the front (partial)
                     month is day-weighted between realized SOFR-to-date and
                     the contract's implied remaining average
  3. SR3 (3M SOFR) : convexity-adjusted IMM-quarter forwards beyond the SR1
                     segment, ACT/360 simple compounding
  4. Eris (YI*)    : forward-starting par swaps -> terminal DF nodes out to 30Y

Everything is expressed on one DiscountCurve (Hagan-West monotone convex).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .calendars import load_fomc
from .contracts import attach_contract_months, sr3_accrual
from .curve_bootstrap import DiscountCurve, yearfrac
from .ingest import databento_io, eris_io, fred_io
from .regime1_fomc import fomc_probabilities
from .regime2_sr3 import adjusted_strip
from .regime3_eris import extend_with_eris, forward_strip


# ---------------------------------------------------------------- data bundle
def load_bundle(refresh: bool = False):
    fut = attach_contract_months(databento_io.load_futures(refresh=refresh))
    fred = fred_io.fred_bps(fred_io.load_fred(refresh=refresh))
    fomc = load_fomc()
    prices, ref, parcurve = eris_io.load_eris()
    return {"fut": fut, "fred": fred, "fomc": fomc,
            "eris_prices": prices, "eris_ref": ref, "eris_parcurve": parcurve}


# ---------------------------------------------------------------- curve build
def _sr1_nodes(fut, fred, asof, max_months=18):
    """Chain SR1 month forwards from asof; returns {month_end_date: df}."""
    df = fut[(fut.root == "SR1") & (fut.date <= asof)]
    px = (df.pivot_table(index="date", columns="contract_month", values="close")
          .sort_index().ffill(limit=60))
    row = px.loc[:asof].iloc[-1].dropna()
    row = row[row.index >= pd.Timestamp(asof.year, asof.month, 1)]

    sofr = fred["SOFR"].dropna()
    r_now = float(sofr.asof(asof)) / 1e4          # bps -> rate units
    nodes, prev_df = {}, 1.0
    for m, price in row.items():
        if len(nodes) >= max_months:
            break
        rate = (100.0 - float(price)) / 100.0           # avg SOFR over month
        m_end = m + pd.offsets.MonthEnd(0)
        if m.month == asof.month and m.year == asof.year:
            # day-weight front month: realized-to-date vs implied remaining
            D, d = m.days_in_month, asof.day
            if D - d >= 1:
                rate = (rate * D - r_now * d) / (D - d)
            tau = (m_end - asof).days / 360.0           # remaining days (inclusive)
        else:
            tau = m.days_in_month / 360.0               # full month, inclusive
        prev_df = prev_df / (1.0 + rate * tau)
        nodes[m_end] = prev_df
    return nodes


def _sr3_nodes(fut, asof, sigma=None, last_node_date=None):
    """Convexity-adjusted SR3 quarter nodes beyond `last_node_date`."""
    strip = adjusted_strip(fut, asof=asof, sigma=sigma)
    nodes = {}
    for r in strip.itertuples():
        end = pd.Timestamp(r.accrual_end)
        if last_node_date is not None and end <= last_node_date:
            continue
        nodes[r.accrual_start] = None                 # marker (start anchor)
    return strip


def build_curve(bundle, asof: pd.Timestamp | None = None, sigma=None,
                sr1_months=2) -> dict:
    """Assemble the unified curve; returns diagnostics bundle.

    Segment policy (liquidity-ordered): SR1 bridges asof -> first IMM SR3
    quarterly (liquid front months only); SR3 quarterlies carry to the strip
    end; Eris anchors the long end. SR1 back months beyond ~2M are thin and
    can dislocate from SR3 quarterlies by >10bp, so they are not used.
    """
    fut, fred = bundle["fut"], bundle["fred"]
    asof = pd.Timestamp(asof or fut["date"].max())

    # 1-2) SR1 months
    nodes = _sr1_nodes(fut, fred, asof, max_months=sr1_months)
    dates = [asof] + list(nodes.keys())
    dfs = [1.0] + list(nodes.values())
    sr1_last = max(nodes.keys()) if nodes else asof

    # 3) SR3 IMM quarters beyond SR1 segment (skip overlapping serial months:
    #    only quarterly-cycle windows — starts in Mar/Jun/Sep/Dec — are liquid
    #    and non-overlapping)
    sr3 = adjusted_strip(fut, asof=asof, sigma=sigma)
    curve = DiscountCurve(asof, dates, dfs)
    for r in sr3.itertuples():
        start, end = pd.Timestamp(r.accrual_start), pd.Timestamp(r.accrual_end)
        if end <= sr1_last or start.month not in (3, 6, 9, 12):
            continue
        tau = (end - start).days / 360.0
        fwd_rate = r.rate_ois_bps / 1e4
        if start > curve.dates[-1]:
            # gap (illiquid SR1 tail): chain from last node over the gap
            p_start = curve.df(start)                 # flat-fwd extrapolation
        else:
            p_start = curve.df(start)
        dates.append(end)
        dfs.append(p_start / (1.0 + fwd_rate * tau))
        curve = DiscountCurve(asof, dates, dfs)
    sr3_curve = curve

    # 4) Eris long end
    estrip = forward_strip(bundle["eris_prices"], bundle["eris_ref"], asof)
    eris_nodes, eris_eval = extend_with_eris(sr3_curve, estrip)
    dates += list(eris_nodes.keys())
    dfs += list(eris_nodes.values())
    curve = DiscountCurve(asof, dates, dfs)

    return {"asof": asof, "curve": curve, "sr3_curve": sr3_curve,
            "sr3_strip": sr3, "eris_strip": estrip, "eris_eval": eris_eval,
            "eris_nodes": eris_nodes, "bundle": bundle}


# ---------------------------------------------------------------- seam checks
def seam_check(res: dict) -> pd.DataFrame:
    """Eris-implied forward par swaps vs SR3-segment curve (overlap 1-3Y).

    For tenors whose maturity lies inside the SR3 segment, we price the Eris
    vintage's implied par (coupon vs price) using ONLY the SR3-segment curve
    annuities — a pure cross-product consistency test.
    """
    from .regime3_eris import implied_par_rates
    sr3_curve = res["sr3_curve"]
    overlap = res["eris_strip"][res["eris_strip"]["maturity"]
                                <= sr3_curve.dates[-1]]
    if overlap.empty:
        return pd.DataFrame()
    implied = implied_par_rates(overlap, sr3_curve)
    rows = []
    for r in implied.itertuples():
        par_sr3, _ = sr3_curve.par_swap(r.effective, r.maturity)
        rows.append({"tenor_years": r.tenor_years, "zcode": r.zcode,
                     "effective": r.effective.date(),
                     "eris_implied_bps": r.par_rate_bps,
                     "sr3_curve_bps": par_sr3 * 1e4})
    out = pd.DataFrame(rows)
    out["gap_bps"] = out["eris_implied_bps"] - out["sr3_curve_bps"]
    return out


# ---------------------------------------------------------------- convenience
def run(refresh: bool = False, asof=None, sigma=None):
    bundle = load_bundle(refresh=refresh)
    if asof is None:
        # as-of alignment: never compare instruments priced on different days
        # (e.g., Monday futures vs Friday Eris settlements)
        asof = min(bundle["fut"]["date"].max(),
                   bundle["eris_prices"]["date"].max(),
                   bundle["fred"].index.max())
    res = build_curve(bundle, asof=asof, sigma=sigma)
    res["seam"] = seam_check(res)
    return res


def master_matrix(res: dict, records: pd.DataFrame,
                  tenors=(2, 3, 5, 7, 10, 15, 20, 30)) -> pd.DataFrame:
    """Unified probability matrix: FOMC meeting columns + tenor columns.

    Rows = 25bp bucket midpoints (bps). Regime-1 columns are labelled by
    meeting date (risk-neutral, FedWatch parity); tenor columns come from
    the Phase-1 model densities (historical-vol based) and are flagged as
    model-based until option-implied PDFs arrive in Phase 2.
    """
    from .densities import historical_vol_densities
    from .regime1_fomc import meeting_matrix

    fomc_mat = meeting_matrix(records, res["asof"])            # meetings x mids
    tenor_mat = historical_vol_densities(res["curve"], tenors=tenors,
                                         fut=res["bundle"]["fut"],
                                         asof=res["asof"])      # mids x tenors

    cols = {}
    for meeting, row in fomc_mat.iterrows():
        cols[pd.Timestamp(meeting).strftime("%Y-%m-%d")] = row
    for T, col in tenor_mat.items():
        cols[f"{T}Y (model)"] = col
    mat = pd.DataFrame(cols).fillna(0.0).sort_index()
    # FedWatch buckets are non-negative; truncate the normal tail below zero
    mat = mat[mat.index >= 0.0]
    mat = mat.div(mat.sum(axis=0), axis=1)
    mat.index.name = "bucket_mid_bps"
    return mat
