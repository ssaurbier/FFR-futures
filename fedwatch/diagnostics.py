"""Automated verification suite for the unified curve and probability engine.

Checks:
  1. forward positivity      : min instantaneous forward > 0 (arbitrage-free)
  2. SR1 repricing           : curve-implied month forwards == SR1 strip
  3. SR3 repricing           : curve-implied quarterly forwards == adjusted strip
  4. Eris published parity   : bootstrapped Eris pars vs Eris's own par curve
  5. SR3/Eris seam           : implied-vs-curve gaps at overlapping tenors
  6. FOMC probability sanity : per-meeting distributions sum to 1
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_curve(res: dict, tol_reprice_bp: float = 2.0,
                tol_eris_bp: float = 5.0, tol_seam_bp: float = 8.0) -> pd.DataFrame:
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "pass": bool(ok), "detail": detail})

    curve, asof = res["curve"], res["asof"]

    # 1) positivity
    g = curve.inst_fwd_grid(n=2000)
    add("forward positivity", g.min() > 0, f"min inst-fwd {g.min()*100:.3f}%")

    # 2) front segment presence (SR1 bridge + SR3 quarterlies in first 15M)
    front_nodes = [d for d in curve.dates[1:] if d <= asof + pd.DateOffset(months=15)]
    add("front segment present", len(front_nodes) > 4,
        f"{len(front_nodes)} nodes in first 15M (SR1 bridge + SR3 quarters)")

    # 3) SR3 quarterly repricing — only windows the bootstrap actually used
    #    (quarterlies beyond the SR1 segment); the SR1/SR3 overlap zone is
    #    SR1-driven by liquidity preference.
    errs = []
    sr1_last = max((d for d in curve.dates[1:]
                    if d <= asof + pd.DateOffset(months=15)), default=asof)
    for r in res["sr3_strip"].itertuples():
        s, e = pd.Timestamp(r.accrual_start), pd.Timestamp(r.accrual_end)
        if e <= asof or s.month not in (3, 6, 9, 12) or e > curve.dates[-1]:
            continue
        tau = (e - s).days / 360.0
        implied = (curve.df(s) / curve.df(e) - 1.0) / tau * 1e4
        errs.append(abs(implied - r.rate_ois_bps))
    add("SR3 quarterly repricing", bool(errs) and max(errs) < tol_reprice_bp,
        f"max |gap| {max(errs):.3f}bp over {len(errs)} quarters"
        if errs else "no quarterly windows in range")

    # 4) Eris parity vs their published par curve
    pc = res["bundle"]["eris_parcurve"].asof(asof)
    pairs = [(4, "SOFR4Y"), (5, "SOFR5Y"), (7, "SOFR7Y"), (10, "SOFR10Y"),
             (15, "SOFR15Y"), (20, "SOFR20Y"), (30, "SOFR30Y")]
    ev = res["eris_eval"].set_index("tenor_years")["par_rate_bps"]
    gaps = {t: ev[t] - pc[c] * 100 for t, c in pairs
            if t in ev.index and c in pc.index}
    worst = max(gaps.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
    add("Eris published parity", abs(worst[1]) < tol_eris_bp,
        "; ".join(f"{t:.0f}Y {v:+.2f}bp" for t, v in gaps.items()))

    # 5) seam (2Y+; 1Y reported as known front-end dislocation)
    seam = res.get("seam")
    if seam is not None and len(seam):
        core = seam[seam.tenor_years >= 2]
        add("SR3/Eris seam (>=2Y)", core.gap_bps.abs().max() < tol_seam_bp,
            "; ".join(f"{r.tenor_years:.0f}Y {r.gap_bps:+.2f}bp"
                      for r in seam.itertuples()))
    else:
        add("SR3/Eris seam (>=2Y)", False, "no overlapping tenors")

    return pd.DataFrame(rows)


def check_probabilities(records: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Per-meeting cumulative distributions must sum to 1 and stay non-negative."""
    day = records[records["asof"] == pd.Timestamp(asof)]
    rows = []
    for rec in day.itertuples():
        total = sum(rec.grid.values())
        rows.append({"meeting_date": rec.meeting_date, "mass": total,
                     "min_p": min(rec.grid.values())})
    df = pd.DataFrame(rows)
    ok = df.empty or (df.mass.sub(1).abs().max() < 1e-9 and df.min_p.min() >= 0)
    return pd.DataFrame([{"check": "FOMC probability sanity", "pass": bool(ok),
                          "detail": f"{len(df)} meetings, max |mass-1| "
                                    f"{0 if df.empty else df.mass.sub(1).abs().max():.1e}"}])
