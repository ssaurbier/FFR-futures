"""Regime 1 (0-18M): classic CME FedWatch reconstruction from ZQ futures.

Methodology (parity with CME FedWatch):
  * Each ZQ contract settles to the calendar-month AVERAGE EFFR, quoted as
    price = 100 - rate. implied_bps = (100 - price) * 100.
  * For a contract month M containing an FOMC decision on day k with D total
    days, the implied month average R blends the pre-meeting rate r and the
    post-meeting expectation E_post:
        R * D = r * k + E_post * (D - k)   =>   E_post = (R*D - r*k) / (D - k)
  * The expected number of 25bp steps at the meeting is
        x = (E_post - r) / 25      (clipped to +/- MAX_SINGLE_STEP)
  * A fractional x is split onto the adjacent integer step outcomes (a
    deterministic two-point approximation of the meeting outcome), and the
    cumulative target-level distribution is the convolution of per-meeting
    step distributions across all remaining meetings.
  * Absolute target level v = effr + 25 * (cumulative steps); levels are
    bucketed into [25*ceil(v/25) - 25, 25*ceil(v/25)), labelled by midpoint
    25*ceil(v/25) - 12.5 — identical to CME's bucket construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------- strip building
def implied_strip(fut: pd.DataFrame, root: str = "ZQ",
                  ffill_limit: int = 90) -> pd.DataFrame:
    """date x contract_month matrix of implied rates (bps) for one root."""
    df = fut[fut["root"] == root].copy()
    df["implied"] = (100.0 - df["close"]) * 100.0
    strip = (df.pivot_table(index="date", columns="contract_month", values="implied")
             .sort_index().ffill(limit=ffill_limit))
    strip.columns = pd.DatetimeIndex(strip.columns)
    return strip


# ---------------------------------------------------------------- step algebra
def step_dist(x: float) -> dict[int, float]:
    """Expected x 25bp-steps -> probability mass on adjacent integers n, n+1."""
    s, ax = np.sign(x), abs(x)
    n, f = int(np.floor(ax)), ax - np.floor(ax)
    d = {s * n: 1.0 - f} if n > 0 else {0: 1.0 - f}
    if f > 1e-12:
        d[s * (n + 1)] = d.get(s * (n + 1), 0.0) + f
    return d


def _convolve(grid: dict[int, float], steps: dict[int, float]) -> dict[int, float]:
    new: dict[int, float] = {}
    for lvl, mass in grid.items():
        for st, p in steps.items():
            j = lvl + st
            if abs(j) <= config.MAX_STEPS:
                new[j] = new.get(j, 0.0) + mass * p
    return new


def bucket_midpoints(levels_bps: np.ndarray | pd.Series) -> np.ndarray:
    """FedWatch bucket label: 25*ceil(v/25) - 12.5 (bps)."""
    return 25.0 * np.ceil(np.asarray(levels_bps, dtype=float) / 25.0) - 12.5


# ---------------------------------------------------------------- engine
def _meetings_by_month(fomc: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map meeting month-start -> decision (last) day of that meeting."""
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    for md in pd.to_datetime(fomc["meeting_date"]).sort_values():
        out[pd.Timestamp(md.year, md.month, 1)] = md
    return out


def solve_fomc_path(strip_row: pd.Series, effr_bps: float, asof: pd.Timestamp,
                    fomc: pd.DataFrame,
                    max_steps: int | None = None,
                    max_single: int | None = None) -> list[dict]:
    """Solve one as-of date's meeting path.

    Returns list of dicts, one per remaining FOMC meeting up to the strip
    horizon, each carrying the post-meeting expected rate and the cumulative
    target-level grid {steps_from_today: probability}.
    """
    max_steps = max_steps or config.MAX_STEPS
    max_single = max_single if max_single is not None else config.MAX_SINGLE_STEP

    meet_by_month = _meetings_by_month(fomc)
    row = strip_row.dropna()
    if row.empty:
        return []
    m0 = pd.Timestamp(year=asof.year, month=asof.month, day=1)
    if sorted(row.index)[0] > m0 + pd.DateOffset(months=2):
        return []          # front contract missing -> cannot anchor path

    r: float = effr_bps
    grid: dict[int, float] = {0: 1.0}
    started = False
    out: list[dict] = []
    for m in sorted(row.index):
        mtg = meet_by_month.get(m)
        if mtg is not None and mtg > asof:
            D, k = m.days_in_month, min(mtg.day, m.days_in_month)
            if D - k >= 1 and not np.isnan(row[m]):
                e_post = (row[m] * D - r * k) / (D - k)
                x = float(np.clip((e_post - r) / 25.0, -max_single, max_single))
                grid = _convolve(grid, step_dist(x))
                out.append({"asof": asof, "meeting_date": mtg,
                            "effr": effr_bps, "e_pre": r, "e_post": e_post,
                            "exp_steps": x, "grid": dict(grid)})
                r, started = e_post, True
        elif (started or m >= m0) and not np.isnan(row[m]):
            r = row[m]    # no meeting this month: forward simply becomes strip
    return out


def fomc_probabilities(fut: pd.DataFrame, fred_bps: pd.DataFrame,
                       fomc: pd.DataFrame, root: str = "ZQ") -> pd.DataFrame:
    """Full-history run: every as-of date x every remaining meeting.

    Returns records frame [asof, meeting_date, effr, e_pre, e_post, exp_steps,
    grid] — `grid` maps cumulative 25bp steps to probability.
    """
    strip = implied_strip(fut, root=root)
    effr = fred_bps["EFFR"].reindex(strip.index).ffill()

    records: list[dict] = []
    for t, row in strip.iterrows():
        e = effr.loc[t]
        if np.isnan(e):
            continue
        records.extend(solve_fomc_path(row, float(e), t, fomc))
    return pd.DataFrame(records)


# ---------------------------------------------------------------- output formats
def meeting_matrix(records: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """FedWatch-style single-date table: rows = meetings, cols = bucket mids.

    Values: probability that the target RANGE MIDPOINT (via EFFR anchor +
    cumulative steps) falls in the bucket after that meeting.
    """
    day = records[records["asof"] == pd.Timestamp(asof)]
    rows = {}
    for rec in day.itertuples():
        levels = rec.effr + 25.0 * np.array(list(rec.grid.keys()), dtype=float)
        mids = bucket_midpoints(levels)
        for mid, mass in zip(mids, rec.grid.values()):
            rows.setdefault(rec.meeting_date, {}).setdefault(mid, 0.0)
            rows[rec.meeting_date][mid] += mass
    mat = pd.DataFrame(rows).T.sort_index()
    mat.index.name = "meeting_date"
    if len(mat):
        mat = mat.div(mat.sum(axis=1), axis=0)      # renormalise to 1 per meeting
    return mat.reindex(columns=sorted(mat.columns)).fillna(0.0)


def expected_path(records: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Meeting-by-meeting expected target (bps) + hike/cut probabilities."""
    day = records[records["asof"] == pd.Timestamp(asof)]
    rows = []
    for rec in day.itertuples():
        probs = np.array(list(rec.grid.values()))
        steps = np.array(list(rec.grid.keys()), dtype=float)
        p_up = sum(p for s, p in rec.grid.items() if s > 0)
        p_dn = sum(p for s, p in rec.grid.items() if s < 0)
        rows.append({"meeting_date": rec.meeting_date,
                     "target_after_bps": rec.effr + 25.0 * float((steps * probs).sum()),
                     "p_hike": p_up, "p_cut": p_dn, "p_hold": rec.grid.get(0, 0.0),
                     "exp_steps": rec.exp_steps})
    return pd.DataFrame(rows)
