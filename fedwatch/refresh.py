"""MANUAL data refresh — pulls all marginal data from every source.

POLICY (per project owner, 2026-11-23): NO scheduled or automatic pulls.
Nothing in this package ever fetches new data unless you explicitly run
`refresh_all()` — from the notebook's MANUAL REFRESH cell or the CLI.
Typical cost per invocation after a gap of N days: ~$0.10 + ~$0.15*N.

Sources refreshed:
  * Databento GLBX.MDP3  — futures ohlcv-1d gap, SR3 + Treasury options gaps
  * FRED                 — realized anchors (EFFR/SOFR/targets/CMTs)
  * Eris public FTP      — daily instruments/settlements + par coupon curve;
                           the full Historical_Prices file only when forced
"""
from __future__ import annotations

import time

import pandas as pd

from . import config
from .ingest import databento_io, eris_io, fred_io


def _options_gap(parquet_path, root, client):
    """Append missing months of one option root's ohlcv to its parquet."""
    from pathlib import Path
    p = Path(parquet_path)
    if not p.exists():
        return 0, f"{root}: cache missing — run the backfill first"
    old = pd.read_parquet(p)
    old["date"] = pd.to_datetime(old["date"])
    start = old["date"].max() + pd.Timedelta(days=1)
    end = pd.Timestamp.today().normalize()
    if start > end:
        return 0, f"{root}: current ({old['date'].max().date()})"
    frames = []
    months = pd.date_range(start, end, freq="MS")
    for i, s in enumerate(months):
        e = (months[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(months) else end
        for a in range(5):
            try:
                res = client.timeseries.get_range(
                    dataset=config.DATABENTO_DATASET, schema="ohlcv-1d",
                    symbols=f"{root}.OPT", stype_in="parent",
                    start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"))
                frames.append(databento_io._normalize(res.to_df(), root))
                break
            except Exception:
                time.sleep(3 * 2 ** a)
        time.sleep(0.6)
    if not frames:
        return 0, f"{root}: no new data pulled"
    merged = pd.concat([old] + frames, ignore_index=True) \
        .drop_duplicates(subset=["instrument_id", "date"], keep="last")
    merged.to_parquet(p, index=False)
    return len(merged) - len(old), f"{root}: {start.date()} -> {merged['date'].max().date()} (+{len(merged)-len(old):,} rows)"


def refresh_all(max_cost_usd: float = 10.0, include_eris_history: bool = False,
                refresh_fomc: bool = False) -> dict:
    """Pull all marginal data. MANUAL ONLY — never scheduled.

    Returns a summary dict; also prints per-source lines.
    """
    log = {}

    # 1) futures (built-in cost guard + interactive prompt suppressed)
    try:
        fut = databento_io.refresh_futures(max_cost_usd=max_cost_usd, interactive=False)
        log["futures"] = f"{fut['date'].max().date()} ({len(fut):,} rows)"
    except Exception as e:
        log["futures"] = f"skipped/failed: {e}"

    # 2) FRED
    try:
        fred = fred_io.load_fred(refresh=True)
        log["fred"] = f"{fred.index.max().date()}"
    except Exception as e:
        log["fred"] = f"failed: {e}"

    # 3) Eris FTP (daily files; full price history only when forced)
    try:
        eris_io.refresh_instruments()
        if include_eris_history:
            eris_io.load_eris(refresh_prices=True)
        if not eris_io.PARCOUPON_PARQUET.exists():
            eris_io.load_eris()
        log["eris"] = "instruments settles + parcurve refreshed"
    except Exception as e:
        log["eris"] = f"failed: {e}"

    # 4) options gaps
    cli = databento_io._client()
    try:
        n, msg = _options_gap(config.DATA_DIR / "options_sr3_ohlcv1d.parquet", "SR3", cli)
        log["sr3_options"] = msg
    except Exception as e:
        log["sr3_options"] = f"failed: {e}"
    for root in ["OZF", "OZN", "OZB"]:
        try:
            n, msg = _options_gap(config.DATA_DIR / "treasury_opt_ohlcv1d.parquet", root, cli)
            log[f"{root.lower()}_options"] = msg
        except Exception as e:
            log[f"{root.lower()}_options"] = f"failed: {e}"

    # 5) FOMC calendar (re-scrape: new meeting dates announced a few times/yr)
    if refresh_fomc:
        try:
            from .calendars import load_fomc
            f = load_fomc(refresh=True)
            log["fomc"] = f"{len(f)} meetings -> {f.meeting_date.max().date()}"
        except Exception as e:
            log["fomc"] = f"failed: {e}"

    for k, v in log.items():
        print(f"  {k:14s} {v}")
    return log
