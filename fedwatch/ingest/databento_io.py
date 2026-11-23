"""Databento GLBX.MDP3 ingestion with local parquet caching.

Strategy:
  * ohlcv-1d pulls use `stype_in="parent"` (symbol = "<ROOT>.FUT") so every
    listed contract of a root arrives without hand-enumerating symbols —
    immune to the 1-digit/2-digit year convention change.
  * Rows are keyed by (instrument_id, date) — instrument_id is stable across
    symbol-normalisation changes, so the legacy CSV cache merges cleanly.
  * `refresh_futures()` only pulls dates after the cached max date and
    refuses to spend more than `max_cost_usd` per call (default $5).
  * Instrument definitions (exact expiries, Eris swap specs, option strikes)
    are cached separately for the contract-date reference tables.
"""
from __future__ import annotations

import time

import pandas as pd

from .. import config
from ..config import (DATABENTO_DATASET, DEFS_PARQUET, FUTURES_PARQUET,
                      LEGACY_FUTURES_CSV)

_COLS = ["date", "instrument_id", "symbol", "root", "open", "high", "low",
         "close", "volume"]


def _client():
    import databento as db
    return db.Historical(config.get_databento_key())


def _normalize(df: pd.DataFrame, root: str) -> pd.DataFrame:
    df = df.reset_index()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize(),
        "instrument_id": df["instrument_id"],
        "symbol": df["symbol"].astype(str),
        "root": root,
        "open": df["open"], "high": df["high"], "low": df["low"],
        "close": df["close"], "volume": df["volume"],
    })
    return out


# ---------------------------------------------------------------- cache primitives
def load_cached_futures() -> pd.DataFrame:
    """Parquet cache, falling back to the legacy root CSV on first run."""
    if FUTURES_PARQUET.exists():
        return pd.read_parquet(FUTURES_PARQUET)
    if LEGACY_FUTURES_CSV.exists():
        df = pd.read_csv(LEGACY_FUTURES_CSV)
        df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize()
        # legacy CSV already carries `root`; normalize manually
        keep = pd.DataFrame({
            "date": df["date"],
            "instrument_id": df["instrument_id"],
            "symbol": df["symbol"].astype(str),
            "root": df["root"],
            "open": df["open"], "high": df["high"], "low": df["low"],
            "close": df["close"], "volume": df["volume"],
        })
        return keep
    return pd.DataFrame(columns=_COLS)


def _save_futures(df: pd.DataFrame) -> None:
    df = df.sort_values(["date", "root", "symbol"]).reset_index(drop=True)
    df.to_parquet(FUTURES_PARQUET, index=False)


def merge_futures(cached: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert on (instrument_id, date); new rows win."""
    key = ["instrument_id", "date"]
    combined = pd.concat([cached, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=key, keep="last")
    return combined.sort_values(["date", "root", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------- pulls
def refresh_futures(roots: list[str] | None = None, max_cost_usd: float = 5.0,
                    end: str | None = None, interactive: bool = True) -> pd.DataFrame:
    """Pull ohlcv-1d for all roots from the day after the cache's last date.

    Prints the pre-pull cost estimate and aborts if it exceeds `max_cost_usd`.
    """
    roots = roots or config.FUTURES_ROOTS
    cached = load_cached_futures()
    if cached.empty:
        start = "2010-06-06"
    else:
        start = (cached["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    if start > end:
        print(f"futures cache already current ({cached['date'].max().date()})")
        return cached

    cli = _client()
    parents = [f"{r}.FUT" for r in roots]
    cost = cli.metadata.get_cost(dataset=DATABENTO_DATASET, schema="ohlcv-1d",
                                 symbols=parents, stype_in="parent",
                                 start=start, end=end)
    print(f"pull {roots} ohlcv-1d {start} -> {end}: ${cost:,.2f}")
    if cost > max_cost_usd:
        if not interactive or input("proceed? [y/N] ").strip().lower() != "y":
            raise RuntimeError(f"pull cost ${cost:.2f} exceeds budget ${max_cost_usd}")

    frames = []
    for root in roots:
        res = cli.timeseries.get_range(
            dataset=DATABENTO_DATASET, schema="ohlcv-1d",
            symbols=f"{root}.FUT", stype_in="parent", start=start, end=end)
        frames.append(_normalize(res.to_df(), root))
        time.sleep(0.5)
    fresh = merge_futures(cached, pd.concat(frames, ignore_index=True))
    _save_futures(fresh)
    print(f"futures cache: {len(fresh):,} rows, {fresh['date'].min().date()} -> "
          f"{fresh['date'].max().date()}, {fresh['symbol'].nunique()} contracts")
    return fresh


def refresh_definitions(roots: list[str] | None = None,
                        start: str = "2018-01-01", max_cost_usd: float = 5.0,
                        end: str | None = None) -> pd.DataFrame:
    """Pull instrument definitions for contract-date reference (SR3/Eris/...).

    Merges into the existing definitions cache keyed by instrument_id.
    Streams in yearly chunks with retries — large single requests are prone
    to premature termination.
    """
    roots = roots or config.FUTURES_ROOTS
    end = pd.Timestamp(end or pd.Timestamp.today().strftime("%Y-%m-%d"))
    cli = _client()
    parents = [f"{r}.FUT" for r in roots]
    cost = cli.metadata.get_cost(dataset=DATABENTO_DATASET, schema="definition",
                                 symbols=parents, stype_in="parent",
                                 start=start, end=end.strftime("%Y-%m-%d"))
    print(f"pull definitions {roots} {start} -> {end.date()}: ${cost:,.2f}")
    if cost > max_cost_usd:
        raise RuntimeError(f"definition pull cost ${cost:.2f} exceeds budget ${max_cost_usd}")

    chunks = pd.date_range(start, end, freq="YS")
    starts = [pd.Timestamp(start)] + [c for c in chunks if pd.Timestamp(start) < c <= end]
    frames = []
    for root in roots:
        for i, s in enumerate(starts):
            e = min(starts[i + 1] - pd.Timedelta(days=1), end) if i + 1 < len(starts) else end
            if s > e:
                continue
            for attempt in range(4):
                try:
                    res = cli.timeseries.get_range(
                        dataset=DATABENTO_DATASET, schema="definition",
                        symbols=f"{root}.FUT", stype_in="parent",
                        start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"))
                    df = res.to_df().reset_index()
                    df["root"] = root
                    frames.append(df)
                    break
                except Exception as exc:               # noqa: BLE001
                    wait = 2 ** attempt * 3
                    print(f"  {root} {s.date()}: retry {attempt + 1} after {exc!r}; sleep {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"definitions failed for {root} @ {s.date()}")
            time.sleep(0.4)

    new = pd.concat(frames, ignore_index=True)
    if DEFS_PARQUET.exists():
        old = pd.read_parquet(DEFS_PARQUET)
        new = pd.concat([old, new], ignore_index=True)
    defs = new.drop_duplicates(subset=["instrument_id"], keep="last")
    defs.to_parquet(DEFS_PARQUET, index=False)
    print(f"definitions cached: {len(defs):,} instruments -> {DEFS_PARQUET.name}")
    return defs


def load_definitions() -> pd.DataFrame:
    return pd.read_parquet(DEFS_PARQUET)


# ---------------------------------------------------------------- convenience
def load_futures(refresh: bool = False, **kwargs) -> pd.DataFrame:
    if refresh:
        return refresh_futures(**kwargs)
    df = load_cached_futures()
    if df.empty:
        return refresh_futures(**kwargs)
    return df
