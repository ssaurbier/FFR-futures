"""Eris public FTP ingestion (files.erisfutures.com/ftp/).

Free authoritative source for the long end:
  * Eris_Historical_Prices.csv      — wide format, per-contract daily
    [NPV(A), AccruedCoupons(B), ErisPAI(C), SettlementPrice] history.
  * Eris_Instruments_YYYYMMDD_Settles.csv — per-contract reference:
    coupon, effective/maturity dates, FairCoupon (their par rate), PV01.
  * Eris_Historical_ParCouponCurve_SOFR.csv — their published 1W-50Y SOFR
    par curve history (validation target for our bootstrap).

Parsed into long-form parquet caches under data/eris/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DATA_DIR

ERIS_DIR = DATA_DIR / "eris"
ERIS_DIR.mkdir(parents=True, exist_ok=True)

FTP = "https://files.erisfutures.com/ftp"
HIST_PRICES_CSV = ERIS_DIR / "Eris_Historical_Prices.csv"
INSTRUMENTS_CSV = ERIS_DIR / "instruments_settles.csv"
PARCOUPON_CSV = ERIS_DIR / "parcoupon_hist.csv"

PRICES_LONG_PARQUET = ERIS_DIR / "prices_long.parquet"
INSTRUMENTS_PARQUET = ERIS_DIR / "instruments.parquet"
PARCOUPON_PARQUET = ERIS_DIR / "parcoupon.parquet"


def _download(url: str, dest: Path, max_mb: int = 200) -> None:
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size > max_mb * 1e6:
        raise RuntimeError(f"{dest.name} unexpectedly large")


# ---------------------------------------------------------------- historical prices
def parse_historical_prices(path: Path = HIST_PRICES_CSV) -> pd.DataFrame:
    """Wide multi-header file -> long [date, zcode, A, B, C, settle].

    Layout: row0 = vintage groups ("Mar 27 Contracts", 4 cols each),
    row1 = short names, row2 = zcodes (ffill across each group's 4 columns),
    row3 = Bloomberg codes, row4 = field names (Date, NPV (A),
    AccruedCoupons (B), ErisPAI (C), SettlementPrice ...), row5+ = data.
    """
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    # locate the field-name row: first column == 'Date'
    field_row = None
    for i in range(min(10, len(raw))):
        if str(raw.iloc[i, 0]).strip() == "Date":
            field_row = i
            break
    if field_row is None:
        raise ValueError("field-name row not found in historical prices file")

    zcodes = raw.iloc[2, 1:].ffill()
    fields = raw.iloc[field_row, 1:]

    body = raw.iloc[field_row + 1:, :].copy()
    body.columns = ["date"] + list(range(1, raw.shape[1]))
    body["date"] = pd.to_datetime(body["date"], format="%m/%d/%Y", errors="coerce")
    body = body.dropna(subset=["date"]).set_index("date")

    frames = []
    for field, name in [("NPV (A)", "A"), ("AccruedCoupons (B)", "B"),
                        ("ErisPAI (C)", "C"),
                        ("SettlementPrice (100+A+B-C)", "settle")]:
        cols = [i for i, f in enumerate(fields, start=1) if str(f).strip() == field]
        if not cols:
            raise ValueError(f"field {field!r} not found in {path.name}")
        sub = body[cols].astype(float)
        sub.columns = [zcodes.iloc[i - 1] for i in cols]
        frames.append(sub.stack().rename(name))
    long = pd.concat(frames, axis=1).reset_index()
    long.columns = ["date", "zcode", "A", "B", "C", "settle"]
    long["date"] = pd.to_datetime(long["date"])
    long = long.dropna(subset=["settle"]).sort_values(["zcode", "date"])
    return long


# ---------------------------------------------------------------- instruments reference
def refresh_instruments(asof: pd.Timestamp | None = None,
                        force: bool = False) -> pd.DataFrame:
    """Download latest Instruments Settles file -> per-contract reference table."""
    if INSTRUMENTS_PARQUET.exists() and not force:
        cached = pd.read_parquet(INSTRUMENTS_PARQUET)
        latest = pd.Timestamp(cached["EvaluationDate"].max())
        if asof is None or latest >= pd.Timestamp(asof) - pd.Timedelta(days=5):
            return cached
    asof = asof or pd.Timestamp.today()
    # walk back up to 7 calendar days to find a published file
    import time
    for back in range(8):
        d = asof - pd.Timedelta(days=back)
        if d.weekday() >= 5:
            continue
        url = f"{FTP}/Eris_Instruments_{d:%Y%m%d}_Settles.csv"
        try:
            _download(url, INSTRUMENTS_CSV)
            break
        except Exception:
            time.sleep(0.3)
    else:
        raise RuntimeError("no Eris Instruments Settles file found in last week")

    df = pd.read_csv(INSTRUMENTS_CSV)
    df["EvaluationDate"] = pd.to_datetime(df["EvaluationDate"])
    df.to_parquet(INSTRUMENTS_PARQUET, index=False)
    return df


def instruments_reference(inst: pd.DataFrame | None = None) -> pd.DataFrame:
    """Slim per-contract reference: zcode, tenor, coupon, dates, fair coupon."""
    inst = inst if inst is not None else pd.read_parquet(INSTRUMENTS_PARQUET)
    ref = pd.DataFrame({
        "zcode": inst["Symbol"],
        "root": inst["ExchangeSymbol (EX005)"],
        "coupon_pct": pd.to_numeric(inst["Coupon (%)"], errors="coerce"),
        "fair_coupon_pct": pd.to_numeric(inst["FairCoupon (%)"], errors="coerce"),
        "effective": pd.to_datetime(inst["EffectiveDate"]),
        "maturity": pd.to_datetime(inst["MaturityDate"]),
        "evaluation_date": pd.to_datetime(inst["EvaluationDate"]),
    })
    # tenor from short name "7Y Dec 2020-2027 SOFR" / root map fallback
    from ..config import ERIS_TENORS
    ref["tenor_years"] = ref["root"].map(ERIS_TENORS)
    m = inst["ShortName"].str.extract(r"^(\d+)Y ")[0]
    ref.loc[m.notna(), "tenor_years"] = pd.to_numeric(m).astype(float)
    return ref.dropna(subset=["zcode"])


# ---------------------------------------------------------------- par coupon curve (validation)
def parse_parcoupon(path: Path = PARCOUPON_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Evaluation Date"] = pd.to_datetime(df["Evaluation Date"])
    return df.set_index("Evaluation Date").sort_index()


# ---------------------------------------------------------------- top-level
def load_eris(refresh_prices: bool = False, asof: pd.Timestamp | None = None):
    """Returns (prices_long, instruments_ref, parcoupon)."""
    if refresh_prices or not PRICES_LONG_PARQUET.exists():
        if not HIST_PRICES_CSV.exists():
            _download(f"{FTP}/Eris_Historical_Prices.csv", HIST_PRICES_CSV,
                      max_mb=200)
        long = parse_historical_prices()
        long.to_parquet(PRICES_LONG_PARQUET, index=False)
    prices = pd.read_parquet(PRICES_LONG_PARQUET)
    prices["date"] = pd.to_datetime(prices["date"])

    if not PARCOUPON_PARQUET.exists():
        if not PARCOUPON_CSV.exists():
            _download(f"{FTP}/Eris_Historical_ParCouponCurve_SOFR.csv", PARCOUPON_CSV)
        parse_parcoupon().to_parquet(PARCOUPON_PARQUET)
    parcurve = pd.read_parquet(PARCOUPON_PARQUET)
    parcurve.index = pd.to_datetime(parcurve.index)

    inst = refresh_instruments(asof=asof)
    ref = instruments_reference(inst)
    return prices, ref, parcurve
