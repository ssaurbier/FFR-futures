"""SR3 option chains: symbol parsing and as-of chain snapshots.

GLBX raw option symbols look like 'SR3F7 C9225' or 'SR3F27 C9225':
  underlying future = root + month-code + year (1- or 2-digit), C/P flag,
  strike in hundredths (9225 -> 92.25 price points = 7.75% rate strike).

Horizon convention: an option's horizon T = as-of -> 3rd Wednesday of the
UNDERLYING futures' contract month (the futures final settlement date), which
is within a couple of business days of the listed option expiry — the
difference is immaterial for T measured in years.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .calendars import third_wednesday
from .contracts import contract_month_start

_OPT_RE = re.compile(r"^(SR3)([FGHJKMNQUVXZ])(\d{1,2})\s+([CP])(\d{4,5})$")


def _decode_strike(digits: str) -> float:
    """Hundredths encoding, mod-10000 wraparound for strikes at/above par.

    '9225' -> 92.25, '0000' -> 100.00, '0025' -> 100.25, '10000' -> 100.00.
    """
    n = int(digits)
    if n >= 5000:
        return n / 100.0                # 50.00 .. 100.00 face value
    return (10000 + n) / 100.0          # wraparound: 100.00 .. 104.99


def parse_option_symbol(sym: str) -> dict | None:
    m = _OPT_RE.match(sym.strip())
    if not m:
        return None
    root, mcode, ydigits, cp, strike = m.groups()
    return {"underlying": f"SR3{mcode}{ydigits}", "cp": cp,
            "month_code": mcode, "year_digits": ydigits,
            "strike_price": _decode_strike(strike)}


def underlying_month(sym_underlying: str, asof: pd.Timestamp) -> pd.Timestamp | None:
    return contract_month_start(sym_underlying, asof)


def option_expiry(underlying_month: pd.Timestamp) -> pd.Timestamp:
    return third_wednesday(underlying_month.year, underlying_month.month)


def chain_snapshot(opt: pd.DataFrame, fut: pd.DataFrame, asof: pd.Timestamp,
                   stale_days: int = 5) -> dict[str, pd.DataFrame]:
    """Latest option chains per underlying future, joined with the underlying price.

    Returns {underlying: DataFrame[cp, strike_price, price, volume,
    strike_bps, fut_price, fut_rate_bps, T_years]}.
    """
    asof = pd.Timestamp(asof)
    opt = opt[opt["date"] <= asof].sort_values("date")
    last = opt.groupby("symbol").tail(1)
    last = last[(asof - last["date"]).dt.days <= stale_days]

    fut_px = (fut[(fut["root"] == "SR3") & (fut["date"] <= asof)]
              .sort_values("date").groupby("symbol").tail(1)
              .set_index("symbol")["close"])

    parsed = [parse_option_symbol(s) for s in last["symbol"]]
    keep = last[[p is not None for p in parsed]].copy()
    keep["meta"] = [p for p in parsed if p is not None]

    chains: dict[str, pd.DataFrame] = {}
    for und, grp in keep.groupby(lambda i: keep.loc[i, "meta"]["underlying"]):
        um = underlying_month(und, asof)
        if um is None or um <= asof or und not in fut_px.index:
            continue
        expiry = option_expiry(um)
        T = (expiry - asof).days / 365.0
        if T <= 0:
            continue
        f_price = float(fut_px[und])
        rows = []
        for i, r in grp.iterrows():
            meta = r["meta"]
            # option premiums arrive in hundredths of a point (33.5 -> 0.335)
            rows.append({"cp": meta["cp"], "strike_price": meta["strike_price"],
                         "price": float(r["close"]) / 100.0,
                         "volume": float(r["volume"])})
        chain = pd.DataFrame(rows)
        chain["strike_bps"] = (100.0 - chain["strike_price"]) * 100.0
        chain["fut_price"] = f_price
        chain["fut_rate_bps"] = (100.0 - f_price) * 100.0
        chain["T_years"] = T
        chain = chain[chain["price"] > 0]
        chains[und] = chain.sort_values(["cp", "strike_price"]).reset_index(drop=True)
    return chains
