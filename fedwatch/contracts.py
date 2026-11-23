"""Contract symbology and accrual windows for every product family.

Symbol conventions handled:
  * 1-digit year  (legacy GLBX native, e.g. ZQZ5)   -> pre-2026 normalisation
  * 2-digit year  (current, e.g. ZQZ26, YIAZ26)
Accrual windows:
  * ZQ  : calendar month of the contract code month (avg EFFR over the month)
  * SR1 : calendar month of the contract code month (avg SOFR over the month)
  * SR3 : 3M window from IMM Wednesday of contract month to IMM Wednesday +3M
          (holiday-adjusted; see sr3_accrual)
  * YI* : Eris swap effective date = contract expiry; tenor from root (see
          regime3_eris for exact per-instrument specs from definitions)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .calendars import modified_following, third_wednesday, us_holidays

_SYM_RE = re.compile(r"^([A-Z0-9]+?)([FGHJKMNQUVXZ])(\d{1,2})$")


def split_symbol(sym: str) -> tuple[str, int, int]:
    """'SR3H27' -> ('SR3', 3, 2027); 'ZQF1' -> ('ZQ', 1, 2011..2021)."""
    m = _SYM_RE.match(sym)
    if not m:
        raise ValueError(f"unparseable futures symbol: {sym!r}")
    root, mcode, ydigits = m.group(1), m.group(2), m.group(3)
    month = config.MONTH_OF[mcode]
    if len(ydigits) == 1:
        yy = int(ydigits)
        years = (2000 + yy, 2010 + yy, 2020 + yy)   # ambiguous decade -> resolved by caller
    else:
        years = (2000 + int(ydigits),)
    return root, month, years[0], years  # first elem = best guess


def contract_month_start(sym: str, asof: pd.Timestamp) -> pd.Timestamp | None:
    """Earliest sensible contract month >= ~2 months before `asof`.

    Disambiguates 1-digit-year symbols by choosing the candidate closest to
    (but not much before) the trade date, matching how listings actually trade.
    """
    m = _SYM_RE.match(sym)
    if not m:
        return None
    root, mcode, ydigits = m.group(1), m.group(2), m.group(3)
    month = config.MONTH_OF[mcode]
    if len(ydigits) == 2:
        years = (2000 + int(ydigits),)
    else:
        yy = int(ydigits)
        years = (2000 + yy, 2010 + yy, 2020 + yy)
    floor = asof - pd.offsets.MonthBegin(2)
    cands = [pd.Timestamp(year=y, month=month, day=1)
             for y in years if pd.Timestamp(year=y, month=month, day=1) >= floor]
    return min(cands) if cands else None


def zq_sr1_accrual(contract_month: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """ZQ/SR1 accrue over the full calendar month of the contract."""
    start = contract_month
    end = contract_month + pd.offsets.MonthEnd(0)
    return start, end


def sr3_accrual(contract_month: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """SR3 accrual: IMM Wednesday of contract month -> IMM Wednesday 3M later.

    Both endpoints follow the modified-following convention on US holidays
    (start rolls forward; end rolls forward within month else backward).
    """
    start_imm = third_wednesday(contract_month.year, contract_month.month)
    end_month = contract_month + pd.DateOffset(months=3)
    end_imm = third_wednesday(end_month.year, end_month.month)
    hol = us_holidays(start_imm - pd.Timedelta(days=10), end_imm + pd.Timedelta(days=10))
    return modified_following(start_imm, hol), modified_following(end_imm, hol)


def implied_rate_bps(close: float) -> float:
    """Futures price (e.g. 95.25) -> implied rate in bps (e.g. 475.0)."""
    return (100.0 - float(close)) * 100.0


def price_from_rate_bps(rate_bps: float) -> float:
    return 100.0 - rate_bps / 100.0


def attach_contract_months(fut: pd.DataFrame) -> pd.DataFrame:
    """Add `contract_month` (Timestamp) + `maturity_yearsfrac` helper columns."""
    cms = [contract_month_start(s, d) for s, d in zip(fut["symbol"], fut["date"])]
    out = fut.copy()
    out["contract_month"] = pd.to_datetime(cms)
    return out.dropna(subset=["contract_month"])


def month_starts_between(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="MS")
