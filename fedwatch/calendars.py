"""Calendars: FOMC meeting schedule, IMM dates, US business days."""
from __future__ import annotations

import re

import pandas as pd

from .config import FOMC_CSV, LEGACY_FOMC_CSV

_MONTHS_RE = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS_RE.split("|"))}


# ---------------------------------------------------------------- FOMC
def scrape_fomc() -> pd.DataFrame:
    """Scrape federalreserve.gov for FOMC meeting dates 2000 -> present.

    Returns frame with columns [year, month_num, month, decision_day, meeting_date];
    decision_day = LAST day of a multi-day meeting (the announcement day).
    Ported from the original research notebook (cell 7).
    """
    import requests

    head = {"User-Agent": "Mozilla/5.0"}
    meetings: list[tuple[int, int, int]] = []

    # historical pages (2000-2020): panel headings like "April 29-30 Meeting - 2008"
    for y in range(2000, 2021):
        text = requests.get(
            f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm",
            headers=head, timeout=30,
        ).text
        for m in re.finditer(
            rf"<h5[^>]*>\s*({_MONTHS_RE})\s+(\d{{1,2}})(?:[-/](\d{{1,2}}))?\s+Meeting"
            rf"(?:\s+\(continued\))?\s+-\s+{y}\s*</h5>", text,
        ):
            last = int(m.group(3) or m.group(2))
            meetings.append((y, _MONTH_NUM[m.group(1)], last))

    # current page (2021+ incl. scheduled): month + day-range blocks
    text = requests.get(
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        headers=head, timeout=30,
    ).text
    parts = re.split(r"(20\d\d)\s+FOMC\s+Meetings", text)
    for i in range(1, len(parts) - 1, 2):
        year, body = int(parts[i]), parts[i + 1]
        for m in re.finditer(
            r"fomc-meeting__month[^>]*><strong>([A-Za-z/]+)</strong>.*?"
            r"fomc-meeting__date[^>]*>([^<]*)<", body, re.S,
        ):
            days = re.findall(r"\d{1,2}", m.group(2).replace("*", ""))
            if m.group(1) in _MONTH_NUM and days:
                meetings.append((year, _MONTH_NUM[m.group(1)], int(days[-1])))

    fomc = pd.DataFrame(sorted(set(meetings)),
                        columns=["year", "month_num", "decision_day"])
    fomc["month"] = fomc["month_num"].map({v: k for k, v in _MONTH_NUM.items()})
    fomc["meeting_date"] = pd.to_datetime(
        dict(year=fomc.year, month=fomc.month_num, day=fomc.decision_day))
    return fomc.sort_values("meeting_date").reset_index(drop=True)


def load_fomc(refresh: bool = False) -> pd.DataFrame:
    """FOMC calendar with caching (data/fomc_meetings.csv)."""
    path = FOMC_CSV if FOMC_CSV.exists() else LEGACY_FOMC_CSV
    if refresh or not path.exists():
        fomc = scrape_fomc()
        fomc.to_csv(FOMC_CSV, index=False)
        return fomc
    return pd.read_csv(path, parse_dates=["meeting_date"])


# ---------------------------------------------------------------- IMM / business days
def third_wednesday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    offset = (2 - d.day_of_week) % 7        # days until first Wednesday
    return d + pd.Timedelta(days=offset + 14)


def imm_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """All IMM (3rd Wednesday of Mar/Jun/Sep/Dec) dates in [start, end]."""
    out = []
    for year in range(start.year, end.year + 1):
        for month in (3, 6, 9, 12):
            d = third_wednesday(year, month)
            if start <= d <= end:
                out.append(d)
    return out


def us_holidays(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    return list(cal.holidays(start, end))


def is_business_day(d: pd.Timestamp, holidays: list[pd.Timestamp] | None = None) -> bool:
    if d.weekday() >= 5:
        return False
    return holidays is None or d.normalize() not in holidays


def modified_following(d: pd.Timestamp, holidays: list[pd.Timestamp]) -> pd.Timestamp:
    """Roll forward to next business day; if month changes, roll backward."""
    if is_business_day(d, holidays):
        return d
    nxt = d
    while not is_business_day(nxt, holidays):
        nxt += pd.Timedelta(days=1)
    if nxt.month != d.month:
        prv = d
        while not is_business_day(prv, holidays):
            prv -= pd.Timedelta(days=1)
        return prv
    return nxt


def preceding_business_day(d: pd.Timestamp, holidays: list[pd.Timestamp]) -> pd.Timestamp:
    while not is_business_day(d, holidays):
        d -= pd.Timedelta(days=1)
    return d
