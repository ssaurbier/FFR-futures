import pandas as pd
import pytest

from fedwatch.contracts import (contract_month_start, implied_rate_bps,
                                split_symbol, sr3_accrual, zq_sr1_accrual)


def test_split_symbol_two_digit():
    root, month, year, _ = split_symbol("SR3H27")
    assert (root, month, year) == ("SR3", 3, 2027)


def test_split_symbol_one_digit_resolved_by_asof():
    cm = contract_month_start("ZQF1", pd.Timestamp(2021, 1, 15))
    assert (cm.year, cm.month) == (2021, 1)
    cm2 = contract_month_start("ZQF1", pd.Timestamp(2010, 12, 15))
    assert (cm2.year, cm2.month) == (2011, 1)
    # expired same-decade candidate is dropped by the 2-month floor
    cm3 = contract_month_start("ZQF1", pd.Timestamp(2011, 6, 1))
    assert (cm3.year, cm3.month) == (2021, 1)


def test_sr3_accrual_imm_quarter():
    s, e = sr3_accrual(pd.Timestamp(2026, 12, 1))
    assert s == pd.Timestamp(2026, 12, 16)      # 3rd Wednesday of Dec 2026
    assert e == pd.Timestamp(2027, 3, 17)       # 3rd Wednesday of Mar 2027


def test_zq_accrual_calendar_month():
    s, e = zq_sr1_accrual(pd.Timestamp(2024, 2, 1))
    assert (s, e) == (pd.Timestamp(2024, 2, 1), pd.Timestamp(2024, 2, 29))


def test_implied_rate_roundtrip():
    assert implied_rate_bps(95.25) == 475.0
    assert implied_rate_bps(100.0) == 0.0


def test_bad_symbol_raises():
    with pytest.raises(ValueError):
        split_symbol("NOTASYMBOL")
