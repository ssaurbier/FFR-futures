import numpy as np
import pandas as pd
import pytest

from fedwatch.curve_bootstrap import DiscountCurve, yearfrac


def _flat_curve(r=0.04):
    asof = pd.Timestamp(2026, 11, 20)
    dates = [asof + pd.Timedelta(days=365 * k) for k in range(1, 11)]
    dfs = [float(np.exp(-r * yearfrac(asof, d))) for d in dates]
    return DiscountCurve(asof, [asof] + dates, [1.0] + dfs)


def test_df_monotone_decreasing():
    c = _flat_curve()
    dfs = [c.df(t) for t in np.linspace(0.1, 10, 50)]
    assert all(x >= y - 1e-12 for x, y in zip(dfs, dfs[1:]))


def test_interval_forwards_conserved():
    """HW construction must reproduce each node interval's average forward."""
    c = _flat_curve(0.04)
    for i in range(len(c.dates) - 1):
        s, e = c.dates[i], c.dates[i + 1]
        tau = yearfrac(s, e)
        fwd = -np.log(c.df(e) / c.df(s)) / tau       # continuous fwd
        assert abs(fwd - 0.04) < 1e-6                # exact by construction


def test_zero_at_today_guard():
    c = _flat_curve()
    assert np.isfinite(c.zero(c.asof))


def test_par_swap_flat_curve():
    """Par rate of an annual swap on a flat 4% curve ~ 4% (ACT/365 vs 360 gaps)."""
    c = _flat_curve(0.04)
    par, ann = c.par_swap(c.asof + pd.Timedelta(days=2),
                          c.asof + pd.DateOffset(years=5))
    assert 0.039 < par < 0.041
    assert 4.0 < ann < 5.0


def test_extrapolation_beyond_last_node():
    c = _flat_curve()
    df35 = c.df(35.0)
    assert 0 < df35 < 1
    assert np.isfinite(df35)


def test_par_swap_short_span_single_period():
    c = _flat_curve(0.04)
    par, ann = c.par_swap(c.asof + pd.Timedelta(days=2),
                          c.asof + pd.Timedelta(days=91))   # 3M swap
    assert 0.039 < par < 0.041
    assert 0.2 < ann < 0.3
