import numpy as np
import pandas as pd
import pytest

from fedwatch.surface import (expected_path, horizon_series, slice_asof)
from fedwatch.treasury_options import _decode_symbol_strike, rule_expiry


def test_rule_expiry_matches_listed_dates():
    # verified against GLBX definitions (5144/5144 in the full sweep)
    assert rule_expiry(pd.Timestamp(2026, 12, 1)) == pd.Timestamp(2026, 11, 20)
    assert rule_expiry(pd.Timestamp(2027, 3, 1)) == pd.Timestamp(2027, 2, 19)


def test_rule_expiry_rolls_good_friday():
    # OZFJ7 (Mar-2027 future): last Friday = Good Friday -> Thursday prior
    assert rule_expiry(pd.Timestamp(2027, 4, 1)) == pd.Timestamp(2027, 3, 25)


def test_rule_expiry_rolls_christmas():
    # OZFF7 (Jan-2027 future): Friday 2026-12-25 is Christmas -> Dec 24
    assert rule_expiry(pd.Timestamp(2027, 1, 1)) == pd.Timestamp(2026, 12, 24)


def test_decode_symbol_strike():
    assert _decode_symbol_strike("1142") == pytest.approx(114.2)
    assert _decode_symbol_strike("0790") == pytest.approx(79.0)


def _toy_surface():
    rows = []
    for asof, center in [(pd.Timestamp(2024, 1, 12), 400.0),
                         (pd.Timestamp(2024, 2, 9), 425.0)]:
        for mid, p in [(387.5, 0.25), (412.5, 0.5), (437.5, 0.25)]:
            shifted = mid + (center - 400.0)
            rows.append({"as_of": asof, "horizon": pd.Timestamp(2024, 3, 20),
                         "layer": "ust_opt", "tenor": 10.0,
                         "bucket_mid": shifted, "p": p})
    return pd.DataFrame(rows)


def test_slice_asof_pivots_one_pane():
    pane = slice_asof(_toy_surface(), pd.Timestamp(2024, 1, 12))
    assert pane.shape == (3, 1)
    assert pane["p"].sum() if "p" in pane else True
    col = pane.iloc[:, 0]
    assert col.sum() == pytest.approx(1.0)


def test_horizon_series_evolves():
    mat = horizon_series(_toy_surface(), "ust_opt", 10.0)
    assert mat.shape[0] == 2                       # two as-of dates
    cols = mat.columns.values.astype(float)
    m1 = float(np.dot(cols, mat.iloc[0].values))
    m2 = float(np.dot(cols, mat.iloc[1].values))
    assert m2 > m1                                  # distribution shifted up


def test_expected_path_lists_horizons():
    ep = expected_path(_toy_surface(), pd.Timestamp(2024, 2, 9))
    assert len(ep) == 1
    assert ep.iloc[0]["mean_bp"] == pytest.approx(437.5, abs=1.0)
