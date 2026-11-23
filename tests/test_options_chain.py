import pandas as pd
import pytest

from fedwatch.options_chain import (chain_snapshot, option_expiry,
                                    parse_option_symbol, underlying_month)


def test_parse_one_and_two_digit_years():
    a = parse_option_symbol("SR3F7 C9225")
    assert a["cp"] == "C" and a["strike_price"] == 92.25
    assert a["underlying"] == "SR3F7"
    b = parse_option_symbol("SR3H27 P10000")
    assert b["cp"] == "P" and b["strike_price"] == 100.00
    assert b["underlying"] == "SR3H27"
    # mod-10000 wraparound strikes at/above par
    c = parse_option_symbol("SR3F7 C0000")
    assert c["strike_price"] == 100.00
    d = parse_option_symbol("SR3F7 C0025")
    assert d["strike_price"] == 100.25


def test_parse_rejects_futures_and_spreads():
    assert parse_option_symbol("SR3Z26") is None
    assert parse_option_symbol("SR3Z26-Z27") is None
    assert parse_option_symbol("GARBAGE") is None


def test_underlying_month_disambiguation():
    asof = pd.Timestamp(2026, 11, 20)
    m = underlying_month("SR3F7", asof)
    assert (m.year, m.month) == (2027, 1)


def test_option_expiry_is_imm_wednesday():
    d = option_expiry(pd.Timestamp(2027, 3, 1))
    assert d == pd.Timestamp(2027, 3, 17) and d.day_name() == "Wednesday"


def _mini_market():
    asof = pd.Timestamp(2026, 11, 20)
    # futures: SR3H7 @ 95.60
    fut = pd.DataFrame([
        {"date": asof, "symbol": "SR3H7", "instrument_id": 1,
         "root": "SR3", "open": 95.6, "high": 95.6, "low": 95.6,
         "close": 95.60, "volume": 100, "contract_month": pd.Timestamp(2027, 3, 1)},
    ])
    # options on SR3H7, struck around the money
    rows = []
    for i, (k, px) in enumerate([(95.25, 0.52), (95.50, 0.35), (95.75, 0.22),
                                 (96.00, 0.12), (96.25, 0.06)]):
        rows.append({"date": asof, "symbol": f"SR3H7 C{k*100:04.0f}".replace(".0", ""),
                     "instrument_id": 100 + i, "root": "SR3-OPT",
                     "open": px, "high": px, "low": px, "close": px,
                     "volume": 10, "contract_month": None})
    opt = pd.DataFrame(rows)
    # fix symbols to explicit hundredths encoding
    opt["symbol"] = [f"SR3H7 C{int(k*100):04d}" for k in
                     (95.25, 95.50, 95.75, 96.00, 96.25)]
    return asof, fut, opt


def test_chain_snapshot_builds_chain():
    asof, fut, opt = _mini_market()
    chains = chain_snapshot(opt, fut, asof)
    assert "SR3H7" in chains
    ch = chains["SR3H7"]
    assert len(ch) == 5
    assert ch["fut_rate_bps"].iloc[0] == pytest.approx(440.0)
    assert ch["T_years"].iloc[0] == pytest.approx((pd.Timestamp(2027, 3, 17) - asof).days / 365.0)
    assert ch["strike_bps"].min() == pytest.approx(375.0)   # K=96.25 -> 3.75%


def test_chain_snapshot_drops_stale():
    asof, fut, opt = _mini_market()
    opt = opt.copy()
    opt["date"] = asof - pd.Timedelta(days=30)
    assert chain_snapshot(opt, fut, asof, stale_days=5) == {}
