import numpy as np
import pandas as pd
import pytest

from fedwatch.sabr_bl import (bachelier_implied_vol, bachelier_price,
                              bl_pdf, bucket_pdf, fit_sabr, pdf_mean_bps,
                              sabr_normal_vol)


def test_bachelier_put_call_parity():
    F, K, s, T = 95.5, 94.0, 0.8, 1.0
    c = bachelier_price(F, K, s, T, "C")
    p = bachelier_price(F, K, s, T, "P")
    assert abs((c - p) - (F - K)) < 1e-10


def test_bachelier_iv_roundtrip():
    F, K, s, T = 95.5, 96.0, 0.65, 0.75
    px = bachelier_price(F, K, s, T, "C")
    iv = bachelier_implied_vol(px, F, K, T, "C")
    assert iv is not None
    assert abs(iv - s) < 1e-8


def test_bachelier_iv_put_side():
    F, K, s, T = 94.0, 95.5, 0.65, 0.75
    px = bachelier_price(F, K, s, T, "P")
    iv = bachelier_implied_vol(px, F, K, T, "P")
    assert iv is not None and abs(iv - s) < 1e-8


def test_bachelier_iv_intrinsic_returns_none():
    assert bachelier_implied_vol(0.0, 96.0, 94.0, 1.0, "C") is None


def test_sabr_vol_atm_is_alpha():
    v = sabr_normal_vol(400.0, 400.0, 1.0, 90.0, -0.3, 0.5)
    assert abs(v - 90.0) < 1e-10


def test_sabr_fit_synthetic_smile():
    """Generate a normal-SABR smile, price Bachelier, fit, recover params."""
    rng = np.random.default_rng(0)
    F, T = 435.0, 1.25                      # bps
    alpha, rho, nu = 92.0, -0.35, 0.45
    strikes = np.arange(335.0, 540.0, 25.0)                    # rate bps
    vols = sabr_normal_vol(strikes, F, T, alpha, rho, nu)      # bps
    # market-convention PRICE-space calls: strike/fwd in points, vol in points
    Kp, Fp = 100.0 - strikes / 100.0, 100.0 - F / 100.0
    prices_pts = bachelier_price(Fp, Kp, vols / 100.0, T, "C")
    chain = pd.DataFrame({
        "cp": "C", "strike_price": 100.0 - strikes / 100.0, "price": prices_pts,
        "volume": 10.0, "fut_price": 100.0 - F / 100.0, "fut_rate_bps": F,
        "T_years": T, "strike_bps": strikes})
    fit = fit_sabr(chain, min_strikes=4)
    assert fit is not None
    assert fit["rmse_bp"] < 0.5                          # recovers its own smile
    assert abs(fit["alpha"] - alpha) < 15.0
    assert abs(fit["rho"] - rho) < 0.2
    assert abs(fit["nu"] - nu) < 0.25


def test_bl_pdf_flat_vol_integrates_and_means():
    """Constant-vol Bachelier => PDF is normal with mean = F."""
    fit = {"alpha": 90.0, "rho": 0.0, "nu": 0.30, "T": 1.0, "F_bps": 435.0}
    pdf = bl_pdf(fit, wing_bp=500.0, step_bp=2.5)
    area = np.trapezoid(pdf.values, pdf.index.values)
    assert abs(area - 1.0) < 1e-6
    assert abs(pdf_mean_bps(pdf) - 435.0) < 1.0         # martingale, <1bp-ish


def test_bucket_pdf_sums_to_one():
    fit = {"alpha": 90.0, "rho": -0.2, "nu": 0.4, "T": 1.5, "F_bps": 430.0}
    b = bucket_pdf(bl_pdf(fit))
    assert abs(b.sum() - 1.0) < 1e-6
    assert (b >= 0).all()
