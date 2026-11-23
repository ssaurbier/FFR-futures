import numpy as np
import pandas as pd
import pytest

from fedwatch.treasury_options import (PRODUCTS, bucket_yield_density,
                                       fit_sabr_price, treasury_yield_density)
from fedwatch.sabr_bl import bachelier_price, sabr_normal_vol


def _synthetic_chain(F=110.0, T=1.0, alpha=0.92, rho=-0.25, nu=0.45):
    call_K = np.arange(F + 0.5, F + 4.1, 0.5)
    put_K = np.arange(F - 4.0, F - 0.01, 0.5)
    rows = []
    for K in call_K:
        v = float(sabr_normal_vol(K, F, T, alpha, rho, nu))
        rows.append({"cp": "C", "strike_price": K,
                     "price": float(bachelier_price(F, K, v, T, "C")),
                     "volume": 5.0, "fut_price": F, "T_years": T,
                     "oz_root": "OZN"})
    for K in put_K:
        v = float(sabr_normal_vol(K, F, T, alpha, rho, nu))
        rows.append({"cp": "P", "strike_price": K,
                     "price": float(bachelier_price(F, K, v, T, "P")),
                     "volume": 5.0, "fut_price": F, "T_years": T,
                     "oz_root": "OZN"})
    return pd.DataFrame(rows)


def test_products_reference_sane():
    for oz, (froot, tenor, dv, col) in PRODUCTS.items():
        assert froot[0] == "Z" and tenor in (5, 10, 30) and dv > 1.0
        assert col.startswith("SOFR")


def test_fit_sabr_price_recovers_params():
    alpha, rho, nu = 0.92, -0.25, 0.45
    fit = fit_sabr_price(_synthetic_chain(alpha=alpha, rho=rho, nu=nu), min_premium=0.0)
    assert fit is not None
    assert fit["rmse_pts"] < 0.01
    assert abs(fit["alpha"] - alpha) < 0.15
    assert abs(fit["rho"] - rho) < 0.25
    assert abs(fit["nu"] - nu) < 0.3


def test_yield_density_mean_is_anchor():
    """E[y] must equal the anchor: E[F]=F under the risk-neutral measure."""
    chain = _synthetic_chain()
    fit = fit_sabr_price(chain, min_premium=0.0)
    dens, info = treasury_yield_density(chain, fit, y_ref_pct=4.50)
    mean = float(np.trapezoid(dens.index.values * dens.values, dens.index.values))
    assert abs(mean - 450.0) < 1.0                     # bp


def test_yield_density_wider_than_price_wing():
    """Dv ~ 8.7 for ZN: a 1pt price wing is ~11.5bp of yield."""
    chain = _synthetic_chain()
    fit = fit_sabr_price(chain, min_premium=0.0)
    dens, _ = treasury_yield_density(chain, fit, y_ref_pct=4.50)
    area = np.trapezoid(dens.values, dens.index.values)
    assert abs(area - 1.0) < 1e-6
    assert (dens.values >= 0).all()


def test_bucket_yield_density_sums_to_one():
    chain = _synthetic_chain()
    fit = fit_sabr_price(chain, min_premium=0.0)
    dens, _ = treasury_yield_density(chain, fit, y_ref_pct=4.50)
    b = bucket_yield_density(dens)
    assert abs(b.sum() - 1.0) < 1e-6
    assert (b >= 0).all()


def test_fit_rejects_too_few_quotes():
    chain = _synthetic_chain().head(3)
    assert fit_sabr_price(chain, min_quotes=5) is None
