import numpy as np

from fedwatch.densities import bucket_density, hw_theta
from fedwatch.curve_bootstrap import DiscountCurve
import pandas as pd


def test_bucket_density_integrates_to_one():
    d = bucket_density(425.0, 90.0)
    assert abs(d.sum() - 1.0) < 1e-9
    assert (d >= 0).all()


def test_bucket_density_centres_on_mean():
    d = bucket_density(425.0, 5.0)                  # near-deterministic
    assert abs(d.idxmax() - 412.5) < 1e-9 or abs(d.idxmax() - 437.5) < 1e-9


def test_bucket_density_zero_sigma():
    d = bucket_density(425.0, 0.0)
    assert abs(d.sum() - 1.0) < 1e-12


def test_hw_theta_finite_flat():
    asof = pd.Timestamp(2026, 1, 1)
    dates = [asof + pd.Timedelta(days=365 * k) for k in range(1, 6)]
    import numpy as np
    dfs = [float(np.exp(-0.04 * k)) for k in range(1, 6)]
    c = DiscountCurve(asof, [asof] + dates, [1.0] + dfs)
    for t in [0.1, 1.0, 3.0]:
        assert np.isfinite(hw_theta(c, t, a=0.03, sigma=0.009))
