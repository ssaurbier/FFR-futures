import numpy as np
import pytest

from fedwatch.volterm import fit_sigma_term, horizon_sigma, sigma_of_T


def _synthetic_pts():
    # rising term structure: 40bp at 0.1y -> ~130bp at 2y
    T = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    v = np.array([45.0, 70.0, 90.0, 110.0, 122.0, 128.0])
    return T, v


def test_fit_recovers_shape():
    T, v = _synthetic_pts()
    fit = fit_sigma_term(T, v)
    assert fit is not None
    model = sigma_of_T(fit, T)
    assert np.max(np.abs(model - v)) < 20.0


def test_extrapolation_is_bounded():
    T, v = _synthetic_pts()
    fit = fit_sigma_term(T, v)
    far = float(sigma_of_T(fit, 30.0))
    assert fit["s_inf"] * 0.5 < far < fit["s_inf"] * 1.5   # no explosion
    assert far > 0


def test_cumulative_sd_grows_but_bounded():
    T, v = _synthetic_pts()
    fit = fit_sigma_term(T, v)
    sd10, sd30 = horizon_sigma(fit, 10.0), horizon_sigma(fit, 30.0)
    assert sd30 > sd10
    # ultimate bound: cumulative sd can never exceed s_inf * sqrt(T)
    assert sd30 < fit["s_inf"] * np.sqrt(30.0) * 1.001
    assert sd10 < fit["s_inf"] * np.sqrt(10.0) * 1.001


def test_too_few_points_returns_none():
    assert fit_sigma_term(np.array([0.5, 1.0]), np.array([90.0, 100.0])) is None


def test_bucket_density_lognormal_preserves_mean():
    import numpy as np
    from fedwatch.densities import bucket_density
    b = bucket_density(456.0, 521.0)          # far rung: sigma ~ mean
    m = float((b.index.values * b.values).sum())
    assert abs(m - 456.0) < 25.0              # within one bucket
    assert abs(b.sum() - 1.0) < 1e-9
    assert b.index.min() >= 0                 # floored at zero


def test_bucket_density_normal_when_sigma_small():
    import numpy as np
    from fedwatch.densities import bucket_density
    b = bucket_density(456.0, 40.0)
    m = float((b.index.values * b.values).sum())
    assert abs(m - 456.0) < 13.0   # <= half-bucket quantization
