"""End-to-end smoke test on the local data cache (no network, no spend)."""
import pandas as pd
import pytest

from fedwatch import pipeline as pl
from fedwatch.diagnostics import check_curve, check_probabilities
from fedwatch.regime1_fomc import fomc_probabilities


@pytest.fixture(scope="module")
def res():
    return pl.run()


def test_curve_builds_out_to_30y(res):
    span = (res["curve"].dates[-1] - res["asof"]).days / 365.25
    assert span > 29


def test_all_curve_diagnostics_pass(res):
    report = check_curve(res)
    failed = report[~report["pass"]]
    assert failed.empty, failed.to_string(index=False)


def test_fomc_engine_runs_and_is_calibrated(res):
    recs = fomc_probabilities(res["bundle"]["fut"], res["bundle"]["fred"],
                              res["bundle"]["fomc"])
    assert len(recs) > 0
    ok = check_probabilities(recs, res["asof"])
    assert ok["pass"].all(), ok.to_string(index=False)


def test_seam_within_tolerance(res):
    seam = res["seam"]
    assert len(seam) >= 2
    assert seam[seam.tenor_years >= 2]["gap_bps"].abs().max() < 8.0


def test_master_matrix(res):
    from fedwatch.regime1_fomc import fomc_probabilities as fp
    recs = fp(res["bundle"]["fut"], res["bundle"]["fred"], res["bundle"]["fomc"])
    mm = pl.master_matrix(res, recs)
    assert mm.index.min() >= 0
    assert ((mm.sum() - 1).abs() < 1e-6).all()
