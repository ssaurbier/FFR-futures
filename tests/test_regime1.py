import numpy as np
import pandas as pd

from fedwatch.regime1_fomc import (bucket_midpoints, meeting_matrix,
                                   solve_fomc_path, step_dist)


def test_step_dist_integer():
    assert step_dist(2.0) == {2: 1.0}


def test_step_dist_fractional_split():
    d = step_dist(1.4)
    assert abs(d[1] - 0.6) < 1e-12 and abs(d[2] - 0.4) < 1e-12


def test_step_dist_negative():
    d = step_dist(-0.25)
    assert abs(d[0] - 0.75) < 1e-12 and abs(d[-1] - 0.25) < 1e-12


def test_bucket_midpoints_fedwatch_convention():
    # levels 388..412 -> buckets labelled by upper-25 minus 12.5
    mids = bucket_midpoints([388.0, 400.0, 412.0])
    assert list(mids) == [387.5, 387.5, 412.5]      # 400 sits on the boundary


def _toy_world():
    """One meeting, one ZQ strip: hand-checkable."""
    asof = pd.Timestamp(2026, 11, 20)
    # FOMC Dec 9 2026; D=31, k=9; choose implied avg R = (418*22 + 388*9)/31
    # so the post-meeting rate solves to exactly 418.0 bps
    strip = pd.Series({pd.Timestamp(2026, 12, 1): (418 * 22 + 388 * 9) / 31})
    fomc = pd.DataFrame({"meeting_date": [pd.Timestamp(2026, 12, 9)]})
    return asof, strip, fomc


def test_day_weight_solver_hand_example():
    asof, strip, fomc = _toy_world()
    out = solve_fomc_path(strip, 388.0, asof, fomc)
    assert len(out) == 1
    rec = out[0]
    assert abs(rec["e_post"] - 418.0) < 0.01
    assert abs(rec["exp_steps"] - 1.2) < 0.01        # (418-388)/25
    total = sum(rec["grid"].values())
    assert abs(total - 1.0) < 1e-12


def test_meeting_matrix_sums_to_one():
    asof, strip, fomc = _toy_world()
    recs = pd.DataFrame([r for r in solve_fomc_path(strip, 388.0, asof, fomc)])
    mat = meeting_matrix(recs, asof)
    assert np.allclose(mat.sum(axis=1), 1.0)
