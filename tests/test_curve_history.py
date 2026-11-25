import numpy as np
import pandas as pd

from fedwatch.curve_history import HORIZONS, forward_surface


def test_horizon_grid_reaches_30y():
    assert HORIZONS.min() < 0.1 and HORIZONS.max() == 30.0
    assert np.all(np.diff(HORIZONS) > 0)


def test_forward_surface_pivots():
    ch = pd.DataFrame({
        "as_of": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-12",
                                 "2024-01-12"]),
        "horizon_years": [1.0, 2.0, 1.0, 2.0],
        "zero_bp": [400.0, 420.0, 410.0, 430.0],
        "par_bp": [400.0, 420.0, 410.0, 430.0]})
    mat = forward_surface(ch)
    assert mat.shape == (2, 2)
    assert mat.loc[pd.Timestamp("2024-01-12"), 2.0] == 430.0


def test_model_layer_schema_and_mass():
    from fedwatch.surface import model_layer
    ch = pd.DataFrame({
        "as_of": pd.to_datetime(["2024-01-05"] * 2),
        "horizon_years": [2.0, 10.0],
        "zero_bp": [420.0, 445.0], "par_bp": [420.0, 445.0]})
    bundle = {"fut": pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-05"]),
        "root": ["SR3"] * 3, "contract_month": [pd.Timestamp("2023-07-01")] * 3,
        "close": [95.0, 95.0, 95.0], "symbol": ["SR3N3"] * 3,
        "instrument_id": [1, 1, 1], "open": [95.0] * 3, "high": [95.0] * 3,
        "low": [95.0] * 3, "volume": [1.0] * 3})}

    df = model_layer(bundle, ch, tenors=(2, 10))
    assert set(df.columns) >= {"as_of", "layer", "tenor", "horizon_years",
                               "bucket_mid", "p"}
    assert (df.layer == "model").all()
    for (a, t), g in df.groupby(["as_of", "tenor"]):
        assert abs(g["p"].sum() - 1.0) < 1e-6
        assert (g["bucket_mid"] >= 0).all()
