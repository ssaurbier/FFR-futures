"""P4 — the 3D probability surface (the project's north star).

Builds the historical-to-present cube:
    Z as-of time  x  X horizon (FOMC meetings / option expiries)  x  Y rate
    level (25bp buckets), value = probability.

Layers:
  fomc    daily      ZQ-implied meeting distributions (full history)
  sr3_opt weekly     SOFR option PDFs (SR3.OPT, 2020->)
  ust_opt weekly     Treasury yield PDFs (OZF/OZN/OZB.OPT, 2010->)

Storage: long-format parquet
  [asof, horizon, layer, tenor, bucket_mid, p]
incremental by layer (rebuild a layer -> drop+rewrite that layer's rows).

UST anchor note (v1): historical densities are centred on the SPOT CMT
yield of the matching tenor (spot-vs-forward roll over ~1M option horizons
is small and affects only the centre, not the market-implied shape).
Forward anchoring needs historical Eris coupons — parked for v2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .options_chain import chain_snapshot
from .regime1_fomc import bucket_midpoints, fomc_probabilities
from .sabr_bl import bl_pdf, bucket_pdf, fit_sabr, pdf_mean_bps
from .treasury_options import (PRODUCTS, fit_sabr_price,
                               treasury_chains, treasury_yield_density,
                               bucket_yield_density)

SURFACE_PARQUET = config.DATA_DIR / "surface.parquet"

LAYERS = ("fomc", "sr3_opt", "ust_opt", "model")


# ---------------------------------------------------------------- FOMC layer
def fomc_layer(bundle: dict, start=None, end=None) -> pd.DataFrame:
    recs = fomc_probabilities(bundle["fut"], bundle["fred"], bundle["fomc"])
    rows = []
    for rec in recs.itertuples():
        levels = rec.effr + 25.0 * np.array(list(rec.grid.keys()), dtype=float)
        probs = np.array(list(rec.grid.values()))
        for mid, p in zip(bucket_midpoints(levels), probs):
            rows.append((pd.Timestamp(rec.asof), rec.meeting_date, np.nan, mid, p))
    df = pd.DataFrame(rows, columns=["as_of", "horizon", "tenor",
                                     "bucket_mid", "p"])
    if start:
        df = df[df["as_of"] >= pd.Timestamp(start)]
    if end:
        df = df[df["as_of"] <= pd.Timestamp(end)]
    df = df.assign(layer="fomc")
    df["horizon_years"] = (df["horizon"] - df["as_of"]).dt.days / 365.0
    return df[["as_of", "horizon", "layer", "tenor", "horizon_years",
               "bucket_mid", "p"]]


# ---------------------------------------------------------------- option layers
def _sr3_layer(bundle: dict, opt: pd.DataFrame, grid) -> pd.DataFrame:
    out = []
    for t in grid:
        try:
            chains = chain_snapshot(opt, bundle["fut"], t)
        except Exception:
            continue
        for und, chain in chains.items():
            fit = fit_sabr(chain, min_strikes=6)
            if fit is None or fit["rmse_bp"] > 30.0:
                continue
            pdf = bl_pdf(fit)
            if abs(pdf_mean_bps(pdf) - fit["F_bps"]) > 10.0:
                continue
            b = bucket_pdf(pdf)
            expiry = pd.Timestamp(chain["expiry"].iloc[0])
            for mid, p in b.items():
                out.append((t, expiry, np.nan, mid, p))
    df = pd.DataFrame(out, columns=["as_of", "horizon", "tenor",
                                    "bucket_mid", "p"]).assign(layer="sr3_opt")
    df["horizon_years"] = (df["horizon"] - df["as_of"]).dt.days / 365.0
    return df


def _ust_layer(bundle: dict, defs: pd.DataFrame, px: pd.DataFrame, grid
               ) -> pd.DataFrame:
    out = []
    for t in grid:
        dgs = bundle["fred"]
        try:
            chains = treasury_chains(bundle["fut"], t, px=px, defs=defs)
        except Exception:
            continue
        for key, chain in chains.items():
            oz_root = chain["oz_root"].iloc[0]
            _, tenor, _, _ = PRODUCTS[oz_root]
            y_ref = float(dgs[f"DGS{tenor}"].asof(t)) / 100.0
            if not np.isfinite(y_ref):
                continue
            fit = fit_sabr_price(chain, min_quotes=6)
            if fit is None:
                continue
            res = treasury_yield_density(chain, fit, y_ref)
            if res is None:
                continue
            density, _ = res
            b = bucket_yield_density(density)
            for mid, p in b.items():
                out.append((t, pd.Timestamp(chain["expiry"].iloc[0]), tenor,
                            mid, p))
    df = pd.DataFrame(out, columns=["as_of", "horizon", "tenor",
                                    "bucket_mid", "p"]).assign(layer="ust_opt")
    df["horizon_years"] = (df["horizon"] - df["as_of"]).dt.days / 365.0
    return df


# ---------------------------------------------------------------- model layer
def model_layer(bundle: dict, curve_hist: pd.DataFrame,
                tenors=(2, 3, 5, 7, 10, 15, 20, 30)) -> pd.DataFrame:
    """Curve-anchored densities at standard tenors for every curve-history
    date (historical-vol method; flagged 'model' until option PDFs exist at
    that horizon). Completes the FedWatch-style 0-30Y forward pane."""
    from .densities import _realized_sigma, bucket_density

    fut = bundle["fut"]
    rows = []
    for t, blk in curve_hist.groupby("as_of"):
        zero = blk.set_index("horizon_years")["zero_bp"]
        for tenor in tenors:
            fwd = zero.get(float(tenor), np.nan)
            if not np.isfinite(fwd):
                continue
            sig = _realized_sigma(fut, tenor, t, fred_bps=bundle.get("fred"))
            if not np.isfinite(sig) or sig <= 0:
                continue
            sig_h = sig * np.sqrt(tenor)
            b = bucket_density(float(fwd), sig_h)
            b = b[b.index >= 0]
            if b.empty:
                continue
            b = b / b.sum()                 # renormalise truncated tail
            for mid, p in b.items():
                rows.append((pd.Timestamp(t), pd.NaT, float(tenor), mid, p))
    df = pd.DataFrame(rows, columns=["as_of", "horizon", "tenor",
                                     "bucket_mid", "p"]).assign(layer="model")
    df["horizon_years"] = df["tenor"]
    return df


# ---------------------------------------------------------------- build/merge
def build_surface(layers=LAYERS, start=None, end=None,
                  out=SURFACE_PARQUET, verbose=True) -> pd.DataFrame:
    from .pipeline import load_bundle
    bundle = load_bundle()
    fut = bundle["fut"]

    frames = []
    if "fomc" in layers:
        df = fomc_layer(bundle, start, end)
        frames.append(df)
        if verbose:
            print(f"fomc layer: {len(df):,} rows, "
                  f"{df["as_of"].nunique()} as-of dates", flush=True)

    if "sr3_opt" in layers:
        opt = pd.read_parquet(config.DATA_DIR / "options_sr3_ohlcv1d.parquet")
        opt["date"] = pd.to_datetime(opt["date"])
        grid = pd.date_range(max(pd.Timestamp(start or opt.date.min()),
                                 opt.date.min()),
                             min(pd.Timestamp(end or opt.date.max()),
                                 opt.date.max()), freq="W-FRI")
        df = _sr3_layer(bundle, opt, grid)
        frames.append(df)
        if verbose:
            print(f"sr3_opt layer: {len(df):,} rows, "
                  f"{df["as_of"].nunique()} as-of dates", flush=True)

    if "model" in layers:
        from .curve_history import load_curve_history
        ch = load_curve_history()
        df = model_layer(bundle, ch)
        frames.append(df)
        if verbose:
            print(f"model layer: {len(df):,} rows, "
                  f"{df['as_of'].nunique()} as-of dates", flush=True)

    if "ust_opt" in layers:
        from .treasury_options import load_opt_defs, load_opt_prices
        defs, px = load_opt_defs(), load_opt_prices()
        grid = pd.date_range(max(pd.Timestamp(start or px.date.min()),
                                 px.date.min()),
                             min(pd.Timestamp(end or px.date.max()),
                                 px.date.max()), freq="W-FRI")
        df = _ust_layer(bundle, defs, px, grid)
        frames.append(df)
        if verbose:
            print(f"ust_opt layer: {len(df):,} rows, "
                  f"{df["as_of"].nunique()} as-of dates", flush=True)

    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if out.exists():                                  # incremental by layer
        old = pd.read_parquet(out)
        old = old[~old["layer"].isin(layers)]
        full = pd.concat([old, fresh], ignore_index=True)
    else:
        full = fresh
    full.to_parquet(out, index=False)
    if verbose:
        print(f"surface: {out.name} -> {len(full):,} rows, "
              f"layers {sorted(full.layer.unique())}", flush=True)
    return full


def load_surface() -> pd.DataFrame:
    df = pd.read_parquet(SURFACE_PARQUET)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["horizon"] = pd.to_datetime(df["horizon"])
    return df


# ---------------------------------------------------------------- queries
def slice_asof(surface: pd.DataFrame, asof) -> pd.DataFrame:
    """One pane: horizon x bucket probability matrix at a given as-of."""
    day = surface[surface["as_of"] == pd.Timestamp(asof)]
    mat = day.pivot_table(index="bucket_mid", columns="horizon",
                          values="p", aggfunc="sum").fillna(0.0)
    return mat.reindex(sorted(mat.columns), axis=1)


def horizon_series(surface: pd.DataFrame, layer: str, tenor: float | None,
                   pick: str = "nearest") -> pd.DataFrame:
    """Distribution evolution for one horizon family: asof x bucket."""
    sub = surface[surface.layer == layer]
    if tenor is not None:
        sub = sub[sub.tenor == tenor]
    rows = {}
    for t, grp in sub.groupby("as_of"):
        # horizon = the nearest expiry available on that date
        h = grp.horizon.min() if pick == "nearest" else grp.horizon.max()
        g = grp[grp.horizon == h]
        rows[t] = g.set_index("bucket_mid")["p"]
    mat = pd.DataFrame(rows).T.fillna(0.0).sort_index()
    mat.columns = [f"{c:.0f}" for c in mat.columns]
    return mat


def expected_path(surface: pd.DataFrame, asof) -> pd.DataFrame:
    """E[level] and sigma per horizon at a given as-of (all layers)."""
    day = surface[surface["as_of"] == pd.Timestamp(asof)]
    recs = []
    grouped = day.groupby(["layer", "horizon", "tenor"], dropna=False)
    for (layer, horizon, tenor), grp in grouped:
        w = grp.set_index("bucket_mid")["p"]
        mean = float((w.index.values * w.values).sum())
        var = float(((w.index.values - mean) ** 2 * w.values).sum())
        recs.append({"layer": layer, "horizon": horizon, "tenor": tenor,
                     "mean_bp": mean, "sd_bp": var ** 0.5})
    out = pd.DataFrame(recs)
    return out.sort_values("horizon") if len(out) else out


# ---------------------------------------------------------------- volatility view
def vol_surface(surface: pd.DataFrame, bucket: str = "Y",
                prefer: tuple[str, ...] = ("sr3_opt", "ust_opt", "model",
                                           "fomc")) -> pd.DataFrame:
    """Distribution width (sd, bps) by as-of x ABSOLUTE FUTURE DATE.

    Where several layers cover a future date, `prefer` order picks the most
    market-implied one. This is the '30-year volatility surface for each
    given date': how uncertain the rate at each future calendar date was,
    seen from each as-of date.
    """
    rank = {l: i for i, l in enumerate(prefer)}
    df = surface.copy()
    df["pref"] = df["layer"].map(rank)
    df = df.dropna(subset=["pref"])
    df["future_date"] = (df["as_of"] + pd.to_timedelta(
        (df["horizon_years"] * 365.25).round().astype(int), unit="D")
    ).dt.to_period(bucket).dt.to_timestamp()

    rows = []
    for (t, fd), grp in df.groupby(["as_of", "future_date"]):
        grp = grp.sort_values("pref")
        # merge layers of equal preference (e.g. all UST tenors) by mixture
        best = grp["pref"].iloc[0]
        g = grp[grp["pref"] == best]
        w = g.groupby("bucket_mid")["p"].sum()
        w = w / w.sum()                             # normalise the mixture
        mean = float((w.index.values * w.values).sum())
        sd = float((w.values * (w.index.values - mean) ** 2).sum() ** 0.5)
        rows.append((t, fd, sd))
    out = pd.DataFrame(rows, columns=["as_of", "future_date", "sd_bp"])
    return out.pivot_table(index="as_of", columns="future_date",
                           values="sd_bp").sort_index()
