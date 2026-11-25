"""Treasury futures options (OZ* roots) -> yield-space market densities.

CBOT options on Treasury futures live under separate Globex roots:
  OZF -> options on ZF (5Y note), OZN -> ZN (10Y note), OZB -> ZB (30Y bond)
Symbol example: 'OZNH7 C1142' = call on ZNH7 struck 114.25 (decipoint-
TRUNCATED encoding — ambiguous), so strikes/expiries come from the cached
DEFINITIONS table (authoritative), not the symbol text.

Method per chain:
  * OTM-only filter (calls K>F, puts K<F) — minimises American-exercise
    distortion and stale deep-ITM quotes;
  * Normal SABR (beta=0) fit in PRICE space (strikes in points);
  * Breeden-Litzenberger density of the futures PRICE;
  * map to yield space:  y(K) = y_ref + (F - K) / Dv  with constant dollar
    duration Dv (points per 1.00 of yield; v1 constants per product, CTD
    tables are a v2 upgrade), Jacobian q_y = q_P * Dv;
  * anchor y_ref = forward swap par (Eris, tenor-matched) minus the standing
    swap spread (Eris spot par - CMT spot), i.e. the forward Treasury yield;
  * bucket into 25bp of yield, FedWatch schema, labelled '(option, UST)'.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config
from .contracts import contract_month_start
from .sabr_bl import bachelier_implied_vol, bachelier_price, sabr_normal_vol

DEFS_PARQUET = config.DATA_DIR / "treasury_opt_defs.parquet"
OHLCTV_PARQUET = config.DATA_DIR / "treasury_opt_ohlcv1d.parquet"

# OZ root -> (futures root, CMT tenor, dollar duration pts/yield, Eris col)
PRODUCTS = {
    "OZF": ("ZF", 5, 4.4, "SOFR5Y"),
    "OZN": ("ZN", 10, 8.7, "SOFR10Y"),
    "OZB": ("ZB", 30, 17.5, "SOFR30Y"),
}


def load_opt_defs() -> pd.DataFrame:
    df = pd.read_parquet(DEFS_PARQUET)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True).dt.tz_localize(None)
    return df


_OPT_SYM_RE = re.compile(r"^(OZ[FBNT])([FGHJKMNQUVXZ])(\d{1,2})\s+([CP])(\d{4})$")


def _decode_symbol_strike(digits: str) -> float:
    """Decipoint encoding, truncated for quarter strikes: '1142' -> 114.2
    (true strike 114.2 or 114.25 — +-0.025pt ambiguity, below tick noise)."""
    return int(digits) / 10.0


def rule_expiry(underlying_month: pd.Timestamp) -> pd.Timestamp:
    """CME rule: options on Treasury futures expire on the last Friday
    preceding by >= 2 business days the last business day of the month
    before the futures delivery month.

    Verified against GLBX definitions for the 2026-27 listings
    (OZNZ6 -> 2026-11-20, OZNH7 -> 2027-02-19).
    """
    from dateutil import easter
    from .calendars import is_business_day, preceding_business_day, us_holidays
    m = underlying_month - pd.offsets.MonthBegin(1)
    month_end = m + pd.offsets.MonthEnd(0)
    hol = us_holidays(m, month_end + pd.Timedelta(days=7))
    # Good Friday is a SIFMA market holiday but NOT a US federal holiday
    for y in (m.year - 1, m.year, m.year + 1):
        gf = pd.Timestamp(easter.easter(y)) - pd.Timedelta(days=2)
        if m <= gf <= month_end + pd.Timedelta(days=7):
            hol.append(gf)
    last_bd = preceding_business_day(month_end, hol)
    d = month_end
    while d.weekday() != 4:                       # last Friday of the month
        d -= pd.Timedelta(days=1)
    while True:
        exp = d
        while not is_business_day(exp, hol):      # Good Friday / Christmas roll
            exp -= pd.Timedelta(days=1)
        # business days strictly between exp and last_bd
        gap, x = 0, exp + pd.Timedelta(days=1)
        while x < last_bd:
            if is_business_day(x, hol):
                gap += 1
            x += pd.Timedelta(days=1)
        if gap >= 2:
            return exp.normalize()
        d -= pd.Timedelta(days=7)


def load_opt_prices() -> pd.DataFrame:
    df = pd.read_parquet(OHLCTV_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


def treasury_chains(fut: pd.DataFrame, asof: pd.Timestamp, stale_days: int = 5,
                    otm_buffer_pts: float = 0.0, px: pd.DataFrame | None = None,
                    defs: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """OTM chains per UNDERLYING future symbol, joined with futures prices.

    Pass pre-loaded px/defs for batch loops (avoids disk reload per date).
    """
    asof = pd.Timestamp(asof)
    if px is None or defs is None:
        defs, px = load_opt_defs(), load_opt_prices()
    px = px[px["date"] <= asof].sort_values("date").groupby("symbol").tail(1)
    px = px[(asof - px["date"]).dt.days <= stale_days]

    # futures closes at asof per root
    fpx = (fut[fut["date"] <= asof].sort_values("date")
           .groupby("symbol").tail(1).set_index("symbol")["close"])

    px = px.set_index("symbol")
    joined = defs.set_index("raw_symbol").join(
        px[["date", "close", "volume"]], how="inner")
    joined = joined.rename(columns={"close": "price"})
    joined["oz_root"] = joined.index.str.slice(0, 3)
    # symbols absent from defs (historical listings): parse from the symbol
    missing = px.index.difference(defs["raw_symbol"])
    parsed_rows = []
    for sym in missing:
        m = _OPT_SYM_RE.match(sym.strip())
        if not m:
            continue
        oz_root, mc, yy, cp, strike = m.groups()
        if oz_root not in PRODUCTS:
            continue
        parsed_rows.append({"raw_symbol": sym, "instrument_class": cp,
                            "strike_price": _decode_symbol_strike(strike),
                            "expiration": pd.NaT, "oz_root": oz_root,
                            "price": float(px.loc[sym, "close"]),
                            "volume": float(px.loc[sym, "volume"])})
    if parsed_rows:
        joined = pd.concat([joined.reset_index(), pd.DataFrame(parsed_rows)],
                           ignore_index=True).set_index("raw_symbol")

    chains: dict[str, pd.DataFrame] = {}
    for oz_root, (f_root, _, _, _) in PRODUCTS.items():
        sub = joined[joined["oz_root"] == oz_root]
        for sym, r in sub.iterrows():
            und = f_root + sym.split()[0][3:]
            if und not in fpx.index:
                continue
            if pd.isna(r["expiration"]):
                cm = contract_month_start(und, asof)
                if cm is None:
                    continue
                expiry = rule_expiry(cm)
            else:
                expiry = pd.Timestamp(r["expiration"])
            if expiry <= asof:
                continue
            F = float(fpx[und])
            K = float(r["strike_price"])
            cp = r["instrument_class"]
            # OTM-only filter for American-style quotes
            if cp == "C" and K < F - otm_buffer_pts:
                continue
            if cp == "P" and K > F + otm_buffer_pts:
                continue
            row = {"cp": cp, "strike_price": K, "price": float(r["price"]),
                   "volume": float(r["volume"]), "fut_price": F,
                   "T_years": (expiry - asof).days / 365.0, "expiry": expiry,
                   "oz_root": oz_root}
            key = f"{und}|{expiry.date()}"
            chains.setdefault(key, []).append(row)
    return {k: pd.DataFrame(v).sort_values("strike_price").reset_index(drop=True)
            for k, v in chains.items()}


def fit_sabr_price(chain: pd.DataFrame, min_quotes: int = 5,
                   min_premium: float = 0.05) -> dict | None:
    """Normal SABR fit in PRICE space (strikes/forwards in points).

    Options priced below `min_premium` (3 ticks = 0.047) are excluded: the
    1/64 tick grid dominates their implied vols.
    """
    F = float(chain["fut_price"].iloc[0])
    T = float(chain["T_years"].iloc[0])
    Ks, vols = [], []
    for r in chain.itertuples():
        if r.price < min_premium:
            continue
        v = bachelier_implied_vol(r.price, F, r.strike_price, T, r.cp)
        if v is None or not np.isfinite(v):
            continue
        Ks.append(r.strike_price)
        vols.append(v)
    if len(Ks) < min_quotes:
        return None
    Ks, vols = map(np.asarray, (Ks, vols))
    atm = float(np.median(vols))
    # vega weights: near-ATM quotes govern the density core; far wings have
    # little vega and their vol quotes carry tick/quote noise
    from scipy.stats import norm as _norm
    w = _norm.pdf((F - Ks) / max(atm * np.sqrt(T), 1e-6))
    w = np.clip(w, 0.02, None)
    w = w / w.sum() * len(w)

    def resid(p):
        return (sabr_normal_vol(Ks, F, T, p[0], np.clip(p[1], -0.999, 0.999), p[2])
                - vols) * w / atm

    try:
        sol = least_squares(resid, x0=[atm, 0.0, 0.3],
                            bounds=([1e-5, -0.999, 1e-4], [30.0, 0.999, 10.0]))
    except Exception:
        return None
    alpha, rho, nu = sol.x
    model = sabr_normal_vol(Ks, F, T, alpha, rho, nu)
    rmse = float(np.sqrt(np.mean(((model - vols) * w) ** 2)))
    if rmse > max(0.15, 0.15 * atm):       # vega-weighted rmse gate, 15% rel
        return None
    return {"alpha": float(alpha), "rho": float(rho), "nu": float(nu),
            "T": T, "F_pts": F, "rmse_pts": rmse, "n_quotes": len(Ks),
            "atm_vol_pts": float(sabr_normal_vol(F, F, T, alpha, rho, nu))}


def yield_anchor(fred_bps: pd.DataFrame, eris_parcurve: pd.DataFrame,
                 asof: pd.Timestamp, eris_col: str, eris_forward: float,
                 tenor: int) -> float | None:
    """Forward Treasury yield ~ Eris forward par minus standing swap spread."""
    pc = eris_parcurve.asof(asof)
    if eris_col not in pc.index:
        return None
    dgs = fred_bps[f"DGS{tenor}"].asof(asof)
    if not np.isfinite(dgs):
        return None
    spread_bps = float(pc[eris_col]) * 100.0 - float(dgs)   # both in bps
    return eris_forward - spread_bps / 100.0                # yield in percent


def treasury_yield_density(chain: pd.DataFrame, fit: dict, y_ref_pct: float,
                           wing_pts: float | None = None
                           ) -> tuple[pd.Series, dict] | None:
    """BL price density mapped to yield bps; returns (density per bp, info)."""
    dv = PRODUCTS[chain["oz_root"].iloc[0]][2]
    F, T = fit["F_pts"], fit["T"]
    wing = wing_pts or max(3.0 * fit["atm_vol_pts"] * np.sqrt(T), 1.5)
    grid = np.arange(F - wing, F + wing, 0.05)              # price pts
    vols = sabr_normal_vol(grid, F, T, fit["alpha"], fit["rho"], fit["nu"])
    calls = bachelier_price(F, grid, vols, T, "C")
    qP = np.clip(np.gradient(np.gradient(calls, grid), grid), 0.0, None)
    if qP.sum() <= 0:
        return None
    qP = qP / np.trapezoid(qP, grid)
    # price -> yield bp:  y_pct = y_ref + (F - p)/Dv ; density Jacobian Dv
    y_bp = (y_ref_pct + (F - grid) / dv) * 100.0
    qy = qP * dv / 100.0                                    # per bp
    s = pd.Series(qy, index=pd.Index(y_bp, name="yield_bp")).sort_index()
    s = s / np.trapezoid(s.values, s.index.values)
    info = {"y_ref_pct": y_ref_pct, "dv": dv,
            "mean_gap_bp": float(np.trapezoid(s.index.values * s.values, s.index.values)
                                 - (y_ref_pct) * 100.0)}
    return s, info


def bucket_yield_density(density: pd.Series, bucket_bp: float = 25.0) -> pd.Series:
    from .regime1_fomc import bucket_midpoints
    mids = bucket_midpoints(density.index.values)
    df = pd.DataFrame({"mid": mids, "p": density.values})
    out = df.groupby("mid")["p"].sum() * bucket_bp
    return out / out.sum()


def treasury_density_columns(res: dict, asof: pd.Timestamp,
                             min_quotes: int = 6
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UST yield density columns for the master matrix + validation table.

    Uses the pipeline bundle for futures, FRED (bps), Eris forwards
    (res['eris_eval']) and the published Eris par curve for the
    swap-spread anchor.
    """
    asof = pd.Timestamp(asof)
    bundle = res["bundle"]
    fut, fred = bundle["fut"], bundle["fred"]
    eris_forward = (res["eris_eval"].set_index("tenor_years")["par_rate_bps"] / 100.0)
    fwd_map = {5: eris_forward.get(5.0), 7: eris_forward.get(7.0),
               10: eris_forward.get(10.0), 15: eris_forward.get(15.0),
               20: eris_forward.get(20.0), 30: eris_forward.get(30.0)}

    chains = treasury_chains(fut, asof)
    cols, checks = {}, []
    for key, chain in sorted(chains.items()):
        oz_root = chain["oz_root"].iloc[0]
        _, tenor, _, eris_col = PRODUCTS[oz_root]
        fwd = fwd_map.get(tenor)
        if fwd is None or not np.isfinite(fwd):
            continue
        fit = fit_sabr_price(chain, min_quotes=min_quotes)
        if fit is None:
            continue
        y_ref = yield_anchor(fred, bundle["eris_parcurve"], asof, eris_col,
                             float(fwd), tenor)
        if y_ref is None:
            continue
        out = treasury_yield_density(chain, fit, y_ref)
        if out is None:
            continue
        density, info = out
        label = f"{key.split('|')[1]} UST{tenor}Y (option)"
        cols[label] = bucket_yield_density(density)
        checks.append({"chain": key, "tenor": tenor, "expiry": chain["expiry"].iloc[0],
                       "n_quotes": fit["n_quotes"], "rmse_pts": fit["rmse_pts"],
                       "atm_vol_pts": fit["atm_vol_pts"],
                       "atm_vol_yield_bp": fit["atm_vol_pts"] / PRODUCTS[oz_root][2] * 100.0,
                       "y_ref_pct": y_ref, "mean_gap_bp": info["mean_gap_bp"]})
    mat = pd.DataFrame(cols).fillna(0.0) if cols else pd.DataFrame()
    if len(mat):
        mat = mat / mat.sum(axis=0)
    return mat, pd.DataFrame(checks)
