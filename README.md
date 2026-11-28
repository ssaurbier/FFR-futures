# fedwatch — 0-30Y Rates Curve & Probability Engine

A recreation of CME FedWatch extended across the whole USD curve: unified
SOFR-discounting curve from CME futures (0–30Y), FOMC hike/cut probability
matrices with FedWatch parity, and model-based rate distributions at every
tenor.

## Architecture

```
fedwatch/                     the engine package
├── config.py                 products, conventions, credentials loader
├── calendars.py              FOMC schedule (scrape+cache), IMM dates, holidays
├── contracts.py              symbol parsing (1/2-digit years), accrual windows
├── ingest/
│   ├── databento_io.py       GLBX.MDP3 ohlcv-1d pulls -> parquet (cost-guarded)
│   ├── fred_io.py            EFFR/SOFR/targets/DGS* series
│   └── eris_io.py            Eris public FTP: settlements, coupons, par curves
├── regime1_fomc.py           ZQ -> day-weighted FOMC step probabilities
├── regime2_sr3.py            SR3 strip + Hull-White convexity adjustment
├── regime3_eris.py           Eris price -> par swap inversion + DF extension
├── curve_bootstrap.py        DiscountCurve (Hagan-West monotone convex forwards)
├── options_chain.py          SR3 option symbol parsing + chain snapshots
├── sabr_bl.py                Bachelier pricing/IV, Normal SABR, BL PDFs
├── phase2.py                 option-density orchestration (fits -> buckets)
├── treasury_options.py       OZF/OZN/OZB chains -> UST yield densities
├── densities.py              historical-vol densities + HW simulation fans
├── diagnostics.py            automated verification suite
├── backtest.py               historical hit-rate calibration harness
├── pipeline.py               end-to-end orchestration + master matrix
└── plotting.py               heatmaps, curve charts, percentile fans
notebooks/fedwatch_30y.ipynb  thin dashboard (kernel: "fedwatch (.venv)")
tests/                        pytest suite (26 tests)
data/                         parquet/csv caches (gitignored)
archive/                      previous research notebooks
```

## Usage

```bash
source .venv/bin/activate
python -m pytest tests/ -q          # verification suite
jupyter lab notebooks/fedwatch_30y.ipynb
```

**Data policy: pulls are MANUAL ONLY — nothing is scheduled or automatic.**
New data arrives solely via the notebook's MANUAL REFRESH cell or
`fedwatch.refresh.refresh_all()` (~$0.10 + ~$0.15/day-gap). Project
meta-docs live in `docs/` (PRD, tech spec, decision log, living handoff).

```python
from fedwatch import pipeline as pl
res = pl.run(refresh=False)         # refresh=True pulls new data (~$0.30/day)
res["curve"].zero("2036-11-20")     # 10Y zero rate
res["seam"]                         # SR3<->Eris cross-product consistency
from fedwatch.regime1_fomc import fomc_probabilities, meeting_matrix
recs = fomc_probabilities(res["bundle"]["fut"], res["bundle"]["fred"], res["bundle"]["fomc"])
meeting_matrix(recs, res["asof"])   # FedWatch-style probability table
```

## Validation status (2026-11-20 snapshot)

| Check | Result |
|---|---|
| SR3 quarterly repricing | max 0.23 bp |
| Eris pars vs Eris's own published curve | ≤ 1 bp (4Y–20Y), −2.5 bp @30Y |
| SR3↔Eris seam (2Y/3Y) | −5.5 / −6.6 bp (futures-swap basis, documented) |
| Forward positivity | min inst-fwd 3.88% > 0 |
| FOMC engine calibration | 77.8% hit ≤7d, 57.2% @3–6m, avg-p tracks hit rate |

## Data sources & cost

- **Databento GLBX.MDP3** (ohlcv-1d): ZQ/SR1/SR3/Treasury futures — one-time
  backfill ≈ $10; daily refresh ≈ $0.30.
- **Eris public FTP** (files.erisfutures.com/ftp/): free — official daily
  settlements, contract coupons, A/B/C price components, published par curves.
- **FRED**: free (EFFR, SOFR, DFEDTAR*, DGS*).
- SR3 options are available on GLBX (`SR3.OPT`) for Phase 2 (~$2/year daily).

## Segment policy

SR1 bridges as-of → first IMM SR3 quarterly (front months only); SR3 quarterlies
carry to ~3Y (Hull-White convexity-adjusted); Eris SOFR swap futures anchor
1Y–30Y par swaps; Hagan-West monotone-convex forwards enforce a smooth,
arbitrage-free P(0,T). SR1 back months beyond ~2M are deliberately unused —
they can dislocate >10bp from liquid SR3 quarterlies.

## Phase 2 — option-implied densities (shipped)

SR3 option chains (GLBX `SR3.OPT`, ohlcv-1d, ~$0.03/month) → Bachelier implied
vols (bracketed root solve) → Normal SABR (β=0, leading-order Hagan) fit per
expiry → dense call surface → Breeden-Litzenberger PDF → 25bp FedWatch buckets
flagged `(option)` in the master matrix. Martingale check (|PDF mean −
forward| < 10bp) rejects unreliable fits; the ATM normal-vol term structure
calibrates the Hull-White σ used in the Regime-2 convexity adjustment.

## Phase 3 — Treasury options for the long end (shipped)

CBOT options on Treasury futures (`OZF`/`OZN`/`OZB` roots on GLBX) →
OTM-filtered chains (American-exercise mitigation, tick-floor premium filter)
→ Normal SABR in price space → BL price density → yield-space mapping via
constant dollar duration → anchored at the forward Treasury yield (Eris
forward par − standing swap spread) → 25bp buckets flagged `(option, UST…)`.
Live validation: UST5Y/10Y/30Y densities with ATM yield vols 107/79/70bp and
a −53bp implied 10Y swap spread consistent with the SOFR/UST curve stack.
v2 upgrades: CTD-based dollar durations (replacing product constants),
Newton-Raphson price→yield in the tails.

## Phase 4 — the 3D probability surface (shipped)

`fedwatch.surface` builds the historical cube — **as-of time x horizon x
25bp buckets** — as a long-format parquet (`data/surface.parquet`,
~500k rows): `fomc` daily since 2010 (4,100 dates), `sr3_opt` weekly since
2022 (SR3 options have no electronic history before then), `ust_opt`
weekly since 2010. Treasury historical chains run defs-free via the CME
expiry rule (verified 5,144/5,144 against listed definitions incl.
Good-Friday/Christmas rolls) and vega-weighted SABR fits (unweighted fits
failed through the COVID smile). Notebook renders the panes, the
time-evolution heatmap, the mean±2sd band, and an animated GIF of the 10Y
distribution morphing 2010→present. Known gaps: UST fits reject
2017-2018 (low-vol smiles under the gate) and some single dates; UST
densities are spot-CMT anchored (v1).

## Phase 5 — the forward expected-rate surface (shipped)

`curve_history` assembles the expected-rate surface across THREE eras
(1,925 weekly dates, 1990-01 → present, 16 horizons each, per-row `era`
and `source` labels): **fred** 1990-2010 FRED CMT spot curves; **hybrid**
2010-2020 ZQ-futures front end + CMT long end; **sofr_full** 2020- full
SR1/SR3/Eris futures stack (true forwards). The expected future rate at
every horizon at every date back to 1990 — the FedWatch-style forward
view, 30Y deep, through 36 years of history. The surface cube carries a
`model` layer (curve-anchored densities at 2-30Y tenors) and a unified
`horizon_years` axis. Plotly `go.Surface` charts: the full-span
expected-rate surface, per-tenor probability evolution, and the 0-30Y
forward probability pane. Era seams are visible by design (spot vs
forward semantics, and the 2020 swap-spread level).

## ⭐ THE 30-year FedWatch (the flagship)

`fedwatch.forward30.forward30(asof)` — FedWatch's 2-year probability
ladder extended to **30 years at any date**: rungs = future dates with
25bp-bucketed rate distributions, `source`-labeled. 0-18M ZQ/FOMC
(FedWatch parity) · ≤2.1Y SR3 option PDFs · 5/10/30Y Treasury option PDFs
· all horizons to 30Y futures-strip mean (real Eris/SR3/SR1 forwards) ×
**market-calibrated vol term structure** (`volterm`: real option ATM vols
fitted with mean-reverting σ(T), bounded extrapolation, floored-lognormal
far rungs so means stay on the strip). This is the answer to "FedWatch,
but 30 years deep."

## Phase 6 — future-dated fan + volatility surface (shipped)

Every 3D chart now uses **absolute future contract dates** on the x-axis:
`future_dated_matrix` pivots the curve history so each as-of row runs 30
years forward in calendar time — the 1995 row spans 1995-2025, the 2026
row spans 2026-2056 (the full forward-curve fan, 67 future-year columns).
`sf.vol_surface` produces the companion **volatility surface** —
distribution sd by future date, most-market-implied layer preferred — now
covering 1990→2026 (σ sources: SR3 realized, Eris parcoupon history, FRED
CMT changes pre-2020, all winsorized and capped at 300bp/yr). Notebook
gallery: full fan, recent zoom, quarterly short-end, volatility surface.

## Caveats

## Caveats

- Convexity adjustment uses a historical-vol σ proxy (a≈0.03); effect <1bp
  within 3Y, calibrated properly in Phase 2.
- 20Y→30Y gap relies on flat-forward extrapolation of intermediate Eris
  payment DFs (−2.5bp at the 30Y point vs published).
- HW simulation percentiles at 15Y+ show mild drift bias from piecewise-θ
  discretization; indicative only.
- Probabilities beyond the FOMC horizon are MODEL-based (flagged `(model)`)
  until option-implied PDFs arrive in Phase 2.
