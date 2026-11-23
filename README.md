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
Treasury option chains (OZF/OZN/OZB) for 5Y+ horizons remain future work.

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
