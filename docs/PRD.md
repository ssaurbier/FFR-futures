# PRD — 0-30Y Rates Probability Engine

**Status:** Phases 1-3 shipped. This document is the product record.
**North star (owner's words, 2026-11-23):** *"Construct the entire
historical-to-present 3-dimensional probability surface of expected future
interest rates — from time-current to as far into the future as possible."*

## 1. Vision

The 3D surface has axes:

```
   rate level (25bp buckets)
        ^  |
        |  |   probability density
        |  |  /
        +--+-----------------------> forecast horizon
       /   3wk  3m  1y  2y  5y  10y  30y
  time /
(as-of) v  one full surface per historical as-of date, back as far as data allows
```

* **X — forecast horizon:** FOMC meetings (0-18M) → SOFR option expiries
  (≤2.1Y) → UST option expiries (5/10/30Y) → model tenors as fallback filler.
* **Y — rate level:** 25bp buckets, FedWatch schema throughout.
* **Z — as-of time:** every historical date the data supports (futures
  2010→, SOFR options 2017→, UST options 2010→, Eris 2012→).

One pane of that surface per day, animated through time, is the deliverable
that CME FedWatch (18-month horizon, current-date only) does not provide.

## 2. Shipped capability (Phases 1-3)

| Regime | Horizon | Method | Validation |
|---|---|---|---|
| FOMC steps | 0-18M | ZQ day-weighted solver, step convolution | 78% hit ≤7d; calibrated |
| SOFR option PDFs | 3wk-2.1Y | SR3 options → Bachelier IV → Normal SABR → BL | martingale <10bp |
| UST yield PDFs | 5/10/30Y | OZF/OZN/OZB → SABR (price space) → BL → duration map | vols 70-107bp; −53bp spread ✓ |
| Unified curve | 0-30Y | SR1→SR3(HW conv.)→Eris, Hagan-West | Eris parity ≤1bp |
| Model densities | filler | historical-vol √t + HW simulation | flagged `(model)` |

## 3. Remaining roadmap

1. **Backtest surface builder (P4):** batch-generate the master matrix for
   every historical as-of date; store as a parquet cube
   (date × horizon × bucket). Then: animated visualisation, calibration
   studies (did the densities predict realized outcomes?), the seam-gap
   signal monitor, FOMC-shock propagation analysis.
2. **Modeling v2 (parked, owner's call):** CTD-based dollar durations,
   Newton-Raphson price→yield in tails, second-order SABR wings, Eris 25Y
   node, Eris historical PDF cross-validation.
3. Data gaps: 2024-08 SR3 month; UB/TN options; weekly Treasury options.

## 4. Operating policy

* **Data pulls are MANUAL ONLY.** No cron, no auto-refresh. The notebook's
  MANUAL REFRESH cell / `fedwatch.refresh.refresh_all()` is the only door.
* Spend guard: every billable pull estimates cost first and aborts above its
  budget parameter.
* Secrets: env-first (`DATABENTO_API_KEY`, `FRED_API_KEY`), gitignored
  `credentials.py` fallback. Keys never enter git.
