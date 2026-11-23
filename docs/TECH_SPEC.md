# Technical Spec — fedwatch package

**Environment:** `.venv/` (Python 3.12 from anaconda); kernel `fedwatch (.venv)`
registered for the notebook. `pip` deps: databento, pandas, numpy, scipy,
matplotlib, pyarrow, pytest, nbclient.

## Module map

```
fedwatch/
  config.py            products, conventions, ERIS_TENORS, credentials loader
  calendars.py         FOMC scrape+cache, IMM 3rd-Wed, US holidays, MF/preceding
  contracts.py         futures symbol parsing (1/2-digit years), SR3 accruals
  ingest/
    databento_io.py    ohlcv-1d parquet cache, parent-stype pulls, cost guards
    fred_io.py         FRED series -> bps
    eris_io.py         Eris FTP: prices (wide->long), instruments (coupons!),
                       published par curve; price identity |100+A+B-C-P|<5e-5
  regime1_fomc.py      day-weight solver E_post=(R·D−r·k)/(D−k), step_dist,
                       convolution, meeting_matrix / expected_path
  regime2_sr3.py       SR3 strip, HW convexity (a≈0.03, σ from options|hist)
  regime3_eris.py      forward_strip -> implied par (linear terminal-DF solve)
  curve_bootstrap.py   DiscountCurve: Hagan-West monotone convex, analytic
                       log-DF integrals, par_swap/annuity (annual Act/360)
  densities.py         bucket_density (normal->25bp), hist-vol, HW simulation
  options_chain.py     SR3.OPT parser: mod-10000 wraparound strikes,
                       premiums in hundredths of a point
  sabr_bl.py           Bachelier (vector σ), brentq IV, leading-order Normal
                       SABR (β=0), BL PDF (double gradient), bucketing
  phase2.py            snapshots -> fits -> gated densities -> master merge
  treasury_options.py  OZF/OZN/OZB: defs-backed chains (tz-normalised
                       expiries), OTM filter, tick-floor premium filter,
                       price-space SABR, yield map via constant Dv,
                       anchor = Eris fwd − (Eris spot − CMT spot)
  pipeline.py          load_bundle / build_curve / seam_check / master_matrix
  diagnostics.py       check_curve / check_options / check_treasury /
                       check_probabilities
  backtest.py          FOMC hit-rate harness
  plotting.py          heatmap, curve, HW fan, option PDF
  refresh.py           MANUAL-ONLY refresh_all() (policy: never scheduled)
```

## Data inventory (data/, gitignored)

| File | Contents | Source |
|---|---|---|
| futures_ohlcv1d.parquet | ZQ/SR1/SR3/ZT/ZF/ZN/ZB/TN + YI* 2010→ | Databento |
| options_sr3_ohlcv1d.parquet | SR3.OPT 2017→(backfill in flight) | Databento |
| treasury_opt_ohlcv1d.parquet | OZF/OZN/OZB.OPT 2010→(backfill in flight) | Databento |
| treasury_opt_defs.parquet | symbol→strike/expiry (exact; sparse quarterly) | Databento |
| eris/ | prices_long, instruments, parcoupon | files.erisfutures.com (free) |
| fred_series.parquet | EFFR/SOFR/targets/DGS* 1990→ | FRED |

## Key conventions

* Rates in **bps** internally; curves ACT/365F time axis; instrument accruals
  per their own conventions (SOFR Act/360).
* SR3/SR1/ZQ implied: `(100 − price) × 100` bps. Eris: `P = 100 + A + B − C`.
* SR3 option strikes: 4-digit hundredths, **mod-10000 wraparound** for ≥par;
  premiums in hundredths of a point. Treasury option strikes: decipoint-
  truncated symbols — ALWAYS use defs table, never decode the symbol.
* FedWatch bucket label: `25·ceil(v/25) − 12.5`; exact-boundary v→lower bucket.
* Density gates: SOFR PDFs — SABR rmse ≤30bp & |mean−fwd| ≤10bp; UST PDFs —
  premium ≥0.05 (3 ticks), rmse ≤ max(0.15, 10%·ATM).

## Segment policy (curve)

SR1 bridges as-of→first IMM SR3 quarterly (front months only — back months
dislocate >10bp); SR3 quarterlies (starts in Mar/Jun/Sep/Dec only, serial
months overlap) to ~3Y; Eris forward-starting vintages 1Y-30Y via linear
terminal-DF solve. Known bases (documented, not bugs): SR3-above-Eris ≈5bp
at 1-3Y; 1Y seam dislocations up to ±10bp.

## Validation gates (diagnostics, all must pass)

forward positivity · SR3 quarterly repricing <2bp · Eris parity vs their
published curve <5bp · seam(≥2Y) <8bp · FOMC mass=1 · SOFR martingale <10bp ·
UST vols 20-250bp yield · relative SABR rmse <10%.
