# HANDOFF — living document (overwrite freely; keep it current)

**Last updated:** 2026-11-23, after Phase 3. **Companion docs:** `PRD.md`
(vision/roadmap), `TECH_SPEC.md` (modules/conventions), `DECISIONS.md`
(append-only log). Repo: FFR-futures; GitHub `ssaurbier/FFR-futures`.

## Where things stand

Phases 1-3 shipped, tested (46/46), committed:
* `ce80c88` Phase 1 — engine (curve, FedWatch, Eris)
* `cf20e6d` Phase 2 — SR3 option PDFs
* `e565942` Phase 3 — Treasury option UST PDFs
Phase 2+3 commits may still be local-only — `git log origin/main..main` to
check; push is manual and owner-approved per event.

**In flight at handoff time:** full-history backfill
(background process; see "backfill status" below).

## How to run

```bash
cd ~/Desktop/FFR-futures && source .venv/bin/activate
python -m pytest tests/ -q                # 46 tests, ~12s
jupyter lab notebooks/fedwatch_30y.ipynb  # kernel: fedwatch (.venv)
```
Notebook flow: (optional) MANUAL REFRESH cell → Run All. `ASOF=None` uses
the latest data date. New data arrives ONLY via the MANUAL REFRESH cell /
`fedwatch.refresh.refresh_all()` — **never scheduled** (owner policy).

## The one-paragraph mental model

`pipeline.run()` builds one SOFR discount curve from SR1→SR3→Eris
(validated ≤1bp vs Eris's own published curve), `regime1_fomc` produces
FedWatch-parity meeting probabilities, `phase2`/`treasury_options` turn
option chains into 25bp-bucketed densities (SABR→BL, double-gated), and
`master_matrix` stacks everything: FOMC dates + `(option)` SOFR expiries +
`(option, UST…)` tenors + `(model)` filler.

## Gotchas learned the hard way

* Databento gateway 504s on large streams → everything pulls in MONTHLY
  chunks with exponential-backoff retries (~5% of chunks fail on a bad
  night; re-run the same job, dedupe makes it idempotent).
* Databento key needs Basic auth (curl `-u KEY:`), not Bearer.
* pandas 3.1: `df.div(s, axis=0)` aligns the SERIES to the row index —
  column-normalize with `axis=1` (cost us an hour once).
* Eris instruments CSV: date columns m/d/Y, tz-aware expiries from
  Databento defs need `tz_localize(None)`.
* SR3 option premiums arrive in hundredths of a point; Treasury premiums
  already decimal points (1/64 grid); Treasury strikes are symbol-ambiguous
  (truncated decipoints) → defs table is the only truth.
* np.log of negative DFs means a unit bug upstream (bps vs rate units bit
  us twice: SOFR anchor, FRED percent).

## Next up (priority order)

1. **Finish/verify the full-history backfill** (Treasury options 2010→,
   SR3 2017→, quarterly defs) — background job writing to
   `data/treasury_opt_ohlcv1d.parquet`, `data/options_sr3_ohlcv1d.parquet`,
   `data/treasury_opt_defs.parquet`. After it lands: rerun notebook, commit.
2. ~~P4 — the 3D surface~~ **SHIPPED (2026-11-25):** `data/surface.parquet`
   (~500k rows; fomc daily 2010→, ust_opt weekly 2010→, sr3_opt weekly
   2022→), `fedwatch/surface.py`, notebook Phase-4 cells incl. the animated
   10Y GIF. Gotchas encoded: expiry-rule holiday rolls, vega-weighted fits,
   `as_of` (not `asof` — pandas method collision), `dropna=False` groupby.
   Remaining ideas: calibration studies vs realized outcomes, seam-gap
   signal monitor, forward-anchored UST densities (needs historical Eris
   coupons), CTD durations, 2017-18 UST gap investigation.
3. Modeling v2 list lives in PRD §3.2 (CTD durations, SABR wings, Eris 25Y).

## Backfill status at handoff

**COMPLETE (2026-11-24).** Full option history cached via Databento batch
API (streaming kept 504ing — see DECISIONS 2026-11-24):

| Cache | Coverage | Rows |
|---|---|---|
| options_sr3_ohlcv1d.parquet | 2020-01-08 → 2026-11-20 (options listed 2020) | 160k |
| treasury_opt_ohlcv1d.parquet | 2010-06-06 → 2026-11-20, all OZF/OZN/OZB | 3.6M |
| treasury_opt_defs.parquet | symbols listed 2026-05 → 2027 | 8.6k |

Caveat for P4 (historical surfaces): treasury defs only cover 2026+ —
historical chains need symbol-decode + rule-based expiry (see DECISIONS
2026-11-24). `data/batch/` holds 1.1GB of raw DBN downloads; deletable,
parquets are canonical. Everything verified: chains build at current asof,
notebook executed end-to-end on the full caches, 46/46 tests.
