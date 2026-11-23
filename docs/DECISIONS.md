# Decision Log — paths taken and paths not taken

Append-only. Format: date / decision / alternatives rejected / why.

---

**2026-11-22 — Data backbone: Databento GLBX.MDP3 + Eris public FTP + FRED.**
Alternatives: CME DataMine (licensing, batch-file friction), free CME delayed
quote scraping (no reliable option chains), manual exports (unworkable).
Why: user mandate "free or cheapest with sufficient granularity"; Databento
key already held; Eris FTP turned out to be a free goldmine (official daily
settlements WITH coupons + their own published par curve as an independent
validation oracle). Databento auth quirk: keys need Basic auth, not Bearer
(the Python client handles this).

**2026-11-22 — Databento pulls use `stype_in="parent"` (`ROOT.FUT`/`ROOT.OPT`).**
Alternative: hand-enumerated symbol lists (the legacy notebook's approach).
Why: immune to the 1-digit→2-digit year-symbol normalization change; dedupe
keyed on stable `instrument_id`. Costs guarded via `metadata.get_cost` before
every billable request.

**2026-11-22 — Long end: Eris SOFR swap futures, one root per tenor.**
Alternatives: Treasury futures + CTD mapping (heavier: conversion factors,
net basis, missing UB), FRED CMT as synthetic swaps (not market-forward).
Why: direct par-SOFR exposure, same discounting index as SR3, FTP gives
coupons so the price→par inversion is exact. Tenor map (erisfutures.com):
YIA1 YIT2 YIC3 YID4 YIW5 YIB7 YIY10 YII12 YIL15 YIO20 YIE30. YII/YIO
never traded in-sample; 12Y/20Y interpolated.

**2026-11-22 — Curve: Hagan-West monotone convex, analytic integration.**
Alternatives: log-linear DF interpolation (kinks), cubic splines (oscillation
risk), full HW tree bootstrap (overkill for node set). HW construction
conserves interval forwards exactly → bootstrap reprices its own instruments
to ~0 by construction (SR3 quarters: 0.23bp max).

**2026-11-22 — SR1 limited to front ~2 months.**
Alternative: SR1 strip out 15-18M. Why rejected: live SR1 back months
dislocate >10-19bp from liquid SR3 quarterlies (verified empirically, both
markets quoting same day). Liquidity-ordered segment policy instead.

**2026-11-22 — SR3 curve uses IMM quarterlies only.**
Serial monthly SR3 contracts overlap; including them double-counts windows.
Quarterlies are contiguous (3rd-Wed → 3rd-Wed) and liquid.

**2026-11-22 — Convexity: Hull-White 1F, σ from options (83bp) / fallback
historical realized (94bp), a=0.03.** Effect <1bp inside 3Y; Phase-2 upgrade
replaced the proxy with ATM option vols. Full HW closed-form is sub-bp here —
parked rather than perfected.

**2026-11-22 — Probabilities beyond FOMC horizon: BOTH historical-vol
densities and HW simulation (owner's explicit choice).** Flagged `(model)`;
option PDFs replace them where chains exist (Phases 2-3 did this for
≤2.1Y SOFR and 5/10/30Y UST).

**2026-11-22 — Eris bootstrapping via forward-starting vintages only.**
In-flight contracts carry B (past payments) and C (PAI) that contaminate the
clean A-vs-annuity inversion; forward vintages have B=0, C≈0 so
`P−100 = A` exactly and the terminal-DF solve is linear.

**2026-11-23 — SR3 options: 25bp-bucketed BL PDFs on leading-order Normal
SABR (β=0).** Alternatives: second-order Hagan wing correction (deferred —
fit rmse already <2% of vol on quarterlies), direct smile interpolation
(rejected: no parametric extrapolation for wings). IV solver switched from
Newton to brentq after Newton diverged on OTM quotes.

**2026-11-23 — Treasury options: OTM-only filter + 3-tick premium floor.**
Why: American exercise distorts ITM quotes; 1/64 tick grid dominates
premiums < 0.05 → wing IVs were noise (rmse 0.5-0.7pts). Struck-at-parable
encodings (mod-10000 for SR3; decipoint truncation for OZ*) resolved only
by trusting the DEFINITIONS table as strike/expiry truth.

**2026-11-23 — Treasury yield mapping: constant dollar durations
(ZF 4.4 / ZN 8.7 / ZB 17.5 pts per 1.00), anchor = Eris forward −
standing swap spread.** Alternatives: full CTD basket + Newton-Raphson
yields (v2 — needs deliverables tables we don't ingest yet). Live check:
implied 10Y spread −53bp, consistent with the negative-swap-spread regime.

**2026-11-23 — OPERATING POLICY: no automatic data pulls, ever.**
Owner's explicit instruction. Single manual entry point:
`fedwatch.refresh.refresh_all()` / notebook MANUAL REFRESH cell. No cron,
no schedulers, nothing backgrounded by the package itself.

**2026-11-23 — Full-history backfill authorized (~$45):** Treasury options
2010→ ($35.70), SR3 options 2017→2023 ($0.44), sparse quarterly defs (~$9),
2024-08 SR3 month ($0.03). ZQ.OPT does not exist on GLBX (rejected: no
Fed Funds option layer; SR3 fills that role).
