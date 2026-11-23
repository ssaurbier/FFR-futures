# Product Requirements Document (PRD) & Technical Specifications
**Project:** 0–30 Year Implied Rate & Probability Engine (`0_30y_fedwatch_engine.ipynb`)  
**Target Architecture:** Self-Contained Single-Notebook Quantitative Pipeline  

---

## Part 1: Product Requirements Document (PRD)

### 1. Product Vision & Scope
Build an end-to-end **0–30 Year Implied Rate & Probability Engine** that extends the CME FedWatch methodology across the entire USD yield curve. While traditional FedWatch terminates at ~18–24 months using discrete FOMC step-probabilities, this system unifies short-term meeting expectations, medium-term quarterly forwards, and long-term par swap/Treasury distributions into one continuous analytical surface.

### 2. Core User Deliverables (Notebook Outputs)
1. **Unified 0–30Y Zero-Coupon & Instantaneous Forward Curve:** A smooth, arbitrage-free discount factor curve $P(0, T)$ and instantaneous forward rate curve $f(0, t)$ stripped from live/EOD CME futures across all three maturity regimes.
2. **Regime-Bridged Probability Matrix ($25\text{ bps}$ Buckets):** A unified probability distribution table and heatmap displaying implied rate probabilities in $25\text{ bps}$ increments across:
   * **Short-End (0–18M):** Scheduled FOMC meeting dates (classic FedWatch format).
   * **Mid-to-Long End (2Y, 3Y, 5Y, 10Y, 20Y, 30Y):** Fixed forward horizons.
3. **Interactive Regime Diagnostic Dashboard:** Visual inspection of:
   * Raw futures-implied rates vs. basis- and convexity-adjusted OIS rates (proving no artificial kinks at the 18M and 5Y–10Y seams).
   * SABR-fitted implied volatility smiles vs. raw option market quotes.
   * Extracted Risk-Neutral Probability Density Functions (PDFs) overlaid with mean forward rates and tail percentiles (5th, 25th, 50th, 75th, 95th).

### 3. Functional Requirements & Maturity Kinks

| Regime | Horizon | Target Rate / Benchmark | Instrument Set | Core Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **Regime 1: FOMC Step** | $0 \rightarrow 18\text{M}$ | EFFR & Overnight SOFR | `ZQ` (30D Fed Funds), `SR1` (1M SOFR) | Calendar-day step interpolation + EFFR/SOFR basis alignment |
| **Regime 2: Quarterly Strip** | $18\text{M} \rightarrow 5\text{Y}$ | 3M Compounded SOFR | `SR3` Futures & `SR3` Options | Hull-White convexity adjustment + Bachelier SABR / Breeden-Litzenberger |
| **Regime 3: Long-End Par** | $5\text{Y} \rightarrow 30\text{Y}$ | 30Y SOFR Swap & CMT Treasury Yields | Eris SOFR Swap Futures (`YIW`, `YIY`, `YIO`, `YIE`) + Treasury Futures/Options (`ZF`, `ZN`, `TN`, `ZB`, `UB`) | Monotone Convex spline bootstrapping + CTD yield mapping + SABR / Breeden-Litzenberger |

### 4. Non-Functional & Modeling Constraints
* **Execution Environment:** Single self-contained `.ipynb` running in Python 3.12+ (`numpy`, `scipy`, `pandas`, `matplotlib`/`plotly`), with synthetic/snapshot fallback payloads so every cell executes deterministically offline or against live CME DataMine / REST API feeds.
* **Arbitrage-Free Guarantee:** Positive forward rates ($f(0, t) > 0$) and non-negative option-implied probability densities ($q(K) \ge 0$) enforced across all strike grids.

---

## Part 2: Technical Specifications & Notebook Architecture

### Cell-by-Cell Execution Plan (`0_30y_fedwatch_engine.ipynb`)

#### Module 1: Configuration, Calendars & Market Data Ingestion
* **Purpose:** Load FOMC meeting schedules, IMM quarterly dates (3rd Wednesday of Mar/Jun/Sep/Dec), US holiday calendars (`Actual/360` for SOFR/swaps, `Actual/Actual` for Treasuries), and market snapshots.
* **Data Schema (`market_snapshot` dictionary/DataFrames):**
  * `zq_strips` & `sr1_strips`: Contract code, settlement price ($100 - R$), accrual start/end dates.
  * `sr3_strips` & `sr3_options`: Quarterly IMM contracts out to 10Y (`SR3`), option expiry $T_{\text{exp}}$, underlying futures price $F$, strike $K$, call/put settlement prices.
  * `long_end_futures` & `treasury_options`: Eris SOFR Swap futures (`YIA` 1Y through `YIE` 30Y, where $\text{Eris Price} = 100 + A(t) + B(t) - C(t)$) and Treasury futures/options (`OZF`, `OZN`, `OZB`, `OUB`) paired with a static **CTD Reference Table** (CUSIP, coupon, maturity, conversion factor).

#### Module 2: Regime 1 Engine — FOMC Step & Basis Solver ($0\text{–}18\text{M}$)
* **EFFR–SOFR Basis Normalization:** Extract the forward EFFR–SOFR spread $b(t) = r_{\text{ZQ}}(t) - r_{\text{SR1}}(t)$ on overlapping non-FOMC months so all instruments map onto a single **SOFR discounting curve**.
* **Calendar-Day Step Solver:** For any month $m$ with $N$ total days and an FOMC meeting on day $d_m$ ($N_1 = d_m - 1$ days before, $N_2 = N - N_1$ days after):
  $$R_{\text{implied}} = \frac{N_1}{N} r_{\text{pre}} + \frac{N_2}{N} r_{\text{post}}$$
* **Discrete Probability Tree:** Solve adjacent $25\text{ bps}$ hike/cut probabilities $p = \frac{r_{\text{post}} - r_{\text{lower}}}{0.25}$ recursively across FOMC meetings, outputting the classic FedWatch probability matrix.

#### Module 3: Regime 2 Engine — Convexity-Adjusted `SR3` Strip ($18\text{M–}5\text{Y}$)
* **Convexity Adjustment:** Convert `SR3` futures rates $f_{\text{fut}}(t_1, t_2) = 100 - P_{\text{SR3}}$ into true forward OIS rates by subtracting the Hull-White 1-factor convexity bias:
  $$f_{\text{OIS}}(t_1, t_2) = f_{\text{fut}}(t_1, t_2) - \frac{\sigma_{\text{HW}}^2}{2 a^2} \left(1 - e^{-a(t_2 - t_1)}\right) \left[ \frac{1 - e^{-a(t_2 - t_1)}}{2} \left(1 - e^{-2at_1}\right) + \left(1 - e^{-at_1}\right)^2 \right]$$
  *(Calibrating $\sigma_{\text{HW}}$ directly to ATM `SR3` option implied volatilities, with mean reversion $a \approx 0.03$).*

#### Module 4: Regime 3 & Global Curve Bootstrapper ($0\text{–}30\text{Y}$ Discount Curve)
* **Par Rate Extraction:**
  * From **Eris SOFR Swap Futures**: Strip the forward par swap rate $S_{\text{par}}(T)$ from the NPV component $A(t)$.
  * From **Treasury Futures**: Compute the forward CTD bond clean price $P_{\text{CTD}} = F_{\text{fut}} \times \text{CF} + \text{Net Basis}$, solve for forward yield-to-maturity $y_{\text{CTD}}$, and adjust by the Eris/Treasury swap spread.
* **Hagan-West Monotone Convex Spline:** Bootstrap zero-coupon discount factors $P(0, T)$ sequentially from $t = 1\text{D}$ to $T = 30\text{Y}$ across `SR1` $\rightarrow$ convexity-adjusted `SR3` $\rightarrow$ 5Y/7Y/10Y/20Y/30Y par swaps. Enforce $C^1$ continuity at the 18M and 5Y kinks so instantaneous forward rates $f(0, t) = -\frac{\partial \ln P(0, t)}{\partial t}$ remain smooth and strictly positive.

#### Module 5: Options Smile Calibration & Breeden-Litzenberger PDF Engine ($18\text{M–}30\text{Y}$)
* **Price-to-Yield Mapping (Treasury Options):** Convert `OZF`/`OZN`/`OZB`/`OUB` price strikes $K_P$ into yield strikes $K_y$ via exact Newton-Raphson bond pricing on each contract's CTD bond (avoiding linear duration approximation errors in the tails).
* **Normal (Bachelier) SABR Calibration:** Fit the Normal SABR volatility surface $\sigma_N(K, F)$ across strikes for each horizon ($2\text{Y}, 3\text{Y}, 5\text{Y}, 10\text{Y}, 20\text{Y}, 30\text{Y}$) via `scipy.optimize.least_squares`, fixing $\beta = 0$ (standard for low/normal rate regimes) and solving for $(\alpha, \rho, \nu)$.
* **Numerical Breeden-Litzenberger Integration:**
  1. Evaluate Bachelier call prices $C(K)$ over a dense $1\text{ bp}$ yield strike grid ($0.00\%$ to $10.00\%$).
  2. Compute the risk-neutral PDF via central finite differences:
     $$q(K_i) = e^{rT} \frac{C(K_{i+1}) - 2C(K_i) + C(K_{i-1})}{(\Delta K)^2}$$
  3. Clip numerical noise ($\max(q(K), 0)$), renormalize $\int q(K)\,dK = 1$, and integrate into **$25\text{ bps}$ bins** (`[3.50-3.75%, 3.75-4.00%, ...]`) to align identically with Regime 1's output schema.

#### Module 6: Unified Output Serialization & Diagnostic Suite
* **Master DataFrame (`fedwatch_30y_matrix`):** Rows = $25\text{ bps}$ rate buckets; Columns = sequential FOMC meeting dates (`0–18M`) + fixed forward tenors (`2Y, 3Y, 5Y, 10Y, 20Y, 30Y`).
* **Diagnostic Verification Suite:** Automated assertions verifying:
  1. **Repricing Error:** Bootstrapped $P(0, T)$ reprices all input `SR1`, `SR3`, and 30Y swap instruments within $< 0.05\text{ bps}$.
  2. **Kink Continuity:** Forward rate jump across the 18M and 5Y boundaries is $< 2\text{ bps}$.
  3. **Martingale Check:** Expected value of each option-implied PDF $\int K \cdot q(K)\,dK$ matches the bootstrapped forward rate within $< 1\text{ bp}$.