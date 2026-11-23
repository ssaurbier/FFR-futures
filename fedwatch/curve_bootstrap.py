"""Discount curve object + Hagan-West monotone convex forward interpolation.

The engine's single SOFR-discounting curve P(0,T):
  * nodes are (date, discount factor) built sequentially from SOFR fixings /
    SR1 month forwards -> convexity-adjusted SR3 forwards -> Eris par swaps;
  * between nodes the instantaneous forward f(0,t) follows the Hagan-West
    (2006) monotone convex scheme: piecewise cubic in log-DF terms,
      f(s) = g_i + (g_{i+1} - g_i) s + c_i s (1-s),
      c_i = 6 (F_i - (g_i + g_{i+1}) / 2),
    which conserves each interval's average forward F_i exactly (so the
    curve reprices its own bootstrap instruments) and is C1 across nodes;
  * flat-forward extrapolation beyond the last node.

Positivity note: full HW positivity enforcement (the "limited" scheme) is not
applied; with current SOFR curves the construction stays positive in
practice, and diagnostics assert this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_YEAR = 365.0


def yearfrac(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return (pd.Timestamp(d1) - pd.Timestamp(d0)).days / DAYS_YEAR


class DiscountCurve:
    """Piecewise-monotone-convex-forward discount curve."""

    def __init__(self, asof: pd.Timestamp, dates, dfs):
        self.asof = pd.Timestamp(asof)
        pairs = sorted(zip([pd.Timestamp(d) for d in dates],
                           np.asarray(dfs, dtype=float)))
        self.dates = pd.DatetimeIndex([p[0] for p in pairs])
        self.t = np.array([yearfrac(self.asof, d) for d in self.dates])
        self._logdf = np.log(np.array([p[1] for p in pairs]))
        self._build_forwards()

    # ------------------------------------------------------------ internals
    def _build_forwards(self) -> None:
        t, ld = self.t, self._logdf
        if len(t) < 2:
            self.F = np.zeros(0)
            self.g = np.zeros(max(len(t), 0))
            return
        dt = np.diff(t)
        self.F = -np.diff(ld) / dt                    # interval average fwds
        F, g = self.F, np.zeros(len(t))
        g[0], g[-1] = F[0], F[-1]
        for i in range(1, len(t) - 1):
            if F[i - 1] * F[i] <= 0:                  # sign change -> flat 0
                g[i] = 0.0
            else:                                     # HW harmonic weighting
                g[i] = (dt[i] * F[i - 1] + dt[i - 1] * F[i]) / (dt[i - 1] + dt[i])
        self.g = g

    def _interval(self, t: float) -> int:
        i = int(np.searchsorted(self.t, t)) - 1
        return max(0, min(i, len(self.t) - 2))

    def _fwd_inst(self, t: float) -> float:
        """Instantaneous forward at year-t (piecewise from HW cubic)."""
        if len(self.t) < 2:
            return self.g[0] if len(self.g) else 0.0
        if t <= 0:
            return self.g[0]
        if t >= self.t[-1]:
            return self.F[-1]
        i = self._interval(t)
        s = (t - self.t[i]) / (self.t[i + 1] - self.t[i])
        gi, gj, Fi = self.g[i], self.g[i + 1], self.F[i]
        c = 6.0 * (Fi - 0.5 * (gi + gj))
        return gi + (gj - gi) * s + c * s * (1.0 - s)

    def _logdf_at(self, t: float) -> float:
        """log P(0,t) via analytic integration of the HW forward cubic."""
        if t <= 0:
            return 0.0
        if len(self.t) < 2:
            return -self.g[0] * t if len(self.g) else 0.0
        if t >= self.t[-1]:
            return self._logdf[-1] - self.F[-1] * (t - self.t[-1])
        i = self._interval(t)
        dt = self.t[i + 1] - self.t[i]
        s = (t - self.t[i]) / dt
        gi, gj, Fi = self.g[i], self.g[i + 1], self.F[i]
        c = 6.0 * (Fi - 0.5 * (gi + gj))
        integral = (gi * s + 0.5 * (gj - gi) * s * s
                    + c * (0.5 * s * s - s ** 3 / 3.0)) * dt
        return self._logdf[i] - integral

    # ------------------------------------------------------------ public API
    def df(self, t: float | pd.Timestamp) -> float:
        """Discount factor P(0, t)."""
        if isinstance(t, pd.Timestamp):
            t = yearfrac(self.asof, t)
        return float(np.exp(self._logdf_at(float(t))))

    def fwd(self, t0, t1) -> float:
        """Simple-compounded forward rate (ACT/365F) over [t0, t1]."""
        if isinstance(t0, pd.Timestamp):
            t0 = yearfrac(self.asof, t0)
        if isinstance(t1, pd.Timestamp):
            t1 = yearfrac(self.asof, t1)
        return (self.df(t0) / self.df(t1) - 1.0) / (t1 - t0)

    def zero(self, t: float | pd.Timestamp) -> float:
        """Continuously-compounded zero rate to t (ACT/365F)."""
        if isinstance(t, pd.Timestamp):
            t = yearfrac(self.asof, t)
        t = float(t)
        if t <= 0:
            return self._fwd_inst(1e-6)          # degenerate: today's short rate
        return -self._logdf_at(t) / t

    def inst_fwd_grid(self, t_max: float | None = None, n: int = 720) -> pd.Series:
        """Instantaneous forward curve on a fine grid (diagnostics/plotting)."""
        t_max = t_max or self.t[-1]
        grid = np.linspace(1e-4, t_max, n)
        return pd.Series([self._fwd_inst(x) for x in grid], index=grid)

    # ------------------------------------------------------------ swaps
    def swap_schedule(self, effective: pd.Timestamp, maturity: pd.Timestamp,
                      freq_months: int = 12):
        """Eris-style annual schedule: MF-adjusted accrual ends, pay +2bd.

        Returns (tau_list_act360, pay_dates, accr_ends).
        """
        from .calendars import modified_following, us_holidays
        effective, maturity = pd.Timestamp(effective), pd.Timestamp(maturity)
        hol = us_holidays(effective, maturity + pd.Timedelta(days=10))
        n_years = round((maturity - effective).days / 365.25)
        ends = [modified_following(effective + pd.DateOffset(years=k), hol)
                for k in range(1, n_years + 1)]
        ends[-1] = maturity
        starts = [effective] + ends[:-1]
        tau = [(e - s).days / 360.0 for s, e in zip(starts, ends)]
        pay = [modified_following(e + pd.Timedelta(days=2), hol) for e in ends]
        return tau, pay, ends

    def par_swap(self, effective: pd.Timestamp, maturity: pd.Timestamp,
                 freq_months: int = 12) -> tuple[float, float]:
        """(par_rate, annuity_years) for a SOFR swap, annual Act/360 fixed."""
        tau, pay, _ = self.swap_schedule(effective, maturity, freq_months)
        annuity = float(sum(t * self.df(d) for t, d in zip(tau, pay)))
        par = (self.df(effective) - self.df(pd.Timestamp(maturity))) / annuity
        return par, annuity

    def annuity(self, effective: pd.Timestamp, maturity: pd.Timestamp,
                freq_months: int = 12) -> float:
        return self.par_swap(effective, maturity, freq_months)[1]
