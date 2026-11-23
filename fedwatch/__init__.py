"""fedwatch: a 0-30Y rates curve & probability engine built on CME futures.

Extends the CME FedWatch methodology across the whole USD curve:
  Regime 1 (0-18M)   ZQ 30D Fed Funds + SR1 1M SOFR  -> FOMC step probabilities
  Regime 2 (18M-5Y+) SR3 3M SOFR futures            -> convexity-adjusted OIS forwards
  Regime 3 (5-30Y)   Eris SOFR swap futures (YI*)   -> par swap anchors

All segments are bridged onto a single SOFR-discounting curve, from which
probability distributions are derived (meeting-level in Regime 1; model-based
in Regimes 2-3 until option-implied PDFs arrive in Phase 2).
"""

__version__ = "0.1.0"
