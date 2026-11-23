"""Central configuration: paths, product universe, conventions, credentials."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- paths
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# legacy CSV caches at repo root (migrated in-place by ingest layer if present)
LEGACY_FUTURES_CSV = REPO_ROOT / "futures_daily_contracts.csv"
LEGACY_FRED_CSV = REPO_ROOT / "fred_curves.csv"
LEGACY_FOMC_CSV = REPO_ROOT / "fomc_meetings.csv"

FUTURES_PARQUET = DATA_DIR / "futures_ohlcv1d.parquet"
DEFS_PARQUET = DATA_DIR / "instrument_definitions.parquet"
FRED_PARQUET = DATA_DIR / "fred_series.parquet"
FOMC_CSV = DATA_DIR / "fomc_meetings.csv"          # keep CSV (human-editable)

# ---------------------------------------------------------------- products
MONTH_CODES = "FGHJKMNQUVXZ"                        # CME month code convention
MONTH_OF = {c: i + 1 for i, c in enumerate(MONTH_CODES)}
QUARTER_CODES = "HMUZ"                              # IMM cycle only

# futures roots pulled from Databento GLBX.MDP3 (ohlcv-1d)
FUTURES_ROOTS = [
    # short end (FedWatch segment)
    "ZQ",   # 30D Federal Funds futures (calendar-month average EFFR)
    "SR1",  # 1M SOFR futures (calendar-month average SOFR)
    # mid curve
    "SR3",  # 3M SOFR futures (IMM quarterly, compounded daily SOFR)
    # treasury futures (sanity/spread overlay only)
    "ZT", "ZF", "ZN", "ZB", "TN",
    # long end: Eris SOFR swap futures, one root per tenor
    "YIA", "YIT", "YIC", "YID", "YIW", "YIB", "YIY", "YII", "YIL", "YIO", "YIE",
]

# Eris tenor (years) by root — per erisfutures.com/sofrcontractspecs.
ERIS_TENORS = {
    "YIA": 1, "YIT": 2, "YIC": 3, "YID": 4, "YIW": 5,
    "YIB": 7, "YIY": 10, "YII": 12, "YIL": 15, "YIO": 20, "YIE": 30,
}

# ---------------------------------------------------------------- conventions
BUCKET_BP = 25.0          # probability bucket width (bps), FedWatch standard
MAX_STEPS = 12            # max cumulative 25bp steps considered per path
MAX_SINGLE_STEP = 6       # clip on single-meeting expected steps

DATABENTO_DATASET = "GLBX.MDP3"

# FRED series: realized anchors & treasury curve (units: percent as delivered)
FRED_SERIES = [
    "EFFR", "DFF", "DFEDTARU", "DFEDTARL", "DFEDTAR", "SOFR",
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
    "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
]

# ---------------------------------------------------------------- credentials
def _from_repo_credentials(name: str) -> str | None:
    """Read a variable from the gitignored credentials.py at repo root."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import credentials  # noqa: PLC0415
        return getattr(credentials, name, None)
    except ImportError:
        return None


def get_fred_key() -> str:
    key = os.environ.get("FRED_API_KEY") or _from_repo_credentials("apikey")
    if not key:
        raise RuntimeError("No FRED API key (set FRED_API_KEY or credentials.apikey)")
    return key


def get_databento_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY") or _from_repo_credentials("db_key")
    if not key:
        raise RuntimeError("No Databento API key (set DATABENTO_API_KEY or credentials.db_key)")
    return key
