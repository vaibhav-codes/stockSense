# =============================================================================
#  backend/utils/__init__.py
#  Shared utility functions used across backend modules.
# =============================================================================

import re
from datetime import datetime


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))


def safe_round(value, digits: int = 2):
    """Round a value safely, returning None if value is None."""
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def now_ist() -> datetime:
    """Return current datetime in IST (UTC+5:30)."""
    import pytz
    return datetime.now(pytz.timezone("Asia/Kolkata"))
