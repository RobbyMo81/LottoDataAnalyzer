"""
Utility helpers used across modules.
Keep these *pure* (no Streamlit calls) so they're easy to unit-test.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------ #
# General-purpose helpers
# ------------------------------------------------------------------ #
def to_datetime(df: pd.DataFrame, col: str = "draw_date") -> pd.Series:
    """Ensures a column is pandas datetime, returns the converted Series."""
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df[col]


def percentile_of_series(series: pd.Series, value: float) -> float:
    """
    Manual percentile without scipy; returns a 0-100 float.
    """
    return 100.0 * (series < value).mean()


def rolling_delta(series: pd.Series, window: int) -> float:
    """
    Difference between the sum of the last *window* observations
    and the preceding *window*.
    """
    if len(series) < 2 * window:
        return np.nan
    recent = series[-window:].sum()
    prev = series[-2*window:-window].sum()
    return recent - prev


# ------------------------------------------------------------------ #
# Colour scales (centralised so charts stay consistent)
# ------------------------------------------------------------------ #
COLOR_HOT = "#d62728"   # red
COLOR_COLD = "#1f77b4"  # blue
COLOR_NEUTRAL = "#2ca02c"  # green
