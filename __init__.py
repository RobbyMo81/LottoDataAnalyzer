"""
Core package for Powerball Insights.
Exposes the datastore singleton so other code can simply:

    from core import store
    df = store.latest()

You can also tuck package-wide constants or the semver here.
"""
from importlib.metadata import version as _v

try:
    __version__ = _v("powerball_insights")  # if you later publish to PyPI
except Exception:
    __version__ = "0.1.1"

# Public shortcut to the datastore
from .storage import get_store as _get_store
store = _get_store()

__all__ = ["store", "__version__"]
