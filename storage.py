# core/storage.py
from pathlib import Path
import pandas as pd
import datetime as dt
import joblib
from typing import Dict, Any

DATA_PATH = Path("data")
META = DATA_PATH / "_meta.joblib"


class _Store:
    """
    Simple versioned data store.

    • `ingest(df)`      → write a *new* versioned parquet file.
    • `latest()`        → read the newest parquet file.
    • `set_latest(df)`  → overwrite the current latest version
                           (no new file, keeps history length constant).
    """

    def __init__(self) -> None:
        DATA_PATH.mkdir(exist_ok=True)
        self.meta: Dict[str, Any] = (
            joblib.load(META) if META.exists() else {"versions": []}
        )

    # -----------------------------------------------------------
    # WRITE METHODS
    # -----------------------------------------------------------
    def ingest(self, df: pd.DataFrame) -> None:
        """Add a *new* versioned file and make it the latest."""
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DATA_PATH / f"history_{ts}.parquet"
        df.to_parquet(path, index=False)
        self.meta.setdefault("versions", []).append(path.name)
        joblib.dump(self.meta, META)

    def set_latest(self, df: pd.DataFrame) -> None:
        """
        Replace the existing latest version with `df`.
        If no versions exist yet, falls back to ingest().
        """
        if not self.meta["versions"]:
            self.ingest(df)
            return

        latest_file = sorted(self.meta["versions"])[-1]
        path = DATA_PATH / latest_file
        df.to_parquet(path, index=False)

    # -----------------------------------------------------------
    # READ METHODS
    # -----------------------------------------------------------
    def latest(self) -> pd.DataFrame:
        """Return the most recently saved DataFrame."""
        if not self.meta["versions"]:
            return pd.DataFrame()  # Return empty DataFrame if no data ingested
        latest_file = sorted(self.meta["versions"])[-1]
        try:
            return pd.read_parquet(DATA_PATH / latest_file)
        except Exception:
            return pd.DataFrame()  # Return empty DataFrame on error


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------
_STORE = _Store()


def get_store() -> _Store:
    return _STORE
