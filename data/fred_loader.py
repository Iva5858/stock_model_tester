from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from .base_loader import BaseLoader, register_loader

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "cache" / "fred"
_CACHE_TTL_HOURS = 24

SERIES = {
    "DGS10": {"name": "treasury_10y", "lag_days": 1},
    "TB3MS": {"name": "tbill_3m", "lag_days": 1},
    "DBAA": {"name": "baa_yield", "lag_days": 1},
    "DAAA": {"name": "aaa_yield", "lag_days": 1},
    "CPIAUCSL": {"name": "cpi", "lag_days": 15},
}


def _cache_path(series_id: str) -> Path:
    return _CACHE_DIR / f"{series_id}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=_CACHE_TTL_HOURS)


def _load_cached(series_id: str) -> pd.Series | None:
    p = _cache_path(series_id)
    if p.exists():
        try:
            return pd.read_parquet(p).squeeze()
        except Exception:
            return None
    return None


def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series | None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set — skipping FRED series '%s'", series_id)
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id, observation_start=start,
                               observation_end=end)
        if data is None or data.empty:
            return None
        data.name = series_id
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(series_id)
        pd.DataFrame(data).to_parquet(cache_file)
        return data
    except Exception as exc:
        logger.warning("FRED fetch failed for '%s': %s", series_id, exc)
        return None


@register_loader("fred")
class FREDLoader(BaseLoader):
    """Fetches macro series from FRED API. Requires FRED_API_KEY env var."""

    # Expose the series config so transforms can reference it
    SERIES = SERIES

    def load(self, start: str, end: str) -> pd.DataFrame:  # type: ignore[override]
        frames: dict[str, pd.Series] = {}

        for series_id, meta in SERIES.items():
            col_name = meta["name"]
            lag_days = meta["lag_days"]

            cache_p = _cache_path(series_id)
            if _is_fresh(cache_p):
                data = _load_cached(series_id)
            else:
                data = _fetch_fred(series_id, start, end)
                if data is None:
                    data = _load_cached(series_id)
                    if data is not None:
                        logger.warning(
                            "Using stale cache for FRED '%s'", series_id)

            if data is None:
                logger.warning(
                    "No data for FRED series '%s' — column '%s' will be missing",
                    series_id, col_name,
                )
                continue

            # Apply publication lag
            data = data.copy()
            data.index = pd.DatetimeIndex(data.index) + pd.Timedelta(days=lag_days)
            data.name = col_name
            frames[col_name] = data

        if not frames:
            return pd.DataFrame()

        # Align to daily business day index and forward-fill
        daily_idx = pd.bdate_range(start=start, end=end)
        result = pd.DataFrame(index=daily_idx)
        for col, series in frames.items():
            series.index = pd.DatetimeIndex(series.index)
            result[col] = series.reindex(daily_idx).ffill()

        # Compute derived series
        if "treasury_10y" in result.columns and "tbill_3m" in result.columns:
            result["term_spread"] = result["treasury_10y"] - result["tbill_3m"]
        if "baa_yield" in result.columns and "aaa_yield" in result.columns:
            result["default_spread"] = result["baa_yield"] - result["aaa_yield"]

        # Drop pure NaN columns
        result = result.dropna(how="all", axis=1)
        return result

    # BaseLoader requires _validate_and_clean but FRED data has no OHLCV
    def _validate_and_clean(self, df, missing="ffill"):  # type: ignore[override]
        return df
