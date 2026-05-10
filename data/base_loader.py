from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):
    """Abstract interface for all data loaders."""

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Return a standardised OHLCV DataFrame with date as index."""
        ...

    def _validate_and_clean(self, df: pd.DataFrame, missing: str = "ffill") -> pd.DataFrame:
        df.columns = [c.lower() for c in df.columns]
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Loader returned DataFrame missing columns: {missing_cols}")

        df.index.name = "date"
        df = df[list(self.REQUIRED_COLUMNS)]

        if missing == "ffill":
            df = df.ffill().dropna()
        else:
            df = df.dropna()

        return df.sort_index()
