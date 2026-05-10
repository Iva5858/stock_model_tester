import pandas as pd
import yfinance as yf

from .base_loader import BaseLoader


class YFinanceLoader(BaseLoader):
    def load(self, ticker: str, start: str, end: str, missing: str = "ffill") -> pd.DataFrame:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"yfinance returned no data for ticker '{ticker}' between {start} and {end}")

        # yfinance returns MultiIndex columns when auto_adjust=True; flatten them
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]

        return self._validate_and_clean(raw, missing=missing)
