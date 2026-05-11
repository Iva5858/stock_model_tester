import pandas as pd

from .base_loader import BaseLoader, register_loader


@register_loader("csv")
class CSVLoader(BaseLoader):
    def load(self, csv_path: str, missing: str = "ffill") -> pd.DataFrame:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return self._validate_and_clean(df, missing=missing)
