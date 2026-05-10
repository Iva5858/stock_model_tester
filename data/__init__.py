from .base_loader import BaseLoader
from .csv_loader import CSVLoader


def get_loader(name: str) -> BaseLoader:
    if name == "yfinance":
        from .yfinance_loader import YFinanceLoader
        return YFinanceLoader()
    if name == "csv":
        return CSVLoader()
    raise ValueError(f"Unknown loader '{name}'. Choose from: ['yfinance', 'csv']")
