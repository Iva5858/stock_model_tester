from .base_loader import BaseLoader, _LOADER_REGISTRY, register_loader, get_loader_class
from .csv_loader import CSVLoader  # noqa: F401 — triggers @register_loader("csv")
from .yfinance_loader import YFinanceLoader  # noqa: F401 — triggers @register_loader("yfinance")
from .fred_loader import FREDLoader  # noqa: F401 — triggers @register_loader("fred")


def get_loader(name: str) -> BaseLoader:
    """Return an instantiated loader by name (backward-compatible entry point)."""
    cls = get_loader_class(name)
    return cls()
