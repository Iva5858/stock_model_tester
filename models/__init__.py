from .base_model import BaseModel, get_model_class, list_models, register

# Core models — always available
from . import baseline, ml_models  # noqa: F401

# Deep learning models — require torch
try:
    from . import dl_models  # noqa: F401
except ImportError:
    pass

# Classical time-series models — require statsmodels (+ arch for GARCH)
try:
    from . import classical_models  # noqa: F401
except ImportError:
    pass

# Regime-switching models — require statsmodels and/or hmmlearn
try:
    from . import regime_models  # noqa: F401
except ImportError:
    pass
