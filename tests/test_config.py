import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import _validate_config


def _base_config():
    return {
        "experiment": {"name": "test"},
        "data": {"loader": "yfinance", "ticker": "AAPL", "start": "2020-01-01", "end": "2021-01-01"},
        "features": {"target": "next_close"},
        "model": {"name": "XGBoost"},
        "evaluation": {"test_size": 0.2},
    }


def test_valid_config_passes():
    _validate_config(_base_config())


def test_missing_top_level_key_raises():
    cfg = _base_config()
    del cfg["model"]
    with pytest.raises(ValueError, match="missing top-level keys"):
        _validate_config(cfg)


def test_missing_model_name_raises():
    cfg = _base_config()
    del cfg["model"]["name"]
    with pytest.raises(ValueError, match="model.name"):
        _validate_config(cfg)


def test_invalid_loader_raises():
    cfg = _base_config()
    cfg["data"]["loader"] = "unknown_loader"
    with pytest.raises(ValueError, match="data.loader"):
        _validate_config(cfg)
