"""Tests for FileResultsStore v1/v2 layout handling."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _write_exp(base: Path, ticker: str, model: str, exp_id: str,
               horizon: int = 1, target: str = "next_return",
               schema_version: int | None = None) -> Path:
    """Write a fake experiment directory in v2 layout by default."""
    exp_dir = base / ticker / model / f"h{horizon}" / target / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "rmse": 0.01,
        "directional_accuracy": 0.55,
        "sharpe": 0.8,
        "cv_method": "holdout",
    }
    if schema_version is not None:
        metrics["schema_version"] = schema_version
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))
    pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "y_true": [0.01, -0.005],
        "y_pred": [0.008, -0.003],
    }).to_csv(exp_dir / "predictions.csv", index=False)
    return exp_dir


def _write_v1_exp(base: Path, ticker: str, model: str, exp_id: str) -> Path:
    """Write a fake v1-format experiment (no schema_version, 3-level path)."""
    exp_dir = base / ticker / model / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"rmse": 0.02, "directional_accuracy": 0.51}
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))
    pd.DataFrame({
        "date": ["2023-01-01"],
        "y_true": [0.01],
        "y_pred": [0.009],
    }).to_csv(exp_dir / "predictions.csv", index=False)
    return exp_dir


# ── list_experiments ──────────────────────────────────────────────────────────

def test_list_experiments_basic(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_001")
    _write_exp(tmp_path, "AAPL", "Ridge", "exp_002")
    store = FileResultsStore(tmp_path)
    records = store.list_experiments()
    assert len(records) == 2
    models = {r.model for r in records}
    assert "XGBoost" in models
    assert "Ridge" in models


def test_list_experiments_filter_ticker(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_001")
    _write_exp(tmp_path, "TSLA", "XGBoost", "exp_002")
    store = FileResultsStore(tmp_path)
    records = store.list_experiments(ticker="AAPL")
    assert len(records) == 1
    assert records[0].ticker == "AAPL"


def test_list_experiments_filter_model(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_001")
    _write_exp(tmp_path, "AAPL", "Ridge", "exp_002")
    store = FileResultsStore(tmp_path)
    records = store.list_experiments(model="Ridge")
    assert len(records) == 1
    assert records[0].model == "Ridge"


def test_list_experiments_filter_horizon(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_h1", horizon=1)
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_h5", horizon=5)
    store = FileResultsStore(tmp_path)
    records = store.list_experiments(horizon=5)
    assert len(records) == 1
    assert records[0].horizon == 5


def test_list_experiments_filter_target(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_ret", target="next_return")
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_dir", target="direction")
    store = FileResultsStore(tmp_path)
    records = store.list_experiments(target="direction")
    assert len(records) == 1
    assert records[0].target == "direction"


def test_list_experiments_sorted(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_20240101")
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_20230101")
    store = FileResultsStore(tmp_path)
    records = store.list_experiments()
    ids = [r.exp_id for r in records]
    assert ids == sorted(ids)


# ── v1 path detection ─────────────────────────────────────────────────────────

def test_v1_path_detected_as_h1_next_return(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_v1_exp(tmp_path, "AAPL", "Ridge", "Ridge_AAPL_20240101_120000")
    store = FileResultsStore(tmp_path)
    records = store.list_experiments()
    assert len(records) == 1
    rec = records[0]
    assert rec.horizon == 1
    assert rec.target == "next_return"
    assert rec.ticker == "AAPL"
    assert rec.model == "Ridge"


def test_v1_and_v2_coexist(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_v1_exp(tmp_path, "AAPL", "Ridge", "Ridge_AAPL_old")
    _write_exp(tmp_path, "AAPL", "XGBoost", "XGBoost_AAPL_new", horizon=1)
    store = FileResultsStore(tmp_path)
    records = store.list_experiments(ticker="AAPL")
    assert len(records) == 2


# ── load_predictions / load_metrics ───────────────────────────────────────────

def test_load_predictions(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_001")
    store = FileResultsStore(tmp_path)
    df = store.load_predictions("exp_001")
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert len(df) == 2


def test_load_metrics(tmp_path):
    from evaluation.results_store import FileResultsStore
    _write_exp(tmp_path, "AAPL", "XGBoost", "exp_001")
    store = FileResultsStore(tmp_path)
    m = store.load_metrics("exp_001")
    assert "rmse" in m
    assert isinstance(m["rmse"], float)


def test_load_predictions_missing_raises(tmp_path):
    from evaluation.results_store import FileResultsStore
    store = FileResultsStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load_predictions("nonexistent_exp")


def test_empty_results_dir(tmp_path):
    from evaluation.results_store import FileResultsStore
    store = FileResultsStore(tmp_path / "empty")
    records = store.list_experiments()
    assert records == []
