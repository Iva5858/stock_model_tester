"""Full pipeline smoke test using synthetic in-memory data."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_synthetic_csv(path: Path, n: int = 150):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    df.index.name = "date"
    df.to_csv(path)


def _make_config(model_name: str, csv_path: str) -> dict:
    return {
        "experiment": {"name": f"smoke_{model_name}"},
        "data": {"loader": "csv", "csv_path": csv_path},
        "features": {
            "target": "next_return",
            "lags": [1, 2, 3],
            "rolling_windows": [5],
            "technical_indicators": False,
        },
        "model": {"name": model_name, "params": {}},
        "evaluation": {
            "test_size": 0.2,
            "metrics": ["rmse", "mae", "r2"],
        },
        "seed": 0,
    }


@pytest.mark.parametrize("model_name", ["NaiveLastValue", "RollingMean", "XGBoost", "RandomForest"])
def test_full_pipeline_smoke(model_name, tmp_path):
    csv_file = tmp_path / "synthetic.csv"
    _make_synthetic_csv(csv_file)

    config = _make_config(model_name, str(csv_file))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from experiments.runner import run_experiment
        results_dir = run_experiment(str(config_file))
    finally:
        os.chdir(original_cwd)

    assert results_dir.exists(), f"Results dir not created: {results_dir}"
    assert (results_dir / "metrics.json").exists()
    assert (results_dir / "predictions.csv").exists()
    assert (results_dir / "config_snapshot.yaml").exists()

    metrics = json.loads((results_dir / "metrics.json").read_text())
    assert "rmse" in metrics
    assert isinstance(metrics["rmse"], float)
    assert metrics["rmse"] >= 0
