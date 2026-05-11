import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from click.testing import CliRunner
from run import cli


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


def _make_config(csv_path: str) -> dict:
    return {
        "experiment": {"name": "cli_test"},
        "data": {"loader": "csv", "csv_path": csv_path},
        "features": {
            "target": "next_return",
            "lags": [1, 2],
            "rolling_windows": [5],
            "technical_indicators": False,
        },
        "model": {"name": "NaiveLastValue", "params": {}},
        "evaluation": {"test_size": 0.2, "metrics": ["rmse", "mae"]},
        "seed": 0,
    }


def test_run_command(tmp_path):
    csv_file = tmp_path / "data.csv"
    _make_synthetic_csv(csv_file)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(_make_config(str(csv_file))))

    runner = CliRunner()
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["run", "--config", str(config_file)])
    finally:
        os.chdir(original_cwd)

    assert result.exit_code == 0, result.output


def test_list_models_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["list-models"])
    assert result.exit_code == 0
    assert "NaiveLastValue" in result.output
    assert "XGBoost" in result.output


def test_compare_command(tmp_path):
    csv_file = tmp_path / "data.csv"
    _make_synthetic_csv(csv_file)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(_make_config(str(csv_file))))

    runner = CliRunner()
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Run an experiment first so results exist
        run_result = runner.invoke(cli, ["run", "--config", str(config_file)])
        assert run_result.exit_code == 0, run_result.output

        # Results live 5 levels deep in v2: results/<ticker>/<model>/h<H>/<target>/<exp>/
        experiment_dirs = sorted((tmp_path / "results").glob("*/*/*/*/*"))
        assert len(experiment_dirs) >= 1, "No experiment directories found"

        result = runner.invoke(cli, ["compare"] + [str(d) for d in experiment_dirs])
        assert result.exit_code == 0
        assert "rmse" in result.output.lower() or "NaiveLastValue" in result.output
    finally:
        os.chdir(original_cwd)
