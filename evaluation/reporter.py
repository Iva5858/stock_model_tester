from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def save_results(
    results_dir: Path,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    config: dict,
    feature_names: list[str],
    *,
    ticker: str = "",
    model: str = "",
    horizon: int = 1,
    target: str = "next_return",
    n_train: int = 0,
    n_test: int = 0,
    elapsed_s: float = 0.0,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 2,
        "ticker": ticker,
        "model": model,
        "horizon": horizon,
        "target": target,
        "n_train": n_train,
        "n_test": n_test,
        "elapsed_s": elapsed_s,
        **metrics,
        "feature_names": feature_names,
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_serial)
    )

    pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred}).to_csv(
        results_dir / "predictions.csv", index=False
    )

    (results_dir / "config_snapshot.yaml").write_text(
        yaml.dump(config, default_flow_style=False)
    )


def compare_results(result_dirs: list[Path], label_fn=None) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of experiment result directories.

    label_fn: optional callable(Path) -> str to customise the row label.
              Defaults to the directory name.
    """
    records = []
    for d in result_dirs:
        metrics_file = d / "metrics.json"
        if not metrics_file.exists():
            continue
        data = json.loads(metrics_file.read_text())
        data.pop("feature_names", None)
        label = label_fn(d) if label_fn else d.name
        records.append({"model": label, **data})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("model")
    if "rmse" in df.columns:
        df = df.sort_values("rmse")
    return df


def compare_multi_ticker(tickers: list[str], results_root: Path,
                          metric: str = "directional_accuracy") -> pd.DataFrame:
    """Build a model × ticker pivot table for a single metric.

    Rows = models, columns = tickers, values = the chosen metric from the
    latest run of each (ticker, model) pair.
    """
    from .results_store import FileResultsStore
    store = FileResultsStore(results_root)
    records: dict[str, dict[str, float]] = {}

    for ticker in tickers:
        ticker_upper = ticker.upper()
        exps = store.list_experiments(ticker=ticker_upper)
        # latest per model
        latest: dict[str, "ExperimentRecord"] = {}
        for rec in exps:
            latest[rec.model] = rec
        for model_name, rec in latest.items():
            if model_name not in records:
                records[model_name] = {}
            records[model_name][ticker_upper] = rec.metrics.get(metric, float("nan"))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).T  # models as rows, tickers as columns
    df.index.name = "model"
    df["avg"] = df.mean(axis=1, skipna=True)
    df = df.sort_values("avg", ascending=False)
    return df


def _json_serial(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
