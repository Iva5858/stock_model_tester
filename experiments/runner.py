from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from data import get_loader
from evaluation import compute_metrics, save_results
from features import FeatureConfig, build_features
from models import get_model_class

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on any key conflict."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_config(config_path: str | Path, ticker_override: str | None = None) -> dict:
    """Load and validate a config, applying a ticker-specific override if one exists.

    Override lookup: configs/tickers/<TICKER>/<config_filename>.yaml
    If found, it is deep-merged on top of the base config before validation.
    If not found, the base config is used as-is.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if ticker_override:
        ticker_upper = ticker_override.upper()
        override_path = path.parent / "tickers" / ticker_upper / path.name
        if override_path.exists():
            with open(override_path) as f:
                override = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, override)
            logger.info("Applied ticker override: %s", override_path)

        cfg["data"]["ticker"] = ticker_upper

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    required_top = {"experiment", "data", "features", "model", "evaluation"}
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Config missing top-level keys: {missing}")
    if "name" not in cfg["model"]:
        raise ValueError("Config must specify model.name")
    if cfg["data"].get("loader") not in ("yfinance", "csv"):
        raise ValueError("data.loader must be 'yfinance' or 'csv'")


def run_experiment(config_path: str | Path, dry_run: bool = False,
                   ticker_override: str | None = None) -> Path:
    cfg = _load_config(config_path, ticker_override=ticker_override)

    seed = cfg.get("seed", 42)
    _set_seed(seed)

    data_cfg = cfg["data"]
    feat_cfg_raw = cfg["features"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]

    # ── Data ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    loader_name = data_cfg["loader"]
    loader = get_loader(loader_name)

    if loader_name == "yfinance":
        ticker = data_cfg["ticker"]
        df = loader.load(
            ticker=ticker,
            start=data_cfg["start"],
            end=data_cfg["end"],
            missing=data_cfg.get("missing", "ffill"),
        )
    else:
        ticker = Path(data_cfg["csv_path"]).stem
        df = loader.load(
            csv_path=data_cfg["csv_path"],
            missing=data_cfg.get("missing", "ffill"),
        )

    logger.info("Data loaded: %d rows, columns: %s", len(df), list(df.columns))

    # ── Features ──────────────────────────────────────────────────────────────
    feat_config = FeatureConfig(
        target=feat_cfg_raw.get("target", "next_return"),
        lags=feat_cfg_raw.get("lags", [1, 2, 3, 5, 10]),
        rolling_windows=feat_cfg_raw.get("rolling_windows", [5, 10, 20]),
        technical_indicators=feat_cfg_raw.get("technical_indicators", False),
        test_size=eval_cfg.get("test_size", 0.2),
    )
    pipeline_out = build_features(df, feat_config)

    logger.info(
        "Features built: %d features, train=%d, test=%d",
        len(pipeline_out.feature_names),
        len(pipeline_out.X_train),
        len(pipeline_out.X_test),
    )

    if dry_run:
        logger.info("Dry run complete — skipping model fit.")
        return Path("results").resolve() / "dry_run"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = model_cfg["name"]
    model_params = model_cfg.get("params", {}) or {}
    model_params["random_state"] = seed
    ModelClass = get_model_class(model_name)
    model = ModelClass(**model_params)

    logger.info("Fitting model: %s with params %s", model_name, model_params)
    model.fit(pipeline_out.X_train, pipeline_out.y_train)

    # ── Prediction ────────────────────────────────────────────────────────────
    y_pred = model.predict(pipeline_out.X_test)
    y_true = pipeline_out.y_test

    # Align lengths — sequence models may return fewer predictions than test rows
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[-min_len:]
    y_pred = y_pred[-min_len:]
    test_dates = pipeline_out.test_dates[-min_len:]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metric_names = eval_cfg.get("metrics", ["rmse", "mae", "r2",
                                            "directional_accuracy", "sharpe"])
    metrics = compute_metrics(y_true, y_pred, metric_names)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)
    logger.info("Metrics: %s", metrics)

    # ── Save ──────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{model_name}_{ticker}_{timestamp}"
    results_dir = Path("results").resolve() / ticker / model_name / experiment_id

    save_results(
        results_dir=results_dir,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        dates=test_dates,
        config=cfg,
        feature_names=pipeline_out.feature_names,
    )

    _print_summary(experiment_id, metrics, elapsed, results_dir)
    return results_dir


def _print_summary(exp_id: str, metrics: dict, elapsed: float, results_dir: Path) -> None:
    width = 50
    print("\n" + "=" * width)
    print(f"  Experiment: {exp_id}")
    print("=" * width)
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.6f}" if isinstance(v, float) else f"  {k:<25} {v}")
    print(f"  {'elapsed':<25} {elapsed:.1f}s")
    print(f"  Results saved to: {results_dir}")
    print("=" * width + "\n")
