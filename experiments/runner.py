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
from features import FeatureConfig, build_features, walk_forward_splits
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
    cv = cfg.get("evaluation", {}).get("cv_method", "holdout")
    if cv not in ("holdout", "expanding", "rolling"):
        raise ValueError(f"evaluation.cv_method must be 'holdout', 'expanding', or 'rolling'; got '{cv}'")


def _save_shap_importance(
    model, X_test: np.ndarray, feature_names: list[str], results_dir: Path
) -> None:
    """Compute and save SHAP values for tree-based models.

    Silently skips if shap is not installed or the model does not support TreeExplainer.
    References: rpaper_8 (Mistol & Möhler 2023), rpaper_4 (Alhomadi 2021).
    """
    try:
        import shap
        import pandas as pd
        underlying = getattr(model, "_model", None)
        if underlying is None:
            return
        explainer = shap.TreeExplainer(underlying)
        shap_vals = explainer.shap_values(X_test)
        if shap_vals is None or not isinstance(shap_vals, np.ndarray):
            return
        pd.DataFrame(shap_vals, columns=feature_names).to_csv(
            results_dir / "shap_values.csv", index=False
        )
        mean_abs = np.abs(shap_vals).mean(axis=0)
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs.tolist(),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True).to_csv(
            results_dir / "feature_importance.csv", index=False
        )
        logger.info("SHAP feature importance saved to %s", results_dir)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("SHAP computation skipped: %s", exc)


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
    cv_method = eval_cfg.get("cv_method", "holdout")
    feat_config = FeatureConfig(
        target=feat_cfg_raw.get("target", "next_return"),
        lags=feat_cfg_raw.get("lags", [1, 2, 3, 5, 10]),
        rolling_windows=feat_cfg_raw.get("rolling_windows", [5, 10, 20]),
        technical_indicators=feat_cfg_raw.get("technical_indicators", False),
        test_size=eval_cfg.get("test_size", 0.2),
        cv_method=cv_method,
        step_size=eval_cfg.get("step_size", 21),
        min_train_size=eval_cfg.get("min_train_size", 500),
        window_size=eval_cfg.get("window_size", 1000),
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

    # ── Model setup ───────────────────────────────────────────────────────────
    model_name = model_cfg["name"]
    model_params = model_cfg.get("params", {}) or {}
    model_params["random_state"] = seed
    ModelClass = get_model_class(model_name)

    # ── Fit / Predict (holdout or walk-forward) ────────────────────────────────
    if cv_method in ("expanding", "rolling"):
        y_true, y_pred, test_dates, y_train_ctx, shap_model, shap_X = \
            _run_walk_forward(
                pipeline_out=pipeline_out,
                ModelClass=ModelClass,
                model_params=model_params,
                cv_method=cv_method,
                step_size=feat_config.step_size,
                min_train_size=feat_config.min_train_size,
                window_size=feat_config.window_size if cv_method == "rolling" else None,
            )
    else:
        model = ModelClass(**model_params)
        logger.info("Fitting model: %s with params %s", model_name, model_params)
        model.fit(pipeline_out.X_train, pipeline_out.y_train)
        y_pred = model.predict(pipeline_out.X_test)
        y_true = pipeline_out.y_test
        test_dates = pipeline_out.test_dates
        y_train_ctx = pipeline_out.y_train
        shap_model = model
        shap_X = pipeline_out.X_test

        # Align lengths — sequence models may return fewer predictions than test rows
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]
        test_dates = test_dates[-min_len:]
        shap_X = shap_X[-min_len:]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metric_names = eval_cfg.get("metrics", ["rmse", "mae", "r2",
                                            "directional_accuracy", "sharpe"])
    metrics = compute_metrics(y_true, y_pred, metric_names, y_train=y_train_ctx)
    metrics["cv_method"] = cv_method

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)
    logger.info("Metrics: %s", {k: v for k, v in metrics.items() if k != "cv_method"})

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

    # SHAP feature importance (tree models only; silent fallback)
    _save_shap_importance(shap_model, shap_X, pipeline_out.feature_names, results_dir)

    _print_summary(experiment_id, metrics, elapsed, results_dir)
    return results_dir


def _run_walk_forward(
    pipeline_out,
    ModelClass,
    model_params: dict,
    cv_method: str,
    step_size: int,
    min_train_size: int,
    window_size: int | None,
):
    """Run walk-forward (expanding or rolling window) cross-validation.

    Returns aggregated (y_true, y_pred, test_dates, y_train_ctx, last_model, last_X_te).
    Each fold re-fits the model and re-scales from scratch to prevent leakage.

    References: rpaper_1 (Turgay 2025), rpaper_8 (Mistol & Möhler 2023),
                fpaper_2 (Goyal & Welch 2008).
    """
    from sklearn.preprocessing import StandardScaler

    splits = walk_forward_splits(
        X_raw=pipeline_out.X_full_raw,
        y=pipeline_out.y_full,
        dates=pipeline_out.dates_full,
        min_train_size=min_train_size,
        step_size=step_size,
        window_size=window_size,
    )

    if not splits:
        raise ValueError(
            f"No walk-forward splits generated. "
            f"Reduce evaluation.min_train_size (current: {min_train_size}) "
            "or fetch more data."
        )

    logger.info("Walk-forward (%s): %d folds", cv_method, len(splits))

    all_y_true, all_y_pred, all_dates = [], [], []
    # Collect all training data preceding the test window for OOS R² benchmark
    y_train_ctx_list = []
    last_model = None
    last_X_te = None

    for X_tr_raw, y_tr, X_te_raw, y_te, test_d in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        m = ModelClass(**model_params)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)

        mn = min(len(y_te), len(preds))
        all_y_true.extend(y_te[-mn:])
        all_y_pred.extend(preds[-mn:])
        all_dates.extend(test_d[-mn:])
        y_train_ctx_list.extend(y_tr)
        last_model = m
        last_X_te = X_te[-mn:]

    y_true = np.array(all_y_true, dtype=np.float32)
    y_pred = np.array(all_y_pred, dtype=np.float32)
    y_train_ctx = np.array(y_train_ctx_list, dtype=np.float32)
    test_dates = pipeline_out.dates_full.__class__(all_dates)

    return y_true, y_pred, test_dates, y_train_ctx, last_model, last_X_te


def _print_summary(exp_id: str, metrics: dict, elapsed: float, results_dir: Path) -> None:
    width = 50
    print("\n" + "=" * width)
    print(f"  Experiment: {exp_id}")
    print("=" * width)
    for k, v in metrics.items():
        if k == "cv_method":
            print(f"  {'cv_method':<25} {v}")
        else:
            print(f"  {k:<25} {v:.6f}" if isinstance(v, float) else f"  {k:<25} {v}")
    print(f"  {'elapsed':<25} {elapsed:.1f}s")
    print(f"  Results saved to: {results_dir}")
    print("=" * width + "\n")
