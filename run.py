#!/usr/bin/env python
"""CLI entrypoint for the stock prediction framework."""

import os
# Must be set before any OpenMP-linked library (torch, sklearn, xgboost) is imported.
# On macOS, PyTorch and Homebrew-installed libs each ship their own libomp.dylib;
# loading both causes an OpenMP barrier deadlock (SIGSEGV in __kmp_fork_barrier).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import concurrent.futures
import logging
import subprocess
import sys
from pathlib import Path

import click
import yaml

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_tickers(value: str | None) -> list[str]:
    """Parse a comma-separated ticker string into an upper-cased list."""
    if not value:
        return []
    return [t.strip().upper() for t in value.split(",") if t.strip()]


@click.group()
def cli():
    """Stock Market Prediction & Evaluation Framework."""


# ── run ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True),
              help="Path to YAML experiment config.")
@click.option("--ticker", default=None,
              help="Override data.ticker in the config (e.g. TSLA). "
                   "Applies ticker-specific overrides from configs/tickers/<TICKER>/ if present.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Validate config and data without fitting the model.")
def run(config: str, ticker: str, dry_run: bool):
    """Run a single experiment from a YAML config."""
    from experiments.runner import run_experiment
    run_experiment(config, dry_run=dry_run, ticker_override=ticker)


# ── run-all ───────────────────────────────────────────────────────────────────

@cli.command("run-all")
@click.option("--ticker", default=None, show_default=True,
              help="Run only configs whose data.ticker matches this symbol. "
                   "Cannot be combined with --tickers.")
@click.option("--tickers", default=None,
              help="Comma-separated list of tickers (e.g. AAPL,TSLA,MSFT). "
                   "Runs ALL configs for each ticker, injecting the ticker at runtime "
                   "and applying per-ticker overrides from configs/tickers/<TICKER>/.")
@click.option("--configs-dir", default="configs", show_default=True, type=click.Path(),
              help="Directory to scan for YAML configs.")
@click.option("--workers", default=1, show_default=True, type=int,
              help="Number of experiments to run in parallel.")
@click.option("--compare/--no-compare", "run_compare", default=True, show_default=True,
              help="Print a comparison table after all runs complete.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Pass --dry-run to every experiment (validate only, no fitting).")
def run_all(ticker: str, tickers: str, configs_dir: str, workers: int,
            run_compare: bool, dry_run: bool):
    """Run all configs in configs/, optionally across multiple tickers.

    Single ticker (legacy):   run-all --ticker AAPL
    Multi-ticker:             run-all --tickers AAPL,TSLA,MSFT --workers 6
    """
    if ticker and tickers:
        click.echo("Error: use --ticker OR --tickers, not both.")
        return

    root = Path(configs_dir)
    config_files = sorted(root.glob("*.yaml"))

    # Build the list of (config_path, ticker_override) jobs
    jobs: list[tuple[Path, str | None]] = []

    if tickers:
        ticker_list = _parse_tickers(tickers)
        if not ticker_list:
            click.echo("No valid tickers provided.")
            return
        # All configs × all tickers (with runtime injection)
        for t in ticker_list:
            for f in config_files:
                jobs.append((f, t))
        # Also pick up standalone ticker-only configs from configs/tickers/<T>/
        for t in ticker_list:
            ticker_dir = root / "tickers" / t
            if ticker_dir.is_dir():
                base_names = {f.name for f in config_files}
                for f in sorted(ticker_dir.glob("*.yaml")):
                    if f.name not in base_names:
                        # Standalone config — runs only for this ticker
                        jobs.append((f, t))

    elif ticker:
        # Legacy single-ticker filter: only configs that hardcode this ticker
        for f in config_files:
            try:
                cfg = yaml.safe_load(f.read_text())
                if cfg.get("data", {}).get("ticker", "").upper() == ticker.upper():
                    jobs.append((f, None))
            except Exception:
                pass
    else:
        jobs = [(f, None) for f in config_files]

    if not jobs:
        click.echo("No matching configs found.")
        return

    total = len(jobs)
    ticker_label = f" across {len(_parse_tickers(tickers))} ticker(s)" if tickers else ""
    click.echo(f"\nFound {total} experiment(s){ticker_label}. Running with {workers} worker(s)...\n")

    completed: list[Path] = []
    failed: list[str] = []

    def _run_one(job: tuple[Path, str | None]) -> tuple[str, int, str, str]:
        config_path, ticker_override = job
        cmd = [sys.executable, str(Path(__file__)), "run",
               "--config", str(config_path)]
        if ticker_override:
            cmd += ["--ticker", ticker_override]
        if dry_run:
            cmd.append("--dry-run")
        result = subprocess.run(cmd, capture_output=True, text=True)
        saved_to = ""
        for line in result.stdout.splitlines():
            if "Results saved to:" in line:
                saved_to = line.split("Results saved to:")[-1].strip()
                break
        label = f"{config_path.stem}" + (f" ({ticker_override})" if ticker_override else "")
        return label, result.returncode, saved_to, result.stderr

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            label, code, saved_to, stderr = future.result()
            done_count += 1
            prefix = f"[{done_count}/{total}]"
            if code == 0:
                click.echo(f"  {prefix} OK    {label}")
                if saved_to:
                    completed.append(Path(saved_to))
            else:
                err_line = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
                click.echo(f"  {prefix} FAIL  {label}  —  {err_line}")
                failed.append(label)

    click.echo(f"\n{len(completed)} succeeded, {len(failed)} failed.")

    if failed:
        click.echo("\nFailed experiments:")
        for name in failed:
            click.echo(f"  {name}")

    if run_compare and completed:
        click.echo("\n" + "=" * 60)
        click.echo("  Comparison")
        click.echo("=" * 60)
        if tickers:
            # Multi-ticker: show pivot table
            from evaluation.reporter import compare_multi_ticker
            ticker_list = _parse_tickers(tickers)
            df = compare_multi_ticker(ticker_list, Path("results").resolve())
            if not df.empty:
                click.echo("\n  Directional Accuracy by Model × Ticker\n")
                click.echo(df.to_string())
        else:
            # v2 path: results/<T>/<M>/h<H>/<target>/<exp> → root is 5 parents up
            save_root = completed[0].parent.parent.parent.parent.parent if completed else None
            _print_comparison(completed, save_dir=save_root)


# ── compare ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("results", nargs=-1, type=click.Path(exists=True))
@click.option("--ticker", default=None,
              help="Auto-compare latest run of every model for one ticker.")
@click.option("--tickers", default=None,
              help="Comma-separated tickers for a cross-ticker pivot table "
                   "(e.g. AAPL,TSLA,MSFT).")
@click.option("--metric", default="directional_accuracy", show_default=True,
              help="Metric to use for the multi-ticker pivot table.")
@click.option("--save/--no-save", default=True, help="Save comparison.csv.")
def compare(results: tuple, ticker: str, tickers: str, metric: str, save: bool):
    """Compare experiment result directories.

    Single ticker:   compare --ticker AAPL
    Multi-ticker:    compare --tickers AAPL,TSLA,MSFT
    Explicit dirs:   compare results/AAPL/XGBoost/exp1 results/AAPL/ARIMA/exp2
    """
    if tickers:
        ticker_list = _parse_tickers(tickers)
        from evaluation.reporter import compare_multi_ticker
        df = compare_multi_ticker(ticker_list, Path("results").resolve(), metric=metric)
        if df.empty:
            click.echo("No results found for the specified tickers.")
            return
        click.echo(f"\n  {metric} — model × ticker\n")
        click.echo(df.to_string())
        if save:
            out = Path("results") / "multi_comparison.csv"
            df.to_csv(out)
            click.echo(f"\nComparison saved to {out}")
        return

    dirs: list[Path] = []
    if ticker:
        root = Path("results") / ticker.upper()
        if not root.exists():
            click.echo(f"No results found for ticker '{ticker.upper()}'.")
            return
        from evaluation.results_store import FileResultsStore
        store = FileResultsStore(Path("results"))
        records = store.list_experiments(ticker=ticker.upper())
        # latest per model
        seen: dict[str, Path] = {}
        for rec in records:
            seen[rec.model] = rec.path
        dirs = list(seen.values())
    elif results:
        dirs = [Path(r) for r in results]
    else:
        click.echo("Provide result directories, --ticker, or --tickers.")
        return

    _print_comparison(dirs, save_dir=dirs[0].parent.parent.parent if dirs else None, save=save)


def _print_comparison(dirs: list, save_dir: Path = None, save: bool = True) -> None:
    from evaluation.reporter import compare_results

    df = compare_results(dirs, label_fn=lambda d: d.parent.name)
    if df.empty:
        click.echo("No valid metrics.json files found.")
        return

    click.echo("\n" + df.to_string())

    if save and save_dir:
        out = Path(save_dir) / "comparison.csv"
        df.to_csv(out)
        click.echo(f"\nComparison saved to {out}")


# ── report ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker", default=None,
              help="Generate a single-ticker report (e.g. AAPL).")
@click.option("--tickers", default=None,
              help="Comma-separated tickers for a cross-ticker comparison report.")
@click.option("--output", default=None, type=click.Path(),
              help="Output HTML path.")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the report in your browser after generating.")
def report(ticker: str, tickers: str, output: str, open_browser: bool):
    """Generate an interactive HTML comparison report.

    Single ticker:   report --ticker AAPL
    Multi-ticker:    report --tickers AAPL,TSLA,MSFT
    """
    import webbrowser
    from evaluation.reporter import compare_results, compare_multi_ticker
    from evaluation.html_reporter import generate_html_report, compute_roi

    if tickers:
        ticker_list = _parse_tickers(tickers)
        _generate_multi_ticker_report(ticker_list, output, open_browser,
                                       compare_multi_ticker, compare_results,
                                       generate_html_report, compute_roi, webbrowser)
        return

    if not ticker:
        click.echo("Provide --ticker or --tickers.")
        return

    ticker = ticker.upper()
    root = Path("results") / ticker
    if not root.exists():
        click.echo(f"No results found for '{ticker}'. Run some experiments first.")
        return

    from evaluation.results_store import FileResultsStore
    store = FileResultsStore(Path("results"))
    records = store.list_experiments(ticker=ticker)
    seen: dict[str, Path] = {}
    for rec in records:
        seen[rec.model] = rec.path
    dirs = list(seen.values())

    if not dirs:
        click.echo("No experiment results found.")
        return

    df = compare_results(dirs, label_fn=lambda d: d.parts[-4] if len(d.parts) >= 4 else d.name)
    if df.empty:
        click.echo("No valid metrics.json files found.")
        return

    roi_data = {}
    for d in dirs:
        model_name = d.parts[-4] if len(d.parts) >= 4 else d.name
        roi = compute_roi(d / "predictions.csv")
        if roi:
            roi_data[model_name] = roi

    out_path = Path(output) if output else Path("results") / ticker / "report.html"
    generate_html_report(df, ticker=ticker, output_path=out_path, roi_data=roi_data)
    click.echo(f"Report saved to {out_path}")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())


def _generate_multi_ticker_report(ticker_list, output, open_browser,
                                   compare_multi_ticker, compare_results,
                                   generate_html_report, compute_roi, webbrowser):
    """Build one HTML file per ticker, plus a summary pivot saved as CSV."""
    import os
    results_root = Path("results").resolve()

    pivot = compare_multi_ticker(ticker_list, results_root)
    if not pivot.empty:
        pivot_path = results_root / "multi_comparison.csv"
        pivot.to_csv(pivot_path)
        click.echo(f"\nCross-ticker pivot (directional_accuracy):\n")
        click.echo(pivot.to_string())
        click.echo(f"\nPivot saved to {pivot_path}")

    from evaluation.results_store import FileResultsStore as _FRS
    _store = _FRS(results_root)
    last_report = None
    for ticker in ticker_list:
        ticker = ticker.upper()
        root = results_root / ticker
        if not root.exists():
            click.echo(f"  Skipping {ticker} — no results found.")
            continue

        records = _store.list_experiments(ticker=ticker)
        seen: dict[str, Path] = {}
        for rec in records:
            seen[rec.model] = rec.path
        dirs = list(seen.values())

        if not dirs:
            continue

        df = compare_results(dirs, label_fn=lambda d: d.parent.name)
        if df.empty:
            continue

        roi_data = {}
        for d in dirs:
            roi = compute_roi(d / "predictions.csv")
            if roi:
                roi_data[d.parent.name] = roi

        out_path = Path(output) if output else results_root / ticker / "report.html"
        generate_html_report(df, ticker=ticker, output_path=out_path, roi_data=roi_data)
        click.echo(f"  {ticker}: report saved to {out_path}")
        last_report = out_path

    if open_browser and last_report:
        import webbrowser as wb
        wb.open(last_report.resolve().as_uri())


# ── list-models ───────────────────────────────────────────────────────────────

@cli.command("list-models")
def list_models():
    """Print all registered models and their configurable parameters."""
    import inspect
    from models import list_models as _list_models

    click.echo("\nRegistered models:")
    for name, cls in _list_models().items():
        try:
            sig = inspect.signature(cls.__init__)
            params = {
                k: v.default
                for k, v in sig.parameters.items()
                if k not in ("self", "kwargs") and v.default is not inspect.Parameter.empty
            }
        except Exception:
            params = {}
        click.echo(f"  {name}")
        for p, default in params.items():
            click.echo(f"    {p}: {default}")
    click.echo()


# ── list-results ──────────────────────────────────────────────────────────────

@cli.command("list-results")
@click.option("--results-dir", default="results", show_default=True,
              type=click.Path(), help="Root results directory.")
@click.option("--ticker", default=None, help="Filter to a specific ticker.")
@click.option("--tickers", default=None, help="Comma-separated tickers to include.")
@click.option("--latest/--all", "latest_only", default=True, show_default=True,
              help="Show only the latest run per model, or all runs.")
def list_results(results_dir: str, ticker: str, tickers: str, latest_only: bool):
    """Print a summary table of saved results."""
    from evaluation.reporter import compare_results
    from evaluation.results_store import FileResultsStore

    root = Path(results_dir)
    if not root.exists():
        click.echo(f"Results directory '{root}' does not exist.")
        return

    store = FileResultsStore(root)
    filter_ticker = ticker.upper() if ticker else (None if not tickers else None)
    if tickers:
        ticker_list = _parse_tickers(tickers)
        records = []
        for t in ticker_list:
            records.extend(store.list_experiments(ticker=t))
    elif ticker:
        records = store.list_experiments(ticker=ticker.upper())
    else:
        records = store.list_experiments()

    if not records:
        click.echo("No experiment results found.")
        return

    if latest_only:
        seen: dict[tuple, object] = {}
        for rec in records:
            key = (rec.ticker, rec.model, rec.horizon, rec.target)
            seen[key] = rec
        records = list(seen.values())

    dirs = [rec.path for rec in records]

    def _label(d: Path) -> str:
        # Try to build a meaningful label from the path
        parts = d.parts
        return "/".join(parts[-4:]) if len(parts) >= 4 else d.name

    df = compare_results(dirs, label_fn=_label)
    if df.empty:
        click.echo("No valid metrics.json files found.")
        return

    click.echo("\n" + df.to_string() + "\n")


# ── ensemble ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ticker", required=True,
              help="Ticker whose latest model predictions to ensemble (e.g. AAPL).")
@click.option("--metric", default="directional_accuracy", show_default=True,
              help="Metric printed in the summary.")
def ensemble(ticker: str, metric: str):
    """Build a simple forecast combination (equal-weight average) across all models.

    Loads the latest predictions.csv from every model for the given ticker,
    aligns on shared dates, averages predictions, then saves the result to
    results/<TICKER>/Ensemble/.

    Research basis: simple forecast combinations consistently outperform individual
    models in out-of-sample tests (rpaper_1, rpaper_2, rpaper_8).
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime as _dt
    from evaluation.metrics import compute_metrics
    from evaluation.reporter import save_results

    ticker = ticker.upper()
    root = Path("results") / ticker
    if not root.exists():
        click.echo(f"No results found for '{ticker}'. Run experiments first.")
        return

    # Collect latest predictions.csv per model using FileResultsStore
    from evaluation.results_store import FileResultsStore as _FRS2
    _store2 = _FRS2(Path("results"))
    records = [r for r in _store2.list_experiments(ticker=ticker) if r.model != "Ensemble"]
    seen_models: dict[str, Path] = {}
    for rec in records:
        seen_models[rec.model] = rec.path

    pred_frames: dict[str, pd.DataFrame] = {}
    for model_name, exp_path in seen_models.items():
        preds_path = exp_path / "predictions.csv"
        if not preds_path.exists():
            continue
        frame = pd.read_csv(preds_path, parse_dates=["date"])
        frame = frame.set_index("date").sort_index()
        pred_frames[model_name] = frame

    if len(pred_frames) < 2:
        click.echo(f"Need at least 2 models with results for '{ticker}'. "
                   f"Found: {list(pred_frames)}")
        return

    # Align on common dates
    common_dates = None
    for frame in pred_frames.values():
        if common_dates is None:
            common_dates = frame.index
        else:
            common_dates = common_dates.intersection(frame.index)

    if len(common_dates) == 0:
        click.echo("No common dates across model predictions. Cannot ensemble.")
        return

    y_true = pred_frames[list(pred_frames)[0]].loc[common_dates, "y_true"].values.astype("float32")
    y_pred = np.mean(
        np.stack([f.loc[common_dates, "y_pred"].values for f in pred_frames.values()]),
        axis=0,
    ).astype("float32")

    metric_names = ["rmse", "mae", "r2", "directional_accuracy", "sharpe",
                    "rank_ic", "max_drawdown", "calmar_ratio"]
    metrics = compute_metrics(y_true, y_pred, metric_names)
    metrics["n_models"] = len(pred_frames)

    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"Ensemble_{ticker}_{timestamp}"
    results_dir = Path("results").resolve() / ticker / "Ensemble" / "h1" / "next_return" / exp_id

    save_results(
        results_dir=results_dir,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        dates=pd.DatetimeIndex(common_dates),
        config={"ensemble": {"ticker": ticker, "models": sorted(pred_frames.keys())}},
        feature_names=[],
        ticker=ticker,
        model="Ensemble",
        horizon=1,
        target="next_return",
    )

    click.echo(f"\n  Ensemble ({len(pred_frames)} models) — {ticker}")
    click.echo(f"  Models: {', '.join(sorted(pred_frames.keys()))}")
    click.echo(f"  {metric}: {metrics.get(metric, float('nan')):.4f}")
    click.echo(f"  directional_accuracy: {metrics.get('directional_accuracy', float('nan')):.4f}")
    click.echo(f"  sharpe:               {metrics.get('sharpe', float('nan')):.4f}")
    click.echo(f"  Results saved to: {results_dir}\n")


# ── migrate ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--results-dir", default="results", show_default=True,
              type=click.Path(), help="Root results directory to migrate.")
def migrate(results_dir: str):
    """Migrate v1 result dirs to v2 path layout.

    Moves results/<T>/<M>/<exp_id>/ → results/<T>/<M>/h1/next_return/<exp_id>/
    and adds schema_version:1 to their metrics.json.
    Skips dirs that already have schema_version.
    """
    import shutil
    import json as _json

    root = Path(results_dir)
    if not root.exists():
        click.echo(f"Results directory '{root}' does not exist.")
        return

    moved = 0
    skipped = 0

    for ticker_dir in sorted(root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for model_dir in sorted(ticker_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for child in sorted(model_dir.iterdir()):
                if not child.is_dir():
                    continue
                # v2 pattern: child name starts with "h" followed by digit
                if child.name.startswith("h") and child.name[1:].isdigit():
                    continue
                # v1 pattern: exp_id dir directly under model_dir
                metrics_file = child / "metrics.json"
                if not metrics_file.exists():
                    continue
                try:
                    data = _json.loads(metrics_file.read_text())
                except Exception:
                    continue
                if "schema_version" in data:
                    skipped += 1
                    continue
                # Move to v2 path
                dest = model_dir / "h1" / "next_return" / child.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(child), str(dest))
                # Add schema_version:1
                data["schema_version"] = 1
                (dest / "metrics.json").write_text(
                    _json.dumps(data, indent=2)
                )
                click.echo(f"  Moved: {child} → {dest}")
                moved += 1

    click.echo(f"\nMigration complete: {moved} moved, {skipped} already versioned.")


# ── experiments group ─────────────────────────────────────────────────────────

@cli.group()
def experiments():
    """Experiment management commands."""


@experiments.command("tune")
@click.option("--ticker", required=True)
@click.option("--model", required=True)
@click.option("--n-trials", default=50, type=int)
@click.option("--config", default=None, type=click.Path(exists=True))
@click.option("--resume", is_flag=True, default=False)
def tune(ticker: str, model: str, n_trials: int, config: str, resume: bool):
    """Hyperparameter tune a model with Optuna (walk-forward OOS R² objective)."""
    try:
        import optuna
    except ImportError:
        click.echo("optuna not installed. Run: pip install optuna")
        return

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from models import get_model_class
    from features import build_features, walk_forward_splits, FeatureConfig
    from data import get_loader
    import yaml

    ModelClass = get_model_class(model)
    search_space = getattr(ModelClass, "search_space", None)
    if not search_space:
        click.echo(f"Model '{model}' has no search_space defined.")
        return

    # Load data once
    cfg_path = config or f"configs/{model.lower()}_{ticker.lower()}.yaml"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        # Use a generic config
        cfg = {
            "data": {"loader": "yfinance", "ticker": ticker.upper(),
                     "start": "2015-01-01", "end": "2024-01-01"},
            "features": {"target": "next_return", "lags": [1, 2, 3, 5, 10],
                         "rolling_windows": [5, 10, 20]},
            "evaluation": {"test_size": 0.2, "min_train_size": 500, "step_size": 21},
        }

    loader = get_loader(cfg["data"]["loader"])
    data_cfg = cfg["data"]
    if cfg["data"]["loader"] == "yfinance":
        df = loader.load(ticker=ticker.upper(), start=data_cfg["start"],
                         end=data_cfg["end"])
    else:
        df = loader.load(csv_path=data_cfg["csv_path"])

    feat_cfg = cfg.get("features", {})
    eval_cfg = cfg.get("evaluation", {})
    feat_config = FeatureConfig(
        target=feat_cfg.get("target", "next_return"),
        lags=feat_cfg.get("lags", [1, 2, 3, 5, 10]),
        rolling_windows=feat_cfg.get("rolling_windows", [5, 10, 20]),
        test_size=eval_cfg.get("test_size", 0.2),
        min_train_size=eval_cfg.get("min_train_size", 500),
        step_size=eval_cfg.get("step_size", 21),
    )
    pipeline = build_features(df, feat_config)
    X_raw = pipeline.X_full_raw
    y_full = pipeline.y_full
    dates_full = pipeline.dates_full
    min_train = feat_config.min_train_size
    step = feat_config.step_size

    def objective(trial):
        params = {}
        for pname, spec in search_space.items():
            ptype = spec["type"]
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, spec["low"], spec["high"])
            elif ptype == "float":
                params[pname] = trial.suggest_float(
                    pname, spec["low"], spec["high"], log=spec.get("log", False))
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec["choices"])
        params["random_state"] = 42

        splits = walk_forward_splits(X_raw, y_full, dates_full,
                                     min_train_size=min_train, step_size=step)
        all_y_true, all_y_pred, all_y_train = [], [], []
        for X_tr_raw, y_tr, X_te_raw, y_te, _ in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)
            m = ModelClass(**params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            mn = min(len(y_te), len(preds))
            all_y_true.extend(y_te[-mn:])
            all_y_pred.extend(preds[-mn:])
            all_y_train.extend(y_tr)

        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)
        y_train_arr = np.array(all_y_train)
        if len(y_true_arr) == 0:
            return float("-inf")
        benchmark = float(np.mean(y_train_arr))
        ss_pred = float(np.sum((y_true_arr - y_pred_arr) ** 2))
        ss_bench = float(np.sum((y_true_arr - benchmark) ** 2))
        if ss_bench == 0:
            return float("-inf")
        return float(1.0 - ss_pred / ss_bench)

    db_dir = Path("results") / ticker.upper()
    db_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_dir}/optuna_{model}.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model}_{ticker}",
        storage=storage,
        load_if_exists=resume,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    click.echo(f"\nBest OOS R²: {study.best_value:.4f}")
    click.echo(f"Best params: {best}")

    out_dir = Path("configs") / "tickers" / ticker.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use model's base config name pattern
    out_file = out_dir / f"{model.lower()}.yaml"
    import yaml as _yaml
    with open(out_file, "w") as f:
        _yaml.dump({"model": {"params": best}}, f, default_flow_style=False)
    click.echo(f"Best params written to {out_file}")


@experiments.command("horizon-sweep")
@click.option("--ticker", required=True)
@click.option("--model", required=True)
@click.option("--config", default=None, type=click.Path(exists=True))
def horizon_sweep(ticker: str, model: str, config: str):
    """Run the model at h=1, h=5, h=21 and print a comparison table."""
    import yaml
    import pandas as pd
    from experiments.runner import run_experiment, _load_config

    horizons = [1, 5, 21]
    rows = []
    for h in horizons:
        click.echo(f"\nRunning h={h}...")
        # Build a temporary config dict with horizon override
        if config:
            with open(config) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {
                "experiment": {"name": f"horizon_sweep_{model}_{ticker}_h{h}"},
                "data": {"loader": "yfinance", "ticker": ticker.upper(),
                         "start": "2015-01-01", "end": "2024-01-01"},
                "features": {"target": "next_return", "lags": [1, 2, 3, 5, 10],
                             "rolling_windows": [5, 10, 20], "horizon": h},
                "model": {"name": model, "params": {}},
                "evaluation": {"test_size": 0.2, "metrics": ["rmse", "r2", "directional_accuracy", "sharpe"]},
                "seed": 42,
            }
        cfg["features"]["horizon"] = h
        if "experiment" not in cfg:
            cfg["experiment"] = {}
        cfg["experiment"]["name"] = f"horizon_sweep_{model}_{ticker}_h{h}"

        import tempfile, json as _json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name
        try:
            results_dir = run_experiment(tmp_path, ticker_override=ticker.upper())
            import json
            m = json.loads((results_dir / "metrics.json").read_text())
            rows.append({
                "horizon": h,
                "rmse": m.get("rmse", float("nan")),
                "r2": m.get("r2", float("nan")),
                "directional_accuracy": m.get("directional_accuracy", float("nan")),
                "sharpe": m.get("sharpe", float("nan")),
            })
        except Exception as e:
            click.echo(f"  h={h} failed: {e}")
        finally:
            import os
            os.unlink(tmp_path)

    if rows:
        df = pd.DataFrame(rows).set_index("horizon")
        click.echo("\nHorizon Sweep Results:")
        click.echo(df.to_string())


# ── strategy group ────────────────────────────────────────────────────────────

@cli.group()
def strategy():
    """Portfolio strategy commands."""


@strategy.command("optimize")
@click.option("--tickers", required=True,
              help="Comma-separated list of tickers.")
@click.option("--criterion", default="sharpe", show_default=True,
              help="Metric to optimize model selection on.")
def optimize(tickers: str, criterion: str):
    """Select best model per ticker using ModelSelector and write strategy_recommendation.yaml."""
    import yaml
    from evaluation.results_store import FileResultsStore
    from strategy.model_selector import ModelSelector

    store = FileResultsStore(Path("results"))
    selector = ModelSelector(store, criterion=criterion)
    ticker_list = _parse_tickers(tickers)
    recommendations = {}
    for t in ticker_list:
        try:
            spec = selector.recommend(t)
            recommendations[t] = {
                "model": spec.model,
                "horizon": spec.horizon,
                "target": spec.target,
                criterion: spec.criterion_value,
                "exp_id": spec.exp_id,
            }
            click.echo(f"  {t}: {spec.model} (h={spec.horizon}, {criterion}={spec.criterion_value:.4f})")
        except Exception as e:
            click.echo(f"  {t}: error — {e}")

    out = Path("strategy_recommendation.yaml")
    with open(out, "w") as f:
        yaml.dump(recommendations, f, default_flow_style=False)
    click.echo(f"\nRecommendations written to {out}")


@strategy.command("portfolio")
@click.option("--tickers", required=True,
              help="Comma-separated list of tickers.")
@click.option("--allocator", default="signal_weighted", show_default=True)
@click.option("--long-only", is_flag=True, default=False)
def portfolio(tickers: str, allocator: str, long_only: bool):
    """Construct a portfolio from latest predictions across tickers."""
    from evaluation.results_store import FileResultsStore
    from strategy.portfolio import PortfolioConstructor

    store = FileResultsStore(Path("results"))
    constraints = {"long_only": long_only}
    constructor = PortfolioConstructor(store, allocator_name=allocator,
                                       constraints=constraints)
    ticker_list = _parse_tickers(tickers)
    result = constructor.build(ticker_list)
    click.echo("\nPortfolio Weights:")
    for t, w in sorted(result.weights.items()):
        click.echo(f"  {t}: {w:+.4f}")
    click.echo(f"\nPortfolio Sharpe: {result.metrics.get('portfolio_sharpe', float('nan')):.4f}")


@strategy.command("backtest")
@click.option("--ticker", required=True)
@click.option("--model", required=True)
@click.option("--cost-bps", default=10, type=float, show_default=True)
@click.option("--config", default=None, type=click.Path(exists=True))
def backtest(ticker: str, model: str, cost_bps: float, config: str):
    """Backtest a model's latest predictions with transaction costs."""
    import pandas as pd
    import numpy as np
    from evaluation.results_store import FileResultsStore
    from strategy.backtest import Backtester
    from strategy.cost_models.fixed_bps import FixedBpsCostModel

    store = FileResultsStore(Path("results"))
    records = store.list_experiments(ticker=ticker.upper(), model=model)
    if not records:
        click.echo(f"No results found for {ticker}/{model}.")
        return

    latest = records[-1]
    preds = store.load_predictions(latest.exp_id)
    preds = preds.sort_values("date")
    y_true = preds["y_true"].values.astype(np.float32)
    y_pred = preds["y_pred"].values.astype(np.float32)
    dates = pd.DatetimeIndex(preds["date"])

    cost_model = FixedBpsCostModel(bps=cost_bps)
    bt = Backtester(y_true=y_true, y_pred=y_pred, dates=dates, cost_model=cost_model)
    result = bt.run()

    out = latest.path / "equity_curve.csv"
    result.equity_curve.to_csv(out, index=False)

    click.echo(f"\nBacktest: {ticker}/{model}")
    click.echo(f"  Gross Sharpe:       {result.gross_sharpe:.4f}")
    click.echo(f"  Net Sharpe:         {result.net_sharpe:.4f}")
    click.echo(f"  Gross Max Drawdown: {result.gross_max_drawdown:.4f}")
    click.echo(f"  Net Max Drawdown:   {result.net_max_drawdown:.4f}")
    click.echo(f"  Turnover Rate:      {result.turnover_rate:.4f}")
    click.echo(f"  Equity curve saved to: {out}")


if __name__ == "__main__":
    cli()
