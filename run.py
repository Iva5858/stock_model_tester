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
            _print_comparison(completed,
                              save_dir=completed[0].parent.parent.parent if completed else None)


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
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            runs = sorted(d for d in model_dir.iterdir() if d.is_dir())
            if runs:
                dirs.append(runs[-1])
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

    dirs = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        runs = sorted(d for d in model_dir.iterdir() if d.is_dir())
        if runs:
            dirs.append(runs[-1])

    if not dirs:
        click.echo("No experiment results found.")
        return

    df = compare_results(dirs, label_fn=lambda d: d.parent.name)
    if df.empty:
        click.echo("No valid metrics.json files found.")
        return

    roi_data = {}
    for d in dirs:
        model_name = d.parent.name
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

    last_report = None
    for ticker in ticker_list:
        ticker = ticker.upper()
        root = results_root / ticker
        if not root.exists():
            click.echo(f"  Skipping {ticker} — no results found.")
            continue

        dirs = []
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            runs = sorted(d for d in model_dir.iterdir() if d.is_dir())
            if runs:
                dirs.append(runs[-1])

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

    root = Path(results_dir)
    if not root.exists():
        click.echo(f"Results directory '{root}' does not exist.")
        return

    if tickers:
        ticker_dirs = [root / t for t in _parse_tickers(tickers)]
    elif ticker:
        ticker_dirs = [root / ticker.upper()]
    else:
        ticker_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]

    dirs = []
    for ticker_dir in ticker_dirs:
        if not ticker_dir.is_dir():
            continue
        for model_dir in sorted(ticker_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            runs = sorted(d for d in model_dir.iterdir() if d.is_dir())
            if not runs:
                continue
            dirs.extend([runs[-1]] if latest_only else runs)

    if not dirs:
        click.echo("No experiment results found.")
        return

    def _label(d: Path) -> str:
        return f"{d.parent.parent.name}/{d.parent.name}"

    df = compare_results(dirs, label_fn=_label)
    if df.empty:
        click.echo("No valid metrics.json files found.")
        return

    click.echo("\n" + df.to_string() + "\n")


if __name__ == "__main__":
    cli()
