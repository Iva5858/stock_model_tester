# Headline results — raw artifacts

Backing data for the [Results table in the top-level README](../../README.md#results). `results/` is
gitignored (it fills up fast with per-run experiment dirs, Optuna DBs, etc.), so these three
runs are copied here and tracked so the numbers in the README are checkable without rerunning
anything.

Each `<TICKER>/` holds:
- `metrics.json` — the model's own evaluation metrics (OOS R², directional accuracy, rank IC, etc.)
- `equity_curve.csv` — gross/net/benchmark equity curves from `strategy backtest --cost-bps 10`
- `config_snapshot.yaml` — exact model params and feature config used for the run

Model: XGBoost, horizon = 1 day, target = `next_return`, `cv_method: expanding` (walk-forward OOS).

To regenerate from scratch:

```bash
python run.py run-all --ticker AAPL --workers 4   # and MSFT, TSLA
python run.py strategy backtest --ticker AAPL --model XGBoost --cost-bps 10
```
