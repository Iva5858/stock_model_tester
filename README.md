# Stock Market Prediction & Evaluation Framework

A Python framework for registering, running, and comparing stock return prediction models. Supports 24 algorithms across 7 families, multi-horizon forecasting (1/5/21 days), classification and regression targets, walk-forward cross-validation, SHAP feature importance, macroeconomic feature enrichment (FRED), automated strategy optimization, portfolio construction, and realistic backtesting with transaction costs.

**v2 architecture:** Registry-everywhere, strict dependency DAG, versioned result schema, `ResultsStore` interface. Every extensible category (loaders, transforms, models, metrics, allocators, cost models) uses the same `@register_*` decorator pattern — adding a new component never requires changes to existing code.

---

## Setup

```bash
source .env/bin/activate
pip install -r requirements.txt
```

Set your FRED API key to enable macroeconomic features (optional — degrades gracefully if unset):

```bash
export FRED_API_KEY=your_key_here
```

---

## Quick Start

```bash
# Run all models for a single ticker
python run.py run-all --ticker AAPL --workers 4

# Run a single experiment
python run.py run --config configs/xgboost_aapl.yaml

# Walk-forward model selection → writes strategy_recommendation.yaml
python run.py strategy optimize --tickers AAPL,MSFT,TSLA

# Build a portfolio from latest predictions
python run.py strategy portfolio --tickers AAPL,MSFT,TSLA --allocator max_sharpe

# Backtest a model with transaction costs
python run.py strategy backtest --ticker AAPL --model XGBoost --cost-bps 10

# Tune hyperparameters with Optuna (walk-forward OOS R² objective)
python run.py experiments tune --ticker AAPL --model XGBoost --n-trials 50

# Evaluate a model at horizons 1, 5, and 21 days
python run.py experiments horizon-sweep --ticker AAPL --model XGBoost

# Migrate v1 results to v2 path format
python run.py migrate
```

---

## CLI Reference

### v1 Commands (unchanged)

| Command | Description |
|---------|-------------|
| `run --config <path>` | Run a single experiment from YAML |
| `run --config <path> --ticker TSLA` | Override the ticker at runtime |
| `run --config <path> --dry-run` | Validate config and data, skip fitting |
| `run-all --ticker AAPL` | Run all configs for one ticker |
| `run-all --tickers AAPL,TSLA,MSFT` | Run all configs × all tickers in parallel |
| `run-all --workers 4` | Parallel workers |
| `ensemble --ticker AAPL` | Equal-weight forecast combination across all models |
| `report --ticker AAPL` | Generate interactive HTML dashboard |
| `report --tickers AAPL,TSLA,MSFT` | Multi-ticker reports |
| `compare --ticker AAPL` | Compare latest run per model |
| `compare --tickers AAPL,TSLA,MSFT --metric sharpe` | Cross-ticker pivot table |
| `list-results` | Table of all saved results |
| `list-models` | Print all registered models |

### v2 — Experiment Commands

| Command | Description |
|---------|-------------|
| `experiments tune --ticker AAPL --model XGBoost --n-trials 50` | Optuna hyperparameter search (walk-forward OOS R² objective). Best params saved to `configs/tickers/<TICKER>/<model>.yaml`. Study persisted to SQLite; resume with `--resume`. |
| `experiments horizon-sweep --ticker AAPL --model XGBoost` | Run model at h=1, h=5, h=21. Prints comparison table; results saved under `h1/`, `h5/`, `h21/` directories. |

### v2 — Strategy Commands

| Command | Description |
|---------|-------------|
| `strategy optimize --tickers AAPL,TSLA,MSFT` | Walk-forward model selection per ticker. Writes `results/<TICKER>/strategy_recommendation.yaml`. |
| `strategy portfolio --tickers AAPL,TSLA,MSFT --allocator signal_weighted` | Construct portfolio from latest predictions using a registered allocator. |
| `strategy backtest --ticker AAPL --model XGBoost --cost-bps 10` | Realistic backtest with transaction costs. Saves `equity_curve.csv` with gross/net/benchmark values. |

### v2 — Maintenance

| Command | Description |
|---------|-------------|
| `migrate` | Move v1 result directories (`results/<T>/<M>/<exp_id>/`) to v2 format (`results/<T>/<M>/h1/next_return/<exp_id>/`) and stamp `schema_version: 1`. Non-destructive; skips already-migrated dirs. |

---

## Results Structure (v2)

```
results/
├── AAPL/
│   ├── report.html
│   ├── comparison.csv
│   ├── strategy_recommendation.yaml      ← written by strategy optimize
│   ├── optuna_XGBoost.db                 ← Optuna study (SQLite)
│   ├── XGBoost/
│   │   ├── h1/
│   │   │   └── next_return/
│   │   │       └── XGBoost_AAPL_20260511_120000/
│   │   │           ├── metrics.json          (schema_version: 2)
│   │   │           ├── predictions.csv
│   │   │           ├── config_snapshot.yaml
│   │   │           ├── equity_curve.csv      (backtest mode)
│   │   │           ├── shap_values.csv       (tree models)
│   │   │           └── feature_importance.csv
│   │   ├── h5/next_return/...
│   │   └── h21/next_return/...
│   └── XGBoostClassifier/
│       └── h1/direction/...
└── TSLA/
    └── ...
```

The `metrics.json` schema (v2):
```json
{
  "schema_version": 2,
  "ticker": "AAPL",
  "model": "XGBoost",
  "horizon": 5,
  "target": "next_return",
  "cv_method": "expanding",
  "rmse": 0.0089,
  "directional_accuracy": 0.542,
  "sharpe": 0.71,
  "oos_r2": 0.012,
  "n_train": 3800,
  "n_test": 950,
  "elapsed_s": 4.2
}
```

---

## Available Models (24)

### Baseline (3)
| Name | Description |
|------|-------------|
| `NaiveLastValue` | Predicts next return = last observed return |
| `RollingMean` | Predicts next return = rolling mean of last N returns |
| `HistoricalMean` | Full training set mean — the Goyal-Welch (2008) prevailing-mean OOS benchmark |

### Linear / Regularised (3)
| Name | Description |
|------|-------------|
| `Ridge` | Ridge regression (L2) |
| `Lasso` | Lasso regression (L1, implicit feature selection) |
| `ElasticNet` | Combined L1 + L2 |

### Tree-Based ML — Regression (5)
| Name | Description |
|------|-------------|
| `XGBoost` | Gradient boosted trees |
| `RandomForest` | Random forest |
| `SVR` | Support vector regression |
| `LightGBM` | Faster gradient boosting, leaf-wise splits |
| `CatBoost` | Gradient boosting with native categorical support |

### Tree-Based ML — Classification (4) — NEW in v2
| Name | Description |
|------|-------------|
| `LogisticRegressionClassifier` | Logistic regression; `predict()` → `{+1,-1}`, `predict_proba()` → `P(up)` |
| `XGBoostClassifier` | XGBoost classifier |
| `RandomForestClassifier` | Random forest classifier |
| `LightGBMClassifier` | LightGBM classifier |

All classifiers use `task = "classification"` and automatically select classification metrics when run.

### Classical Time Series (6)
| Name | Description |
|------|-------------|
| `ARIMA` | AutoRegressive Integrated Moving Average |
| `SARIMA` | Seasonal ARIMA (weekly seasonality via `m=5`) |
| `ETS` | Exponential smoothing |
| `GARCH` | ARMA-GARCH conditional mean forecast |
| `MonteCarlo` | GBM Monte Carlo — mean of N simulated return paths |
| `OrnsteinUhlenbeck` | Mean-reverting OU process |

### Regime Models (2)
| Name | Description |
|------|-------------|
| `HMM` | Gaussian Hidden Markov Model (3 regimes by default) |
| `MarkovSwitching` | Hamilton's Markov-Switching AR(1) |

### Ensemble (1)
| Name | Description |
|------|-------------|
| `Ensemble` | Equal-weight average of all model predictions. Run via `ensemble --ticker AAPL`. Consistently outperforms individual models OOS (rpaper_1, rpaper_2, rpaper_8). |

> **Note:** PyTorch-based models (LSTM, Transformer, TCN) are disabled due to an OpenMP conflict on macOS ARM. Classes remain in `dl_models.py` and can be re-enabled on a CUDA Linux system.

---

## Config Schema (v2)

```yaml
schema_version: 2            # optional — omit for v1 compatibility

experiment:
  name: xgboost_aapl_macro_h5
  description: "XGBoost on AAPL, 5-day horizon, with macro features"

data:
  loader: yfinance            # yfinance | csv | fred
  ticker: AAPL
  start: "2004-01-01"
  end: "2024-01-01"

features:
  target: next_return         # next_return | next_close | direction (NEW)
  horizon: 5                  # NEW: 1 | 5 | 21 (default: 1)
  lags: [1, 2, 3, 5, 10]     # v1 flag-based features (backward-compatible)
  rolling_windows: [5, 10, 20]
  technical_indicators: true
  # v2 transform list (opt-in, overrides flag-based features when present):
  transforms:
    - lag_returns:   {lags: [1, 2, 3, 5, 10]}
    - rolling_stats: {windows: [5, 10, 20]}
    - volume_delta:  {}
    - technical:     {}
    - macro_fred:    {}       # requires FRED_API_KEY env var

model:
  name: XGBoost
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05

evaluation:
  cv_method: expanding        # holdout | expanding | rolling
  test_size: 0.2              # holdout only
  step_size: 21               # walk-forward refit frequency (trading days)
  min_train_size: 500
  metrics: [rmse, mae, r2, directional_accuracy, sharpe, oos_r2, rank_ic,
            max_drawdown, calmar_ratio]

seed: 42
```

**Backward compatibility:** v1 configs (no `schema_version`) run unchanged. The v2 `transforms` list and `horizon` field are opt-in. No existing config needs updating.

### `direction` target

For classification models, set `target: direction`. The target is `sign(close[t+h] − close[t])` encoded as `{+1, -1}`. Use with the `*Classifier` model names and `cv_method: expanding`.

---

## Metrics

### Regression Metrics

| Metric | Plain English | Direction | Research Basis |
|--------|--------------|-----------|----------------|
| `rmse` | Average prediction error | Lower | Standard |
| `mae` | Typical prediction error | Lower | Standard |
| `r2` | Predictive power | Higher | Standard |
| `directional_accuracy` | % of days the model correctly called up vs. down | Higher | Standard |
| `sharpe` | Risk-adjusted return of long/short strategy | Higher | Standard |
| `oos_r2` | Improvement in MSE over historical mean (Campbell-Thompson 2008) | Higher | fpaper_2, fpaper_3 |
| `rank_ic` | Spearman rank correlation between predictions and actuals | Higher | rpaper_5, rpaper_8 |
| `max_drawdown` | Worst peak-to-trough loss of long/short strategy | Closer to 0 | rpaper_5 |
| `calmar_ratio` | Annualised return ÷ absolute max drawdown | Higher | rpaper_5 |

### Classification Metrics (v2 — auto-selected for `*Classifier` models)

| Metric | Plain English | Direction |
|--------|--------------|-----------|
| `auc_roc` | Area under ROC curve | Higher |
| `log_loss` | Cross-entropy of predicted probabilities | Lower |
| `brier_score` | MSE of predicted probabilities vs. binary outcomes | Lower |
| `precision_up` | Precision of the up (+1) class | Higher |
| `recall_up` | Recall of the up (+1) class | Higher |
| `f1_up` | Harmonic mean of precision and recall (up class) | Higher |

### Portfolio & Backtest Metrics (v2 — `strategy backtest`)

| Metric | Description |
|--------|-------------|
| `gross_sharpe` / `net_sharpe` | Sharpe before / after transaction costs |
| `gross_calmar` / `net_calmar` | Calmar before / after costs |
| `gross_max_drawdown` / `net_max_drawdown` | Max drawdown before / after costs |
| `turnover_rate` | Fraction of days with a position sign change |

---

## Strategy Layer (v2)

### Model Selection

`ModelSelector` reads walk-forward results through the `ResultsStore` interface and selects the best `(model, horizon)` per ticker by a configurable criterion (default: `sharpe`). Selection is strictly OOS-safe.

```bash
python run.py strategy optimize --tickers AAPL,MSFT --criterion sharpe
```

Writes `results/<TICKER>/strategy_recommendation.yaml` — a fully-specified experiment config that can be passed directly to `run --config`.

### Portfolio Construction

Four registered allocators:

| Allocator | Description |
|-----------|-------------|
| `equal` | Equal weight across all tickers with non-zero signal |
| `signal_weighted` | Weight proportional to signal magnitude |
| `min_variance` | Minimum variance via scipy + Ledoit-Wolf shrinkage |
| `max_sharpe` | Maximum Sharpe tangency portfolio |

Default constraints: `max_position_weight=0.40`, `long_only=false` (short selling allowed).

### Backtesting

`Backtester` accepts arrays (not file paths). The CLI loads `predictions.csv` and passes arrays in. A trade occurs when `sign(position)` changes. Cost is applied only on trade days.

```bash
python run.py strategy backtest --ticker AAPL --model XGBoost --cost-bps 10
```

Saves `equity_curve.csv` with `date`, `gross_value`, `net_value`, `benchmark_value` columns.

---

## FRED Macro Features (v2)

Add `macro_fred: {}` to the transforms list to include macroeconomic predictors:

- 10-year Treasury yield (`DGS10`)
- 3-month T-bill rate (`TB3MS`)
- Term spread (10y − 3mo)
- BAA corporate yield (`DBAA`)
- AAA corporate yield (`DAAA`)
- Default spread (BAA − AAA)
- CPI inflation (`CPIAUCSL`)

Requires `FRED_API_KEY` env variable. Responses are cached in `data/cache/fred/` with a 24-hour TTL. Publication lags are enforced per series. If FRED is unavailable, the transform logs a warning and continues without macro columns — the pipeline never fails.

---

## Hyperparameter Tuning (v2 — Optuna)

```bash
python run.py experiments tune --ticker AAPL --model XGBoost --n-trials 50
python run.py experiments tune --ticker AAPL --model XGBoost --n-trials 100 --resume
```

- Objective: walk-forward OOS R² (data loaded once; search runs over pre-built arrays)
- Study persisted to `results/<TICKER>/optuna_XGBoost.db` (SQLite) — resumable
- Best parameters written to `configs/tickers/<TICKER>/<model>.yaml`

Supported models: `XGBoost`, `LightGBM`, `CatBoost`, `RandomForest`, `Ridge`, `Lasso`, `ElasticNet`.

---

## Walk-Forward Evaluation

```yaml
evaluation:
  cv_method: expanding   # train grows from start; refit every step_size days
  # cv_method: rolling   # fixed-length sliding window
  step_size: 21
  min_train_size: 500
  window_size: 1000      # rolling only
```

Each fold re-fits the scaler on train data only. Predictions are aggregated across all folds before metrics are computed. The `step_size` defaults to `max(horizon, 5)` when `horizon > 1` to avoid overlapping test windows.

---

## Adding New Components

### New model
```python
from models.base_model import BaseModel, register

@register("MyModel")
class MyModel(BaseModel):
    name = "MyModel"
    task = "regression"   # or "classification"
    search_space = {"alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True}}

    def fit(self, X_train, y_train): ...
    def predict(self, X_test): ...
    def get_params(self): return {}
```

Import it in `models/__init__.py`. It is automatically picked up by `run-all`, `list-models`, and `experiments tune`.

### New data loader
```python
from data.base_loader import BaseLoader, register_loader

@register_loader("myapi")
class MyAPILoader(BaseLoader):
    def load(self, **kwargs) -> pd.DataFrame: ...
```

### New feature transform
```python
from features.base_transform import FeatureTransform, register_transform

@register_transform("my_transform")
class MyTransform(FeatureTransform):
    task = "both"
    def fit_transform(self, df, feat, is_train): ...
```

### New allocator
```python
from strategy.allocators.base_allocator import BaseAllocator, register_allocator

@register_allocator("my_allocator")
class MyAllocator(BaseAllocator):
    def allocate(self, signals, cov_matrix, constraints): ...
```

---

## Ticker-Specific Hyperparameter Overrides

```
configs/tickers/<TICKER>/<config_filename>.yaml
```

The override is deep-merged on top of the base config. Only specify the keys that differ.

```yaml
# configs/tickers/TSLA/xgboost_aapl.yaml
model:
  params:
    n_estimators: 300
    max_depth: 8
features:
  lags: [1, 2, 3, 5, 10, 20]
```

---

## Architecture

Strict dependency DAG — no layer imports from a layer above it:

```
run.py (CLI — only place all layers converge)
    │
    ├── strategy/      ← reads evaluation/ only; never touches data/ or features/
    ├── experiments/   ← imports data/, features/, models/, evaluation/
    ├── evaluation/    ← pure math; no project imports
    ├── models/        ← pure ML; no project imports
    ├── features/      ← imports data/ only
    └── data/          ← no project imports
```

The `strategy/` layer reads all experiment results through the `ResultsStore` interface (`FileResultsStore` in v2; swappable to `SQLiteResultsStore` in v3 with no strategy code changes).

---

## Running Tests

```bash
.env/bin/python -m pytest tests/ -v
```

89 tests — all pass.

| Test file | Coverage |
|-----------|----------|
| `test_baseline.py` | Baseline model predict/fit |
| `test_metrics.py` | All 15 metrics including classification |
| `test_feature_pipeline.py` | Feature shapes, no-leakage, dates |
| `test_integration.py` | Full pipeline smoke test (4 models) |
| `test_config.py` | Config validation |
| `test_cli.py` | CLI command smoke tests |
| `test_registry.py` | Registry completeness + DAG enforcement |
| `test_classification.py` | Direction target, classification metrics, classifier predict/proba |
| `test_results_store.py` | v1/v2 path detection, filtering, load_predictions/load_metrics |
