# Stock Market Prediction & Evaluation Framework

A Python framework for registering, running, and comparing stock return prediction models across 18 algorithms. Supports multi-ticker analysis with per-ticker hyperparameter overrides. Designed as a clean evaluation harness — not a trading system.

---

## Setup

```bash
source .env/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run all models for a single ticker (4 parallel workers)
python run.py run-all --ticker AAPL --workers 4

# Run all models across multiple tickers simultaneously
python run.py run-all --tickers AAPL,TSLA,MSFT --workers 4

# Generate an interactive HTML report (opens in browser)
python run.py report --ticker AAPL

# Generate reports for multiple tickers at once
python run.py report --tickers AAPL,TSLA,MSFT

# Run a single experiment
python run.py run --config configs/xgboost_aapl.yaml

# Run a config but override the ticker (uses ticker-specific hyperparams if they exist)
python run.py run --config configs/xgboost_aapl.yaml --ticker TSLA

# Compare latest run of every model for a ticker
python run.py compare --ticker AAPL

# Cross-ticker pivot table: rows = models, columns = tickers
python run.py compare --tickers AAPL,TSLA,MSFT --metric directional_accuracy
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `run --config <path>` | Run a single experiment |
| `run --config <path> --ticker TSLA` | Run a config with a different ticker |
| `run --config <path> --dry-run` | Validate config and data, skip fitting |
| `run-all --ticker AAPL` | Run all configs for one ticker |
| `run-all --tickers AAPL,TSLA,MSFT` | Run all configs × all tickers in parallel |
| `run-all --workers 4` | Parallel workers (default: 2) |
| `run-all --no-compare` | Skip the auto-comparison at the end |
| `report --ticker AAPL` | Generate interactive HTML dashboard (auto-opens) |
| `report --tickers AAPL,TSLA,MSFT` | Generate reports for multiple tickers |
| `report --no-open` | Generate HTML without opening browser |
| `compare --ticker AAPL` | Auto-compare latest run per model for one ticker |
| `compare --tickers AAPL,TSLA,MSFT` | Cross-ticker pivot table for all models |
| `compare --tickers ... --metric sharpe` | Choose the pivot metric (default: directional_accuracy) |
| `compare <dir1> <dir2> ...` | Compare specific result directories |
| `list-results` | Table of all saved results (latest run per model) |
| `list-results --ticker AAPL` | Filter by ticker |
| `list-results --tickers AAPL,TSLA` | Filter by multiple tickers |
| `list-results --all` | Show every run, not just the latest |
| `list-models` | Print all registered models and their parameters |

---

## HTML Report

`python run.py report --ticker AAPL` generates a self-contained `results/AAPL/report.html` with:

- **Top performers at a glance** — best direction predictor, best money return, best signal quality
- **$10,000 investment simulation** — shows what a $10k portfolio becomes using each model's signals (long/short and long-only), vs. buy-and-hold benchmark
- **Portfolio value bar chart** — colour-coded green (beat B&H) / yellow (profit) / red (loss)
- **Colour-coded metrics table** — green = best in column, red = worst
- **Bar charts** — direction accuracy, Sharpe ratio, prediction error
- **Metric glossary** — plain-English explanation of every statistic

---

## Results Structure

```
results/
├── AAPL/
│   ├── report.html             ← HTML dashboard
│   ├── comparison.csv          ← auto-saved after run-all or compare
│   ├── XGBoost/
│   │   └── XGBoost_AAPL_20260510_120000/
│   │       ├── metrics.json
│   │       ├── predictions.csv
│   │       └── config_snapshot.yaml
│   └── ...
├── TSLA/
│   └── ...
└── MSFT/
    └── ...
```

---

## Available Models (18)

### Baseline
| Name | Description |
|------|-------------|
| `NaiveLastValue` | Predicts next return = last observed return |
| `RollingMean` | Predicts next return = rolling mean of last N returns |

### Linear / Regularised
| Name | Description |
|------|-------------|
| `Ridge` | Ridge regression (L2) |
| `Lasso` | Lasso regression (L1, implicit feature selection) |
| `ElasticNet` | Combined L1 + L2 |

### Tree-Based ML
| Name | Description |
|------|-------------|
| `XGBoost` | Gradient boosted trees |
| `RandomForest` | Random forest |
| `SVR` | Support vector regression |
| `LightGBM` | Faster gradient boosting, leaf-wise splits |
| `CatBoost` | Gradient boosting with native categorical support |

### Classical Time Series
| Name | Description |
|------|-------------|
| `ARIMA` | AutoRegressive Integrated Moving Average |
| `SARIMA` | Seasonal ARIMA (weekly seasonality via `m=5`) |
| `ETS` | Exponential smoothing |
| `GARCH` | ARMA-GARCH conditional mean forecast |
| `MonteCarlo` | GBM Monte Carlo — mean of N simulated return paths |
| `OrnsteinUhlenbeck` | Mean-reverting OU process |

### Regime Models
| Name | Description |
|------|-------------|
| `HMM` | Gaussian Hidden Markov Model (3 regimes by default) |
| `MarkovSwitching` | Hamilton's Markov-Switching AR(1) |

> **Note:** PyTorch-based models (LSTM, Transformer, TCN, N-BEATS, N-HiTS) are disabled due to an OpenMP library conflict on macOS ARM between PyTorch's bundled `libomp` and Homebrew's `libomp`. The model classes remain in the codebase for reference and can be re-enabled on a CUDA Linux system.

---

## Config Schema

```yaml
experiment:
  name: xgboost_aapl
  description: "XGBoost on AAPL daily returns"

data:
  loader: yfinance          # yfinance | csv
  ticker: AAPL
  start: "2004-01-01"
  end: "2024-01-01"
  # csv_path: data/raw/aapl.csv   # used when loader: csv

features:
  target: next_return         # next_return (default) | next_close
  lags: [1, 2, 3, 5, 10]
  rolling_windows: [5, 10, 20]
  technical_indicators: true  # RSI, MACD, Bollinger Bands via ta

model:
  name: XGBoost
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8

evaluation:
  test_size: 0.2
  metrics: [rmse, mae, r2, directional_accuracy, sharpe]

seed: 42
```

### `next_return` vs `next_close`

Always use `next_return` (the default). Predicting raw price levels (`next_close`) on a trending asset produces strongly negative R² because test prices fall far outside the training distribution. Predicting daily percentage returns keeps train and test distributions aligned and produces meaningful metrics.

---

## Adding a New Model

1. Create a class extending `BaseModel` in `models/`
2. Decorate it with `@register("YourModelName")`
3. Import it in `models/__init__.py`

```python
from models.base_model import BaseModel, register

@register("MyModel")
class MyModel(BaseModel):
    name = "MyModel"
    def __init__(self, alpha: float = 1.0, **kwargs):
        self.alpha = alpha
    def fit(self, X_train, y_train): ...
    def predict(self, X_test): ...
    def get_params(self): return {"alpha": self.alpha}
```

Add a YAML config in `configs/` and it is picked up by `run-all` automatically. No changes to the runner or CLI needed.

---

## Ticker-Specific Hyperparameter Overrides

Every config in `configs/` acts as the base configuration for all tickers. To tune hyperparameters for a specific ticker, create a partial YAML override at:

```
configs/tickers/<TICKER>/<config_filename>.yaml
```

The override is **deep-merged** on top of the base config — you only need to specify the keys that differ.

**Example:** give TSLA more trees and deeper splits for XGBoost

`configs/tickers/TSLA/xgboost_aapl.yaml`:
```yaml
model:
  params:
    n_estimators: 300
    max_depth: 8
    learning_rate: 0.03
features:
  lags: [1, 2, 3, 5, 10, 20]
```

**Example:** tune ARIMA order for MSFT

`configs/tickers/MSFT/arima_aapl.yaml`:
```yaml
model:
  params:
    p: 10
    d: 0
    q: 1
```

Override files are optional — if none exists for a (ticker, config) pair, the base config is used unchanged. See `configs/tickers/README.md` for the full spec.

---

## Metrics

| Metric | Plain English | Direction |
|--------|--------------|-----------|
| `rmse` | Average prediction error | Lower is better |
| `mae` | Typical prediction error | Lower is better |
| `r2` | Predictive power (near 0 is normal for daily returns) | Higher is better |
| `directional_accuracy` | % of days the model correctly called up vs. down | Higher is better |
| `sharpe` | Risk-adjusted return of following model signals (long/short) | Higher is better |

---

## Running Tests

```bash
.env/bin/python -m pytest tests/ -v
```

All 31 tests should pass.
