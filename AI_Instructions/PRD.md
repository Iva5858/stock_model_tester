# PRD: Stock Market Prediction & Model Evaluation Framework

**Version:** 1.0 — MVP
**Status:** Draft
**Author:** —
**Last Updated:** May 2026

---

## 1. Overview

### 1.1 Purpose

This document defines the requirements for the MVP of a Python-based stock market prediction framework. The primary goal is **not** to build a production trading system, but to create a clean, extensible evaluation harness that allows a data scientist to:

- Plug in different datasets with minimal friction
- Register and run multiple prediction models (baseline, ML, DL)
- Compare model performance using standardised metrics
- Log and reproduce every experiment

This MVP is the foundation of a long-lived project. Every architectural decision must prioritise **extensibility** over premature optimisation.

### 1.2 Goals

| Goal | Description |
|------|-------------|
| **Model agnosticism** | Any model can be registered and evaluated through a common interface |
| **Dataset agnosticism** | Any ticker, index, or local CSV can be used interchangeably |
| **Reproducibility** | Every experiment run is fully logged and re-runnable from its config |
| **Portfolio value** | The codebase must demonstrate professional ML engineering practices |

### 1.3 Non-Goals (MVP)

- Live trading or brokerage integration
- Real-time data feeds or streaming
- Portfolio-level optimisation
- A web UI or dashboard
- Hyperparameter tuning automation (e.g. Optuna, Ray Tune)
- Deployment or serving infrastructure

---

## 2. Background & Motivation

Stock market prediction is a standard benchmark domain for time series modelling. The goal of this project is to build a **model evaluation framework** that mirrors real ML research workflows: define an experiment in a config, run it, and get a structured result that can be compared against others.

The MVP focuses on establishing the skeleton cleanly. Future versions will layer on more sophisticated models, feature sets, and evaluation techniques on top of a stable base.

---

## 3. Users

This is a single-user developer project. The primary user is the author — a data scientist building this as a portfolio project and personal research tool. No multi-user or access-control requirements apply to the MVP.

---

## 4. System Architecture

### 4.1 High-Level Component Map

```
stock_predictor/
│
├── configs/                  # YAML experiment definitions
│   ├── baseline_aapl.yaml
│   └── xgboost_aapl.yaml
│
├── data/                     # Data ingestion layer
│   ├── __init__.py
│   ├── base_loader.py        # Abstract loader interface
│   ├── yfinance_loader.py    # yfinance implementation
│   └── csv_loader.py         # Local CSV fallback
│
├── features/                 # Feature engineering
│   ├── __init__.py
│   └── feature_pipeline.py  # Builds feature matrix from raw OHLCV
│
├── models/                   # Model registry
│   ├── __init__.py
│   ├── base_model.py         # Abstract model interface
│   ├── baseline.py           # Naive last-value & rolling mean models
│   ├── ml_models.py          # XGBoost, RandomForest, SVR
│   └── dl_models.py          # LSTM, Transformer (PyTorch)
│
├── evaluation/               # Metrics & reporting
│   ├── __init__.py
│   ├── metrics.py            # All metric implementations
│   └── reporter.py           # Formats and saves results
│
├── experiments/              # Orchestration
│   ├── __init__.py
│   └── runner.py             # Reads config → runs experiment → saves results
│
├── results/                  # Auto-generated output (gitignored raw data)
│   └── <experiment_id>/
│       ├── metrics.json
│       ├── predictions.csv
│       └── config_snapshot.yaml
│
├── notebooks/                # Exploratory analysis (not part of pipeline)
│
├── run.py                    # CLI entrypoint
├── requirements.txt
└── README.md
```

### 4.2 Data Flow

```
YAML Config
    │
    ▼
DataLoader (yfinance / CSV)
    │   raw OHLCV DataFrame
    ▼
FeaturePipeline
    │   feature matrix (X) + target vector (y)
    ▼
Train / Test Split
    │
    ├──► Model.fit(X_train, y_train)
    │
    └──► Model.predict(X_test) ──► Metrics ──► Reporter ──► results/
```

---

## 5. Functional Requirements

### 5.1 Data Layer

| ID | Requirement |
|----|-------------|
| D-01 | The system must support fetching OHLCV data via `yfinance` given a ticker symbol, start date, and end date |
| D-02 | The system must support loading OHLCV data from a local CSV file as a drop-in alternative to D-01 |
| D-03 | All loaders must return a standardised `pandas.DataFrame` with columns: `open`, `close`, `high`, `low`, `volume`, `date` (index) |
| D-04 | The data layer must handle missing values via forward-fill, with a configurable fallback to drop |
| D-05 | The loader to use must be selectable from the YAML config |

### 5.2 Feature Engineering

| ID | Requirement |
|----|-------------|
| F-01 | The feature pipeline must accept a raw OHLCV DataFrame and return `(X: np.ndarray, y: np.ndarray)` |
| F-02 | The target variable (`y`) must be configurable: `next_close` (regression) or `direction` (binary classification, future version) |
| F-03 | MVP features must include: lagged close prices (configurable window), rolling mean, rolling std, daily return, and volume delta |
| F-04 | Optional technical indicators (RSI, MACD, Bollinger Bands) must be togglable via the YAML config using the `ta` library |
| F-05 | The pipeline must perform train/test split **before** any fitting (e.g. scaling) to prevent data leakage |
| F-06 | Feature names must be logged alongside experiment results |

### 5.3 Models

#### 5.3.1 Base Interface

All models must implement the following interface:

```python
class BaseModel(ABC):
    name: str
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None
    def predict(self, X_test: np.ndarray) -> np.ndarray
    def get_params(self) -> dict
```

| ID | Requirement |
|----|-------------|
| M-01 | All models must inherit from `BaseModel` and satisfy its interface |
| M-02 | Models must be selectable by name string from the YAML config |
| M-03 | Model hyperparameters must be passable from the YAML config as a `params` block |
| M-04 | A model registry (dict or decorator pattern) must map name strings to classes |

#### 5.3.2 Baseline Models (required for MVP)

The baseline exists to establish a minimum performance floor. Any model that cannot beat the baseline on a given dataset is not useful.

| Model | Description |
|-------|-------------|
| `NaiveLastValue` | Predicts the next close as equal to the current close (random walk assumption) |
| `RollingMeanBaseline` | Predicts the next close as the rolling mean of the last N closes (configurable window) |

#### 5.3.3 ML Models (MVP)

| Model | Library |
|-------|---------|
| `XGBoostModel` | `xgboost` |
| `RandomForestModel` | `scikit-learn` |
| `SVRModel` | `scikit-learn` — must handle internal scaling transparently |

#### 5.3.4 Deep Learning Models (MVP)

| Model | Notes |
|-------|-------|
| `LSTMModel` | PyTorch. Must use sliding window sequences. Must implement early stopping with configurable patience |
| `TransformerModel` | PyTorch encoder-only. Positional encoding required. Same sequence/early-stopping pattern as LSTM |

Both DL models must:
- Auto-detect device (CUDA > MPS > CPU)
- Expose an internal train/val split for early stopping
- Accept `seq_len`, `epochs`, `batch_size`, `lr`, and `patience` via the YAML config

### 5.4 Evaluation

| ID | Requirement |
|----|-------------|
| E-01 | The evaluation module must compute the following metrics from `(y_true, y_pred)` arrays |
| E-02 | **Regression metrics:** RMSE, MAE, MAPE, R² |
| E-03 | **Financial metrics:** Directional Accuracy (% of correctly predicted up/down moves), Sharpe Ratio of a simple long/short strategy based on predictions |
| E-04 | All metrics must be returned as a flat `dict` for easy serialisation |
| E-05 | The reporter must save metrics as `metrics.json` and predictions as `predictions.csv` under `results/<experiment_id>/` |
| E-06 | A comparison utility must accept multiple `metrics.json` files and produce a ranked summary table (printed to console and saved as `comparison.csv`) |

**Metric definitions:**

```
RMSE              = sqrt(mean((y_true - y_pred)²))
MAE               = mean(|y_true - y_pred|)
MAPE              = mean(|y_true - y_pred| / |y_true|) × 100
R²                = 1 - SS_res / SS_tot
Directional Acc.  = mean(sign(Δy_true) == sign(Δy_pred))
Sharpe Ratio      = mean(strategy_returns) / std(strategy_returns) × sqrt(252)
```

### 5.5 Experiment Runner

| ID | Requirement |
|----|-------------|
| X-01 | The runner must accept a path to a YAML config and execute the full pipeline end-to-end |
| X-02 | Each experiment must be assigned a unique `experiment_id` (format: `<model>_<ticker>_<timestamp>`) |
| X-03 | The YAML config used for the run must be snapshot-copied into the results directory |
| X-04 | The runner must print a summary to stdout on completion |
| X-05 | The runner must support a `--dry-run` flag that validates the config and data without fitting |

### 5.6 CLI

| ID | Requirement |
|----|-------------|
| C-01 | `python run.py run --config configs/xgboost_aapl.yaml` — runs a single experiment |
| C-02 | `python run.py compare --results results/exp1 results/exp2` — compares two or more experiment result directories |
| C-03 | `python run.py list-models` — prints all registered models and their configurable parameters |
| C-04 | `python run.py list-results` — prints a summary table of all results in the `results/` directory |
| C-05 | CLI must be implemented with `click` |

---

## 6. Configuration Schema

Every experiment is defined by a single YAML file. Example:

```yaml
experiment:
  name: xgboost_aapl_20y
  description: "XGBoost on AAPL with 20 years of data"

data:
  loader: yfinance          # yfinance | csv
  ticker: AAPL
  start: "2004-01-01"
  end: "2024-01-01"
  # csv_path: data/raw/aapl.csv   # used only when loader: csv

features:
  target: next_close          # next_close (regression)
  lags: [1, 2, 3, 5, 10]     # lag windows for close price
  rolling_windows: [5, 10, 20]
  technical_indicators: true  # enables RSI, MACD, Bollinger Bands via ta

model:
  name: XGBoost               # must match registry key
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8

evaluation:
  test_size: 0.2              # proportion of data held out for test
  metrics: [rmse, mae, mape, r2, directional_accuracy, sharpe]
```

---

## 7. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Reproducibility** | Random seeds must be settable globally via config; results must be identical on re-run with same seed |
| **Modularity** | Adding a new model must require only: (1) creating a class that extends `BaseModel`, (2) registering it — no changes to the runner or CLI |
| **Error handling** | The runner must fail fast with a descriptive error if the config is malformed, the ticker is invalid, or a model name is unrecognised |
| **Logging** | All runs must log start time, data shape, feature count, train/test sizes, and completion time to stdout |
| **Performance** | ML model training on 20 years of daily data (~5000 rows) must complete in under 60 seconds on a standard laptop CPU |
| **Test coverage** | Unit tests must cover: all metric functions, the feature pipeline (no-leakage check), and the baseline models |

---

## 8. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11+ | Industry standard for ML |
| Data | `yfinance`, `pandas` | Free, reliable, well-documented |
| Features | `scikit-learn`, `ta` | Standard preprocessing; `ta` for technical indicators |
| ML models | `xgboost`, `scikit-learn` | Industry-standard libraries |
| DL models | `PyTorch` | More explicit than Keras; better for custom architectures in future versions |
| CLI | `click` | Cleaner than argparse; composable commands |
| Config | `PyYAML` | Human-readable, widely used in ML projects |
| Testing | `pytest` | Standard |

---

## 9. MVP Scope vs. Future Versions

### MVP (v1.0) — this document

- Baseline, ML, and DL models
- yfinance + CSV data loaders
- Regression target only (`next_close`)
- Manual experiment runs via CLI
- Results saved as JSON/CSV

### v2.0 — Candidate features

- Classification target (`direction`: up/down)
- Stochastic models: Geometric Brownian Motion, Monte Carlo simulation, Ornstein-Uhlenbeck
- Classical time series: ARIMA, SARIMA, Exponential Smoothing
- Walk-forward / expanding window cross-validation (replacing static train/test split)
- Hyperparameter search (Optuna integration)

### v3.0 — Candidate features

- Multi-asset experiments (portfolio of tickers)
- Alternative data sources (Alpha Vantage, Polygon.io)
- Backtesting engine with realistic transaction costs
- Experiment dashboard (Streamlit or similar)
- MLflow or similar experiment tracking integration

---

## 10. Testing Plan

| Test | Type | Description |
|------|------|-------------|
| Metric correctness | Unit | Assert RMSE, MAE, etc. against manually computed values |
| No data leakage | Unit | Verify scaler is fit on train only; assert test indices not in train |
| Baseline correctness | Unit | NaiveLastValue predictions equal lagged actuals |
| Config validation | Unit | Malformed configs raise clear exceptions |
| Full pipeline smoke test | Integration | Run full experiment with a small synthetic dataset; assert results directory is created and metrics.json is valid |
| CLI commands | Integration | Test `run`, `compare`, `list-models` via `click.testing.CliRunner` |

---

## 11. Open Questions

| # | Question | Priority |
|---|----------|----------|
| 1 | Should the Transformer use a full encoder-decoder or encoder-only architecture for MVP? | Medium |
| 2 | Should `directional_accuracy` use raw price direction or return direction? | Low |
| 3 | Should results be append-only (never overwrite) or allow re-running an experiment ID? | Low |
| 4 | Should the feature pipeline be sklearn `Pipeline`-compatible for future GridSearch integration? | High |

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **OHLCV** | Open, High, Low, Close, Volume — standard candlestick data |
| **Experiment** | One run of the full pipeline: data + features + model + evaluation, defined by a single YAML config |
| **Experiment ID** | Unique identifier for a run, format: `<model>_<ticker>_<YYYYMMDD_HHMMSS>` |
| **Baseline model** | The simplest possible predictor; used as a minimum performance floor |
| **Data leakage** | When information from the test set illegitimately influences training (e.g. scaling on all data before splitting) |
| **Directional accuracy** | The fraction of timesteps where the model correctly predicted the sign of the price move |
| **Walk-forward validation** | A time-series-aware cross-validation strategy that always trains on the past and tests on the future (v2.0) |