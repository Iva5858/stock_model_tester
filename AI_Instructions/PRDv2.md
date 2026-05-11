# PRD: Stock Prediction Framework — Version 2.0

**Version:** 2.0  
**Status:** Draft  
**Prerequisite:** v1.0 MVP complete (see `PRD.md`)  
**Last Updated:** May 2026

---

## 1. Overview

### 1.1 What v1 Built

v1.0 established a clean model evaluation harness:

- **20 models** across 6 families (baseline, linear, tree-ML, classical time-series, stochastic, regime)
- **Static holdout** and **walk-forward** (expanding + rolling) cross-validation
- **Research-grounded metrics**: RMSE, MAE, R², directional accuracy, Sharpe, OOS R² (Campbell-Thompson 2008), Rank IC, max drawdown, Calmar ratio
- **SHAP feature importance** for tree-based models
- **Forecast ensemble** (equal-weight combination)
- Per-ticker hyperparameter overrides, multi-ticker parallel runs, interactive HTML reports

The framework answers: *"For a given ticker and model, how good are the predictions?"*

### 1.2 What v2 Must Answer

v2 answers a harder, more valuable question:

> **"Given a universe of tickers, what is the optimal prediction strategy — which model, at what horizon, on which stocks, and how should capital be allocated?"**

This requires three new functional capabilities on top of v1:

1. **Strategy Optimization** — automated walk-forward discovery of the best (model, horizon, refit schedule) per ticker
2. **Portfolio Construction** — combining per-ticker signals into capital allocation decisions
3. **Realistic Backtesting** — evaluating strategies under market friction

**And one architectural evolution** that makes all three possible without accumulating technical debt:

4. **Registry-everywhere, layered architecture** — every extensible category (loaders, feature transforms, models, metrics, allocators, cost models) uses the same plug-in registry pattern. The dependency graph is a strict DAG. Results have a versioned, stable schema accessible through a uniform interface.

Scalability is not a v3 concern — it is baked into every design decision in v2. The complexity of the project will only increase. Every abstraction added now that does not follow the registry + interface pattern will need to be refactored later under deadline pressure.

### 1.3 Goals

| Goal | Description |
|------|-------------|
| **Optimal strategy discovery** | Automatically identify the best (model, horizon) for each ticker via walk-forward evidence |
| **Portfolio-level thinking** | Move from single-ticker evaluation to cross-ticker capital allocation |
| **Classification target** | Add direction prediction (up/down) to complement regression |
| **Macro feature enrichment** | Add the macro predictors the academic literature demonstrates are informative |
| **Realistic economics** | All performance figures account for transaction costs |
| **Reproducible decisions** | Strategy recommendations are fully derived from walk-forward results — no look-ahead |
| **Scalable architecture** | Every category uses a registry; the dependency graph is a DAG; new components require no changes to existing code |

### 1.4 Non-Goals (v2)

- Live trading, brokerage integration, or order execution
- Real-time or intraday data
- Natural language / sentiment data (v3 candidate — see rpaper_5, rpaper_10)
- Neural network models on macOS ARM (OpenMP conflict; re-enable on CUDA Linux — see rpaper_10)
- A web dashboard (v3 candidate)
- Distributed computation (v3 candidate)

---

## 2. Background & Research Grounding

Every major component in v2 is directly motivated by the research in `AI_Instructions/research/`.

| v2 Component | Primary Research Basis |
|---|---|
| Classification target | rpaper_10 (LSTM 72% directional accuracy), rpaper_8 (direction-based long-short portfolios) |
| Multi-horizon forecasting | rpaper_2 (monthly horizon more predictable), rpaper_8 (horizon sensitivity analysis) |
| Macro predictor features | fpaper_2 (Goyal & Welch: D/P, E/P, term spread), fpaper_3 (Campbell & Thompson: sign-restricted macro predictors), rpaper_1 (Turgay: macro variables improve OOS) |
| Walk-forward model selection | rpaper_1 (expanding window OOS is the correct paradigm), rpaper_8 (model selection from OOS R²) |
| Portfolio construction | rpaper_2 (mean-variance investor Euler equation framework), rpaper_8 (long-short portfolio performance) |
| Backtesting with transaction costs | rpaper_5 (10 bps per trade benchmark), rpaper_8 (costs materially reduce Sharpe) |
| Hyperparameter search | PRD v1 open question #4; walk-forward objective prevents in-sample overfitting |
| Regime-conditioned strategy | fpaper_4 (Sarno & Valente: regime models outperform in market timing) |

The foundational insight across all papers: static model selection is unreliable — predictors that work in-sample frequently fail OOS (fpaper_2, rpaper_1). Walk-forward model selection (pick the model with the best recent OOS evidence) is the correct paradigm.

---

## 3. Architecture

### 3.1 Scalability Principles

These are non-negotiable constraints on every design decision in v2. Violating any of them creates technical debt that compounds at v3.

---

**Principle 1: Every extensible category has a registry.**

v1 established the `@register("Name")` decorator for models. v2 extends this pattern to every category where more implementations will be added over time:

| Category | v1 | v2 addition |
|---|---|---|
| Models | `@register("XGBoost")` | unchanged |
| Data loaders | hard-coded `get_loader()` dict | `@register_loader("fred")` |
| Feature transforms | `if technical_indicators:` branch | `@register_transform("rsi")` |
| Metrics | `_METRIC_FNS` dict (already registry-like) | add `task` field (regression/classification) |
| Allocation strategies | switch case in `portfolio.py` | `@register_allocator("max_sharpe")` |
| Backtest cost models | hard-coded 10 bps | `@register_cost_model("fixed_bps")` |

Adding a new implementation of anything = one class + one decorator. Zero changes to orchestration code.

---

**Principle 2: Strict layer dependency DAG.**

The dependency graph is a directed acyclic graph. No layer may import from a layer above it or from a sibling layer. This prevents circular imports at v3 when sentiment, live data, and deep learning are all active simultaneously.

```
                     run.py (CLI)
                         │
            ┌────────────┼────────────────┐
            ▼            ▼                ▼
        strategy/   experiments/      evaluation/
            │            │                │
            └────────────┼────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           data/      features/  models/
              │
           (external APIs, files)
```

Rules:
- `data/` imports nothing from this project
- `features/` imports from `data/` only
- `models/` imports nothing from this project (pure ML)
- `evaluation/` imports nothing from this project (pure math)
- `experiments/` imports from `data/`, `features/`, `models/`, `evaluation/`
- `strategy/` imports from `experiments/` and `evaluation/` only — never directly from `data/` or `features/`
- `run.py` imports from everything (CLI is the only place where all layers converge)

---

**Principle 3: Results have a versioned, stable schema accessible through a uniform interface.**

The result schema (`metrics.json`) gains a `schema_version` field. Existing keys are never renamed or removed — only new keys are added. The results directory path encodes all experiment dimensions so the strategy layer can filter without reading every file.

**v2 path structure:**
```
results/<TICKER>/<MODEL>/h<HORIZON>/<TARGET>/<exp_id>/
    metrics.json
    predictions.csv
    config_snapshot.yaml
    shap_values.csv          (tree models only)
    feature_importance.csv   (tree models only)
    equity_curve.csv         (backtest mode only)
```

Example: `results/AAPL/XGBoost/h5/next_return/XGBoost_AAPL_20260510_120000/`

The `strategy/` layer never scans raw paths. It reads through a `ResultsStore` interface (Principle 4 below). The path structure is an implementation detail of the v2 file-based store.

`metrics.json` schema:
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
  "rank_ic": 0.048,
  "max_drawdown": -0.14,
  "calmar_ratio": 1.2,
  "feature_names": ["return_lag_1", "..."],
  "n_train": 3800,
  "n_test": 950,
  "elapsed_s": 4.2
}
```

The `schema_version` field lets the runner and strategy layer detect and handle configs from different versions.

---

**Principle 4: The strategy layer reads results through a `ResultsStore` interface.**

The strategy layer (`strategy/`) never accesses the file system directly. All result reading goes through `ResultsStore`:

```python
class ResultsStore(ABC):
    @abstractmethod
    def list_experiments(
        self,
        ticker: str | None = None,
        model: str | None = None,
        horizon: int | None = None,
        target: str | None = None,
    ) -> list[ExperimentRecord]: ...

    @abstractmethod
    def load_predictions(self, exp_id: str) -> pd.DataFrame: ...

    @abstractmethod
    def load_metrics(self, exp_id: str) -> dict: ...
```

v2 ships `FileResultsStore` (reads from `results/`). v3 can add `SQLiteResultsStore` or `MLflowResultsStore` without changing any strategy layer code. `ModelSelector`, `PortfolioConstructor`, and `Backtester` all accept a `ResultsStore` at construction time.

---

**Principle 5: Composable, registry-based feature pipeline.**

The feature pipeline becomes a sequence of registered `FeatureTransform` objects, not a growing function with flag branches. This directly enables v3 additions (sentiment, earnings, alternative data) with zero changes to `build_features()`.

```python
class FeatureTransform(ABC):
    name: str
    task: str = "both"        # "regression" | "classification" | "both"

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame: ...
```

Registered transforms (v2):
- `@register_transform("lag_returns")` — existing lagged return features
- `@register_transform("rolling_stats")` — existing rolling mean/std features
- `@register_transform("volume_delta")` — existing volume change feature
- `@register_transform("technical")` — existing RSI, MACD, Bollinger Bands
- `@register_transform("macro_fred")` — NEW: FRED macro predictors
- `@register_transform("volatility_realized")` — NEW: rolling realized volatility regimes

The `features` config block becomes a list:
```yaml
features:
  target: next_return
  horizon: 1
  transforms:
    - lag_returns:   {lags: [1, 2, 3, 5, 10]}
    - rolling_stats: {windows: [5, 10, 20]}
    - volume_delta:  {}
    - technical:     {}
    - macro_fred:    {}   # NEW in v2
```

Adding sentiment in v3 = one new `@register_transform("sentiment_news")` class. No other changes.

---

### 3.2 Directory Structure

```
stock_predictor/
│
├── data/                         ← v1 + fred_loader
│   ├── base_loader.py            ← BaseLoader ABC (with @register_loader decorator)
│   ├── yfinance_loader.py        ← @register_loader("yfinance")
│   ├── csv_loader.py             ← @register_loader("csv")
│   ├── fred_loader.py            ← NEW: @register_loader("fred")
│   └── cache/fred/               ← NEW: local FRED response cache
│
├── features/                     ← v1 refactored + new transforms
│   ├── base_transform.py         ← NEW: BaseFeatureTransform + @register_transform
│   ├── feature_pipeline.py       ← updated: loop over registered transforms
│   ├── transforms/               ← NEW: one file per transform family
│   │   ├── lag_returns.py
│   │   ├── rolling_stats.py
│   │   ├── volume_delta.py
│   │   ├── technical.py          ← existing _add_technical_indicators extracted
│   │   └── macro_fred.py         ← NEW: FRED-sourced predictors
│   └── __init__.py
│
├── models/                       ← v1 unchanged + classification
│   ├── base_model.py             ← BaseModel, @register (unchanged)
│   ├── baseline.py               ← unchanged
│   ├── ml_models.py              ← unchanged
│   ├── classical_models.py       ← unchanged
│   ├── regime_models.py          ← unchanged
│   ├── dl_models.py              ← unchanged (disabled on ARM)
│   └── classification_models.py  ← NEW: @register_classifier pattern
│
├── evaluation/                   ← v1 + classification metrics + ResultsStore
│   ├── metrics.py                ← add classification metrics + task metadata
│   ├── reporter.py               ← unchanged
│   ├── html_reporter.py          ← add new metrics
│   └── results_store.py          ← NEW: ResultsStore ABC + FileResultsStore
│
├── experiments/                  ← v1 unchanged (path encoding updated)
│   └── runner.py                 ← updated: encode horizon+target in path
│
├── strategy/                     ← NEW: strategy optimization layer
│   ├── __init__.py
│   ├── model_selector.py         ← uses ResultsStore; outputs recommendation YAML
│   ├── allocators/               ← one file per allocation strategy
│   │   ├── base_allocator.py     ← BaseAllocator + @register_allocator
│   │   ├── equal_weight.py       ← @register_allocator("equal")
│   │   ├── signal_weighted.py    ← @register_allocator("signal_weighted")
│   │   ├── min_variance.py       ← @register_allocator("min_variance")
│   │   └── max_sharpe.py         ← @register_allocator("max_sharpe")
│   ├── portfolio.py              ← PortfolioConstructor (delegates to allocator registry)
│   ├── backtest.py               ← Backtester (accepts arrays, not file paths)
│   └── cost_models/              ← one file per transaction cost model
│       ├── base_cost_model.py    ← BaseCostModel + @register_cost_model
│       └── fixed_bps.py          ← @register_cost_model("fixed_bps")
│
├── configs/
│   ├── ...v1 experiment configs (unchanged)...
│   ├── search_spaces/            ← NEW: Optuna search space per model
│   │   ├── xgboost.yaml
│   │   ├── lightgbm.yaml
│   │   └── ...
│   └── strategy/                 ← NEW: strategy-level configs
│       └── tech_portfolio.yaml
│
└── run.py                        ← CLI with command groups
```

### 3.3 Updated Data Flow

```
  Experiment Config (YAML, schema_version: 2)
          │
          ▼
  DataLoader registry.get("yfinance") ──── DataLoader registry.get("fred")
          │  OHLCV DataFrame                │  Macro DataFrame
          └──────────────┬─────────────────┘
                         ▼
  FeaturePipeline  [transform_1, transform_2, ..., transform_n]
          │  feature matrix (X) + target vector (y)
          ▼
  walk_forward_splits / holdout split
          │
  Model.fit(X_train, y_train)  →  Model.predict(X_test)
          │
  compute_metrics(y_true, y_pred, y_train, task="regression"|"classification")
          │
  FileResultsStore.write(results/<TICKER>/<MODEL>/h<H>/<TARGET>/<exp_id>/)
                         │
  ─────────────────────────────────────────────────────────
  Strategy Layer (reads through ResultsStore interface only)
  ─────────────────────────────────────────────────────────
                         │
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
  ModelSelector   PortfolioConstructor  Backtester
  (OOS evidence)  (allocator registry)  (cost model registry)
          │              │                  │
          ▼              ▼                  ▼
  recommendation  allocation weights   net equity curve
  .yaml           + portfolio metrics  + gross/net metrics
```

---

## 4. Functional Requirements

### 4.1 Classification Target & Models

| ID | Requirement |
|----|-------------|
| CL-01 | Add `direction` as a valid `features.target` value: `y = sign(next_return)` encoded as `{+1, -1}` |
| CL-02 | Classification models implement `BaseModel` with `predict()` returning `{+1, -1}`. An optional `predict_proba()` returns `P(up) ∈ [0, 1]` for signal-strength-weighted allocation |
| CL-03 | Classification models are registered with a `task = "classification"` attribute, enabling the runner to automatically select the correct metric set |
| CL-04 | Classification models to implement: `LogisticRegression`, `XGBoostClassifier`, `RandomForestClassifier`, `LightGBMClassifier` — all via `@register` |
| CL-05 | All existing regression metrics continue to work unchanged when `target: direction` (directional accuracy and Sharpe are already defined on `{+1, -1}` signals) |
| CL-06 | Classification metrics are registered in `_METRIC_FNS` with a `task: "classification"` field. The runner selects the correct subset automatically based on the model's `task` attribute |

### 4.2 Multi-Horizon Forecasting

| ID | Requirement |
|----|-------------|
| H-01 | Add `horizon: int` to `FeatureConfig` (default: `1`). Horizon = number of trading days ahead to predict |
| H-02 | Target construction: `next_return → forward_return = (close[t+h] - close[t]) / close[t]`; `direction → sign(forward_return)` |
| H-03 | All lagged features are shifted by `horizon` steps relative to the target to prevent look-ahead leakage. Verified by unit test |
| H-04 | Walk-forward `step_size` defaults to `max(horizon, 5)` to avoid overlapping test prediction windows |
| H-05 | `horizon` is encoded in both the results path (`h<horizon>/`) and `metrics.json` so the `ResultsStore` can filter by horizon without reading file content |
| H-06 | A `horizon-sweep` CLI command runs one model × [h=1, h=5, h=21] and prints a comparison table. Results are saved under the respective `h<N>/` directories |
| H-07 | `ModelSelector.recommend(ticker)` returns the best `(model, horizon)` pair — not just the best model |

### 4.3 Macro Predictor Features

All foundational papers (fpaper_2, fpaper_3, rpaper_1) demonstrate that macroeconomic variables are among the most consistently informative predictors for equity returns. These are the variables the academic literature actually uses — unlike technical indicators which are widely implemented but more contested.

| ID | Requirement |
|----|-------------|
| MF-01 | `fred_loader.py` fetches from the FRED API: **D/P** (dividend-price ratio), **E/P** (earnings-price ratio), **3-month T-bill rate**, **10-year Treasury yield**, **term spread** (10yr − 3mo), **default spread** (BAA − AAA corporate bond yield), **CPI inflation** (YoY) |
| MF-02 | `macro_fred.py` transform registers as `@register_transform("macro_fred")`. It calls `fred_loader` internally and joins the macro series onto the OHLCV index via forward-fill (monthly/weekly series → daily) |
| MF-03 | All macro features enforce publication lag: each value visible at time t uses the release available at t, not the period-end value (e.g., Q1 GDP is not visible until May, not March) |
| MF-04 | FRED responses are cached in `data/cache/fred/<series_id>.parquet` with a 24-hour TTL. Cache is invalidated if the requested date range extends beyond the cached range |
| MF-05 | The transform degrades gracefully: if FRED is unavailable and no cache exists, it logs a warning and returns the DataFrame without macro columns. The pipeline does not fail |
| MF-06 | `macro_features` config flag (boolean, backward-compatible default: `false`) enables the transform when set to `true`. In the v2 composable pipeline, it simply adds `macro_fred: {}` to the transforms list |

### 4.4 Walk-Forward Model Selection (`strategy/model_selector.py`)

The core v2 capability. The system learns which strategy works best for each ticker from walk-forward evidence, rather than relying on in-sample comparisons.

| ID | Requirement |
|----|-------------|
| MS-01 | `ModelSelector` accepts a `ResultsStore` and a selection `criterion` (default: `sharpe`). It never reads file paths directly |
| MS-02 | Selection is strictly walk-forward-safe: only OOS results from the rolling selection window (default: 252 days of OOS data) are used. No look-ahead permitted |
| MS-03 | `ModelSelector.recommend(ticker) → StrategySpec` where `StrategySpec` contains: `{model, horizon, cv_method, params, criterion_value, oos_r2, directional_accuracy, sharpe}` |
| MS-04 | `StrategySpec` serialises to a valid v2 experiment YAML (`strategy_recommendation.yaml`) that can be passed directly to `run --config` |
| MS-05 | The `optimize` CLI command: (1) runs all models × all horizons × `cv_method: expanding` for a given ticker via `run-all`, (2) calls `ModelSelector.recommend()`, (3) writes `results/<TICKER>/strategy_recommendation.yaml` |
| MS-06 | Regime-conditioned selection (optional, `regime_conditioned: true`): reads the HMM/MarkovSwitching regime state from the latest run, groups OOS results by regime, and returns per-regime recommendations |

### 4.5 Portfolio Construction (`strategy/portfolio.py`, `strategy/allocators/`)

| ID | Requirement |
|----|-------------|
| PF-01 | `BaseAllocator` is an abstract class with a single method: `allocate(signals: dict[str, float], cov_matrix: pd.DataFrame | None, constraints: dict) → dict[str, float]` |
| PF-02 | Four allocators are registered in v2: `@register_allocator("equal")`, `@register_allocator("signal_weighted")`, `@register_allocator("min_variance")`, `@register_allocator("max_sharpe")` |
| PF-03 | `PortfolioConstructor` accepts the allocator name from config and resolves it via `get_allocator(name)`. Adding a new allocation strategy requires only a new `BaseAllocator` subclass with `@register_allocator` — no changes to `PortfolioConstructor` or the CLI |
| PF-04 | For `min_variance` and `max_sharpe`, covariance is estimated from historical returns using `LedoitWolf` shrinkage (already in scikit-learn). Estimation window is configurable (default: 252 days) |
| PF-05 | Constraints passed to every allocator: `max_position_weight` (default: 0.40), `min_position_weight` (default: 0.0), `long_only` (default: `false`) |
| PF-06 | `signals` input is the dict of latest predicted values per ticker. For regression models, `signal = predicted_return`. For classification models, `signal = predict_proba(up) - 0.5` (centred confidence score). The allocator is agnostic to signal source |
| PF-07 | Portfolio-level metrics computed after allocation: `portfolio_sharpe`, `portfolio_calmar`, `portfolio_max_drawdown`, `portfolio_annual_return`, `benchmark_return` (equal-weight B&H), `excess_return`, `information_ratio` |
| PF-08 | Missing ticker signals default to zero weight, not an error |

### 4.6 Backtesting Engine (`strategy/backtest.py`, `strategy/cost_models/`)

All prior performance figures are gross. v2 makes the economics explicit.

| ID | Requirement |
|----|-------------|
| BT-01 | `Backtester.__init__` accepts `y_true: np.ndarray`, `y_pred: np.ndarray`, `dates: pd.DatetimeIndex`, and a `CostModel` instance. It never reads `predictions.csv` directly — the CLI does the loading and passes arrays in |
| BT-02 | `BaseCostModel` has a single method: `cost_per_step(position_before: float, position_after: float) → float` (returns the fractional cost deducted from the return at that step) |
| BT-03 | `FixedBpsCostModel` (default, 10 bps) is registered as `@register_cost_model("fixed_bps")`. Adding a new cost model (e.g., percentage spread, volume-weighted impact) requires only a new class |
| BT-04 | A trade occurs when `sign(position_before) ≠ sign(position_after)`. Cost is applied only on trade days |
| BT-05 | The backtester outputs: `equity_curve.csv` (date, gross_value, net_value, benchmark_value), and a metrics dict with `gross_sharpe`, `net_sharpe`, `gross_calmar`, `net_calmar`, `gross_max_drawdown`, `net_max_drawdown`, `turnover_rate` |
| BT-06 | Signal construction: regression mode → `signal = sign(y_pred)`; classification mode → `signal = sign(y_pred)` with optional magnitude scaling by `abs(predict_proba - 0.5)` |
| BT-07 | `backtest` CLI command: loads latest `predictions.csv` for the specified ticker + model, converts to arrays, instantiates `Backtester`, saves results to `results/<TICKER>/<MODEL>/h<H>/<TARGET>/<exp_id>/equity_curve.csv` |

### 4.7 Hyperparameter Search (Optuna)

| ID | Requirement |
|----|-------------|
| OPT-01 | Each tunable model class carries a `search_space: dict` class attribute defining the Optuna parameter space. The search space is defined in Python (not external YAML) so it stays co-located with the model and is always in sync |
| OPT-02 | The Optuna objective function uses **walk-forward OOS R²** as the sole optimisation target. In-sample metrics are never used as the objective — this is the primary protection against overfitting during tuning |
| OPT-03 | `tune` CLI command: wraps Optuna with the walk-forward objective, runs `--n-trials` trials (default: 50), writes best hyperparameters to `configs/tickers/<TICKER>/<model>.yaml` (the existing override format) |
| OPT-04 | Tuning is supported for all models with a defined `search_space`: `XGBoost`, `LightGBM`, `CatBoost`, `RandomForest`, `Ridge`, `Lasso`, `ElasticNet` |
| OPT-05 | `tune` accepts `--multi-objective` flag that optimises for (OOS R², directional accuracy) simultaneously using Optuna's Pareto front (NSGA-II sampler). The CLI prints the Pareto-optimal hyperparameter set closest to the ideal point |
| OPT-06 | Optuna study is persisted to `results/<TICKER>/optuna_<model>.db` (SQLite) so trials are resumable with `tune --resume` |

---

## 5. Configuration Schema

### 5.1 Extended Experiment Config

```yaml
schema_version: 2            # NEW: runner validates this; missing = assumed v1

experiment:
  name: xgboost_aapl_macro_h5
  description: "XGBoost on AAPL, 5-day horizon, with macro features"

data:
  loader: yfinance
  ticker: AAPL
  start: "2004-01-01"
  end: "2024-01-01"

features:
  target: next_return           # next_return | next_close | direction
  horizon: 5                    # NEW: 1 | 5 | 21 (default: 1)
  transforms:                   # NEW: composable feature list
    - lag_returns:   {lags: [1, 2, 3, 5, 10]}
    - rolling_stats: {windows: [5, 10, 20]}
    - volume_delta:  {}
    - technical:     {}
    - macro_fred:    {}         # opt-in; degrades gracefully if FRED unavailable

model:
  name: XGBoost
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05

evaluation:
  cv_method: expanding          # holdout | expanding | rolling
  test_size: 0.2                # used for holdout only
  step_size: 21                 # walk-forward refit frequency (trading days)
  min_train_size: 500
  metrics: [rmse, mae, r2, directional_accuracy, sharpe, oos_r2, rank_ic,
            max_drawdown, calmar_ratio]

seed: 42
```

**Backward compatibility:** The `schema_version` field is optional. If absent, the runner treats the config as v1 and uses v1 feature construction (non-transform-list `features` block). Both formats are supported simultaneously. The v1 `features.technical_indicators: true` flag is translated internally to `transforms: [technical: {}]`.

### 5.2 Strategy Config

```yaml
schema_version: 2
strategy:
  name: tech_portfolio_v1
  description: "Optimal strategy for tech universe"

tickers: [AAPL, MSFT, GOOGL, AMZN, META, TSLA]

model_selection:
  criterion: sharpe             # oos_r2 | directional_accuracy | sharpe | calmar_ratio
  selection_window: 252         # rolling OOS lookback (trading days)
  regime_conditioned: false

portfolio:
  allocator: signal_weighted    # equal | signal_weighted | min_variance | max_sharpe
  max_position_weight: 0.35
  long_only: false
  rebalance_frequency: 21

backtest:
  cost_model: fixed_bps         # registered cost model name
  cost_bps: 10
  initial_capital: 100000

horizons: [1, 5, 21]
```

---

## 6. CLI Structure

v2 introduces command **groups** to contain CLI sprawl. All existing v1 commands remain at the top level for backward compatibility. New v2 commands are grouped under `experiments`, `analyze`, and `strategy`.

```bash
# v1 commands — unchanged, still work at top level
python run.py run --config configs/xgboost_aapl.yaml
python run.py run-all --tickers AAPL,TSLA --workers 4
python run.py compare --ticker AAPL
python run.py report --ticker AAPL
python run.py ensemble --ticker AAPL
python run.py list-models
python run.py list-results

# v2 experiment commands
python run.py experiments tune        --ticker AAPL --model XGBoost --n-trials 50
python run.py experiments horizon-sweep --ticker AAPL --model XGBoost

# v2 strategy commands
python run.py strategy optimize  --tickers AAPL,TSLA,MSFT
python run.py strategy portfolio --tickers AAPL,TSLA,MSFT --allocator signal_weighted
python run.py strategy backtest  --ticker AAPL --model XGBoost --cost-bps 10
python run.py strategy backtest  --config configs/strategy/tech_portfolio.yaml
```

| Command | Group | Description |
|---------|-------|-------------|
| `experiments tune` | experiments | Optuna hyperparameter search (walk-forward OOS objective) |
| `experiments horizon-sweep` | experiments | Evaluate one model at h=1, 5, 21 |
| `strategy optimize` | strategy | Walk-forward model selection → `strategy_recommendation.yaml` |
| `strategy portfolio` | strategy | Construct portfolio from latest predictions |
| `strategy backtest` | strategy | Realistic backtest with transaction costs |

---

## 7. New Metrics

### 7.1 Classification Metrics (added to `evaluation/metrics.py`)

All classification metrics are registered in `_METRIC_FNS` with `task="classification"` so the runner selects them automatically for classification targets.

| Metric key | Description | Better |
|---|---|---|
| `auc_roc` | Area under ROC curve for direction prediction | Higher |
| `log_loss` | Cross-entropy of predicted probabilities | Lower |
| `brier_score` | MSE of predicted probabilities vs. binary outcomes | Lower |
| `precision_up` | Precision of the +1 (up) class | Higher |
| `recall_up` | Recall of the +1 (up) class | Higher |
| `f1_up` | Harmonic mean of precision and recall for up class | Higher |

### 7.2 Portfolio & Backtest Metrics

| Metric key | Description | Better |
|---|---|---|
| `gross_sharpe` | Sharpe before transaction costs | Higher |
| `net_sharpe` | Sharpe after transaction costs | Higher |
| `gross_calmar` | Calmar before transaction costs | Higher |
| `net_calmar` | Calmar after transaction costs | Higher |
| `gross_max_drawdown` | Max drawdown before costs | Closer to 0 |
| `net_max_drawdown` | Max drawdown after costs | Closer to 0 |
| `turnover_rate` | Fraction of days with a position change | Context-dependent |
| `portfolio_sharpe` | Portfolio-level Sharpe | Higher |
| `benchmark_return` | Equal-weight buy-and-hold annualised return | Benchmark |
| `excess_return` | Strategy annualised return − benchmark return | Higher |
| `information_ratio` | Excess return ÷ tracking error vs. benchmark | Higher |

---

## 8. Tech Stack Additions

| Layer | Addition | Rationale |
|-------|---------|-----------|
| Macro data | `fredapi>=0.5` | Standard FRED API client; all foundational papers cite these series |
| Hyperparameter search | `optuna>=3.0` | Walk-forward-aware objective; multi-objective Pareto (NSGA-II) |
| Portfolio optimisation | `scipy.optimize` (already installed) | Mean-variance QP |
| Covariance estimation | `sklearn.covariance.LedoitWolf` (already installed) | Shrinkage estimator |
| Classification | `scikit-learn` (already installed) | LogisticRegression, RandomForestClassifier |
| Results persistence | `sqlite3` (stdlib) | Optuna study storage; no new dependency |

---

## 9. v1 → v2 Migration

### What Does Not Change

- `BaseModel` interface and `@register` decorator — completely unchanged
- All 20 v1 models — completely unchanged
- All 9 v1 evaluation metrics — unchanged; 6 new ones are purely additive
- All v1 CLI commands — all still work identically
- `results/<TICKER>/<MODEL>/<exp_id>/` path for v1 experiments — new experiments use the extended path; old results remain valid
- All 34 existing tests — all still pass

### What Extends

| Component | Change |
|---|---|
| `features/feature_pipeline.py` | Add transform-list path alongside legacy flag path |
| `features/feature_pipeline.py` | Add `horizon` to `FeatureConfig` |
| `experiments/runner.py` | Encode horizon + target in results path when non-default |
| `evaluation/metrics.py` | Add 6 classification metrics with `task` metadata |
| `evaluation/results_store.py` | New file — `ResultsStore` ABC + `FileResultsStore` |
| `models/classification_models.py` | New file |
| `strategy/` | New package — all new |
| `run.py` | Add `experiments` and `strategy` command groups |

### Backward Compatibility Guarantee

Any v1 YAML config (without `schema_version`) runs unchanged and produces results in the existing path structure. The v2 transform-list feature config is opt-in. No existing config needs to be updated.

---

## 10. Testing Plan

### Existing (all must continue passing)
All 34 v1 tests remain unchanged and green.

### New Unit Tests

| Test | Description |
|------|-------------|
| Registry coverage | Assert that every registered model, loader, transform, allocator, cost model is importable and instantiable |
| Transform no-leakage (h > 1) | h=5: assert no test-set feature uses data within 5 steps of train boundary |
| Macro feature publication lag | Assert FRED values at time t use data released on or before t |
| `oos_r2` with classification target | Assert `oos_r2` NaN when y_train not provided; finite when provided |
| Classification metrics | Assert AUC, Brier, log_loss against analytically known values |
| ModelSelector walk-forward safety | Assert selector never uses OOS results from after the selection date |
| Portfolio weights sum | Assert `sum(weights) ≤ 1.0` and `max(weights) ≤ max_position_weight` for all allocators |
| Backtest cost accounting | Net equity = gross equity × (1 - cost) at each trade day |
| Turnover arithmetic | Turnover = 0 when signal never changes; = 1.0 when signal flips every step |
| ResultsStore contract | `FileResultsStore.list_experiments(ticker="AAPL", horizon=5)` returns only h=5 results |
| Dependency DAG | Static import check: assert no module in `data/` or `features/` imports from `strategy/` or `experiments/` |
| Config schema versioning | v1 config (no schema_version) runs without error; v2 config with wrong version raises clear error |
| Optuna OOS objective | Assert the tuning objective function calls `walk_forward_splits`, not `build_features` directly |

### New Integration Tests

| Test | Description |
|------|-------------|
| Full optimize smoke test | Run `optimize --ticker SYNTHETIC` on synthetic data; assert `strategy_recommendation.yaml` produced and valid |
| Full portfolio smoke test | Run `portfolio` on 3 synthetic tickers; assert weights valid and output files written |
| Full backtest smoke test | Run `backtest` on synthetic predictions; assert `equity_curve.csv` and net/gross metrics present |
| Horizon-sweep smoke test | Run `horizon-sweep` on synthetic data; assert three result directories created under `h1/`, `h5/`, `h21/` |
| Classification end-to-end | Run `run --config` with `target: direction`; assert classification metrics in `metrics.json`, regression metrics absent |

---

## 11. Open Questions

| # | Question | Priority |
|---|----------|----------|
| 1 | **Portfolio default:** Should `long_only: false` (allowing short positions) be the default, or `true`? Research (rpaper_8) uses long-short, but long-only is more accessible to practitioners. | High |
| 2 | **Macro publication lag:** FRED data at time t — use the value from the most recent available release, or shift all series by their standard publication lag (e.g., CPI is released ~15 days after month end)? | High |
| 3 | **Multi-horizon classification target:** `direction` at h=5 — `sign(close[t+5] − close[t])` or `sign(sum of 5 daily returns)`? The latter is slightly noisier but avoids the gap return problem. | Medium |
| 4 | **ModelSelector criterion:** Single metric (e.g., Sharpe) or composite score (e.g., 0.6 × norm_sharpe + 0.4 × norm_directional_accuracy)? Composite is more robust but adds a weighting choice. | Medium |
| 5 | **Legacy config detection:** Should the runner print a deprecation warning when a v1 config (no `schema_version`) is run in v2, or silently support both? | Low |
| 6 | **Optuna study scope:** Per-ticker only, or cross-ticker shared tuning (find hyperparameters robust across multiple tickers simultaneously)? | Low |

---

## 12. Scope Boundaries

### v2.0 — this document

- Registry pattern for loaders, transforms, allocators, cost models
- `ResultsStore` interface with `FileResultsStore` implementation
- Versioned results path and `schema_version` in configs
- `BaseAllocator`, `BaseCostModel` ABCs with registered implementations
- Classification target + 4 classification model variants + 6 classification metrics
- Multi-horizon forecasting (h = 1, 5, 21) with leakage-proof feature construction
- FRED macro predictor transform (7 series, cached, publication-lag-aware)
- Walk-forward model selection (`ModelSelector`, `optimize` command)
- Portfolio construction (4 registered allocators, `portfolio` command)
- Backtest engine (registered cost models, `backtest` command)
- Optuna hyperparameter search (walk-forward OOS objective, `tune` command)
- Command groups (`experiments`, `strategy`)

### v3.0 — Future Candidates

- `SentimentTransform`: financial news → sentiment score (rpaper_5, rpaper_10). Slot-in via `@register_transform("sentiment_news")` — no pipeline changes needed.
- `SQLiteResultsStore` / `MLflowResultsStore`: swap ResultsStore implementation for scalable experiment tracking. No strategy layer changes needed.
- Deep learning re-enabled: LSTM, Transformer, TCN on CUDA Linux (rpaper_10). Slot-in via `@register` — already in `dl_models.py`.
- `PolygonLoader` / `AlphaVantageLoader`: alternative data sources. Slot-in via `@register_loader` — no pipeline changes needed.
- Streamlit dashboard: reads from `ResultsStore`, replaces static HTML. No strategy changes needed.
- Live paper trading: consumes `StrategySpec` output from `ModelSelector`. No model or evaluation changes needed.

---

## 13. Updated Glossary

| Term | Definition |
|------|------------|
| **Registry** | A decorator-based mapping (`@register_X("name")`) from name strings to class implementations. Adding a new implementation requires only the decorated class — no changes to callers. |
| **BaseAllocator** | Abstract class for capital allocation strategies. Implementations: `equal`, `signal_weighted`, `min_variance`, `max_sharpe`. |
| **BaseCostModel** | Abstract class for transaction cost models. Default implementation: `fixed_bps` (10 basis points per trade side). |
| **ResultsStore** | Interface for reading experiment results. v2 ships `FileResultsStore`. v3 can add `SQLiteResultsStore` without changing any strategy code. |
| **StrategySpec** | The output of `ModelSelector.recommend()`: a fully-specified, serialisable description of the best (model, horizon, params) for a ticker. Serialises to a valid experiment config YAML. |
| **Dependency DAG** | The directed acyclic import graph: `data → features → experiments, strategy reads evaluation outputs`. No layer imports from a layer above it. |
| **schema_version** | Integer field in experiment and strategy YAMLs. Enables the runner to detect version mismatches and apply the correct parsing logic. |
| **Direction target** | Binary classification target: `+1` if `forward_return > 0`, `-1` otherwise. |
| **Horizon** | Number of trading days ahead being predicted (1, 5, or 21). |
| **FeatureTransform** | A registered, composable unit of feature engineering. Applied in sequence by the pipeline. |
| **Gross / Net performance** | Gross = before transaction costs. Net = after. Net metrics are the operationally relevant figures. |
| **Turnover** | Fraction of trading days on which the portfolio position changes sign. Directly drives net cost. |
| **OOS R²** | Campbell-Thompson / Goyal-Welch metric: improvement in MSE over the prevailing mean baseline. Primary selection criterion for `ModelSelector`. |
| **FRED** | Federal Reserve Economic Data — free macro data API (St. Louis Fed). The macro predictor source used by all foundational papers. |
| **Ledoit-Wolf** | Covariance shrinkage estimator used by `min_variance` and `max_sharpe` allocators. Stable with limited history. |
