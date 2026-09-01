# PRD: Stock Prediction Framework — Version 3.0

**Version:** 3.0  
**Status:** Tentative Draft  
**Prerequisite:** v2.0 complete (see `PRDv2.md`)  
**Last Updated:** May 2026

---

## 1. Overview

### 1.1 What v2 Built

v2.0 elevated the framework from a single-ticker model evaluator into a full strategy research system:

- **24 models** across 7 families — 4 new classification models with `predict_proba()` and calibrated probability metrics
- **Multi-horizon forecasting** (h = 1, 5, 21) with leakage-proof target construction
- **Registry-everywhere architecture** — `@register_loader`, `@register_transform`, `@register_allocator`, `@register_cost_model`, `@register` for models
- **FRED macro features** — 7 macroeconomic series, 24h cache, publication-lag-aware
- **`ResultsStore` interface** — `FileResultsStore` with versioned `schema_version: 2` schema; both v1 and v2 path layouts supported
- **Strategy layer** — `ModelSelector` (walk-forward OOS model selection), `PortfolioConstructor` (4 allocators, Ledoit-Wolf covariance), `Backtester` (gross/net equity curves, transaction costs)
- **Optuna hyperparameter search** — walk-forward OOS R² objective, SQLite-persisted studies, multi-objective Pareto flag
- **New CLI commands** — `experiments tune`, `experiments horizon-sweep`, `strategy optimize/portfolio/backtest`, `migrate`

The framework answers: *"Given a universe of tickers, what is the optimal prediction strategy — which model, at what horizon, on which stocks, and how should capital be allocated?"*

### 1.2 What v3 Must Answer

v3 answers two harder, more operationally valuable questions:

> **"How statistically confident should I be in the strategy v2 selected — and does it hold up with richer signals and a production-scale universe?"**

> **"Can I run this system in production — with live data, an interactive dashboard, deep learning models, and a live paper trading loop?"**

This requires four new capability layers on top of v2:

1. **Statistical validation** — move from point estimates to significance-tested conclusions: Diebold-Mariano test wired into `compare`, Model Confidence Sets, multiple testing correction across a large model universe
2. **Signal enrichment** — sentiment from financial news (FinBERT / lexicon), earnings calendar features, Fama-French factor loadings, alternative data loaders (Polygon, AlphaVantage)
3. **Deep learning** — LSTM, Transformer, TCN re-enabled on CUDA Linux; GAN-based synthetic data augmentation for low-data regimes; cross-sectional learning-to-rank models
4. **Production infrastructure** — `SQLiteResultsStore` / `MLflowResultsStore` swapped in with zero strategy code changes, Streamlit dashboard, live paper trading loop consuming `StrategySpec` output from `ModelSelector`

**And two scaling investments** that become necessary at a universe of 50+ tickers:

5. **Distributed walk-forward** — Dask or Ray for parallelising fold computation across cores and machines
6. **Universe-level portfolio** — cross-sectional factor risk model, CVaR-constrained optimisation, multi-period rebalancing

v2's architecture was designed to absorb all of these at v3. Every component listed above plugs into an existing registry or interface. No v2 strategy or experiment code changes.

### 1.3 Goals

| Goal | Description |
|------|-------------|
| **Statistical rigour** | Report whether model outperformance is statistically significant — not just numerically positive |
| **Sentiment signals** | Integrate financial news sentiment as a registered `FeatureTransform` |
| **Earnings features** | Add earnings calendar, surprise, and revision features — the most consistently informative short-term catalysts |
| **Factor loadings** | Add Fama-French 5-factor exposures as features (D/P, E/P, size, value, momentum, profitability) |
| **Deep learning** | Re-enable LSTM, Transformer, TCN on CUDA Linux; add GAN augmentation for small samples |
| **Cross-sectional ranking** | Add listwise learning-to-rank models that predict relative return ranks across tickers |
| **Production result store** | SQLiteResultsStore and MLflowResultsStore that drop in with zero strategy code changes |
| **Interactive dashboard** | Streamlit app reading from ResultsStore — replaces static HTML reports |
| **Paper trading** | Live data → StrategySpec → position management → P&L tracking, without real capital |
| **Scale** | Dask/Ray distributed walk-forward for universes of 50–500 tickers |
| **Advanced portfolio** | CVaR-constrained optimisation, factor risk model, multi-period rebalancing |

### 1.4 Non-Goals (v3)

- Live brokerage order execution or real capital deployment
- Sub-daily (intraday) data or tick data
- Cryptocurrency or derivatives (equity focus maintained)
- Proprietary data sources requiring institutional data agreements
- Multi-machine distributed training of deep learning models (single-GPU scope)

---

## 2. Background & Research Grounding

Every major v3 component traces to the research in `AI_Instructions/research/`.

| v3 Component | Primary Research Basis |
|---|---|
| Sentiment transform | rpaper_5 (Wang 2025: sentiment factors improve Rank IC), rpaper_10 (Vishwas 2025: FinBERT for financial NLP) |
| Earnings features | rpaper_5 (earnings surprise as short-term catalyst), rpaper_8 (analyst revision signals) |
| Fama-French factor features | fpaper_2 (Goyal & Welch: D/P, E/P as predictors), fpaper_3 (Campbell & Thompson: sign-restricted macro) |
| Deep learning (LSTM, Transformer) | rpaper_10 (Vishwas 2025: LSTM 72% directional accuracy, Transformer attention), rpaper_8 (neural nets best OOS) |
| GAN data augmentation | rpaper_10 (Vishwas 2025: synthetic minority oversampling for imbalanced direction targets) |
| Cross-sectional ranking | rpaper_2 (Rossi 2018: cross-sectional momentum), rpaper_8 (cross-model Rank IC) |
| Diebold-Mariano in compare | rpaper_8 (Mistol & Möhler 2023: DM test is the standard for comparing predictive accuracy) |
| Model Confidence Set | rpaper_8 (MCS is the rigorous multi-model comparison test) |
| Multiple testing correction | fpaper_2 (Goyal & Welch: data-snooping bias is the primary validity threat in predictability research) |
| Time-varying model weights | rpaper_6/7 (Cheng et al. 2025: locally stationary predictors — predictability is state-dependent) |
| Bayesian LASSO | rpaper_3 (Huang 2024: high-dimensional regularisation, Partially Protected LASSO) |
| Factor risk model in portfolio | fpaper_2, fpaper_3 (factor exposures as a risk decomposition) |
| CVaR-constrained optimisation | rpaper_5 (Wang 2025: downside risk metrics in portfolio construction) |
| Distributed walk-forward | rpaper_1 (Turgay 2025: expanding window at scale is the correct OOS paradigm) |

The foundational insight motivating the statistical validation layer: a positive OOS R² or Sharpe computed on a single model-ticker pair is weak evidence when you have searched over 24 models × 3 horizons × 3 targets = 216 combinations per ticker (fpaper_2). v3's statistical layer quantifies the probability that the best observed strategy outperforms by luck alone.

---

## 3. Architecture

### 3.1 v2 Architectural Contracts That v3 Must Honour

v3 adds components but does not modify any v2 interface. The following are frozen:

- `BaseModel` / `@register` — unchanged
- `BaseLoader` / `@register_loader` — unchanged
- `FeatureTransform` / `@register_transform` — unchanged
- `BaseAllocator` / `@register_allocator` — unchanged
- `BaseCostModel` / `@register_cost_model` — unchanged
- `ResultsStore` ABC — unchanged; v3 adds new implementations
- `StrategySpec` schema — new fields additive only
- `metrics.json` schema — `schema_version: 3` added; no existing keys renamed or removed

**Any v2 config (schema_version: 2 or absent) runs unchanged in v3.**

### 3.2 New Dependency DAG Extensions

The v2 DAG is extended with two new optional layers:

```
                     run.py (CLI)
                         │
    ┌────────────────────┼────────────────────────┐
    ▼                    ▼                        ▼
strategy/           experiments/              dashboard/   ← NEW (reads ResultsStore only)
    │                    │                        │
    └────────────────────┼────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           data/      features/  models/
```

New rules:
- `dashboard/` imports from `evaluation/` (ResultsStore) only — never from `strategy/`, `experiments/`, `data/`, or `features/` directly
- `live/` (paper trading layer) imports from `strategy/` and `live/feed.py` (`BaseDataFeed`) only — it never imports from `data/` loaders directly. The `DataFeed` abstraction wraps all data access; `live/` is a thin adapter between the strategy layer and that abstraction
- `analysis/` (statistical testing layer) imports from `evaluation/` only — pure statistics on arrays

### 3.3 Updated Directory Structure

```
stock_predictor/
│
├── data/
│   ├── ...v2 unchanged...
│   ├── polygon_loader.py         ← NEW: @register_loader("polygon")
│   └── alphavantage_loader.py    ← NEW: @register_loader("alphavantage")
│
├── features/
│   ├── ...v2 unchanged...
│   └── transforms/
│       ├── ...v2 unchanged...
│       ├── sentiment_news.py     ← NEW: @register_transform("sentiment_news")
│       ├── earnings_calendar.py  ← NEW: @register_transform("earnings")
│       └── fama_french.py        ← NEW: @register_transform("fama_french")
│
├── models/
│   ├── ...v2 unchanged...
│   ├── dl_models.py              ← UPDATED: re-enable LSTM, Transformer, TCN on CUDA
│   ├── ranking_models.py         ← NEW: cross-sectional LTR models
│   └── bayesian_models.py        ← NEW: BayesianLASSO, PartiallyProtectedLASSO
│
├── evaluation/
│   ├── ...v2 unchanged...
│   ├── results_store.py          ← UPDATED: add SQLiteResultsStore, MLflowResultsStore
│   └── statistical_tests.py      ← NEW: MCS, DM at scale, multiple testing correction
│
├── experiments/
│   └── runner.py                 ← UPDATED: schema_version: 3, distributed fold option
│
├── strategy/
│   ├── ...v2 unchanged...
│   ├── allocators/
│   │   ├── ...v2 unchanged...
│   │   ├── cvar.py               ← NEW: @register_allocator("cvar")
│   │   └── factor_risk.py        ← NEW: @register_allocator("factor_risk")
│   └── model_selector.py         ← UPDATED: time-varying weights, MCS filtering
│
├── analysis/                     ← NEW package (statistical testing layer)
│   ├── __init__.py
│   ├── dm_test.py                ← pairwise DM test across all saved experiments
│   ├── model_confidence_set.py   ← MCS implementation (Hansen, Lunde, Nason 2011)
│   └── multiple_testing.py       ← Benjamini-Hochberg FDR correction
│
├── dashboard/                    ← NEW package (Streamlit app)
│   ├── __init__.py
│   ├── app.py                    ← Streamlit entrypoint
│   └── pages/
│       ├── model_comparison.py
│       ├── strategy_overview.py
│       ├── portfolio_monitor.py
│       └── backtest_detail.py
│
├── live/                         ← NEW package (paper trading)
│   ├── __init__.py
│   ├── feed.py                   ← BaseDataFeed ABC + @register_feed decorator
│   ├── yfinance_feed.py          ← @register_feed("yfinance_live") — polling-based, testing only
│   ├── polygon_feed.py           ← @register_feed("polygon") — websocket, production use
│   └── paper_trader.py           ← PaperTrader: consumes StrategySpec + BaseDataFeed, tracks P&L
│
├── configs/
│   ├── ...v2 unchanged...
│   └── strategy/
│       └── tech_portfolio_v3.yaml  ← example v3 strategy config
│
└── run.py                        ← CLI with new v3 command groups
```

---

## 4. Functional Requirements

### 4.1 Statistical Validation Layer (`analysis/`)

The core v3 insight: a model that beats the prevailing mean by OOS R² = 0.01 on a single experiment is weak evidence. v3 quantifies whether outperformance survives rigorous statistical testing across the full model universe.

#### 4.1.1 Diebold-Mariano at Scale

v2 has `diebold_mariano()` as a standalone function. v3 wires it into the `compare` command.

| ID | Requirement |
|----|-------------|
| DM-01 | `analysis/dm_test.py` runs pairwise DM tests across all experiments for a ticker, building an N×N p-value matrix where entry (i,j) = P(model i is not better than model j) |
| DM-02 | `compare --ticker AAPL --dm-test` prints the DM p-value for each model vs. the best model. Models with p > 0.10 are marked "not significantly different from best" |
| DM-03 | DM test uses the Harvey-Leybourne-Newbold small-sample correction (already in v2 `diebold_mariano()`) |
| DM-04 | p-values are corrected for multiple comparisons using Benjamini-Hochberg FDR at `alpha = 0.10` (default) |

#### 4.1.2 Model Confidence Set

| ID | Requirement |
|----|-------------|
| MCS-01 | `analysis/model_confidence_set.py` implements the Hansen, Lunde, Nason (2011) MCS procedure. Input: a dict of {model_name: prediction_array}. Output: the set of models that cannot be statistically eliminated at confidence level α |
| MCS-02 | `analyze mcs --ticker AAPL --alpha 0.10` prints the Model Confidence Set — the models that survive pairwise elimination |
| MCS-03 | `ModelSelector.recommend()` gains an optional `require_mcs: bool` flag. When true, model selection is restricted to the MCS rather than the full experiment set — only statistically significant winners can be recommended |
| MCS-04 | MCS results are cached in `results/<TICKER>/mcs_<criterion>.json` with the models and their p-values |

#### 4.1.3 Multiple Testing Correction

| ID | Requirement |
|----|-------------|
| MT-01 | When `--dm-test` is run across a ticker with N models, Benjamini-Hochberg correction is applied to the N(N-1)/2 pairwise p-values |
| MT-02 | `analysis/multiple_testing.py` exposes `benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray` returning a boolean mask of rejected nulls |
| MT-03 | The `compare` output optionally shows adjusted p-values alongside the raw metrics |

### 4.2 Signal Enrichment

#### 4.2.1 Sentiment Transform (`@register_transform("sentiment_news")`)

| ID | Requirement |
|----|-------------|
| SE-01 | `features/transforms/sentiment_news.py` fetches financial news headlines for the ticker via a configurable news API (default: `newsapi`, configurable via `NEWSAPI_KEY` env var). Falls back to a simple positive/negative word-count lexicon (Loughran-McDonald dictionary) if the API is unavailable |
| SE-02 | Features produced: `sentiment_score` (rolling 5-day mean of daily headline sentiment), `sentiment_volatility` (rolling std), `sentiment_momentum` (1-day change in sentiment). All shifted by 1 day to prevent look-ahead |
| SE-03 | Headlines are cached in `data/cache/sentiment/<TICKER>_<date>.parquet` with a 24h TTL. Degradation: if both the API and cache are unavailable, log a warning and skip — never fail |
| SE-04 | The transform respects the publication lag: the sentiment score visible at close of day t uses only headlines published before market close on day t |
| SE-05 | `sentiment_news: {api: newsapi, window: 5}` in the transforms list activates the transform. Compatible with both regression and classification targets (`task: "both"`) |

Research basis: rpaper_5 (Wang 2025) demonstrates that sentiment factors improve Rank IC by 15–30% in OOS tests. rpaper_10 (Vishwas 2025) establishes FinBERT as the current best-in-class financial sentiment model.

#### 4.2.2 Earnings Calendar Transform (`@register_transform("earnings")`)

| ID | Requirement |
|----|-------------|
| EC-01 | `features/transforms/earnings_calendar.py` fetches earnings dates, EPS actuals, and EPS consensus estimates for the ticker. Primary source: `yfinance.Ticker.earnings_dates` for data from 2018 onwards. For dates before 2018, earnings features degrade gracefully to `NaN` — the transform logs a warning and continues. AlphaVantage is used as a fallback only when `ALPHAVANTAGE_KEY` is explicitly configured; it is never activated on the free tier due to rate limits that would silently corrupt walk-forward folds |
| EC-02 | Features produced: `days_to_earnings` (integer countdown), `earnings_surprise_lag1` (previous quarter's normalised EPS surprise), `earnings_revision_30d` (analyst EPS revision momentum over 30 days). All shifted to prevent look-ahead |
| EC-03 | `days_to_earnings` = 0 on the earnings date. Features are forward-filled between earnings dates |
| EC-04 | The transform degrades gracefully if no earnings data is available for the ticker (e.g., ETFs, indices) |

#### 4.2.3 Fama-French Factor Transform (`@register_transform("fama_french")`)

| ID | Requirement |
|----|-------------|
| FF-01 | `features/transforms/fama_french.py` downloads the Fama-French 5-Factor daily data from Ken French's data library (publicly available CSV). Cached in `data/cache/fama_french/` with weekly TTL |
| FF-02 | Features produced: `mkt_rf`, `smb`, `hml`, `rmw`, `cma` (daily factor returns, shifted by 1 day) and rolling 21-day factor exposure estimates (`beta_mkt`, `beta_smb`, `beta_hml`, etc.) computed from rolling OLS within each walk-forward fold |
| FF-03 | Rolling factor betas are estimated on the training window only (no look-ahead). The estimate at test day t uses only data up to t |
| FF-04 | `fama_french: {factors: [mkt_rf, smb, hml, rmw, cma], rolling_window: 63}` in the transforms list activates the transform |

Research basis: fpaper_2 (Goyal & Welch 2008) uses D/P and E/P (proxied by HML and CMA loading) as predictors. fpaper_3 (Campbell & Thompson) finds sign-restricted factor forecasts improve OOS R².

### 4.3 Deep Learning Models (re-enabled on CUDA Linux)

| ID | Requirement |
|----|-------------|
| DL-01 | `models/dl_models.py` is updated to detect the runtime environment. On CUDA Linux (`torch.cuda.is_available()` and `sys.platform != "darwin"`), LSTM, Transformer, and TCN are registered. On macOS ARM, they remain unregistered with a clear log message |
| DL-02 | `@register("LSTM")` — LSTM with configurable hidden size, layers, and dropout. `predict_proba()` supported for `direction` targets using a sigmoid output head |
| DL-03 | `@register("Transformer")` — encoder-only Transformer with positional encoding. Same interface as LSTM |
| DL-04 | `@register("TCN")` — Temporal Convolutional Network with dilated causal convolutions |
| DL-05 | All DL models carry a `search_space` dict for Optuna tuning (learning rate, hidden size, dropout, n_layers) |
| DL-06 | Early stopping is applied within each walk-forward fold using a 10% validation split of the training window |
| DL-07 | GAN data augmentation: `@register_transform("gan_augment")` — trains a conditional GAN on the training fold and augments with synthetic samples when `n_train < 1000`. Before synthetic samples are used, a KS test is run comparing the generated distribution to the real training data. If the KS test p-value < 0.05 (distributions are significantly different), the synthetic samples are discarded, a warning is logged, and the fold proceeds with real data only. Active only when the `gan_augment` transform is in the transforms list (opt-in). Never applied to the test fold |

Research basis: rpaper_10 (Vishwas 2025) — LSTM achieves 72% directional accuracy; Transformer attention captures long-range dependencies; GAN augmentation addresses the class imbalance problem in direction classification.

### 4.4 Cross-Sectional Ranking Models

| ID | Requirement |
|----|-------------|
| CR-01 | `models/ranking_models.py` adds cross-sectional learning-to-rank models using listwise ranking objectives. These models predict the *rank* of a ticker's return within a universe rather than the return level |
| CR-02 | `@register("LambdaMART")` — gradient boosted trees with LambdaMART ranking loss (via LightGBM's `rank:pairwise` objective) |
| CR-03 | Cross-sectional models accept stacked feature matrices `(n_dates × n_tickers, n_features)` and output a rank score per row. The runner handles the stacking when `cv_mode: cross_sectional` is set in the config |
| CR-04 | `rank_ic` (Spearman) is the primary evaluation metric for cross-sectional models. The `compare` command reports it by default for these models |

Research basis: rpaper_2 (Rossi 2018) — cross-sectional momentum is among the most robust return predictors. rpaper_8 — Rank IC is the standard metric for cross-sectional signal quality.

### 4.5 Bayesian Models

| ID | Requirement |
|----|-------------|
| BM-01 | `models/bayesian_models.py` adds `@register("BayesianLASSO")` — Bayesian LASSO with spike-and-slab prior, implemented via `pymc` or `scikit-learn`'s `BayesianRidge` as a fallback |
| BM-02 | `@register("PartiallyProtectedLASSO")` — Lasso with a protected subset of features that are never zeroed out (theory-grounded variables: the FRED macro series from v2). Protected features are specified in `model.params.protected_features` |
| BM-03 | Both models expose `get_posterior_intervals()` returning 90% credible intervals on predictions, stored in `predictions.csv` as `y_pred_lo` and `y_pred_hi` columns |

Research basis: rpaper_3 (Huang 2024) — Bayesian regularisation is more principled than grid-searched penalisation in high dimensions. Partially Protected LASSO directly encodes theory priors.

### 4.6 Production Result Stores

| ID | Requirement |
|----|-------------|
| RS-01 | `SQLiteResultsStore` implements the `ResultsStore` ABC. Reads from / writes to a single SQLite database at `results/experiments.db`. Schema matches the v2 `metrics.json` fields plus indexed columns for efficient filtering |
| RS-02 | `MLflowResultsStore` implements `ResultsStore` using MLflow's tracking API. Requires `MLFLOW_TRACKING_URI` env var. Falls back to `FileResultsStore` if MLflow is unavailable |
| RS-03 | The active store is selected in a top-level `results_store:` config block (default: `file`). All strategy, analysis, and dashboard code receives the interface — no concrete type |
| RS-04 | `FileResultsStore` gains an index file (`results/.index.db`, SQLite) that caches the directory scan results. The first access builds the index; subsequent accesses query it. Cache invalidated on file modification time changes |
| RS-05 | `run.py results-store reindex` rebuilds the FileResultsStore index. Useful after bulk migrations |

### 4.7 Streamlit Dashboard (`dashboard/`)

| ID | Requirement |
|----|-------------|
| DB-01 | `python run.py serve` launches the Streamlit app on `localhost:8501` |
| DB-02 | **Model Comparison page** — interactive version of the current HTML report. Filterable by ticker, model family, horizon, and target. Sortable by any metric. Colour-coded heatmap |
| DB-03 | **Strategy Overview page** — shows `strategy_recommendation.yaml` contents for each ticker. One-click re-run of `strategy optimize` from the UI |
| DB-04 | **Portfolio Monitor page** — current allocation weights, live P&L if paper trading is active, historical equity curve from `equity_curve.csv` |
| DB-05 | **Backtest Detail page** — gross vs. net equity curve chart, rolling Sharpe, drawdown chart, trade log from `equity_curve.csv` |
| DB-06 | The dashboard reads exclusively through `ResultsStore` — no raw file path access. Swapping to `SQLiteResultsStore` requires zero dashboard code changes |
| DB-07 | Dashboard is read-only. It never calls `run_experiment()` or modifies any result file |

### 4.8 Paper Trading (`live/`)

| ID | Requirement |
|----|-------------|
| PT-01 | `BaseDataFeed` ABC defined in `live/feed.py`: `def get_latest(ticker: str) -> pd.Series` returns the most recent OHLCV bar. `@register_feed` decorator follows the same registry pattern. This is the only data interface `live/` uses — it never imports from `data/` loaders directly |
| PT-02 | `@register_feed("yfinance_live")` — polling-based feed that calls `yfinance.download()` on a schedule to retrieve the most recent daily bar. Data is delayed by approximately 15 minutes and subject to yfinance's unofficial API stability. **For testing and development only — not suitable for production use.** `@register_feed("polygon")` — Polygon.io websocket feed for production paper trading. Requires `POLYGON_API_KEY` env var |
| PT-03 | `PaperTrader` accepts a `StrategySpec` (from `ModelSelector.recommend()`) and a `BaseDataFeed`. On each new bar: loads latest features, runs the model's `predict()` using the **frozen model from the last `run` experiment** (no live retrain), constructs a signal, records the position |
| PT-04 | `PaperTrader` tracks: `positions: dict[str, float]`, `cash: float`, `equity_history: list[tuple[datetime, float]]`, `trades: list[Trade]`. All persisted to `live/paper_trading_log.jsonl` |
| PT-05 | `python run.py live start --tickers AAPL,MSFT --feed yfinance_live` launches the paper trading loop. Runs until `live stop` is called or the process exits |
| PT-06 | `python run.py live status` prints current positions, unrealised P&L, and today's trades |
| PT-07 | The paper trading loop never modifies `results/` — it writes only to `live/`. The strategy layer is read-only from the live layer's perspective |

### 4.9 Advanced Portfolio Construction

#### 4.9.1 CVaR Allocator

| ID | Requirement |
|----|-------------|
| CV-01 | `@register_allocator("cvar")` minimises Conditional Value at Risk (CVaR, also called Expected Shortfall) at a configurable confidence level (default: 95%) |
| CV-02 | CVaR optimisation uses `scipy.optimize.linprog` (linear programming formulation of CVaR). Requires historical return scenarios — uses the walk-forward OOS returns from `predictions.csv` |
| CV-03 | Constraint: `max_position_weight` (default 0.40), `long_only` (default false), `min_cvar_improvement: 0.05` (refuse allocation if CVaR is not at least 5% better than equal weight) |

#### 4.9.2 Factor Risk Model Allocator

| ID | Requirement |
|----|-------------|
| FR-01 | `@register_allocator("factor_risk")` constructs a covariance matrix from a Fama-French factor risk model rather than from historical returns directly. Factor exposures come from the `fama_french` transform |
| FR-02 | Covariance = BΛBᵀ + Δ where B is the factor loading matrix, Λ is the factor covariance, and Δ is the diagonal idiosyncratic variance |
| FR-03 | Falls back to Ledoit-Wolf if factor loadings are not available (e.g., fama_french transform not in pipeline) |

#### 4.9.3 Multi-Period Rebalancing

| ID | Requirement |
|----|-------------|
| MP-01 | `PortfolioConstructor.build()` gains a `rebalance_frequency: int` parameter (trading days between rebalances, default: 21). Between rebalance dates, positions drift with market returns |
| MP-02 | `BacktestResult` reports `avg_turnover_per_rebalance` (average weight change at each rebalance event) alongside the existing `turnover_rate` |

### 4.10 Distributed Walk-Forward

| ID | Requirement |
|----|-------------|
| DW-01 | `walk_forward_splits()` gains a `distributed: bool` parameter (default: False). When True and Dask/Ray is available, fold computation is dispatched across workers |
| DW-02 | The distributed path is activated via `evaluation.distributed: true` in the config. Falls back silently to sequential if Dask/Ray is not installed |
| DW-03 | Fold independence is guaranteed by design (each fold is stateless beyond its input arrays), so distribution requires no architectural changes — only the execution layer changes |
| DW-04 | `python run.py run-all --tickers <500-ticker list> --workers 32 --distributed` runs walk-forward across all tickers using a Dask LocalCluster |

---

## 5. Configuration Schema (v3 extensions)

All v2 configs run unchanged. v3 adds optional top-level blocks:

```yaml
schema_version: 3

experiment:
  name: transformer_aapl_sentiment_h5

data:
  loader: polygon              # NEW: polygon | alphavantage | yfinance | csv | fred

features:
  target: direction
  horizon: 5
  transforms:
    - lag_returns:      {lags: [1, 2, 3, 5, 10]}
    - rolling_stats:    {windows: [5, 10, 20]}
    - volume_delta:     {}
    - technical:        {}
    - macro_fred:       {}
    - sentiment_news:   {api: newsapi, window: 5}    # NEW
    - earnings:         {}                            # NEW
    - fama_french:      {factors: [mkt_rf, smb, hml, rmw, cma], rolling_window: 63}  # NEW
    - gan_augment:      {min_train_size: 1000}        # NEW — opt-in only

model:
  name: Transformer            # re-enabled on CUDA Linux
  params:
    d_model: 64
    nhead: 4
    num_layers: 2
    dropout: 0.1

evaluation:
  cv_method: expanding
  step_size: 21
  min_train_size: 500
  distributed: false           # NEW: Dask/Ray distributed fold execution
  metrics: [auc_roc, log_loss, brier_score, directional_accuracy, rank_ic]

# NEW: statistical validation block
analysis:
  dm_test: true                # run DM test after experiment
  dm_alpha: 0.10
  mcs: true                    # compute MCS for ticker
  fdr_correction: true

seed: 42
```

### Strategy Config (v3 extensions)

```yaml
schema_version: 3
strategy:
  name: tech_portfolio_v3

tickers: [AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, AMD, QCOM, INTC]

model_selection:
  criterion: sharpe
  selection_window: 252
  require_mcs: true            # NEW: restrict to MCS survivors only
  regime_conditioned: false

portfolio:
  allocator: cvar              # NEW: cvar | factor_risk | max_sharpe | min_variance | signal_weighted | equal
  max_position_weight: 0.20
  long_only: false
  rebalance_frequency: 21      # NEW: days between rebalances

backtest:
  cost_model: fixed_bps
  cost_bps: 10
  initial_capital: 100000

results_store: sqlite          # NEW: file | sqlite | mlflow
horizons: [1, 5, 21]
```

---

## 6. CLI Structure

All v2 commands are unchanged. New v3 commands are grouped under `analyze`, `live`, and `serve`.

```bash
# v2 commands — all still work at top level
python run.py run / run-all / compare / report / ensemble / list-models / list-results / migrate
python run.py experiments tune / horizon-sweep
python run.py strategy optimize / portfolio / backtest

# v3 — statistical analysis commands
python run.py analyze dm-test  --ticker AAPL --alpha 0.10
python run.py analyze mcs      --ticker AAPL --alpha 0.10
python run.py analyze compare  --ticker AAPL --dm-test --fdr

# v3 — live / paper trading commands
python run.py live start   --tickers AAPL,MSFT --feed yfinance_live
python run.py live stop
python run.py live status

# v3 — dashboard
python run.py serve        # launches Streamlit on localhost:8501

# v3 — result store management
python run.py results-store reindex
python run.py results-store migrate-to-sqlite
```

| Command | Group | Description |
|---------|-------|-------------|
| `analyze dm-test` | analyze | Pairwise Diebold-Mariano test across all models for a ticker |
| `analyze mcs` | analyze | Model Confidence Set — statistically survivable models only |
| `analyze compare` | analyze | Enhanced compare with optional DM test and FDR correction |
| `live start` | live | Start paper trading loop |
| `live stop` | live | Gracefully stop paper trading loop |
| `live status` | live | Print current positions and P&L |
| `serve` | top-level | Launch Streamlit dashboard |
| `results-store reindex` | results-store | Rebuild FileResultsStore index |
| `results-store migrate-to-sqlite` | results-store | Copy all results from FileResultsStore into SQLiteResultsStore |

---

## 7. New Metrics

### 7.1 Statistical Test Metrics

| Metric key | Description |
|---|---|
| `dm_stat` | Diebold-Mariano test statistic vs. HistoricalMean benchmark |
| `dm_pvalue` | DM p-value (HLN-corrected) |
| `dm_pvalue_adj` | Benjamini-Hochberg adjusted p-value |
| `in_mcs` | Boolean — whether this model survives the Model Confidence Set at α = 0.10 |

### 7.2 Deep Learning Specific

| Metric key | Description |
|---|---|
| `val_loss_curve` | Saved as JSON array in `metrics.json` — training and validation loss per epoch for the last fold |
| `early_stopping_epoch` | Epoch at which early stopping triggered in the last fold |

### 7.3 Cross-Sectional Ranking

| Metric key | Description |
|---|---|
| `ndcg_at_5` | NDCG@5 — ranking quality at the top 5 positions |
| `precision_at_5` | Fraction of top-5 predicted tickers that are true top-5 returners |
| `cross_sectional_rank_ic` | Rank IC computed across tickers rather than across time |

### 7.4 Portfolio v3

| Metric key | Description |
|---|---|
| `cvar_95` | 95% Conditional Value at Risk of the strategy |
| `factor_r2` | Fraction of portfolio variance explained by Fama-French factors |
| `avg_turnover_per_rebalance` | Average absolute weight change at each rebalance event |

---

## 8. Tech Stack Additions

| Layer | Addition | Rationale |
|-------|---------|-----------|
| Sentiment | `transformers>=4.0` (FinBERT) or `nltk` (Loughran-McDonald lexicon) | rpaper_5, rpaper_10 |
| News data | `newsapi-python>=0.2` | Financial headline retrieval |
| Factor data | `pandas-datareader>=0.10` | Ken French data library download |
| Deep learning | `torch>=2.2` (already in requirements; CUDA path re-enabled) | rpaper_10 |
| Dashboard | `streamlit>=1.30` | Interactive replacement for static HTML |
| Result store | `mlflow>=2.0` (optional) | Experiment tracking at scale |
| Distributed | `dask>=2024.0` or `ray>=2.0` (optional) | Walk-forward at scale |
| CVaR | `scipy.optimize` (already installed) | Linear programming CVaR |
| Statistical tests | `statsmodels>=0.14` (already installed) | MCS implementation |
| Alternative data | `polygon-api-client>=1.0` (optional) | Real-time and historical equity data |

---

## 9. v2 → v3 Migration

### What Does Not Change

- All 24 v2 models — completely unchanged
- All v2 CLI commands — all still work identically
- All v2 configs (schema_version: 2 or absent) — run without modification
- `BaseModel`, `BaseLoader`, `FeatureTransform`, `BaseAllocator`, `BaseCostModel`, `ResultsStore` ABCs — frozen
- All 89 v2 tests — all still pass in v3

### What Extends

| Component | Change |
|---|---|
| `models/dl_models.py` | Enable LSTM/Transformer/TCN when CUDA detected |
| `evaluation/results_store.py` | Add `SQLiteResultsStore`, `MLflowResultsStore`, `FileResultsStore` index |
| `evaluation/metrics.py` | Add statistical test metrics with `task` metadata |
| `strategy/model_selector.py` | Add `require_mcs` flag, time-varying weight option |
| `strategy/allocators/` | Add `cvar.py`, `factor_risk.py` |
| `strategy/backtest.py` | Add `rebalance_frequency`, `avg_turnover_per_rebalance` |
| `experiments/runner.py` | Add `schema_version: 3`, distributed fold dispatch |
| `run.py` | Add `analyze`, `live`, `results-store` command groups; add `serve` |

### New Packages (zero impact on v2 code)

`analysis/`, `dashboard/`, `live/` are fully additive. No existing file imports from them until `run.py` wires up the new command groups.

---

## 10. Testing Plan

### Existing (all must continue passing)
All 89 v2 tests remain unchanged and green.

### New Unit Tests

| Test | Description |
|------|-------------|
| DM test correctness | Assert DM stat and p-value match analytically computed values for known prediction pairs |
| BH correction | Assert Benjamini-Hochberg rejects exactly the right hypotheses for a known p-value vector |
| MCS elimination | Assert MCS correctly eliminates the strictly inferior predictor in a 3-model toy example |
| Sentiment transform no-leakage | Assert sentiment features at time t use only headlines from before t |
| Earnings transform forward-fill | Assert `days_to_earnings` is correct at known earnings dates; assert pre-2018 dates return NaN without error |
| Fama-French beta estimation | Assert rolling OLS betas are estimated on training data only, not test |
| LSTM predict shape | Assert LSTM.predict() returns correct shape on synthetic data |
| GAN KS gate | Assert that when generated samples fail KS test, fold proceeds with real data only and logs a warning |
| CVaR allocator | Assert CVaR of resulting allocation ≤ CVaR of equal-weight baseline |
| Factor risk covariance | Assert BΛBᵀ + Δ is positive semi-definite |
| SQLiteResultsStore contract | Assert it satisfies the same test suite as FileResultsStore (parametrize existing tests over store type) |
| Paper trader position tracking | Assert positions update correctly after signal change; assert cash decreases by cost_bps on trade; assert frozen model is used (no retrain) |
| DataFeed DAG enforcement | Assert `paper_trader.py` imports only from `live/feed.py` and not from `data/` loaders directly |
| Registry completeness v3 | Assert all new transforms, allocators, feeds are importable and instantiable |
| DAG v3 | Assert `dashboard/` does not import from `strategy/`, `data/`, or `features/`; assert `live/` does not import from `data/` |

### New Integration Tests

| Test | Description |
|------|-------------|
| Sentiment end-to-end | Run `run --config` with `sentiment_news: {}` on synthetic data; assert feature columns present |
| Full statistical pipeline | Run `analyze mcs` on 3 synthetic models; assert MCS eliminates the worst one |
| SQLiteResultsStore round-trip | Write 3 experiments via runner; assert SQLiteResultsStore retrieves them with correct filtering |
| Paper trading smoke test | Start paper trader on synthetic feed for 5 steps; assert log file written and positions non-empty |
| Streamlit import smoke | Assert `dashboard/app.py` is importable without launching a server |
| CVaR portfolio smoke | Run `strategy portfolio --allocator cvar` on 3 synthetic tickers; assert output CVaR ≤ equal-weight CVaR |

---

## 11. Resolved Design Decisions

These questions were identified during v3 design and are resolved here. v4 depends on these answers being stable.

| # | Question | Resolution |
|---|----------|------------|
| 1 | **Sentiment API default** | Loughran-McDonald lexicon is the default when no `NEWSAPI_KEY` is set. newsapi is used when the key is present. Both are always supported; the lexicon never requires network access |
| 2 | **MCS implementation** | Bootstrap-based MCS (Hansen et al. 2011). More rigorous than sequential pairwise DM elimination. Acceptable performance at N ≤ 24 models (v3 universe). If N grows beyond 50 in a future version, revisit |
| 3 | **Default result store** | `SQLiteResultsStore` is the v3 default. Zero external dependencies, fast filter queries, no server required. MLflow is opt-in via `results_store: mlflow` in config |
| 4 | **DL hardware detection** | Silent skip on macOS ARM with a clear log message. An explicit error would break `run-all` on mixed hardware — the silent skip with a logged reason is the better failure mode |
| 5 | **Cross-sectional model data format** | Separate runner function (`cv_mode: cross_sectional`). Stacking into the existing runner would couple it to a fixed universe and break the single-ticker path |
| 6 | **Paper trading retrain** | **Frozen model.** The model fitted by the last `run` experiment is used unchanged throughout the paper trading session. Rolling live retrain is v5 scope — it requires a separate degradation-detection framework before it is safe to use with capital |
| 7 | **GAN augmentation stability** | **KS-test gate.** Before synthetic samples are used in a fold, a KS test compares the generated distribution to real training data. If p < 0.05, synthetic samples are discarded and the fold uses real data only. Warning is logged |
| 8 | **Earnings data availability** | **yfinance post-2018 only.** For dates before 2018, earnings features (`days_to_earnings`, `earnings_surprise_lag1`, `earnings_revision_30d`) are set to NaN. AlphaVantage fallback activates only when `ALPHAVANTAGE_KEY` is explicitly set — never on the free tier |

---

## 12. Scope Boundaries

### v3.0 — this document

- Statistical validation: DM test in compare, Model Confidence Set, Benjamini-Hochberg FDR
- Sentiment transform (newsapi + Loughran-McDonald fallback), earnings transform, Fama-French transform
- GAN augmentation transform (opt-in, with KS-test gate)
- LSTM, Transformer, TCN re-enabled on CUDA Linux
- BayesianLASSO, PartiallyProtectedLASSO models
- Cross-sectional LambdaMART ranking model
- `SQLiteResultsStore` (default) and `MLflowResultsStore` (opt-in)
- Streamlit dashboard (4 pages, read-only)
- Paper trading loop (`yfinance_live` polling feed for testing; Polygon websocket feed for production)
- CVaR and factor-risk allocators
- Multi-period rebalancing in Backtester
- Distributed walk-forward via Dask/Ray (opt-in)
- Polygon and AlphaVantage data loaders
- `analyze`, `live`, `results-store` CLI command groups; `serve` command

### v4.0 — see `PRDv4.md`

- Real brokerage execution: Alpaca (primary) and IBKR (optional)
- `BaseBroker` ABC, `OrderManager`, `PositionManager`, `RiskGuard`, `ExecutionLoop`, `ExecutionStore`
- Execution Monitor dashboard page (5th Streamlit page)
- `execution live`, `execution status`, `execution history`, `execution reconcile` CLI commands

### v5.0 — Future Candidates

- Rolling live model retrain with degradation detection
- Options / derivatives pricing signals
- Fixed income + cross-asset regime models
- Agent-based RL trading
- Federated learning
- Intraday / HF data
- Earnings call NLP
- Knowledge graph features
- Automated feature discovery

---

## 13. Updated Glossary

| Term | Definition |
|------|------------|
| **Model Confidence Set (MCS)** | The set of models that cannot be statistically eliminated as inferior at a given confidence level. Implements Hansen, Lunde, Nason (2011) bootstrap procedure. The v3 replacement for simple "best model" selection when the universe is large |
| **Diebold-Mariano (DM) test** | Statistical test of equal predictive accuracy between two models (Harvey-Leybourne-Newbold corrected for small samples). Available in v2 as a function; wired into `compare` in v3 |
| **FDR (False Discovery Rate)** | The expected fraction of rejected null hypotheses that are true nulls. Controlled via Benjamini-Hochberg correction when running pairwise DM tests across N models |
| **CVaR (Conditional Value at Risk)** | Expected loss in the worst α% of scenarios (also called Expected Shortfall). Used by the `cvar` allocator to construct downside-risk-aware portfolios |
| **Factor Risk Model** | Covariance estimated as BΛBᵀ + Δ using Fama-French factor loadings B and factor covariance Λ. More stable than sample covariance for large universes |
| **Cross-sectional model** | A model that predicts the *rank* of a ticker's return within a universe on a given date, rather than the return level. Evaluated by cross-sectional Rank IC and NDCG |
| **LambdaMART** | Gradient boosted trees with a learning-to-rank loss function. Optimises NDCG directly, making it the standard cross-sectional ranking model |
| **GAN augmentation** | Generative Adversarial Network trained on the training fold to produce synthetic return samples. Subject to a KS-test gate before use — synthetic samples are discarded if their distribution differs significantly from real training data |
| **BaseDataFeed** | Abstract interface in `live/feed.py` for live data access. All `live/` code uses this interface only — never `data/` loaders directly. Implementations: `yfinance_live` (polling, 15-minute delayed, testing only), `polygon` (websocket, production) |
| **PaperTrader** | The v3 live simulation layer. Consumes a `StrategySpec` and a `BaseDataFeed`; updates positions and tracks P&L without real capital. Uses a frozen model — no live retrain |
| **SQLiteResultsStore** | v3 default implementation of `ResultsStore`. Backed by a single SQLite database. Zero-dependency, fast for filter queries |
| **Sentiment score** | Rolling mean of daily headline sentiment for a ticker (Loughran-McDonald lexicon default, or FinBERT via newsapi). Shifted by 1 day to prevent look-ahead |
| **Earnings surprise** | Normalised difference between reported EPS and consensus estimate: `(actual - estimate) / abs(estimate)`. Available for post-2018 data only via yfinance |
| **Publication lag** | The delay between when an economic data point is measured and when it is released. Enforced in v2 for FRED series; extended in v3 to earnings revisions and factor data |
| **Frozen model** | The model fitted by the last `run` experiment, used unchanged during paper trading. No live retrain occurs in v3 |