## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current

## Research-Driven Development

All model, feature, and evaluation design decisions must be grounded in the research papers in `AI_Instructions/research/`. Before adding or changing any modelling component, check whether the literature supports the approach. Use the following paper inventory as context:

**Foundational Papers:**
- `fpaper_1` — Stambaugh (1999): Predictive regression bias with lagged stochastic regressors. Justifies using returns (not prices) as the target.
- `fpaper_2` — Goyal & Welch (2008): Most predictors fail OOS. The `HistoricalMean` prevailing-mean baseline is the canonical hard-to-beat benchmark; OOS R² vs. this mean is the primary evaluation metric.
- `fpaper_3` — Campbell & Thompson (2008): Sign restrictions on forecasts improve OOS R². Small but economically meaningful OOS predictability is achievable.
- `fpaper_4` — Sarno & Valente (2003): Regime-switching models outperform on market timing; supports `HMM` and `MarkovSwitching`.

**Research Papers:**
- `rpaper_1` — Turgay (2025): Expanding-window and rolling-window OOS are the correct evaluation paradigm; static holdout is insufficient. Forecast combinations outperform individual models.
- `rpaper_2` — Rossi (2018): Boosted Regression Trees (BRT/XGBoost) handle high-dimensional conditioning without overfitting; volatility forecasting alongside returns improves portfolio allocation.
- `rpaper_3` — Huang (2024): Bayesian LASSO and high-dimensional regularisation; Partially Protected LASSO for theory-informed variable selection.
- `rpaper_4` — Alhomadi (2021): Feature importance analysis across ML models; confirms difficulty of OOS prediction.
- `rpaper_5` — Wang (2025): Rank IC (Spearman correlation) and maximum drawdown as evaluation metrics; sentiment factors improve prediction.
- `rpaper_6/7` — Cheng et al. (2025): Nonparametric predictive regression with locally stationary predictors; time-varying predictability.
- `rpaper_8` — Mistol & Möhler (2023): Neural networks + ensemble combinations best overall; SHAP values for cross-model interpretability; Diebold-Mariano test for significance; OOS R² standard metric.
- `rpaper_9` — Nhon et al. (2025): DEA efficiency scores + automatic feature engineering improve prediction.
- `rpaper_10` — Vishwas et al. (2025): Hybrid AI models review; LSTM, Transformer, GAN-based data augmentation, sentiment analysis.

## Scalability & Architecture Rules

These are non-negotiable. The project will grow to v3+ — every shortcut taken now creates refactor debt under pressure.

### Registry pattern everywhere
Every extensible category uses a decorator-based registry. Adding a new instance of anything = one class + one decorator + zero changes elsewhere.
- Models: `@register("Name")` in `models/base_model.py` — already in place
- Data loaders: `@register_loader("name")` — add in v2
- Feature transforms: `@register_transform("name")` — add in v2
- Metrics: `_METRIC_FNS` dict with `task` metadata — extend as needed
- Allocation strategies: `@register_allocator("name")` — add in v2
- Cost models: `@register_cost_model("name")` — add in v2

### Strict dependency DAG
No layer may import from a layer above it or from a sibling layer:
- `data/` → nothing from this project
- `features/` → `data/` only
- `models/` → nothing from this project
- `evaluation/` → nothing from this project
- `experiments/` → `data/`, `features/`, `models/`, `evaluation/`
- `strategy/` → `experiments/`, `evaluation/` only (never directly `data/` or `features/`)
- `run.py` → everything (only the CLI converges all layers)

Before adding any import, verify it does not violate this DAG.

### Result schema stability
- `metrics.json` gains `schema_version` in v2. Never rename or remove existing keys — only add new ones.
- Results path encodes all experiment dimensions: `results/<TICKER>/<MODEL>/h<HORIZON>/<TARGET>/<exp_id>/`
- The `strategy/` layer reads results only through `ResultsStore` interface — never raw file paths.

### Interface over concrete references
- `strategy/` code must accept `ResultsStore` (interface), `BaseAllocator` (interface), `BaseCostModel` (interface) — not concrete implementations. This allows swapping `FileResultsStore` → `SQLiteResultsStore` in v3 without touching strategy logic.
- `Backtester` accepts arrays (`y_true`, `y_pred`, `dates`) — not file paths. The CLI is the only layer that does file I/O.

### Composable feature pipeline
Features are a list of registered `FeatureTransform` objects applied in sequence. Adding sentiment, earnings, or alternative data in v3 = one new `@register_transform` class. No changes to `build_features()` core logic.

## README Maintenance

After every session that adds features, changes the model list, adds CLI commands, or changes the config schema, update `README.md` to reflect the current state. The README is the single source of truth for users. Keep the Available Models table, CLI Reference, Metrics table, and Config Schema sections current.

## Articles/techniques to expore

- https://arxiv.org/abs/2412.20138    
- https://github.com/TauricResearch/TradingAgents
- https://tradingagents-ai.github.io/   
