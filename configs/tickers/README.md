# Ticker-Specific Config Overrides

Place YAML files here to override specific fields of a base config for a particular ticker.

## How it works

When you run `run-all --tickers AAPL,TSLA,MSFT`, the runner:

1. Loads each base config from `configs/*.yaml`
2. Checks for `configs/tickers/<TICKER>/<same_filename>.yaml`
3. If found, **deep-merges** the override on top of the base (override wins on any key conflict)
4. Runs the merged config with the ticker injected

## Override file (partial YAML — only specify what changes)

`configs/tickers/TSLA/xgboost_aapl.yaml`:
```yaml
model:
  params:
    n_estimators: 300   # TSLA is more volatile — more trees
    max_depth: 8
```

Everything not mentioned inherits from the base config.

## Ticker-only config (standalone — does NOT merge with a base)

A file whose name does NOT match any base config is treated as a standalone
experiment that only runs for that ticker:

`configs/tickers/TSLA/xgboost_tsla_aggressive.yaml` — runs only for TSLA, full config required.

## Quick reference

| File location | Behaviour |
|---|---|
| `configs/xgboost_aapl.yaml` | Base — runs for all tickers unless overridden |
| `configs/tickers/TSLA/xgboost_aapl.yaml` | Merges with base for TSLA only |
| `configs/tickers/TSLA/xgboost_tsla_v2.yaml` | Standalone TSLA-only config |
