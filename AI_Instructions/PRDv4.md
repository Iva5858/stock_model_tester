# PRD: Stock Prediction Framework — Version 4.0

**Version:** 4.0  
**Status:** Draft  
**Prerequisite:** v3.0 complete (see `PRDv3.md`)  
**Last Updated:** June 2026

---

## 1. Overview

### 1.1 What v3 Built

v3.0 elevated the framework from a strategy research system into a statistically rigorous, production-ready research platform:

- **Statistical validation layer** (`analysis/`) — Diebold-Mariano tests, Model Confidence Sets, Benjamini-Hochberg FDR correction wired into `compare`
- **Signal enrichment** — sentiment (`sentiment_news`), earnings calendar (`earnings`), Fama-French 5-factor (`fama_french`) transforms
- **Deep learning** — LSTM, Transformer, TCN re-enabled on CUDA Linux; GAN augmentation (opt-in)
- **Cross-sectional models** — LambdaMART ranking; Bayesian LASSO and PartiallyProtectedLASSO
- **Production result stores** — `SQLiteResultsStore`, `MLflowResultsStore`, `FileResultsStore` with index cache
- **Streamlit dashboard** — 4-page interactive read-only UI replacing static HTML reports
- **Paper trading loop** (`live/`) — `PaperTrader` consuming `StrategySpec`, tracking P&L against a live data feed without real capital
- **Advanced portfolio construction** — CVaR allocator, factor risk model allocator, multi-period rebalancing
- **Distributed walk-forward** — Dask/Ray opt-in for 50–500 ticker universes

The framework answers: *"Is my strategy statistically significant — and does it hold up in a realistic simulated live environment?"*

### 1.2 What v4 Must Answer

v3 paper trading proved the system works end-to-end in simulation. v4 answers the operational question:

> **"Can I execute this strategy with real capital, through a real broker, with the same codebase that developed and validated it?"**

This requires one new layer on top of v3:

**Live brokerage execution** — a thin adapter layer that replaces `PaperTrader`'s simulated fill logic with real broker API calls, while reusing every upstream component unchanged: `StrategySpec`, `ModelSelector`, `PortfolioConstructor`, `DataFeed`, risk controls, and the dashboard.

The design constraint is strict: **v4 adds `execution/` as a new package. No v3 code changes.**

### 1.3 Prerequisites Before Using v4

These are gates, not suggestions. Running v4 with real capital before completing them is how people lose money quickly:

1. **v3 paper trading must have run for a minimum of 60 trading days** on the target ticker universe
2. **Net Sharpe (after costs at `cost_bps=10`) must be positive** on the paper trading equity curve
3. **The strategy must be in the Model Confidence Set** (`require_mcs: true` in the strategy config)
4. **Open Question resolution from v3:**
   - Earnings data: use `yfinance.Ticker.earnings_dates` for post-2018 data only; degrade gracefully for earlier history. Do not use AlphaVantage as fallback unless a key is configured — the free tier rate limits will silently corrupt walk-forward folds.
   - PaperTrader retrain: use **frozen model** (the last model fitted by `run`). Rolling live retrain is excluded from v4 scope — it introduces lookahead risk and operational complexity that is not justified before a system has a track record.
   - GAN augmentation: add a KS-test validation gate before using synthetic samples. If the KS test p-value < 0.05 (generated distribution significantly differs from real), log a warning and fall back to real data only. This resolves v3 Open Question #7.

### 1.4 Goals

| Goal | Description |
|------|-------------|
| **Broker abstraction** | `@register_broker` decorator; `BaseBroker` ABC with a clean interface. Adding a new broker never touches existing code |
| **Alpaca integration** | First broker implementation. Paper and live via base URL swap. Identical code path — the only difference is an environment variable |
| **IBKR integration** | Second broker implementation via `ib_insync`. Requires IB Gateway running separately. Optional dependency |
| **Order management** | Market, limit, and bracket orders. Position reconciliation between what the model expects and what the broker holds |
| **Risk guard** | Hard stops enforced before every order: max position weight, daily loss kill switch, minimum OOS R² threshold. Non-negotiable — cannot be disabled |
| **Execution logging** | Every order, fill, and rejection logged to `execution/logs/` in structured JSONL format. Fully auditable |
| **Dashboard integration** | The v3 Streamlit dashboard gains a 5th page: live execution monitor. No dashboard architecture changes — reads from `execution/logs/` via a new `ExecutionStore` interface |
| **Regulatory awareness** | PDT rule detection for US-resident accounts. Position sizing that respects broker margin requirements |

### 1.5 Non-Goals (v4)

- Intraday or high-frequency execution (daily signal, end-of-day orders only — consistent with v2/v3 model horizons)
- Options, futures, or derivatives
- Multi-broker order routing or smart order routing
- Automated tax lot selection or wash-sale tracking
- Sub-second order management or co-location
- Cryptocurrency (equity focus maintained from v1)
- Rolling live model retrain (v5 candidate — requires a separate validation framework)

---

## 2. Architecture

### 2.1 v3 Architectural Contracts That v4 Must Honour

The following are frozen — v4 does not modify them:

- `BaseModel` / `@register` — unchanged
- `BaseLoader` / `@register_loader` — unchanged
- `FeatureTransform` / `@register_transform` — unchanged
- `BaseAllocator` / `@register_allocator` — unchanged
- `ResultsStore` ABC — unchanged
- `BaseDataFeed` / `@register_feed` — unchanged; `execution/` consumes feeds, does not redefine them
- `StrategySpec` schema — v4 adds `execution_config` block; all existing fields unchanged
- `metrics.json` schema — `schema_version: 4` added; no existing keys renamed or removed

### 2.2 Updated Dependency DAG

```
                          run.py (CLI)
                               │
    ┌──────────────────────────┼──────────────────────────┐
    ▼                          ▼                          ▼
execution/   ←NEW         dashboard/                   live/
    │                          │                          │
    │         ┌────────────────┘                          │
    ▼         ▼                                           ▼
strategy/ ←──────────────────────────────────────────────┘
    │
    └──── evaluation/ ──── models/ ──── features/ ──── data/
```

Rules:
- `execution/` imports from `strategy/` (reads `StrategySpec`) and `live/` (consumes `DataFeed`) only
- `execution/` never imports from `data/`, `features/`, `models/`, or `evaluation/` directly
- `dashboard/` gains a read path to `execution/logs/` via `ExecutionStore` only — no direct broker API calls from the dashboard

### 2.3 New Directory Structure

Only new files shown. All v3 files unchanged.

```
stock_predictor/
│
├── execution/                        ← NEW package
│   ├── __init__.py
│   ├── base_broker.py                ← BaseBroker ABC + @register_broker
│   ├── brokers/
│   │   ├── __init__.py
│   │   ├── alpaca_broker.py          ← @register_broker("alpaca")
│   │   └── ibkr_broker.py            ← @register_broker("ibkr") — optional dep
│   ├── order_manager.py              ← order construction, submission, fill tracking
│   ├── position_manager.py           ← reconcile broker positions vs model expectations
│   ├── risk_guard.py                 ← hard stops: max weight, daily loss, OOS R² gate
│   ├── execution_loop.py             ← daily scheduler: signal → size → risk check → submit
│   ├── execution_store.py            ← ExecutionStore: reads/writes execution/logs/
│   └── logs/                         ← JSONL execution logs (gitignored)
│       └── .gitkeep
│
├── dashboard/
│   └── pages/
│       └── execution_monitor.py      ← NEW 5th page; reads ExecutionStore only
│
└── configs/
    └── execution/
        └── alpaca_example.yaml       ← example execution config
```

---

## 3. Functional Requirements

### 3.1 Broker Abstraction (`execution/base_broker.py`)

The `BaseBroker` ABC defines the interface every broker implementation must satisfy.

| ID | Requirement |
|----|-------------|
| BR-01 | `BaseBroker` defines: `connect()`, `disconnect()`, `get_positions() -> dict[str, float]`, `get_cash() -> float`, `get_account_value() -> float`, `submit_order(order: Order) -> OrderResult`, `cancel_order(order_id: str) -> bool`, `get_order_status(order_id: str) -> OrderStatus` |
| BR-02 | `@register_broker("name")` decorator follows the identical pattern to `@register`, `@register_loader`, etc. Adding a new broker is a single decorated class — no changes to `execution_loop.py`, `order_manager.py`, or any other file |
| BR-03 | `Order` is a dataclass: `ticker`, `side` (`buy`/`sell`), `order_type` (`market`/`limit`/`bracket`), `qty`, `limit_price` (optional), `stop_price` (optional), `time_in_force` (`day`/`gtc`) |
| BR-04 | `OrderResult` is a dataclass: `order_id`, `status`, `filled_qty`, `avg_fill_price`, `timestamp`, `error_message` (optional) |
| BR-05 | All broker implementations are **isolated** — a failure in one broker's API call raises `BrokerError`, which `execution_loop.py` catches and logs. It never propagates to the strategy layer |

### 3.2 Alpaca Broker (`execution/brokers/alpaca_broker.py`)

Alpaca is the primary broker implementation. It is chosen because: commission-free, REST + WebSocket API, paper and live environments are identical except for base URL, no minimum balance for paper, well-maintained Python SDK.

| ID | Requirement |
|----|-------------|
| AL-01 | `@register_broker("alpaca")` wraps the `alpaca-trade-api` SDK. Required env vars: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`. Mode (paper/live) controlled by `ALPACA_PAPER=true/false` env var — **never** a code change |
| AL-02 | Paper mode base URL: `https://paper-api.alpaca.markets`. Live mode: `https://api.alpaca.markets`. The broker reads `ALPACA_PAPER` at `connect()` time and sets the URL accordingly |
| AL-03 | `submit_order()` submits a market order by default (consistent with the v2/v3 model design: end-of-day signal, market-on-open execution next day). Limit and bracket orders available via `order_type` field |
| AL-04 | `get_positions()` returns a dict mapping ticker symbol to current net position in shares (positive = long, negative = short). Fractional shares are supported by Alpaca and returned as floats |
| AL-05 | Alpaca enforces the US Pattern Day Trader rule for accounts under $25,000. `alpaca_broker.py` detects PDT-flagged accounts at `connect()` and logs a warning. `risk_guard.py` uses this flag to limit same-day round-trips |
| AL-06 | Rate limits: Alpaca free tier allows 200 requests/minute. `alpaca_broker.py` implements exponential backoff with jitter on 429 responses. Never raises on rate limit — retries up to 3 times then raises `BrokerError` |
| AL-07 | `alpaca_broker.py` is available without `ibkr` dependencies. Install: `pip install alpaca-trade-api` |

### 3.3 IBKR Broker (`execution/brokers/ibkr_broker.py`)

IBKR is the second implementation, suitable for non-US residents (no PDT rule, broader market access) and accounts requiring more sophisticated order types.

| ID | Requirement |
|----|-------------|
| IB-01 | `@register_broker("ibkr")` wraps `ib_insync`. Optional dependency: `pip install ib_insync`. If not installed, `@register_broker("ibkr")` is skipped with a log message — identical to how DL models are skipped on macOS ARM |
| IB-02 | Requires IB Gateway or TWS running locally. Connection params: `host` (default `127.0.0.1`), `port` (default `7497` for TWS paper, `7496` for TWS live, `4002` for Gateway paper, `4001` for Gateway live), `client_id` (default `1`). All configurable via `execution_config` in `StrategySpec` |
| IB-03 | IBKR does not have a paper/live URL swap — paper and live use different ports. The `ibkr_broker.py` selects port based on `IBKR_PAPER=true/false` env var |
| IB-04 | `submit_order()` submits via `ib_insync.MarketOrder` by default. `ib_insync` is synchronous-async — `ibkr_broker.py` wraps calls with `ib.run()` to make them blocking from `execution_loop.py`'s perspective |
| IB-05 | IBKR requires contract specification for each ticker. `ibkr_broker.py` resolves `STK`/`SMART`/`USD` contracts automatically for US equity tickers. Non-US tickers require explicit `exchange` and `currency` in config |
| IB-06 | `ibkr_broker.py` reconnects automatically if the IB Gateway connection drops. Reconnect attempts: 3, with 30-second delays. After 3 failures, raises `BrokerError` and halts the execution loop for that day |

### 3.4 Order Manager (`execution/order_manager.py`)

| ID | Requirement |
|----|-------------|
| OM-01 | `OrderManager.build_orders(target_weights: dict[str, float], current_positions: dict[str, float], account_value: float) -> list[Order]` computes the set of orders needed to move from current positions to target weights |
| OM-02 | Target weights come from `PortfolioConstructor` (v3). `OrderManager` converts weights to share quantities using `account_value` and current market prices fetched from the `DataFeed` |
| OM-03 | Share quantities are rounded to whole shares by default. Alpaca supports fractional shares — enable via `fractional_shares: true` in execution config |
| OM-04 | Orders are sorted: sells before buys. This ensures cash is available before new positions are opened, avoiding margin calls on cash accounts |
| OM-05 | Minimum order threshold: orders below `min_order_value` (default $10) are skipped and logged. Prevents excessive small orders from accumulating transaction costs |
| OM-06 | `OrderManager` never submits orders directly — it returns `list[Order]` to `execution_loop.py`, which passes them through `RiskGuard` first |

### 3.5 Position Manager (`execution/position_manager.py`)

| ID | Requirement |
|----|-------------|
| PM-01 | `PositionManager.reconcile(model_positions: dict[str, float], broker_positions: dict[str, float]) -> ReconciliationReport` compares what the model expects to hold against what the broker actually holds |
| PM-02 | `ReconciliationReport` contains: `matched`, `model_only` (positions the model expects but broker doesn't hold), `broker_only` (positions the broker holds that the model doesn't expect), `diverged` (positions held on both sides but with quantity differences exceeding `reconcile_threshold`) |
| PM-03 | On divergence, `execution_loop.py` logs the full report and **does not trade**. A human must resolve the divergence before the next execution cycle. This is intentional — silent drift correction is more dangerous than halting |
| PM-04 | `reconcile_threshold` defaults to 5% of position value. Configurable in execution config |
| PM-05 | `PositionManager` writes reconciliation reports to `execution/logs/reconciliation_<date>.jsonl` |

### 3.6 Risk Guard (`execution/risk_guard.py`)

The risk guard is the last line of defence before any order reaches the broker. It cannot be disabled via config — only by modifying the source code, which is intentional friction.

| ID | Requirement |
|----|-------------|
| RG-01 | **Max position weight** — no single ticker may exceed `max_position_weight` (default 20%) of account value. Orders that would breach this are reduced to the maximum permitted size, not rejected entirely |
| RG-02 | **Daily loss kill switch** — if the account's unrealised + realised P&L for the current trading day falls below `daily_loss_limit` (default -2% of account value), all pending orders are cancelled and no new orders are submitted for the remainder of that day. The event is logged and a warning is printed to stdout |
| RG-03 | **OOS R² gate** — before executing any signal from a model, `RiskGuard` checks the model's most recent `oos_r2` from `ResultsStore`. If `oos_r2 < min_oos_r2` (default 0.0), the signal is suppressed and logged. The system holds current positions rather than trading on a degraded model |
| RG-04 | **PDT protection** (Alpaca only) — if the account is PDT-flagged, `RiskGuard` blocks any order that would constitute a same-day round-trip (buy and sell the same ticker on the same day) |
| RG-05 | **Market hours check** — `RiskGuard` rejects orders submitted outside market hours. For Alpaca, this means 09:30–16:00 ET. Orders submitted via the scheduler outside these hours are queued for market-on-open, not rejected |
| RG-06 | Every `RiskGuard` decision (permit, reduce, block) is logged to `execution/logs/risk_<date>.jsonl` with the reason |

### 3.7 Execution Loop (`execution/execution_loop.py`)

| ID | Requirement |
|----|-------------|
| EL-01 | The execution loop runs **once per trading day**, triggered either by a scheduler (cron, APScheduler) or manually via CLI. It is not a continuous process — it is a daily batch job |
| EL-02 | Execution sequence (strict order, no skipping): (1) connect to broker, (2) fetch account state, (3) reconcile positions — halt if divergence detected, (4) load `StrategySpec` from `ResultsStore`, (5) fetch latest features via `DataFeed`, (6) generate signal via model `predict()`, (7) construct target weights via `PortfolioConstructor`, (8) build orders via `OrderManager`, (9) check orders through `RiskGuard`, (10) submit approved orders, (11) log all results, (12) disconnect |
| EL-03 | Each step is wrapped in a try/except. A failure at any step halts execution for that day and logs the error. The system never partially executes a rebalance — it is all-or-nothing per day |
| EL-04 | `execution_loop.py` is stateless between runs. All state (positions, P&L, trade history) lives in `execution/logs/` and the broker's account. The loop reconstructs everything it needs from these sources at each run |
| EL-05 | Dry-run mode: `python run.py execution live --dry-run` executes all steps up to and including `RiskGuard`, prints the orders that would be submitted, and stops. Nothing reaches the broker |

### 3.8 Execution Store (`execution/execution_store.py`)

| ID | Requirement |
|----|-------------|
| ES-01 | `ExecutionStore` reads and writes structured JSONL logs in `execution/logs/`. It is the only way any other component (dashboard, CLI) accesses execution data — no direct file reads |
| ES-02 | Log files: `orders_<date>.jsonl` (submitted orders and fills), `risk_<date>.jsonl` (risk guard decisions), `reconciliation_<date>.jsonl` (position reconciliation reports), `loop_<date>.jsonl` (execution loop run summary) |
| ES-03 | `ExecutionStore.get_equity_history() -> pd.DataFrame` returns a time series of account value, compatible with the v3 dashboard's equity curve chart |
| ES-04 | `ExecutionStore.get_trade_log() -> pd.DataFrame` returns all fills with ticker, side, qty, fill price, timestamp, and gross/net P&L |
| ES-05 | `ExecutionStore` is read-only from the dashboard's perspective. Only `execution_loop.py` writes to it |

### 3.9 Dashboard: Execution Monitor Page (`dashboard/pages/execution_monitor.py`)

| ID | Requirement |
|----|-------------|
| DM-01 | 5th Streamlit page: **Execution Monitor**. Reads exclusively from `ExecutionStore` — no broker API calls from the dashboard |
| DM-02 | Displays: current positions (ticker, shares, market value, unrealised P&L), today's orders (submitted, filled, rejected), account equity curve since live trading started, daily P&L bar chart |
| DM-03 | Risk guard event log: table of all risk interventions with timestamp, type, and reason |
| DM-04 | Reconciliation status: green if last reconciliation passed, red if divergence was detected, with the full report expandable |
| DM-05 | The page auto-refreshes every 60 seconds during market hours. Outside market hours it displays a static snapshot of the last close |

### 3.10 CLI Commands

All v3 commands unchanged. New v4 commands:

| Command | Description |
|---------|-------------|
| `python run.py execution live --broker alpaca` | Run the execution loop once (live mode) |
| `python run.py execution live --broker alpaca --dry-run` | Simulate execution — print orders, submit nothing |
| `python run.py execution live --broker ibkr` | Run with IBKR broker |
| `python run.py execution status --broker alpaca` | Print current positions, today's P&L, last reconciliation result |
| `python run.py execution history` | Print trade log from `ExecutionStore` |
| `python run.py execution reconcile --broker alpaca` | Run position reconciliation manually, print report |

---

## 4. Configuration Schema (v4 extensions)

All v2, v3 configs run unchanged. v4 adds an optional `execution_config` block to `StrategySpec`:

```yaml
schema_version: 4

# All existing v3 strategy config fields unchanged
strategy:
  name: tech_portfolio_v4
tickers: [AAPL, MSFT, GOOGL]
model_selection:
  criterion: sharpe
  require_mcs: true
portfolio:
  allocator: cvar
  max_position_weight: 0.20
  long_only: false
  rebalance_frequency: 21
backtest:
  cost_model: fixed_bps
  cost_bps: 10
  initial_capital: 100000
results_store: sqlite

# NEW: execution block — absent = no live execution
execution_config:
  broker: alpaca                  # alpaca | ibkr
  fractional_shares: false        # alpaca only
  min_order_value: 10             # USD — skip orders below this
  reconcile_threshold: 0.05       # 5% position value divergence triggers halt

  risk:
    max_position_weight: 0.20     # overrides portfolio.max_position_weight for live execution
    daily_loss_limit: -0.02       # -2% of account value
    min_oos_r2: 0.0               # suppress signal if OOS R² below this
    pdt_protection: true          # alpaca only; ignored for ibkr

  schedule:
    time: "15:45"                 # ET — 15 minutes before close; generates signal for MOO next day
    timezone: "America/New_York"

  # IBKR-specific (ignored for alpaca)
  ibkr:
    host: "127.0.0.1"
    port: 4002                    # 4002 = IB Gateway paper; 4001 = IB Gateway live
    client_id: 1
```

### Environment Variables

| Variable | Broker | Description |
|----------|--------|-------------|
| `ALPACA_API_KEY` | Alpaca | API key |
| `ALPACA_SECRET_KEY` | Alpaca | Secret key |
| `ALPACA_PAPER` | Alpaca | `true` = paper trading, `false` = live. Default: `true` |
| `IBKR_PAPER` | IBKR | `true` = paper port, `false` = live port. Default: `true` |

**Paper mode is the default for both brokers.** Switching to live requires explicitly setting the env var to `false`. This is intentional — accidental live trading is a worse failure mode than accidental paper trading.

---

## 5. Tech Stack Additions

| Addition | Rationale |
|----------|-----------|
| `alpaca-trade-api>=3.0` | Alpaca REST + WebSocket SDK |
| `ib_insync>=0.9` | IBKR TWS/Gateway Python wrapper (optional) |
| `apscheduler>=3.10` | Daily execution loop scheduling (optional — can use cron instead) |

All three are optional at import time. `alpaca-trade-api` is only imported inside `alpaca_broker.py`. `ib_insync` is only imported inside `ibkr_broker.py`. Neither is a hard dependency — the framework runs without them.

---

## 6. Testing Plan

### Existing (all must continue passing)

All 89 v2 tests and all v3 tests remain unchanged and green.

### New Unit Tests

| Test | Description |
|------|-------------|
| `test_order_manager.py` | Assert `build_orders()` produces correct buy/sell quantities for known weight changes; assert sells before buys; assert min order value threshold respected |
| `test_position_manager.py` | Assert `reconcile()` correctly identifies matched, model-only, broker-only, and diverged positions for known inputs |
| `test_risk_guard.py` | Assert max position weight reduces (not rejects) oversized orders; assert daily loss kill switch fires at correct threshold; assert OOS R² gate suppresses signal below threshold; assert PDT protection blocks same-day round-trips |
| `test_execution_store.py` | Assert JSONL logs written correctly; assert `get_equity_history()` and `get_trade_log()` return correct DataFrames for known log files |
| `test_broker_registry.py` | Assert `@register_broker` registers correctly; assert uninstalled optional brokers (ibkr without ib_insync) are skipped without error |
| `test_alpaca_broker_paper.py` | Mock Alpaca SDK; assert paper URL used when `ALPACA_PAPER=true`; assert live URL used when `ALPACA_PAPER=false`; assert exponential backoff on 429 |
| `test_dag_v4.py` | Assert `execution/` does not import from `data/`, `features/`, `models/`, or `evaluation/` directly — DAG enforcement |

### New Integration Tests

| Test | Description |
|------|-------------|
| `test_execution_loop_dry_run.py` | Run `execution live --dry-run` with mock broker and synthetic `StrategySpec`; assert all 12 steps execute; assert no orders submitted to broker |
| `test_reconciliation_halt.py` | Inject a position divergence; assert execution loop halts at step 3 and logs the reconciliation report |
| `test_risk_guard_daily_loss.py` | Simulate account value drop below `daily_loss_limit`; assert all orders are cancelled and loop halts for the day |
| `test_alpaca_end_to_end_paper.py` | Full loop against Alpaca paper API (requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in test environment); assert order submitted, fill returned, log written |

---

## 7. Resolved v3 Open Questions

These were flagged as unresolved in v3. v4 depends on specific answers:

| # | v3 Question | v4 Resolution |
|---|-------------|---------------|
| 6 | PaperTrader retrain frequency | **Frozen model.** The model fitted by the last `run` experiment is used. Rolling live retrain is v5 scope. |
| 7 | GAN augmentation stability | **KS-test gate added.** If generated samples fail KS test (p < 0.05 vs real training data), fall back to real data and log a warning. |
| 8 | Earnings data availability | **yfinance post-2018 only.** For dates before 2018, the earnings transform degrades to `days_to_earnings=NaN` and `earnings_surprise_lag1=NaN`. AlphaVantage fallback only if key is configured — never silently on free tier. |

---

## 8. Scope Boundaries

### v4.0 — this document

- `BaseBroker` ABC and `@register_broker` decorator
- Alpaca broker (paper + live via env var)
- IBKR broker (optional dependency, IB Gateway required)
- `OrderManager`, `PositionManager`, `RiskGuard`, `ExecutionLoop`, `ExecutionStore`
- Execution Monitor dashboard page (5th Streamlit page)
- `execution live`, `execution status`, `execution history`, `execution reconcile` CLI commands
- Resolution of v3 Open Questions 6, 7, 8

### v5.0 — Future Candidates

- **Rolling live retrain** — model refits on a schedule using live data. Requires a separate validation framework to detect model degradation before capital is at risk
- **Intraday execution** — minute-bar data, intraday signals, multiple orders per day
- **Options and derivatives** — implied volatility surface as feature, options order types
- **Multi-broker routing** — split orders across brokers for liquidity or cost reasons
- **Agent-based RL trading** — reinforcement learning policy trained against the backtester as an environment
- **Automated tax lot selection** — wash-sale rule awareness, FIFO/LIFO/specific lot selection
- **Earnings call NLP** — transcript-level sentiment from earnings calls
- **Federated learning** — cross-institutional model training without sharing raw data

---

## 9. Glossary Additions

| Term | Definition |
|------|------------|
| **BaseBroker** | Abstract interface all broker implementations must satisfy. Methods: `connect`, `disconnect`, `get_positions`, `get_cash`, `get_account_value`, `submit_order`, `cancel_order`, `get_order_status` |
| **ExecutionStore** | Read/write interface for structured execution logs in `execution/logs/`. The only way the dashboard or CLI accesses execution data |
| **RiskGuard** | Pre-submission filter applied to every order before it reaches the broker. Enforces max position weight, daily loss limit, OOS R² gate, and PDT protection. Cannot be disabled via config |
| **ReconciliationReport** | Output of `PositionManager.reconcile()` — compares model-expected positions to broker-held positions. Divergence halts execution |
| **Dry-run mode** | Execution loop flag that simulates all steps through RiskGuard but submits nothing to the broker. Used for testing and validation |
| **PDT rule** | US Pattern Day Trader rule: accounts under $25,000 may not make more than 3 day trades in a rolling 5-business-day window. Enforced by RiskGuard for Alpaca accounts |
| **Market-on-Open (MOO)** | Order type that executes at the opening auction price. Default execution style for v4 — signal generated at close, order submitted pre-market, fills at open |
| **IB Gateway** | Interactive Brokers application that provides API access to IBKR without requiring the full TWS desktop application. Required for `ibkr_broker.py` |