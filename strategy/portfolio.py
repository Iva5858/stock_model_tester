from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.results_store import ResultsStore
from strategy.allocators.base_allocator import get_allocator


@dataclass
class PortfolioResult:
    weights: dict[str, float]
    signals: dict[str, float]
    metrics: dict


class PortfolioConstructor:
    def __init__(
        self,
        store: ResultsStore,
        allocator_name: str = "signal_weighted",
        constraints: dict | None = None,
    ):
        self._store = store
        self._allocator_name = allocator_name
        self._constraints = constraints or {"long_only": False, "max_position_weight": 0.40}

    def build(self, tickers: list[str], as_of_date: str | None = None) -> PortfolioResult:
        signals: dict[str, float] = {}

        for ticker in tickers:
            records = self._store.list_experiments(ticker=ticker.upper())
            if not records:
                signals[ticker] = 0.0
                continue

            latest = records[-1]
            try:
                preds = self._store.load_predictions(latest.exp_id)
            except Exception:
                signals[ticker] = 0.0
                continue

            preds = preds.sort_values("date")
            if preds.empty:
                signals[ticker] = 0.0
                continue

            y_pred_last = float(preds["y_pred"].iloc[-1])
            model_task = latest.metrics.get("task", "regression")

            if model_task == "classification":
                # P(up) - 0.5 gives signed signal in [-0.5, 0.5]
                signals[ticker] = y_pred_last - 0.5
            else:
                signals[ticker] = y_pred_last

        allocator_cls = get_allocator(self._allocator_name)
        allocator = allocator_cls()
        weights = allocator.allocate(signals, cov_matrix=None,
                                     constraints=self._constraints)

        # Compute simple portfolio metrics
        portfolio_returns = []
        for ticker, w in weights.items():
            if abs(w) < 1e-9:
                continue
            records = self._store.list_experiments(ticker=ticker.upper())
            if not records:
                continue
            try:
                preds = self._store.load_predictions(records[-1].exp_id)
                preds = preds.sort_values("date")
                r = preds["y_true"].values * w
                portfolio_returns.append(r)
            except Exception:
                pass

        metrics: dict = {}
        if portfolio_returns:
            # Align lengths
            min_len = min(len(r) for r in portfolio_returns)
            combined = np.sum(np.stack([r[-min_len:] for r in portfolio_returns]), axis=0)
            std = float(np.std(combined))
            metrics["portfolio_sharpe"] = (
                float(np.mean(combined) / std * np.sqrt(252)) if std > 0 else float("nan")
            )
            cum = np.cumprod(1.0 + combined)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / np.where(peak == 0, 1.0, peak)
            mdd = float(np.min(dd))
            ann_ret = float(np.mean(combined) * 252)
            metrics["portfolio_calmar"] = ann_ret / abs(mdd) if mdd < 0 else float("nan")

        return PortfolioResult(weights=weights, signals=signals, metrics=metrics)
