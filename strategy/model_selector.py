from __future__ import annotations

from dataclasses import dataclass

import yaml

from evaluation.results_store import ResultsStore, ExperimentRecord


@dataclass
class StrategySpec:
    ticker: str
    model: str
    horizon: int
    target: str
    cv_method: str
    params: dict
    criterion: str
    criterion_value: float
    oos_r2: float
    directional_accuracy: float
    sharpe: float
    exp_id: str

    def to_yaml(self) -> str:
        data = {
            "experiment": {"name": f"strategy_{self.ticker}_{self.model}"},
            "data": {"loader": "yfinance", "ticker": self.ticker},
            "features": {"target": self.target, "horizon": self.horizon},
            "model": {"name": self.model, "params": self.params},
            "evaluation": {"cv_method": self.cv_method},
            "strategy": {
                "criterion": self.criterion,
                "criterion_value": self.criterion_value,
                "exp_id": self.exp_id,
            },
            "schema_version": 2,
        }
        return yaml.dump(data, default_flow_style=False)


class ModelSelector:
    def __init__(self, store: ResultsStore, criterion: str = "sharpe",
                 selection_window: int = 252):
        self._store = store
        self.criterion = criterion
        self.selection_window = selection_window

    def recommend(self, ticker: str) -> StrategySpec:
        records = self._store.list_experiments(ticker=ticker)
        if not records:
            raise ValueError(
                f"No experiments found for ticker '{ticker}'. "
                "Run experiments first with 'python run.py run'."
            )

        best: ExperimentRecord | None = None
        best_val = float("-inf")

        for rec in records:
            val = rec.metrics.get(self.criterion, None)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if val > best_val:
                best_val = val
                best = rec

        if best is None:
            raise ValueError(
                f"No experiment for '{ticker}' has metric '{self.criterion}' in its results."
            )

        m = best.metrics
        return StrategySpec(
            ticker=best.ticker,
            model=best.model,
            horizon=best.horizon,
            target=best.target,
            cv_method=m.get("cv_method", "holdout"),
            params=m.get("model_params", {}),
            criterion=self.criterion,
            criterion_value=best_val,
            oos_r2=float(m.get("oos_r2", float("nan"))),
            directional_accuracy=float(m.get("directional_accuracy", float("nan"))),
            sharpe=float(m.get("sharpe", float("nan"))),
            exp_id=best.exp_id,
        )
