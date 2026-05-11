from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.cost_models.base_cost_model import BaseCostModel


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    gross_sharpe: float
    net_sharpe: float
    gross_calmar: float
    net_calmar: float
    gross_max_drawdown: float
    net_max_drawdown: float
    turnover_rate: float


class Backtester:
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: pd.DatetimeIndex,
        cost_model: BaseCostModel,
        initial_capital: float = 100_000.0,
    ):
        self._y_true = y_true
        self._y_pred = y_pred
        self._dates = dates
        self._cost_model = cost_model
        self._initial_capital = initial_capital

    def run(self) -> BacktestResult:
        n = len(self._y_true)
        signals = np.sign(self._y_pred)

        gross_rets = np.zeros(n)
        net_rets = np.zeros(n)
        bench_rets = self._y_true.copy()
        trades = 0

        prev_pos = 0.0
        for i in range(n):
            pos = float(signals[i])
            cost = self._cost_model.cost_per_step(prev_pos, pos)
            if cost > 0:
                trades += 1
            gross_rets[i] = pos * self._y_true[i]
            net_rets[i] = gross_rets[i] - cost
            prev_pos = pos

        turnover = trades / n if n > 0 else 0.0

        def _equity(rets: np.ndarray) -> np.ndarray:
            return self._initial_capital * np.cumprod(1.0 + rets)

        def _sharpe(rets: np.ndarray) -> float:
            std = float(np.std(rets))
            return float(np.mean(rets) / std * np.sqrt(252)) if std > 0 else float("nan")

        def _mdd(rets: np.ndarray) -> float:
            cum = np.cumprod(1.0 + rets)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / np.where(peak == 0, 1.0, peak)
            return float(np.min(dd))

        def _calmar(rets: np.ndarray) -> float:
            mdd = _mdd(rets)
            ann_ret = float(np.mean(rets) * 252)
            return ann_ret / abs(mdd) if mdd < 0 else float("nan")

        gross_eq = _equity(gross_rets)
        net_eq = _equity(net_rets)
        bench_eq = _equity(bench_rets)

        equity_curve = pd.DataFrame({
            "date": self._dates,
            "gross_value": gross_eq,
            "net_value": net_eq,
            "benchmark_value": bench_eq,
        })

        return BacktestResult(
            equity_curve=equity_curve,
            gross_sharpe=_sharpe(gross_rets),
            net_sharpe=_sharpe(net_rets),
            gross_calmar=_calmar(gross_rets),
            net_calmar=_calmar(net_rets),
            gross_max_drawdown=_mdd(gross_rets),
            net_max_drawdown=_mdd(net_rets),
            turnover_rate=turnover,
        )
