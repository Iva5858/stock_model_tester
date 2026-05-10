from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of steps where predicted sign matches actual sign.

    For next_return targets (recommended): compares sign(y_true) vs sign(y_pred).
    Returns NaN when the array is empty.
    """
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray,
                 annualisation: float = 252.0) -> float:
    """Annualised Sharpe of a long/short strategy driven by predictions.

    Signal: long (+1) when y_pred > 0, short (-1) otherwise.
    Strategy daily return: signal * y_true (i.e. y_true is already a return).
    Designed for next_return targets.
    """
    signal = np.sign(y_pred)
    strategy_ret = signal * y_true
    std = float(np.std(strategy_ret))
    if std == 0:
        return float("nan")
    return float(np.mean(strategy_ret) / std * np.sqrt(annualisation))


_METRIC_FNS = {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "r2": r2,
    "directional_accuracy": directional_accuracy,
    "sharpe": sharpe_ratio,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_names: list[str] | None = None) -> dict:
    if metric_names is None:
        metric_names = list(_METRIC_FNS)
    return {name: _METRIC_FNS[name](y_true, y_pred) for name in metric_names}
