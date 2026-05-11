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


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray,
           y_train: np.ndarray | None = None) -> float:
    """Campbell-Thompson (2008) / Goyal-Welch (2008) out-of-sample R².

    Measures improvement in MSE over the prevailing (historical training) mean.
    Positive → model beats the mean; negative → model is worse than predicting mean.
    Requires y_train to compute the benchmark; returns NaN if y_train is None.

    References: fpaper_2 (Goyal & Welch 2008), fpaper_3 (Campbell & Thompson 2008),
                rpaper_8 (Mistol & Möhler 2023).
    """
    if y_train is None or len(y_train) == 0:
        return float("nan")
    benchmark = float(np.mean(y_train))
    ss_pred = float(np.sum((y_true - y_pred) ** 2))
    ss_bench = float(np.sum((y_true - benchmark) ** 2))
    if ss_bench == 0:
        return float("nan")
    return float(1.0 - ss_pred / ss_bench)


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank Information Coefficient between predictions and actuals.

    Measures the model's ability to rank returns; used for investment signal quality.
    References: rpaper_5 (Wang 2025), rpaper_8 (Mistol & Möhler 2023).
    """
    from scipy.stats import spearmanr
    if len(y_true) < 3:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else float("nan")


def max_drawdown(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum drawdown of the long/short strategy driven by predictions.

    Returns a value ≤ 0; closer to 0 is better.
    References: rpaper_5 (Wang 2025).
    """
    signal = np.sign(y_pred)
    strategy_ret = signal * y_true
    cum = np.cumprod(1.0 + strategy_ret)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / np.where(peak == 0, 1.0, peak)
    return float(np.min(drawdown))


def calmar_ratio(y_true: np.ndarray, y_pred: np.ndarray,
                 annualisation: float = 252.0) -> float:
    """Calmar ratio: annualised strategy return divided by absolute max drawdown.

    Higher is better. Penalises large peak-to-trough losses relative to returns.
    References: rpaper_5 (Wang 2025).
    """
    signal = np.sign(y_pred)
    strategy_ret = signal * y_true
    ann_return = float(np.mean(strategy_ret) * annualisation)
    mdd = max_drawdown(y_true, y_pred)
    if mdd >= 0:
        return float("nan")
    return float(ann_return / abs(mdd))


def diebold_mariano(y_true: np.ndarray, y_pred1: np.ndarray,
                    y_pred2: np.ndarray) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold corrected Diebold-Mariano test (h=1, MSE loss).

    H0: equal predictive accuracy. Returns (dm_stat, p_value), two-sided.
    Negative dm_stat → pred1 more accurate than pred2.
    References: rpaper_8 (Mistol & Möhler 2023).
    """
    from scipy import stats
    if len(y_true) < 4:
        return float("nan"), float("nan")
    e1 = (y_true - y_pred1) ** 2
    e2 = (y_true - y_pred2) ** 2
    d = e1 - e2
    T = len(d)
    d_mean = float(np.mean(d))
    d_var = float(np.var(d, ddof=1))
    if d_var == 0:
        return float("nan"), float("nan")
    dm_stat = d_mean / np.sqrt(d_var / T)
    p_val = float(2.0 * stats.t.sf(np.abs(dm_stat), df=T - 1))
    return float(dm_stat), p_val


# Metrics that require y_train context (not computable from y_true/y_pred alone)
_CONTEXT_METRICS = {"oos_r2"}

_METRIC_FNS = {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "r2": r2,
    "directional_accuracy": directional_accuracy,
    "sharpe": sharpe_ratio,
    "oos_r2": oos_r2,
    "rank_ic": rank_ic,
    "max_drawdown": max_drawdown,
    "calmar_ratio": calmar_ratio,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_names: list[str] | None = None,
                    y_train: np.ndarray | None = None) -> dict:
    """Compute evaluation metrics.

    Pass y_train to enable oos_r2 (Campbell-Thompson OOS R² vs. prevailing mean).
    """
    if metric_names is None:
        metric_names = [m for m in _METRIC_FNS if m not in _CONTEXT_METRICS]
    result = {}
    for name in metric_names:
        fn = _METRIC_FNS[name]
        if name in _CONTEXT_METRICS:
            result[name] = fn(y_true, y_pred, y_train)
        else:
            result[name] = fn(y_true, y_pred)
    return result
