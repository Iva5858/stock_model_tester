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


# ── Classification metrics ─────────────────────────────────────────────────────

def auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area under the ROC curve.  y_pred interpreted as P(up) in [0,1] when
    possible, else as a signed score with 0 as boundary."""
    from sklearn.metrics import roc_auc_score
    try:
        labels = np.sign(y_true)
        labels_bin = (labels > 0).astype(int)
        if len(np.unique(labels_bin)) < 2:
            return float("nan")
        # If y_pred already in [0,1] use as-is; else normalize to probability
        score = y_pred if y_pred.max() <= 1.0 and y_pred.min() >= 0 else \
            (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-12)
        return float(roc_auc_score(labels_bin, score))
    except Exception:
        return float("nan")


def log_loss_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Log-loss for classification. y_pred as probabilities P(up)."""
    from sklearn.metrics import log_loss
    try:
        labels_bin = (np.sign(y_true) > 0).astype(int)
        if len(np.unique(labels_bin)) < 2:
            return float("nan")
        proba = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return float(log_loss(labels_bin, proba))
    except Exception:
        return float("nan")


def brier_score_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score: MSE between predicted probabilities and binary outcomes."""
    try:
        labels_bin = (np.sign(y_true) > 0).astype(float)
        proba = np.clip(y_pred, 0.0, 1.0)
        return float(np.mean((proba - labels_bin) ** 2))
    except Exception:
        return float("nan")


def precision_up(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision for predicting the 'up' class (+1)."""
    from sklearn.metrics import precision_score
    try:
        labels = np.sign(y_true)
        preds = np.sign(y_pred)
        return float(precision_score(labels, preds, pos_label=1.0,
                                     zero_division=0))
    except Exception:
        return float("nan")


def recall_up(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for predicting the 'up' class (+1)."""
    from sklearn.metrics import recall_score
    try:
        labels = np.sign(y_true)
        preds = np.sign(y_pred)
        return float(recall_score(labels, preds, pos_label=1.0,
                                  zero_division=0))
    except Exception:
        return float("nan")


def f1_up(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for the 'up' class (+1)."""
    from sklearn.metrics import f1_score
    try:
        labels = np.sign(y_true)
        preds = np.sign(y_pred)
        return float(f1_score(labels, preds, pos_label=1.0, zero_division=0))
    except Exception:
        return float("nan")


# ── Metric registry ───────────────────────────────────────────────────────────

# Metrics that require y_train context (not computable from y_true/y_pred alone)
_CONTEXT_METRICS = {"oos_r2"}

_METRIC_FNS: dict = {
    # Regression metrics
    "rmse":                {"fn": rmse,                "task": "regression"},
    "mae":                 {"fn": mae,                 "task": "regression"},
    "mape":                {"fn": mape,                "task": "regression"},
    "r2":                  {"fn": r2,                  "task": "regression"},
    "directional_accuracy":{"fn": directional_accuracy,"task": "both"},
    "sharpe":              {"fn": sharpe_ratio,        "task": "regression"},
    "oos_r2":              {"fn": oos_r2,              "task": "regression"},
    "rank_ic":             {"fn": rank_ic,             "task": "regression"},
    "max_drawdown":        {"fn": max_drawdown,        "task": "regression"},
    "calmar_ratio":        {"fn": calmar_ratio,        "task": "regression"},
    # Classification metrics
    "auc_roc":             {"fn": auc_roc,             "task": "classification"},
    "log_loss":            {"fn": log_loss_metric,     "task": "classification"},
    "brier_score":         {"fn": brier_score_metric,  "task": "classification"},
    "precision_up":        {"fn": precision_up,        "task": "classification"},
    "recall_up":           {"fn": recall_up,           "task": "classification"},
    "f1_up":               {"fn": f1_up,               "task": "classification"},
}

_DEFAULT_REGRESSION_METRICS = [
    m for m, v in _METRIC_FNS.items()
    if v["task"] in ("regression", "both") and m not in _CONTEXT_METRICS
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    metric_names: list[str] | None = None,
                    y_train: np.ndarray | None = None,
                    task: str = "regression") -> dict:
    """Compute evaluation metrics.

    Pass y_train to enable oos_r2 (Campbell-Thompson OOS R² vs. prevailing mean).
    Pass task='classification' to get classification-only default metrics.
    """
    if metric_names is None:
        if task == "classification":
            metric_names = [
                m for m, v in _METRIC_FNS.items()
                if v["task"] in ("classification", "both") and m not in _CONTEXT_METRICS
            ]
        else:
            metric_names = _DEFAULT_REGRESSION_METRICS

    result = {}
    for name in metric_names:
        entry = _METRIC_FNS.get(name)
        if entry is None:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(_METRIC_FNS)}")
        fn = entry["fn"]
        if name in _CONTEXT_METRICS:
            result[name] = fn(y_true, y_pred, y_train)
        else:
            result[name] = fn(y_true, y_pred)
    return result
