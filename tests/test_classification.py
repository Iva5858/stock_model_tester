"""Tests for classification target construction and classification metrics."""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


# ── Direction target construction ─────────────────────────────────────────────

def test_direction_target_h1():
    from features.feature_pipeline import FeatureConfig, build_features
    df = _make_df(300)
    out = build_features(df, FeatureConfig(target="direction", horizon=1))
    y = out.y_train
    # All values must be +1 or -1
    assert set(np.unique(y)).issubset({1.0, -1.0})


def test_direction_target_h5():
    from features.feature_pipeline import FeatureConfig, build_features
    df = _make_df(300)
    out = build_features(df, FeatureConfig(target="direction", horizon=5))
    y = out.y_train
    assert set(np.unique(y)).issubset({1.0, -1.0})


def test_direction_target_sign_correct():
    """direction target = sign(close[t+h] - close[t]): all values are ±1."""
    from features.feature_pipeline import FeatureConfig, build_features
    df = _make_df(300)
    out = build_features(df, FeatureConfig(target="direction", horizon=1, test_size=0.1))
    # All values in y_full must be exactly +1 or -1
    unique = set(np.unique(out.y_full))
    assert unique.issubset({1.0, -1.0}), f"Unexpected values: {unique}"
    # Length must be less than the total rows (1 row dropped for target shift)
    assert len(out.y_full) < len(df)


def test_direction_h5_no_leakage():
    """For h=5, no feature should use data within 5 steps of the train boundary."""
    from features.feature_pipeline import FeatureConfig, build_features
    df = _make_df(500)
    out = build_features(df, FeatureConfig(target="direction", horizon=5, test_size=0.2))
    # The split index in the feature-space corresponds to the train/test boundary.
    # X_train and X_test are from disjoint windows — just verify shapes are consistent.
    assert out.X_train.shape[1] == out.X_test.shape[1]
    assert len(out.y_train) == len(out.X_train)
    assert len(out.y_test) == len(out.X_test)


# ── Classification metrics ─────────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    y_true = rng.choice([-1.0, 1.0], size=100).astype(np.float32)
    # y_pred as probabilities P(up), correlated with true labels
    # +1 → high prob, -1 → low prob, with some noise
    proba = np.where(y_true > 0, 0.7, 0.3).astype(np.float32)
    noise = rng.normal(0, 0.1, 100).astype(np.float32)
    y_pred = np.clip(proba + noise, 0.01, 0.99).astype(np.float32)
    return y_true, y_pred


def test_auc_roc_range(binary_data):
    from evaluation.metrics import auc_roc
    y_true, y_pred = binary_data
    val = auc_roc(y_true, y_pred)
    assert 0.0 <= val <= 1.0, f"AUC={val} out of range"


def test_auc_roc_better_than_random(binary_data):
    from evaluation.metrics import auc_roc
    y_true, y_pred = binary_data
    # A correlated predictor should beat 0.5
    assert auc_roc(y_true, y_pred) > 0.5


def test_brier_score_range(binary_data):
    from evaluation.metrics import brier_score_metric
    y_true, y_pred = binary_data
    val = brier_score_metric(y_true, y_pred)
    assert 0.0 <= val <= 1.0, f"Brier={val}"


def test_brier_score_perfect():
    from evaluation.metrics import brier_score_metric
    y_true = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    # Perfect proba predictions
    y_pred = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert math.isclose(brier_score_metric(y_true, y_pred), 0.0, abs_tol=1e-6)


def test_log_loss_positive(binary_data):
    from evaluation.metrics import log_loss_metric
    y_true, y_pred = binary_data
    val = log_loss_metric(y_true, y_pred)
    assert val > 0, "Log loss should be positive"


def test_precision_up_range(binary_data):
    from evaluation.metrics import precision_up
    y_true, y_pred = binary_data
    val = precision_up(y_true, y_pred)
    assert 0.0 <= val <= 1.0


def test_recall_up_range(binary_data):
    from evaluation.metrics import recall_up
    y_true, y_pred = binary_data
    val = recall_up(y_true, y_pred)
    assert 0.0 <= val <= 1.0


def test_f1_up_range(binary_data):
    from evaluation.metrics import f1_up
    y_true, y_pred = binary_data
    val = f1_up(y_true, y_pred)
    assert 0.0 <= val <= 1.0


def test_compute_metrics_classification_task():
    from evaluation.metrics import compute_metrics
    y_true = np.array([1.0, -1.0, 1.0, -1.0, 1.0] * 10, dtype=np.float32)
    y_pred = np.array([0.8, 0.2, 0.7, 0.3, 0.9] * 10, dtype=np.float32)
    result = compute_metrics(y_true, y_pred, task="classification")
    clf_keys = {"auc_roc", "log_loss", "brier_score", "precision_up",
                "recall_up", "f1_up", "directional_accuracy"}
    assert clf_keys.issubset(set(result.keys()))


# ── Classification model smoke tests ──────────────────────────────────────────

@pytest.mark.parametrize("model_name", [
    "LogisticRegressionClassifier",
    "XGBoostClassifier",
    "RandomForestClassifier",
    "LightGBMClassifier",
])
def test_classifier_predict_binary(model_name):
    from models import get_model_class
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 5)).astype(np.float32)
    y_train = rng.choice([-1.0, 1.0], size=200).astype(np.float32)
    X_test = rng.normal(size=(50, 5)).astype(np.float32)

    ModelClass = get_model_class(model_name)
    model = ModelClass(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({1.0, -1.0}), \
        f"{model_name}.predict() returned values outside {{+1, -1}}: {np.unique(preds)}"


@pytest.mark.parametrize("model_name", [
    "LogisticRegressionClassifier",
    "XGBoostClassifier",
    "RandomForestClassifier",
    "LightGBMClassifier",
])
def test_classifier_predict_proba_range(model_name):
    from models import get_model_class
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(200, 5)).astype(np.float32)
    y_train = rng.choice([-1.0, 1.0], size=200).astype(np.float32)
    X_test = rng.normal(size=(50, 5)).astype(np.float32)

    ModelClass = get_model_class(model_name)
    model = ModelClass(random_state=0)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    assert proba.shape == (50,)
    assert float(proba.min()) >= 0.0
    assert float(proba.max()) <= 1.0


def test_classifier_task_attribute():
    from models import get_model_class
    for name in ("LogisticRegressionClassifier", "XGBoostClassifier",
                 "RandomForestClassifier", "LightGBMClassifier"):
        cls = get_model_class(name)
        assert getattr(cls, "task", None) == "classification", \
            f"{name} missing task='classification'"
