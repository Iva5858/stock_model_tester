import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline import NaiveLastValue, RollingMeanBaseline


def _make_data(n: int = 100):
    rng = np.random.default_rng(1)
    y = 100 + np.cumsum(rng.normal(0, 1, n)).astype(np.float32)
    X = np.stack([y[:-1]], axis=1)  # simple 1-feature X
    return X[:-1], y[:-1], y[1:]   # X_train, y_train, y_test (shifted)


def test_naive_last_value_prediction():
    _, y_train, _ = _make_data()
    model = NaiveLastValue()
    model.fit(np.zeros((len(y_train), 1)), y_train)
    preds = model.predict(np.zeros((10, 1)))
    assert np.all(preds == y_train[-1])


def test_naive_last_value_length():
    _, y_train, _ = _make_data()
    model = NaiveLastValue()
    model.fit(np.zeros((len(y_train), 1)), y_train)
    preds = model.predict(np.zeros((20, 1)))
    assert len(preds) == 20


def test_rolling_mean_prediction_shape():
    _, y_train, _ = _make_data()
    model = RollingMeanBaseline(window=5)
    model.fit(np.zeros((len(y_train), 1)), y_train)
    preds = model.predict(np.zeros((15, 1)))
    assert len(preds) == 15


def test_rolling_mean_value():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    model = RollingMeanBaseline(window=5)
    model.fit(np.zeros((5, 1)), y_train)
    preds = model.predict(np.zeros((1, 1)))
    expected = np.mean(y_train)
    assert abs(preds[0] - expected) < 1e-5


def test_get_params():
    model = RollingMeanBaseline(window=10)
    assert model.get_params() == {"window": 10}
