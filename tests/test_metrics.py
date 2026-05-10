import math
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import rmse, mae, mape, r2, directional_accuracy, sharpe_ratio, compute_metrics


Y_TRUE = np.array([100.0, 102.0, 101.0, 105.0, 103.0])
Y_PRED = np.array([101.0, 101.0, 102.0, 104.0, 104.0])


def test_rmse():
    errors = Y_TRUE - Y_PRED
    expected = math.sqrt(np.mean(errors ** 2))
    assert math.isclose(rmse(Y_TRUE, Y_PRED), expected, rel_tol=1e-6)


def test_mae():
    expected = np.mean(np.abs(Y_TRUE - Y_PRED))
    assert math.isclose(mae(Y_TRUE, Y_PRED), expected, rel_tol=1e-6)


def test_mape():
    expected = np.mean(np.abs((Y_TRUE - Y_PRED) / Y_TRUE)) * 100
    assert math.isclose(mape(Y_TRUE, Y_PRED), expected, rel_tol=1e-6)


def test_r2_perfect():
    assert math.isclose(r2(Y_TRUE, Y_TRUE), 1.0)


def test_r2_range():
    val = r2(Y_TRUE, Y_PRED)
    assert val <= 1.0


def test_directional_accuracy_perfect():
    assert directional_accuracy(Y_TRUE, Y_TRUE) == 1.0


def test_directional_accuracy_range():
    val = directional_accuracy(Y_TRUE, Y_PRED)
    assert 0.0 <= val <= 1.0


def test_sharpe_returns_float():
    val = sharpe_ratio(Y_TRUE, Y_PRED)
    assert isinstance(val, float)


def test_compute_metrics_keys():
    result = compute_metrics(Y_TRUE, Y_PRED)
    assert set(result) == {"rmse", "mae", "mape", "r2", "directional_accuracy", "sharpe"}


def test_compute_metrics_subset():
    result = compute_metrics(Y_TRUE, Y_PRED, ["rmse", "mae"])
    assert set(result) == {"rmse", "mae"}
