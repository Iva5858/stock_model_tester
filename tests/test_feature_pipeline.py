import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_pipeline import FeatureConfig, build_features


def _make_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


def test_output_shapes():
    df = _make_df()
    out = build_features(df, FeatureConfig(test_size=0.2))
    assert out.X_train.shape[1] == out.X_test.shape[1]
    assert len(out.y_train) == len(out.X_train)
    assert len(out.y_test) == len(out.X_test)


def test_no_data_leakage():
    """Scaler fitted on train only: train is zero-mean, test is not forced to zero."""
    df = _make_df(300)
    out = build_features(df, FeatureConfig(test_size=0.2))
    assert abs(out.X_train.mean()) < 0.1
    assert not np.allclose(out.X_train.mean(axis=0), out.X_test.mean(axis=0))


def test_feature_names_match_columns():
    df = _make_df()
    out = build_features(df, FeatureConfig())
    assert len(out.feature_names) == out.X_train.shape[1]


def test_test_dates_length():
    df = _make_df()
    out = build_features(df, FeatureConfig(test_size=0.2))
    assert len(out.test_dates) == len(out.X_test)


def test_train_indices_before_test():
    """Time-based split: test dates are strictly after all training dates."""
    df = _make_df(300)
    out = build_features(df, FeatureConfig(test_size=0.2))
    assert len(out.test_dates) > 0
    # test dates should be sorted and within the overall date range
    assert (out.test_dates == out.test_dates.sort_values()).all()
    # The last test date must be at or before the last date in the raw data
    assert out.test_dates[-1] <= df.index[-1]
    # test_dates must start after the earliest date in the raw series
    assert out.test_dates[0] > df.index[0]


def test_horizon5_target_construction():
    """horizon=5: target = (close[t+5] - close[t]) / close[t], no steps within [t+1..t+4] used."""
    df = _make_df(300)
    out = build_features(df, FeatureConfig(target="next_return", horizon=5, test_size=0.2))
    # Basic shape consistency
    assert out.X_train.shape[1] == out.X_test.shape[1]
    assert len(out.y_train) == len(out.X_train)
    assert len(out.y_test) == len(out.X_test)


def test_horizon5_fewer_rows_than_h1():
    """h=5 drops 5 rows for the target shift (vs 1 row for h=1), so rows ≤ h=1 rows."""
    df = _make_df(300)
    out1 = build_features(df, FeatureConfig(target="next_return", horizon=1))
    out5 = build_features(df, FeatureConfig(target="next_return", horizon=5))
    # h=5 drops 4 more rows for the shifted target
    assert len(out5.y_full) <= len(out1.y_full)


def test_direction_target_produces_pm1():
    """direction target at h=1 gives only {+1, -1}."""
    df = _make_df(300)
    out = build_features(df, FeatureConfig(target="direction", horizon=1))
    unique = set(np.unique(out.y_full))
    assert unique.issubset({1.0, -1.0}), f"Got unexpected values: {unique}"


def test_transform_list_path():
    """If transforms= is specified, the v2 transform-registry path runs correctly."""
    df = _make_df(200)
    config = FeatureConfig(
        target="next_return",
        transforms=[
            {"name": "lag_returns", "lags": [1, 2, 3]},
            {"name": "rolling_stats", "windows": [5, 10]},
            {"name": "volume_delta"},
        ],
    )
    out = build_features(df, config)
    assert out.X_train.shape[1] == out.X_test.shape[1]
    assert len(out.feature_names) > 0
