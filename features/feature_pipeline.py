from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    target: str = "next_return"
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    technical_indicators: bool = False
    test_size: float = 0.2
    # Walk-forward cross-validation settings (rpaper_1, rpaper_8)
    # cv_method: "holdout" | "expanding" | "rolling"
    cv_method: str = "holdout"
    step_size: int = 21          # trading days between refits
    min_train_size: int = 500    # minimum rows in the initial training window
    window_size: int = 1000      # fixed window length for cv_method="rolling"
    horizon: int = 1             # prediction horizon in trading days
    transforms: Optional[List[dict]] = field(default=None)


@dataclass
class PipelineOutput:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    test_dates: pd.DatetimeIndex
    # Full unscaled arrays — used by walk-forward evaluation
    X_full_raw: Optional[np.ndarray] = field(default=None)
    y_full: Optional[np.ndarray] = field(default=None)
    dates_full: Optional[object] = field(default=None)


def build_features(df: pd.DataFrame, config: FeatureConfig) -> PipelineOutput:
    feat = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change()
    h = config.horizon

    if config.transforms is not None:
        # v2 transform-registry path
        # YAML parses `- lag_returns: {lags: [...]}` as {"lag_returns": {...}},
        # so each tspec is a single-key dict: {name: params}
        from features.base_transform import get_transform
        for tspec in config.transforms:
            if "name" in tspec:
                tname = tspec["name"]
                tparams = {k: v for k, v in tspec.items() if k != "name"}
            else:
                tname, tparams = next(iter(tspec.items()))
                tparams = tparams or {}
            transform_cls = get_transform(tname)
            transform = transform_cls(**tparams)
            transform.fit_transform(df, feat, is_train=True)
    else:
        # v1 flag-based path (backward compat)
        if config.target in ("next_return", "direction"):
            for lag in config.lags:
                feat[f"return_lag_{lag}"] = ret.shift(lag)
            for window in config.rolling_windows:
                feat[f"rolling_ret_mean_{window}"] = ret.shift(1).rolling(window).mean()
                feat[f"rolling_ret_std_{window}"] = ret.shift(1).rolling(window).std()
            feat["volume_delta"] = df["volume"].pct_change().shift(1)
        elif config.target == "next_close":
            for lag in config.lags:
                feat[f"close_lag_{lag}"] = df["close"].shift(lag)
            for window in config.rolling_windows:
                feat[f"rolling_mean_{window}"] = df["close"].shift(1).rolling(window).mean()
                feat[f"rolling_std_{window}"] = df["close"].shift(1).rolling(window).std()
            feat["daily_return"] = ret.shift(1)
            feat["volume_delta"] = df["volume"].pct_change().shift(1)
        else:
            raise ValueError(
                f"Unsupported target '{config.target}'. "
                "Use 'next_return', 'next_close', or 'direction'."
            )

        if config.technical_indicators:
            _add_technical_indicators(df, feat)

    # Target construction — supports h>1
    close = df["close"]
    if config.target == "next_return":
        feat["target"] = (close.shift(-h) - close) / close
    elif config.target == "next_close":
        feat["target"] = close.shift(-h)
    elif config.target == "direction":
        diff = close.shift(-h) - close
        feat["target"] = np.where(diff > 0, 1.0, -1.0)
    else:
        raise ValueError(
            f"Unsupported target '{config.target}'. "
            "Use 'next_return', 'next_close', or 'direction'."
        )

    feat = feat.dropna()

    feature_names = [c for c in feat.columns if c != "target"]
    X = feat[feature_names].values.astype(np.float32)
    y = feat["target"].values.astype(np.float32)
    dates = feat.index

    X_full_raw = X.copy()
    y_full = y.copy()
    dates_full = dates

    split_idx = int(len(X) * (1 - config.test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = dates[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return PipelineOutput(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
        test_dates=test_dates,
        X_full_raw=X_full_raw,
        y_full=y_full,
        dates_full=dates_full,
    )


def walk_forward_splits(
    X_raw: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    min_train_size: int = 500,
    step_size: int = 21,
    window_size: int | None = None,
) -> list:
    """Generate (X_tr_raw, y_tr, X_te_raw, y_te, test_dates) tuples for walk-forward CV.

    cv_method 'expanding': window_size=None — training window grows from the start.
    cv_method 'rolling':   window_size=int  — fixed-length sliding training window.

    Scaling is intentionally excluded: callers must scale within each fold to
    prevent data leakage (scaler.fit on train only, .transform on test).

    References: rpaper_1 (Turgay 2025), rpaper_8 (Mistol & Möhler 2023),
                fpaper_2 (Goyal & Welch 2008).
    """
    n = len(X_raw)
    if min_train_size >= n:
        raise ValueError(
            f"min_train_size ({min_train_size}) >= total rows ({n}). "
            "Reduce min_train_size or fetch more data."
        )
    splits = []
    t = min_train_size
    while t < n:
        end = min(t + step_size, n)
        start = 0 if window_size is None else max(0, t - window_size)
        splits.append((
            X_raw[start:t],
            y[start:t],
            X_raw[t:end],
            y[t:end],
            dates[t:end],
        ))
        t = end
    return splits


def _add_technical_indicators(df: pd.DataFrame, feat: pd.DataFrame) -> None:
    try:
        import ta
    except ImportError:
        raise ImportError("Install 'ta' to use technical_indicators: pip install ta")

    close = df["close"]

    feat["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close)
    feat["macd"] = macd.macd()
    feat["macd_signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close=close)
    feat["bb_upper"] = bb.bollinger_hband()
    feat["bb_lower"] = bb.bollinger_lband()
    feat["bb_width"] = bb.bollinger_wband()
