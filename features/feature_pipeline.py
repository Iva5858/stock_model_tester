from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

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


@dataclass
class PipelineOutput:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    test_dates: pd.DatetimeIndex


def build_features(df: pd.DataFrame, config: FeatureConfig) -> PipelineOutput:
    feat = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change()

    if config.target == "next_return":
        # Return-based features — stationary, no distribution shift across train/test
        for lag in config.lags:
            feat[f"return_lag_{lag}"] = ret.shift(lag)
        for window in config.rolling_windows:
            feat[f"rolling_ret_mean_{window}"] = ret.shift(1).rolling(window).mean()
            feat[f"rolling_ret_std_{window}"] = ret.shift(1).rolling(window).std()
        feat["volume_delta"] = df["volume"].pct_change().shift(1)
        feat["target"] = ret.shift(-1)

    elif config.target == "next_close":
        # Raw price features — kept for short-horizon experiments only
        for lag in config.lags:
            feat[f"close_lag_{lag}"] = df["close"].shift(lag)
        for window in config.rolling_windows:
            feat[f"rolling_mean_{window}"] = df["close"].shift(1).rolling(window).mean()
            feat[f"rolling_std_{window}"] = df["close"].shift(1).rolling(window).std()
        feat["daily_return"] = ret.shift(1)
        feat["volume_delta"] = df["volume"].pct_change().shift(1)
        feat["target"] = df["close"].shift(-1)

    else:
        raise ValueError(f"Unsupported target '{config.target}'. Use 'next_return' or 'next_close'.")

    if config.technical_indicators:
        _add_technical_indicators(df, feat)

    feat = feat.dropna()

    feature_names = [c for c in feat.columns if c != "target"]
    X = feat[feature_names].values.astype(np.float32)
    y = feat["target"].values.astype(np.float32)
    dates = feat.index

    split_idx = int(len(X) * (1 - config.test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = dates[split_idx:]

    # Fit scaler on train only — no leakage
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
    )


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
