import pandas as pd
from features.base_transform import FeatureTransform, register_transform


@register_transform("lag_returns")
class LagReturnsTransform(FeatureTransform):
    name = "lag_returns"
    task = "both"

    def __init__(self, lags: list[int] | None = None):
        self.lags = lags or [1, 2, 3, 5, 10]

    def fit_transform(self, df: pd.DataFrame, feat: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        ret = df["close"].pct_change()
        for lag in self.lags:
            feat[f"return_lag_{lag}"] = ret.shift(lag)
        return feat
