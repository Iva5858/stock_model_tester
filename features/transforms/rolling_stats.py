import pandas as pd
from features.base_transform import FeatureTransform, register_transform


@register_transform("rolling_stats")
class RollingStatsTransform(FeatureTransform):
    name = "rolling_stats"
    task = "both"

    def __init__(self, windows: list[int] | None = None):
        self.windows = windows or [5, 10, 20]

    def fit_transform(self, df: pd.DataFrame, feat: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        ret = df["close"].pct_change()
        for window in self.windows:
            feat[f"rolling_ret_mean_{window}"] = ret.shift(1).rolling(window).mean()
            feat[f"rolling_ret_std_{window}"] = ret.shift(1).rolling(window).std()
        return feat
