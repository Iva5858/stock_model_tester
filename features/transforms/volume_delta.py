import pandas as pd
from features.base_transform import FeatureTransform, register_transform


@register_transform("volume_delta")
class VolumeDeltaTransform(FeatureTransform):
    name = "volume_delta"
    task = "both"

    def fit_transform(self, df: pd.DataFrame, feat: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        feat["volume_delta"] = df["volume"].pct_change().shift(1)
        return feat
