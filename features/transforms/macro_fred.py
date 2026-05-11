from __future__ import annotations

import logging

import pandas as pd
from features.base_transform import FeatureTransform, register_transform

logger = logging.getLogger(__name__)


@register_transform("macro_fred")
class MacroFREDTransform(FeatureTransform):
    """Joins FRED macro predictors onto the feature DataFrame."""
    name = "macro_fred"
    task = "both"

    def fit_transform(self, df: pd.DataFrame, feat: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        try:
            from data.fred_loader import FREDLoader
        except ImportError:
            logger.warning("FREDLoader not available — skipping macro_fred transform")
            return feat

        start = df.index.min().strftime("%Y-%m-%d")
        end = df.index.max().strftime("%Y-%m-%d")

        try:
            loader = FREDLoader()
            macro = loader.load(start=start, end=end)
        except Exception as exc:
            logger.warning("macro_fred transform failed to load: %s", exc)
            return feat

        if macro.empty:
            return feat

        # Publication lag already applied inside FREDLoader; join on index
        macro.index = pd.DatetimeIndex(macro.index)
        feat_index = pd.DatetimeIndex(feat.index)
        macro_aligned = macro.reindex(feat_index).ffill()

        for col in macro_aligned.columns:
            feat[f"fred_{col}"] = macro_aligned[col].values

        return feat
