import pandas as pd
from features.base_transform import FeatureTransform, register_transform


@register_transform("technical")
class TechnicalTransform(FeatureTransform):
    name = "technical"
    task = "both"

    def fit_transform(self, df: pd.DataFrame, feat: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        try:
            import ta
        except ImportError:
            raise ImportError("Install 'ta' to use technical transform: pip install ta")
        close = df["close"]
        feat["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd = ta.trend.MACD(close=close)
        feat["macd"] = macd.macd()
        feat["macd_signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(close=close)
        feat["bb_upper"] = bb.bollinger_hband()
        feat["bb_lower"] = bb.bollinger_lband()
        feat["bb_width"] = bb.bollinger_wband()
        return feat
