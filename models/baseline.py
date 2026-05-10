import numpy as np

from .base_model import BaseModel, register


@register("NaiveLastValue")
class NaiveLastValue(BaseModel):
    """Predicts next value = last observed value (random walk assumption)."""

    name = "NaiveLastValue"

    def __init__(self, **kwargs):
        self._last_train_value: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._last_train_value = float(y_train[-1])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.full(len(X_test), self._last_train_value, dtype=np.float32)

    def get_params(self) -> dict:
        return {}


@register("RollingMean")
class RollingMeanBaseline(BaseModel):
    """Predicts next value = rolling mean of last N training values."""

    name = "RollingMean"

    def __init__(self, window: int = 5, **kwargs):
        self.window = int(window)
        self._prediction: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._prediction = float(np.mean(y_train[-self.window:]))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.full(len(X_test), self._prediction, dtype=np.float32)

    def get_params(self) -> dict:
        return {"window": self.window}
