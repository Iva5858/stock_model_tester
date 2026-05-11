from abc import ABC, abstractmethod
import pandas as pd


_TRANSFORM_REGISTRY: dict = {}


def register_transform(name: str):
    def decorator(cls):
        _TRANSFORM_REGISTRY[name] = cls
        return cls
    return decorator


def get_transform(name: str):
    if name not in _TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform '{name}'. Registered: {list(_TRANSFORM_REGISTRY)}")
    return _TRANSFORM_REGISTRY[name]


class FeatureTransform(ABC):
    name: str
    task: str = "both"  # "regression" | "classification" | "both"

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, feat: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Add columns to feat in-place and return feat."""
        ...
