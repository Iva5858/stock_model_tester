from abc import ABC, abstractmethod

import numpy as np


_REGISTRY: dict = {}


def register(name: str):
    """Decorator to register a model class under a string key."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Registered models: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> dict:
    return {name: cls for name, cls in _REGISTRY.items()}


class BaseModel(ABC):
    name: str

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_params(self) -> dict: ...
