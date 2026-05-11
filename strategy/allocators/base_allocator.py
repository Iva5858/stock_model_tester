from abc import ABC, abstractmethod
import pandas as pd

_ALLOCATOR_REGISTRY: dict = {}


def register_allocator(name: str):
    def decorator(cls):
        _ALLOCATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_allocator(name: str):
    if name not in _ALLOCATOR_REGISTRY:
        raise ValueError(f"Unknown allocator '{name}'. Registered: {list(_ALLOCATOR_REGISTRY)}")
    return _ALLOCATOR_REGISTRY[name]


class BaseAllocator(ABC):
    @abstractmethod
    def allocate(
        self,
        signals: dict[str, float],
        cov_matrix: pd.DataFrame | None,
        constraints: dict,
    ) -> dict[str, float]:
        """Return ticker → weight mapping. Weights may be negative (short). Sum ≤ 1."""
        ...
