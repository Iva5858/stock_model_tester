from abc import ABC, abstractmethod

_COST_REGISTRY: dict = {}


def register_cost_model(name: str):
    def decorator(cls):
        _COST_REGISTRY[name] = cls
        return cls
    return decorator


def get_cost_model(name: str):
    if name not in _COST_REGISTRY:
        raise ValueError(f"Unknown cost model '{name}'. Registered: {list(_COST_REGISTRY)}")
    return _COST_REGISTRY[name]


class BaseCostModel(ABC):
    @abstractmethod
    def cost_per_step(self, position_before: float, position_after: float) -> float:
        """Return fractional cost to deduct from return at this step."""
        ...
