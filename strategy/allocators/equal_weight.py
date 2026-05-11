import pandas as pd
from .base_allocator import BaseAllocator, register_allocator


@register_allocator("equal")
class EqualWeightAllocator(BaseAllocator):
    def allocate(
        self,
        signals: dict[str, float],
        cov_matrix: pd.DataFrame | None,
        constraints: dict,
    ) -> dict[str, float]:
        long_only = constraints.get("long_only", False)
        active = {t: s for t, s in signals.items() if s != 0}
        if long_only:
            active = {t: s for t, s in active.items() if s > 0}
        if not active:
            return {t: 0.0 for t in signals}
        w = 1.0 / len(active)
        return {t: (w if t in active else 0.0) for t in signals}
