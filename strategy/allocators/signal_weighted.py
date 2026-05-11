import pandas as pd
from .base_allocator import BaseAllocator, register_allocator


@register_allocator("signal_weighted")
class SignalWeightedAllocator(BaseAllocator):
    def allocate(
        self,
        signals: dict[str, float],
        cov_matrix: pd.DataFrame | None,
        constraints: dict,
    ) -> dict[str, float]:
        long_only = constraints.get("long_only", False)
        max_pos = constraints.get("max_position_weight", 0.40)

        if long_only:
            active = {t: s for t, s in signals.items() if s > 0}
        else:
            active = {t: s for t, s in signals.items() if s != 0}

        if not active:
            return {t: 0.0 for t in signals}

        total_abs = sum(abs(v) for v in active.values())
        weights = {}
        for t in signals:
            if t not in active:
                weights[t] = 0.0
            else:
                raw_w = active[t] / total_abs
                # Apply max_position_weight cap on absolute basis
                if abs(raw_w) > max_pos:
                    raw_w = max_pos * (1.0 if raw_w > 0 else -1.0)
                weights[t] = raw_w

        return weights
