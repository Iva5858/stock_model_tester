import numpy as np
import pandas as pd
from .base_allocator import BaseAllocator, register_allocator


@register_allocator("max_sharpe")
class MaxSharpeAllocator(BaseAllocator):
    def allocate(
        self,
        signals: dict[str, float],
        cov_matrix: pd.DataFrame | None,
        constraints: dict,
    ) -> dict[str, float]:
        long_only = constraints.get("long_only", False)
        max_pos = constraints.get("max_position_weight", 0.40)
        tickers = list(signals.keys())

        if cov_matrix is None or cov_matrix.empty:
            from .equal_weight import EqualWeightAllocator
            return EqualWeightAllocator().allocate(signals, cov_matrix, constraints)

        try:
            from scipy.optimize import minimize

            common = [t for t in tickers if t in cov_matrix.columns]
            if len(common) < 2:
                from .equal_weight import EqualWeightAllocator
                return EqualWeightAllocator().allocate(signals, cov_matrix, constraints)

            cov = cov_matrix.loc[common, common].values
            mu = np.array([signals.get(t, 0.0) for t in common])
            n = len(common)

            bounds = [(0, max_pos) if long_only else (-max_pos, max_pos)] * n
            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            def neg_sharpe(w):
                ret = float(w @ mu)
                var = float(w @ cov @ w)
                return -ret / (np.sqrt(var) + 1e-12)

            w0 = np.ones(n) / n
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"ftol": 1e-9, "maxiter": 200})

            weights = {}
            for t in tickers:
                if t in common:
                    idx = common.index(t)
                    weights[t] = float(res.x[idx]) if res.success else 1.0 / n
                else:
                    weights[t] = 0.0
            return weights

        except Exception:
            from .equal_weight import EqualWeightAllocator
            return EqualWeightAllocator().allocate(signals, cov_matrix, constraints)
