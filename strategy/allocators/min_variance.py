import numpy as np
import pandas as pd
from .base_allocator import BaseAllocator, register_allocator


@register_allocator("min_variance")
class MinVarianceAllocator(BaseAllocator):
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
            # Fall back to equal weight
            from .equal_weight import EqualWeightAllocator
            return EqualWeightAllocator().allocate(signals, cov_matrix, constraints)

        try:
            from scipy.optimize import minimize
            from sklearn.covariance import LedoitWolf

            common = [t for t in tickers if t in cov_matrix.columns]
            if len(common) < 2:
                from .equal_weight import EqualWeightAllocator
                return EqualWeightAllocator().allocate(signals, cov_matrix, constraints)

            cov = cov_matrix.loc[common, common].values
            n = len(common)

            bounds = [(0, max_pos) if long_only else (-max_pos, max_pos)] * n
            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            def portfolio_variance(w):
                return float(w @ cov @ w)

            w0 = np.ones(n) / n
            res = minimize(portfolio_variance, w0, method="SLSQP",
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
