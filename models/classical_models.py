"""
Classical time-series models. These are sequence-native: fit() stores y_train;
predict() forecasts len(X_test) steps ahead. X_train is ignored by all models here.
"""
from __future__ import annotations

import numpy as np

from .base_model import BaseModel, register


@register("ARIMA")
class ARIMAModel(BaseModel):
    """ARIMA(p,d,q). With next_return target use d=0 (returns are already stationary)."""

    name = "ARIMA"

    def __init__(self, p: int = 5, d: int = 0, q: int = 0, **kwargs):
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self._result = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("pip install statsmodels")
        self._result = ARIMA(y_train, order=(self.p, self.d, self.q)).fit()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.asarray(self._result.forecast(steps=len(X_test))).astype(np.float32)

    def get_params(self) -> dict:
        return dict(p=self.p, d=self.d, q=self.q)


@register("SARIMA")
class SARIMAModel(BaseModel):
    """Seasonal ARIMA. m=5 captures weekly seasonality in daily equity data."""

    name = "SARIMA"

    def __init__(self, p: int = 1, d: int = 0, q: int = 0,
                 P: int = 1, D: int = 0, Q: int = 0, m: int = 5, **kwargs):
        self.order = (int(p), int(d), int(q))
        self.seasonal_order = (int(P), int(D), int(Q), int(m))
        self._result = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("pip install statsmodels")
        self._result = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.asarray(self._result.forecast(steps=len(X_test))).astype(np.float32)

    def get_params(self) -> dict:
        return dict(order=self.order, seasonal_order=self.seasonal_order)


@register("ETS")
class ETSModel(BaseModel):
    """Exponential Smoothing (Error-Trend-Seasonal). trend=None suits stationary returns."""

    name = "ETS"

    def __init__(self, error: str = "add", trend: str = None,
                 seasonal: str = None, damped_trend: bool = False, **kwargs):
        self.ets_params = dict(
            error=error,
            trend=trend,
            seasonal=seasonal,
            damped_trend=bool(damped_trend),
        )
        self._result = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel as _ETS
        except ImportError:
            raise ImportError("pip install statsmodels")
        self._result = _ETS(y_train, **self.ets_params).fit(disp=False)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return np.asarray(self._result.forecast(steps=len(X_test))).astype(np.float32)

    def get_params(self) -> dict:
        return self.ets_params


@register("GARCH")
class GARCHModel(BaseModel):
    """
    ARMA-GARCH model (arch library). Predicts the conditional mean return.
    The variance (volatility) forecast is available via get_params() after fitting.
    """

    name = "GARCH"

    def __init__(self, p: int = 1, q: int = 1,
                 mean: str = "AR", lags: int = 1, **kwargs):
        self.p = int(p)
        self.q = int(q)
        self.mean = mean
        self.lags = int(lags)
        self._result = None
        self._n_train = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("pip install arch")
        self._n_train = len(y_train)
        # Scale to % returns for numerical stability
        model = arch_model(
            y_train * 100,
            mean=self.mean,
            lags=self.lags,
            vol="Garch",
            p=self.p,
            q=self.q,
            rescale=False,
        )
        self._result = model.fit(disp="off")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        n = len(X_test)
        fc = self._result.forecast(horizon=n, reindex=False)
        # Mean forecast, rescaled back from %
        mean_preds = fc.mean.iloc[-1].values / 100.0
        return mean_preds.astype(np.float32)

    def get_params(self) -> dict:
        return dict(p=self.p, q=self.q, mean=self.mean, lags=self.lags)


@register("MonteCarlo")
class MonteCarloModel(BaseModel):
    """
    GBM Monte Carlo simulation. Estimates drift (mu) and volatility (sigma)
    from y_train returns, simulates n_paths, and returns the mean path.
    """

    name = "MonteCarlo"

    def __init__(self, n_paths: int = 1000, random_state: int = 42, **kwargs):
        self.n_paths = int(n_paths)
        self.random_state = int(random_state)
        self._mu = 0.0
        self._sigma = 1e-4

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._mu = float(np.mean(y_train))
        self._sigma = max(float(np.std(y_train)), 1e-4)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n = len(X_test)
        # Each simulated path: r_t = mu + sigma * Z_t
        Z = rng.standard_normal((self.n_paths, n))
        paths = self._mu + self._sigma * Z
        return paths.mean(axis=0).astype(np.float32)

    def get_params(self) -> dict:
        return dict(n_paths=self.n_paths, random_state=self.random_state)


@register("OrnsteinUhlenbeck")
class OrnsteinUhlenbeckModel(BaseModel):
    """
    Ornstein-Uhlenbeck mean-reverting process.
    Estimates theta (reversion speed), mu (long-run mean), and sigma via OLS.
    Best suited for spread/pairs trading where the series is mean-reverting.
    """

    name = "OrnsteinUhlenbeck"

    def __init__(self, dt: float = 1.0, random_state: int = 42, **kwargs):
        self.dt = float(dt)
        self.random_state = int(random_state)
        self._theta = 1.0
        self._mu_ou = 0.0
        self._sigma = 1e-4
        self._last_value = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # OLS estimator: delta_r = theta*mu*dt - theta*dt*r_{t-1} + noise
        y = y_train
        delta = np.diff(y)
        r_lag = y[:-1]
        A = np.column_stack([np.ones(len(r_lag)), r_lag])
        coeffs, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)
        theta_dt = -coeffs[1]
        self._theta = theta_dt / self.dt if self.dt > 0 else float(theta_dt)
        self._mu_ou = float(coeffs[0] / theta_dt) if abs(theta_dt) > 1e-10 else float(np.mean(y))
        resid = delta - A @ coeffs
        self._sigma = max(float(np.std(resid) / np.sqrt(self.dt)), 1e-4)
        self._last_value = float(y[-1])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n = len(X_test)
        preds = np.empty(n, dtype=np.float32)
        r = self._last_value
        for i in range(n):
            dr = (self._theta * (self._mu_ou - r) * self.dt
                  + self._sigma * np.sqrt(self.dt) * rng.standard_normal())
            r += dr
            preds[i] = r
        return preds

    def get_params(self) -> dict:
        return dict(theta=round(self._theta, 6),
                    mu=round(self._mu_ou, 6),
                    sigma=round(self._sigma, 6))
