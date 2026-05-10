"""
Regime-switching models. Fit on y_train (return series) to detect latent market
regimes (bull / bear / sideways). Predict by propagating regime probabilities
forward using the learned transition matrix.
"""
from __future__ import annotations

import numpy as np

from .base_model import BaseModel, register


@register("HMM")
class HMMModel(BaseModel):
    """
    Gaussian Hidden Markov Model (hmmlearn).
    Identifies n_components latent regimes and predicts the expected return
    per step by weighting each regime's mean by the transition probability.
    """

    name = "HMM"

    def __init__(self, n_components: int = 3, covariance_type: str = "full",
                 n_iter: int = 100, random_state: int = 42, **kwargs):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)
        self._model = None
        self._last_state = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("pip install hmmlearn")
        obs = y_train.reshape(-1, 1)
        self._model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(obs)
        _, state_seq = self._model.decode(obs, algorithm="viterbi")
        self._last_state = int(state_seq[-1])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        n = len(X_test)
        preds = np.empty(n, dtype=np.float32)
        A = self._model.transmat_       # (n_components, n_components)
        means = self._model.means_      # (n_components, 1)
        state = self._last_state
        for i in range(n):
            next_probs = A[state]
            preds[i] = float(np.dot(next_probs, means[:, 0]))
            state = int(np.argmax(A[state]))
        return preds

    def get_params(self) -> dict:
        return dict(n_components=self.n_components, covariance_type=self.covariance_type)


@register("MarkovSwitching")
class MarkovSwitchingModel(BaseModel):
    """
    Hamilton's Markov-Switching Autoregression (statsmodels).
    Fits k_regimes AR(order) processes with regime-specific parameters.
    Out-of-sample prediction uses the final smoothed regime probabilities
    to weight each regime's AR forecast — a regime-probability-weighted AR.
    """

    name = "MarkovSwitching"

    def __init__(self, k_regimes: int = 2, order: int = 1,
                 switching_ar: bool = True, **kwargs):
        self.k_regimes = int(k_regimes)
        self.order = int(order)
        self.switching_ar = bool(switching_ar)
        self._result = None
        self._n_train = 0
        self._y_train: np.ndarray = np.array([])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        except ImportError:
            raise ImportError("pip install statsmodels")
        self._n_train = len(y_train)
        model = MarkovAutoregression(
            y_train,
            k_regimes=self.k_regimes,
            order=self.order,
            switching_ar=self.switching_ar,
        )
        self._result = model.fit(disp=False)
        self._y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        n = len(X_test)
        res = self._result
        k = self.k_regimes

        # filtered_marginal_probabilities is a numpy array (nobs, k) in statsmodels >=0.14
        fmp = np.asarray(res.filtered_marginal_probabilities)
        last_probs = fmp[-1].astype(float)  # (k,)

        # Estimate transition matrix from the smoothed state sequence —
        # avoids internal statsmodels API differences across versions.
        smp = np.asarray(res.smoothed_marginal_probabilities)  # (nobs, k)
        state_seq = smp.argmax(axis=1)
        A = np.zeros((k, k))
        for t in range(len(state_seq) - 1):
            A[state_seq[t], state_seq[t + 1]] += 1
        row_sums = A.sum(axis=1, keepdims=True)
        A = np.where(row_sums > 0, A / row_sums, 1.0 / k)  # (k, k)

        # params layout for switching_ar=True, k_regimes=2, order=1:
        # [p01, p10, const_0, const_1, ar_0, ar_1, sigma2]
        # Transition probs occupy the first k*(k-1) entries; skip them.
        params = res.params
        n_trans = k * (k - 1)
        regime_params = params[n_trans:]
        intercepts = np.array([regime_params[i] for i in range(k)])
        ar_coefs = (np.array([regime_params[k + i] for i in range(k)])
                    if len(regime_params) > k else np.zeros(k))

        preds = np.empty(n, dtype=np.float32)
        probs = last_probs.copy()
        last_y = float(self._y_train[-1])

        for i in range(n):
            regime_forecasts = intercepts + ar_coefs * last_y
            preds[i] = float(np.dot(probs, regime_forecasts))
            last_y = preds[i]
            probs = probs @ A

        return preds

    def get_params(self) -> dict:
        return dict(k_regimes=self.k_regimes, order=self.order)
