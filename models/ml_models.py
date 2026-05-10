import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .base_model import BaseModel, register


@register("XGBoost")
class XGBoostModel(BaseModel):
    name = "XGBoost"

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 1.0,
                 random_state: int = 42, **kwargs):
        self.params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            random_state=int(random_state),
            verbosity=0,
        )
        self._model = XGBRegressor(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("RandomForest")
class RandomForestModel(BaseModel):
    name = "RandomForest"

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 random_state: int = 42, **kwargs):
        self.params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth is not None else None,
            random_state=int(random_state),
            n_jobs=-1,
        )
        self._model = RandomForestRegressor(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("SVR")
class SVRModel(BaseModel):
    """SVR regression. Input is already scaled by the feature pipeline."""

    name = "SVR"

    def __init__(self, C: float = 1.0, epsilon: float = 0.1,
                 kernel: str = "rbf", **kwargs):
        self.params = dict(C=float(C), epsilon=float(epsilon), kernel=kernel)
        self._model = SVR(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("Ridge")
class RidgeModel(BaseModel):
    name = "Ridge"

    def __init__(self, alpha: float = 1.0, **kwargs):
        self.params = dict(alpha=float(alpha))
        self._model = Ridge(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("Lasso")
class LassoModel(BaseModel):
    name = "Lasso"

    def __init__(self, alpha: float = 0.001, **kwargs):
        self.params = dict(alpha=float(alpha))
        self._model = Lasso(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("ElasticNet")
class ElasticNetModel(BaseModel):
    name = "ElasticNet"

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5, **kwargs):
        self.params = dict(alpha=float(alpha), l1_ratio=float(l1_ratio))
        self._model = ElasticNet(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("LightGBM")
class LightGBMModel(BaseModel):
    name = "LightGBM"

    def __init__(self, n_estimators: int = 200, max_depth: int = -1,
                 learning_rate: float = 0.05, num_leaves: int = 31,
                 subsample: float = 0.8, random_state: int = 42, **kwargs):
        self.params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            subsample=float(subsample),
            random_state=int(random_state),
            verbose=-1,
        )
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("pip install lightgbm")
        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LightGBMModel.fit() must be called before predict()")
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("CatBoost")
class CatBoostModel(BaseModel):
    name = "CatBoost"

    def __init__(self, iterations: int = 200, depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42, **kwargs):
        self.params = dict(
            iterations=int(iterations),
            depth=int(depth),
            learning_rate=float(learning_rate),
            random_seed=int(random_state),
            verbose=0,
        )
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("pip install catboost")
        self._model = CatBoostRegressor(**self.params)
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CatBoostModel.fit() must be called before predict()")
        return self._model.predict(X_test).astype(np.float32)

    def get_params(self) -> dict:
        return self.params
