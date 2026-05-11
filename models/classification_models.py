import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as _RFClassifier

from .base_model import BaseModel, register


@register("LogisticRegressionClassifier")
class LogisticRegressionClassifier(BaseModel):
    name = "LogisticRegressionClassifier"
    task = "classification"

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 random_state: int = 42, **kwargs):
        self.params = dict(C=float(C), max_iter=int(max_iter),
                           random_state=int(random_state))
        self._model = LogisticRegression(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train.astype(int))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X_test)
        return np.where(raw >= 0, 1.0, -1.0).astype(np.float32)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(X_test)
        classes = list(self._model.classes_)
        pos_idx = classes.index(1) if 1 in classes else -1
        if pos_idx == -1:
            return np.zeros(len(X_test), dtype=np.float32)
        return proba[:, pos_idx].astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("XGBoostClassifier")
class XGBoostClassifier(BaseModel):
    name = "XGBoostClassifier"
    task = "classification"

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
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        from xgboost import XGBClassifier
        # Map {-1, +1} → {0, 1} for XGBoost
        y_bin = (y_train > 0).astype(int)
        self._model = XGBClassifier(**self.params, eval_metric="logloss")
        self._model.fit(X_train, y_bin)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X_test)
        return np.where(raw >= 0.5, 1.0, -1.0).astype(np.float32)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("RandomForestClassifier")
class RandomForestClassifier(BaseModel):
    name = "RandomForestClassifier"
    task = "classification"

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 random_state: int = 42, **kwargs):
        self.params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth is not None else None,
            random_state=int(random_state),
            n_jobs=-1,
        )
        self._model = _RFClassifier(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, (y_train > 0).astype(int))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X_test)
        return np.where(raw >= 0.5, 1.0, -1.0).astype(np.float32)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    def get_params(self) -> dict:
        return self.params


@register("LightGBMClassifier")
class LightGBMClassifier(BaseModel):
    name = "LightGBMClassifier"
    task = "classification"

    def __init__(self, n_estimators: int = 200, max_depth: int = -1,
                 learning_rate: float = 0.05, num_leaves: int = 31,
                 random_state: int = 42, **kwargs):
        self.params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            random_state=int(random_state),
            verbose=-1,
        )
        self._model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("pip install lightgbm")
        self._model = lgb.LGBMClassifier(**self.params)
        self._model.fit(X_train, (y_train > 0).astype(int))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X_test)
        return np.where(raw >= 0.5, 1.0, -1.0).astype(np.float32)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X_test)[:, 1].astype(np.float32)

    def get_params(self) -> dict:
        return self.params
