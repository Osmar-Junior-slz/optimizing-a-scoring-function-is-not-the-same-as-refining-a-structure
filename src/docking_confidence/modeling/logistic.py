from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticBaseModel:
    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", max_iter: int = 5000):
        if C <= 0 or max_iter < 1:
            raise ValueError("invalid logistic-regression configuration")
        self.C = float(C)
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            class_weight=class_weight,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("invalid logistic-regression training arrays")
        if not np.isfinite(X).all():
            raise ValueError("logistic-regression features must be finite")
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
            raise ValueError("logistic-regression training requires both binary classes")
        Xs = self.scaler.fit_transform(X)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(Xs, y)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError("logistic-regression optimization did not converge")
        if not np.isfinite(self.model.coef_).all() or not np.isfinite(self.model.intercept_).all():
            raise RuntimeError("logistic-regression produced non-finite parameters")
        return self

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("logistic-regression query features must be a finite matrix")
        Xs = self.scaler.transform(X)
        scores = np.asarray(self.model.decision_function(Xs), dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError("logistic-regression produced non-finite scores")
        return scores

    def export(self, feature_columns: list[str] | None = None) -> dict:
        if not hasattr(self.model, "coef_"):
            raise RuntimeError("logistic-regression model is not fitted")
        scale = np.where(self.scaler.scale_ == 0.0, 1.0, self.scaler.scale_)
        original_coef = self.model.coef_[0] / scale
        original_intercept = float(
            self.model.intercept_[0]
            - np.sum(self.model.coef_[0] * self.scaler.mean_ / scale)
        )
        return {
            "method": "standardized_logistic_regression_v1",
            "positive_class": 1,
            "score_orientation": "larger_means_more_likely_preserved",
            "formula": "raw_score = intercept + sum(((x_j - mean_j) / scale_j) * coef_j)",
            "C": self.C,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "solver": "lbfgs",
            "n_iter": int(self.model.n_iter_[0]),
            "feature_columns": feature_columns,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": scale.tolist(),
            "coef": self.model.coef_[0].tolist(),
            "intercept": float(self.model.intercept_[0]),
            "original_scale_coef": original_coef.tolist(),
            "original_scale_intercept": original_intercept,
        }
