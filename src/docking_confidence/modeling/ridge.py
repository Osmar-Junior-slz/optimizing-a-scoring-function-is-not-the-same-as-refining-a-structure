"""A frozen linear score for a continuous preservation margin."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from docking_confidence.modeling.extra_trees import complex_weights


class RidgeBaseModel:
    """Fit one standardized Ridge regression, without parameter tuning."""

    def __init__(self, alpha: float = 1.0):
        if isinstance(alpha, bool):
            raise ValueError("ridge alpha must be finite and positive")
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("ridge alpha must be finite and positive")
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, fit_intercept=True, solver="svd")

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
            raise ValueError("ridge features must be a nonempty matrix")
        if not np.isfinite(values).all():
            raise ValueError("ridge features must be finite")
        return values

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None) -> "RidgeBaseModel":
        values = self._features(X)
        outcomes = np.asarray(y, dtype=float)
        if outcomes.ndim != 1 or len(outcomes) != len(values):
            raise ValueError("ridge outcomes must be a vector aligned with features")
        if not np.isfinite(outcomes).all():
            raise ValueError("ridge outcomes must be finite")
        self.molecular_weighting = groups is not None
        if groups is None:
            # Preserve the historical unweighted fitting path for V1 replay tests.
            scaled = self.scaler.fit_transform(values)
            self.model.fit(scaled, outcomes)
        else:
            weights = complex_weights(groups, len(values))
            self.scaler.fit(values, sample_weight=weights)
            scaled = self.scaler.transform(values)
            self.model.fit(scaled, outcomes, sample_weight=weights)
        if not np.isfinite(self.model.coef_).all() or not np.isfinite(self.model.intercept_):
            raise RuntimeError("ridge regression produced non-finite parameters")
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self.model, "coef_"):
            raise RuntimeError("ridge-regression model is not fitted")

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = self._features(X)
        scores = np.asarray(self.model.predict(self.scaler.transform(values)), dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError("ridge regression produced non-finite scores")
        return scores

    def export(self, feature_columns: list[str] | None = None) -> dict:
        self._require_fitted()
        if feature_columns is not None and len(feature_columns) != len(self.model.coef_):
            raise ValueError("ridge feature names must match the fitted feature count")
        scale = self.scaler.scale_
        coef = self.model.coef_
        original_coef = coef / scale
        original_intercept = float(self.model.intercept_ - np.sum(original_coef * self.scaler.mean_))
        exported = {
            "method": "standardized_ridge_regression_v1",
            "score_orientation": "larger_means_larger_preservation_margin",
            "formula": "raw_score = intercept + sum(((x_j - mean_j) / scale_j) * coef_j)",
            "alpha": self.alpha,
            "fit_intercept": True,
            "solver": "svd",
            "feature_columns": list(feature_columns) if feature_columns is not None else None,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": scale.tolist(),
            "coef": coef.tolist(),
            "intercept": float(self.model.intercept_),
            "original_scale_coef": original_coef.tolist(),
            "original_scale_intercept": original_intercept,
        }
        if self.molecular_weighting:
            exported["weighting"] = "equal_total_weight_per_molecular_complex"
        return exported
