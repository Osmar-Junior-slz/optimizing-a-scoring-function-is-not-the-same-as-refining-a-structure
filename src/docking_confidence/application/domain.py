from __future__ import annotations

import numpy as np


class FeatureDomain:
    def __init__(self, lower_quantile: float, upper_quantile: float):
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("domain quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower: np.ndarray | None = None
        self.upper: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureDomain":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("domain training data must be a non-empty finite matrix")
        self.lower = np.quantile(X, self.lower_quantile, axis=0)
        self.upper = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def contains(self, X: np.ndarray) -> np.ndarray:
        if self.lower is None or self.upper is None:
            raise RuntimeError("domain not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.lower):
            raise ValueError("domain query has incompatible feature shape")
        if not np.isfinite(X).all():
            return np.zeros(len(X), dtype=bool)
        return ((X >= self.lower) & (X <= self.upper)).all(axis=1)

    def export(self) -> dict:
        if self.lower is None or self.upper is None:
            raise RuntimeError("domain not fitted")
        return {
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }
