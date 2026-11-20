"""Classical counterparts evaluated on exactly the same representations.

* :class:`NearestCentroid` - cosine nearest centroid on the encoded unit vectors.
  It is the infinite-shot limit of the *superposition* protocol, i.e. what the
  quantum circuit computes when sampling noise vanishes.
* :class:`CosineKNN` - k-nearest neighbours with cosine similarity, the classical
  member of the distance-based family.
* :class:`LogisticProbe` - L2-regularised logistic regression. On the compressed
  vectors it measures how much *linearly decodable* sentiment information survives
  the compression (a "probe"); on the full embedding it is the reference that
  defines 100% retention.
"""
from __future__ import annotations

import numpy as np

__all__ = ["NearestCentroid", "CosineKNN", "LogisticProbe"]


class NearestCentroid:
    """argmax_c <x, mu_c> with mu_c the mean of the unit training vectors of class c."""

    name = "nearest-centroid"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        self.means_ = np.stack([x[y == c].mean(0) for c in (0, 1)])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        s = x @ self.means_.T
        return (s[:, 1] > s[:, 0]).astype(int)


class CosineKNN:
    name = "cosine-knn"

    def __init__(self, k: int = 15):
        self.k = k

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CosineKNN":
        self.x_, self.y_ = x, y
        return self

    def predict(self, x: np.ndarray, batch: int = 512) -> np.ndarray:
        out = []
        for start in range(0, len(x), batch):
            sim = x[start:start + batch] @ self.x_.T
            nn = np.argpartition(-sim, self.k, axis=1)[:, : self.k]
            out.append((self.y_[nn].mean(1) > 0.5).astype(int))
        return np.concatenate(out)


class LogisticProbe:
    """Standardised features + logistic regression (C chosen on the dev split if given)."""

    name = "logistic-regression"

    def __init__(self, c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0), seed: int = 0):
        self.c_grid, self.seed = c_grid, seed

    def fit(self, x: np.ndarray, y: np.ndarray, x_dev: np.ndarray | None = None,
            y_dev: np.ndarray | None = None) -> "LogisticProbe":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def model(c: float):
            return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, random_state=self.seed))

        if x_dev is None or len(self.c_grid) == 1:
            self.c_ = self.c_grid[len(self.c_grid) // 2]
        else:
            scores = [(model(c).fit(x, y).score(x_dev, y_dev), -c) for c in self.c_grid]
            self.c_ = -max(scores)[1]
        self.model_ = model(self.c_).fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(x).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of the positive class."""
        return self.model_.predict_proba(x)[:, 1]

