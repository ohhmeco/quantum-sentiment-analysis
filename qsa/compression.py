"""Classical compression of sentence embeddings to 2^n dimensions.

An n-qubit amplitude encoding holds a vector of 2^n real numbers, so a d-dimensional
embedding (d = 300 for GloVe, 768 for BERT) has to be compressed to k = 2^n
components first. This module implements the compressors compared in the report:

* :class:`PCACompressor` - unsupervised projection on the k leading principal
  components (the method of the project proposal);
* :class:`FisherSelector` - supervised selection of the k original coordinates with
  the largest Fisher score (the selector used in the face-expression thesis);
* :class:`PCAFisherCompressor` - PCA to a pool of components, then the k components
  with the largest Fisher score: the *directions* of PCA chosen with the *supervised*
  criterion of the thesis (best of both, see the report);
* :class:`RandomProjection` - Gaussian random projection, a data-independent control.

All compressors are fitted on the training split only. :class:`AmplitudeNormalizer`
then maps the compressed vectors to valid quantum states (centring + L2 norm).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Compressor", "PCACompressor", "FisherSelector", "RandomProjection", "AmplitudeNormalizer",
           "build_compressor", "fisher_scores", "PCAFisherCompressor"]


def fisher_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-coordinate Fisher score (mu1 - mu0)^2 / (var0 + var1)."""
    x0, x1 = x[y == 0], x[y == 1]
    return (x1.mean(0) - x0.mean(0)) ** 2 / (x0.var(0) + x1.var(0) + 1e-12)


class Compressor(ABC):
    """Maps d-dimensional embeddings to k dimensions; fitted on training data only."""

    name = "compressor"
    supervised = False

    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "Compressor": ...

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(x, y).transform(x)

    def retained_variance(self, x: np.ndarray) -> float:
        """Fraction of the total (centred) variance of ``x`` kept by the compression."""
        xc = x - x.mean(0)
        z = self.transform(x)
        zc = z - z.mean(0)
        return float((zc**2).sum() / max((xc**2).sum(), 1e-300))


class PCACompressor(Compressor):
    """Projection on the k leading principal components of the training embeddings."""

    name = "pca"

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "PCACompressor":
        if self.k > min(x.shape):
            raise ValueError(f"k={self.k} larger than min(n_samples, dim)={min(x.shape)}")
        self.mean_ = x.mean(0)
        _, s, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        self.components_ = vt[: self.k]
        self.explained_variance_ratio_ = (s[: self.k] ** 2) / (s**2).sum()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) @ self.components_.T


class FisherSelector(Compressor):
    """Keeps the k original coordinates with the highest Fisher score (uses labels)."""

    name = "fisher"
    supervised = True

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "FisherSelector":
        if y is None:
            raise ValueError("FisherSelector is supervised: pass the training labels")
        self.scores_ = fisher_scores(x, y)
        self.indices_ = np.argsort(-self.scores_, kind="stable")[: self.k]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x[:, self.indices_]

    def retained_variance(self, x: np.ndarray) -> float:  # selected coordinates only
        xc = x - x.mean(0)
        return float((xc[:, self.indices_] ** 2).sum() / max((xc**2).sum(), 1e-300))


class PCAFisherCompressor(Compressor):
    """PCA to ``pool`` components, then keep the k components with the highest Fisher score.

    Plain PCA ranks directions by variance, which need not be sentiment; Fisher
    selection on raw coordinates ranks single axes, which in an embedding carry no
    meaning on their own. Ranking principal *directions* by their Fisher score keeps
    the decorrelation of PCA and the label-awareness of the thesis' selector.
    ``pool=None`` uses ``min(d, n_samples, max(64, 2k))`` components.
    """

    name = "pca-fisher"
    supervised = True

    def __init__(self, k: int, pool: int | None = None):
        super().__init__(k)
        self.pool = pool

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "PCAFisherCompressor":
        if y is None:
            raise ValueError("PCAFisherCompressor is supervised: pass the training labels")
        pool = self.pool or min(x.shape[1], x.shape[0], max(64, 2 * self.k))
        if self.k > pool:
            raise ValueError(f"k={self.k} larger than the pool of {pool} components")
        self.pca_ = PCACompressor(pool).fit(x)
        z = self.pca_.transform(x)
        self.scores_ = fisher_scores(z, y)
        self.indices_ = np.argsort(-self.scores_, kind="stable")[: self.k]
        self.components_ = self.pca_.components_[self.indices_]
        self.mean_ = self.pca_.mean_
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) @ self.components_.T


class RandomProjection(Compressor):
    """Projection on a random k-dimensional subspace (orthonormalised Gaussian matrix).

    It is independent of the data, so it isolates the effect of *choosing* the
    subspace (PCA, Fisher) from the effect of its size alone.
    """

    name = "random"

    def __init__(self, k: int, seed: int = 0):
        super().__init__(k)
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "RandomProjection":
        rng = np.random.default_rng(self.seed)
        q, _ = np.linalg.qr(rng.normal(size=(x.shape[1], self.k)))
        self.matrix_ = q[:, : self.k]
        self.mean_ = x.mean(0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) @ self.matrix_


def build_compressor(name: str, k: int, seed: int = 0) -> Compressor:
    if name == "pca":
        return PCACompressor(k)
    if name == "fisher":
        return FisherSelector(k)
    if name == "pca-fisher":
        return PCAFisherCompressor(k)
    if name == "random":
        return RandomProjection(k, seed)
    raise ValueError(f"unknown compressor {name!r}")


class AmplitudeNormalizer:
    """Turns compressed vectors into real amplitude vectors of unit norm.

    ``center=True`` subtracts the training mean before normalising. Without it,
    vectors whose coordinates share the same sign (e.g. raw selected coordinates)
    all point into the same orthant and their pairwise overlaps are close to 1,
    which is exactly what the interference classifier cannot discriminate. PCA
    and random projections are already centred by construction.
    Zero vectors (empty sentences) cannot be encoded and are reported as ``None``.
    """

    def __init__(self, center: bool = True):
        self.center = center

    def fit(self, z: np.ndarray) -> "AmplitudeNormalizer":
        self.mean_ = z.mean(0) if self.center else np.zeros(z.shape[1])
        return self

    def transform(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(states, valid_mask)``; invalid rows are left as zeros."""
        zc = z - self.mean_
        norms = np.linalg.norm(zc, axis=1)
        valid = norms > 1e-12
        states = np.zeros_like(zc)
        states[valid] = zc[valid] / norms[valid, None]
        return states, valid
