"""Evaluation metrics: classification quality and retention of semantic information.

Classification: accuracy, precision, recall and F1 of the positive class, plus the
Matthews correlation coefficient. Bootstrap confidence intervals quantify the
uncertainty due to the finite test set.

Retention, at compression size k = 2^n:

* ``variance_retention``    - fraction of embedding variance kept by the compressor;
* ``probe_retention``       - accuracy of a logistic probe on the k-dimensional
  vectors divided by the probe accuracy on the full embedding (how much linearly
  decodable sentiment survives);
* ``pipeline_retention``    - same ratio for the quantum classifier;
* ``neighbourhood_overlap`` - mean fraction of the 10 nearest cosine neighbours of a
  sentence that are still among its 10 nearest neighbours after compression
  (geometry, i.e. *semantic* similarity structure, independent of the labels);
* ``similarity_correlation``- Spearman correlation between pairwise cosine
  similarities before and after compression.
"""
from __future__ import annotations

import numpy as np

__all__ = ["classification_metrics", "bootstrap_ci", "neighbourhood_overlap", "similarity_correlation",
           "unit_rows"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"accuracy": (tp + tn) / len(y_true), "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval of the accuracy over test sentences."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    samples = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def _knn(x: np.ndarray, k: int) -> np.ndarray:
    sim = unit_rows(x) @ unit_rows(x).T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k, axis=1)[:, :k]


def neighbourhood_overlap(full: np.ndarray, compressed: np.ndarray, k: int = 10) -> float:
    """Mean |kNN_full(i) & kNN_compressed(i)| / k over the rows (cosine similarity)."""
    a, b = _knn(full, k), _knn(compressed, k)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))]))


def similarity_correlation(full: np.ndarray, compressed: np.ndarray) -> float:
    """Spearman correlation of the pairwise cosine similarities (upper triangle)."""
    from scipy.stats import spearmanr

    iu = np.triu_indices(len(full), k=1)
    s_full = (unit_rows(full) @ unit_rows(full).T)[iu]
    s_comp = (unit_rows(compressed) @ unit_rows(compressed).T)[iu]
    return float(spearmanr(s_full, s_comp).statistic)
