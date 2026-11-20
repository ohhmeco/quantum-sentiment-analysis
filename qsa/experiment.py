"""Experiment orchestration: configuration, the main grid, and result files.

An experiment is described by a JSON file (see ``configs/``) and produces, in
``results/<name>/``:

* ``metrics.csv``    - one row per (embedder, compressor, qubits, model, protocol,
  shots, seed) with accuracy, precision, recall, F1, MCC, 95% bootstrap interval and
  quantum diagnostics (tie rate, single-vote accuracy, acceptance probability);
* ``retention.csv``  - one row per (embedder, compressor, qubits) with variance,
  neighbourhood and similarity retention;
* ``config.json`` and ``environment.json`` - what was run and with which versions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd

from .baselines import CosineKNN, LogisticProbe, NearestCentroid
from .compression import AmplitudeNormalizer, build_compressor
from .data import load_sst2
from .embeddings import EmbeddingCache, build_embedder
from .metrics import bootstrap_ci, classification_metrics, neighbourhood_overlap, similarity_correlation
from .quantum.classifier import InterferenceClassifier

LOGGER = logging.getLogger("qsa")

__all__ = ["ExperimentConfig", "ExperimentRunner", "Representation"]


@dataclass
class ExperimentConfig:
    name: str
    data_dir: str = "data/sst2"
    cache_dir: str = "data/embeddings/cache"
    output_dir: str = "results"
    embedders: list[dict] = field(default_factory=lambda: [{"kind": "glove-sif"}])
    compressors: list[str] = field(default_factory=lambda: ["pca"])
    qubits: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    protocols: list[dict] = field(default_factory=lambda: [{"protocol": "pairs", "n_pairs": 100}])
    shots: list[int | None] = field(default_factory=lambda: [1024])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    center: bool = True
    baselines: list[str] = field(default_factory=lambda: ["logistic-regression", "nearest-centroid", "cosine-knn"])
    retention_sample: int = 1000
    bootstrap: int = 1000

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text()))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


@dataclass
class Representation:
    """Compressed and encoded vectors of every split for one (embedder, compressor, k)."""

    embedder: str
    compressor: str
    qubits: int
    compressed: dict[str, np.ndarray]   # k-dimensional vectors
    states: dict[str, np.ndarray]       # unit-norm amplitude vectors
    valid: dict[str, np.ndarray]        # False for vectors that cannot be encoded
    variance_retention: float

    @property
    def k(self) -> int:
        return 1 << self.qubits


def build_representation(embedder_name: str, embeddings: dict[str, np.ndarray], y_train: np.ndarray,
                         compressor_name: str, qubits: int, center: bool = True, seed: int = 0) -> Representation:
    """Fit the compressor and the normaliser on the training split and transform every split."""
    comp = build_compressor(compressor_name, 1 << qubits, seed=seed).fit(embeddings["train"], y_train)
    compressed = {s: comp.transform(x) for s, x in embeddings.items()}
    norm = AmplitudeNormalizer(center=center).fit(compressed["train"])
    states, valid = {}, {}
    for split, z in compressed.items():
        states[split], valid[split] = norm.transform(z)
    return Representation(embedder_name, compressor_name, qubits, compressed, states, valid,
                          comp.retained_variance(embeddings["train"]))


class ExperimentRunner:
    """Runs the grid described by an :class:`ExperimentConfig`."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.out_dir = Path(config.output_dir) / config.name
        self.metrics: list[dict[str, Any]] = []
        self.retention: list[dict[str, Any]] = []

    # helpers ---------------------------------------------------------------------------------

    def _record(self, base: dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray, **extra: Any) -> None:
        row = {**base, **classification_metrics(y_true, y_pred)}
        if self.config.bootstrap:
            row["ci_low"], row["ci_high"] = bootstrap_ci(y_true, y_pred, n_boot=self.config.bootstrap)
        row.update(extra)
        self.metrics.append(row)

    def _flush(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.metrics).to_csv(self.out_dir / "metrics.csv", index=False)
        pd.DataFrame(self.retention).to_csv(self.out_dir / "retention.csv", index=False)

    def _write_provenance(self) -> None:
        import qiskit
        import sklearn

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_json(self.out_dir / "config.json")
        env = {"python": platform.python_version(), "numpy": np.__version__, "qiskit": qiskit.__version__,
               "scikit-learn": sklearn.__version__, "platform": platform.platform(),
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.out_dir / "environment.json").write_text(json.dumps(env, indent=2))

    # main loop -------------------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        cfg = self.config
        self._write_provenance()
        data = load_sst2(cfg.data_dir)
        y = {"train": data.train.labels, "test": data.test.labels}
        if data.dev is not None:
            y["dev"] = data.dev.labels
        cache = EmbeddingCache(cfg.cache_dir)
        rng = np.random.default_rng(12345)
        retention_idx = rng.choice(len(data.test), size=min(cfg.retention_sample, len(data.test)), replace=False)

        for spec in cfg.embedders:
            embedder = build_embedder(spec)
            emb = cache.get_or_compute(embedder, data)
            name, dim = embedder.name, emb["train"].shape[1]
            LOGGER.info("[%s] dimension %d", name, dim)

            # Full-embedding reference (defines 100% retention).
            probe = LogisticProbe().fit(emb["train"], y["train"], emb.get("dev"), y.get("dev"))
            self._record({"embedder": name, "compressor": "none", "qubits": np.nan, "k": dim,
                          "model": "logistic-regression", "protocol": "", "shots": np.nan, "seed": 0},
                         y["test"], probe.predict(emb["test"]), c=probe.c_)

            for comp_name in cfg.compressors:
                for q in cfg.qubits:
                    if (1 << q) > dim:
                        continue
                    t0 = time.perf_counter()
                    rep = build_representation(name, emb, y["train"], comp_name, q, cfg.center)
                    self._evaluate_representation(rep, y, emb, retention_idx)
                    self._flush()
                    LOGGER.info("[%s/%s] %d qubits done in %.1fs", name, comp_name, q, time.perf_counter() - t0)
        self._flush()
        return pd.DataFrame(self.metrics)

    def _evaluate_representation(self, rep: Representation, y: dict[str, np.ndarray],
                                 emb: dict[str, np.ndarray], retention_idx: np.ndarray) -> None:
        cfg = self.config
        base = {"embedder": rep.embedder, "compressor": rep.compressor, "qubits": rep.qubits, "k": rep.k}
        full_test = emb["test"][retention_idx]
        self.retention.append({
            **base, "variance_retention": rep.variance_retention,
            "neighbourhood_overlap": neighbourhood_overlap(full_test, rep.compressed["test"][retention_idx]),
            "similarity_correlation": similarity_correlation(full_test, rep.compressed["test"][retention_idx]),
            "invalid_test_vectors": int((~rep.valid["test"]).sum()),
        })
        tr_ok = rep.valid["train"]
        x_tr, y_tr = rep.states["train"][tr_ok], y["train"][tr_ok]

        # Classical counterparts on the same representation.
        if "logistic-regression" in cfg.baselines:
            probe = LogisticProbe().fit(rep.compressed["train"], y["train"], rep.compressed.get("dev"), y.get("dev"))
            self._record({**base, "model": "logistic-regression", "protocol": "", "shots": np.nan, "seed": 0},
                         y["test"], probe.predict(rep.compressed["test"]), c=probe.c_)
        if "nearest-centroid" in cfg.baselines:
            self._record({**base, "model": "nearest-centroid", "protocol": "", "shots": np.nan, "seed": 0},
                         y["test"], NearestCentroid().fit(x_tr, y_tr).predict(rep.states["test"]))
        if "cosine-knn" in cfg.baselines:
            self._record({**base, "model": "cosine-knn", "protocol": "", "shots": np.nan, "seed": 0},
                         y["test"], CosineKNN().fit(x_tr, y_tr).predict(rep.states["test"]))

        # Quantum interference classifier.
        for proto in cfg.protocols:
            for shots in cfg.shots:
                for seed in cfg.seeds:
                    clf = InterferenceClassifier(**proto, shots=shots, seed=seed).fit(x_tr, y_tr)
                    report = clf.predict_report(rep.states["test"])
                    extra = {"tie_rate": float(report.ties.mean()), "mean_acceptance": float(report.acceptance.mean())}
                    if report.pair_votes is not None:
                        votes = report.pair_votes
                        hit = np.where(votes == -1, 0.5, (votes == y["test"][:, None]).astype(float))
                        extra["single_vote_accuracy"] = float(hit.mean())
                        extra["n_pairs"] = votes.shape[1]
                    self._record({**base, "model": "quantum-interference", "protocol": proto["protocol"],
                                  "shots": np.nan if shots is None else shots, "seed": seed},
                                 y["test"], report.labels, **extra)
