"""The interference-based quantum classifier as a scikit-learn-like estimator.

Three protocols are implemented (see :mod:`qsa.quantum.circuits` for the circuits):

``pairs``
    For every test sentence, ``n_pairs`` (negative, positive) reference pairs are
    drawn from the training set; each pair is one execution of the pair circuit with
    ``shots`` measurements and votes for the class measured more often after
    post-selection (a = 0). The final label is the majority vote over the pairs
    (the "WholeSet" protocol of Mengoni et al.); the accuracy of the single votes is
    also reported ("SingleFace").

``centroid``
    One execution of the pair circuit per test sentence, whose two references are the
    *normalised* class centroids ``mu_c / ||mu_c||`` of the training states (computed
    classically once). In the infinite-shot limit it compares the cosine similarity of
    the test state with the two class directions; ``superposition`` compares the plain
    inner products ``<t|mu_c>``, which differ only when the two class means have different
    norms. The margin ``<t|mu_1/||mu_1|| - mu_0/||mu_0||>/4`` is not shrunk by the small
    norm of the class means, so it needs far fewer shots, and no QRAM.

``superposition``
    One circuit per test sentence with all ``n_references`` references (half per
    class) indexed in superposition by a QRAM. It needs a logarithmic number of extra
    qubits, but the signal shrinks: ``P(a=0,c)`` differs between classes only by
    ``<t|mu_1 - mu_0>/4``.

Backends
--------
``statevector`` (default) computes the exact output distribution of each circuit in
closed form, ``P(a=0, c) = (1 + <t|x_c>)/4``, and samples ``shots`` outcomes from it
(``shots=None`` gives the infinite-shot limit). This is mathematically identical to
simulating the noiseless circuit (validated against Qiskit Aer in the tests and in
``scripts/validate_circuits.py``) and fast enough for the whole test set.
``aer`` builds and simulates the actual circuits with Qiskit Aer (Grover-Rudolph or
explicit QRAM loaders); use it on small subsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuits import interpret_counts, pair_circuit, qram_pair_circuit, superposition_circuit

__all__ = ["InterferenceClassifier", "PredictionReport"]


@dataclass
class PredictionReport:
    """Predictions plus the diagnostics used in the analysis."""

    labels: np.ndarray                      # final labels (ties broken by a fair coin)
    ties: np.ndarray                        # True where the vote was a tie
    acceptance: np.ndarray                  # mean P(a=0) per test sentence (exact)
    margin: np.ndarray                      # exact P(a=0,c=1) - P(a=0,c=0) (mean over pairs)
    p_positive: np.ndarray | None = field(default=None)  # exact P(c=1 | a=0), mean over circuits
    pair_votes: np.ndarray | None = field(default=None)  # (n_test, n_pairs), -1 = tie


class InterferenceClassifier:
    """Distance-based interference classifier (Schuld et al., 2017).

    Parameters
    ----------
    protocol : ``"pairs"``, ``"centroid"`` or ``"superposition"``.
    n_pairs : number of reference pairs per test sentence (``pairs`` protocol).
    n_references : references in the superposition circuit (``None`` = all training).
    shots : measurements per circuit; ``None`` = exact probabilities.
    backend : ``"statevector"`` or ``"aer"``.
    loader : ``"grover-rudolph"``, ``"qram-qrom"`` or ``"qram-bb"`` (``aer`` backend, pairs only).
    qram_bits : precision of the stored angles for the QRAM loaders.
    seed : seed of every random choice (references, shots, tie breaking).
    """

    def __init__(self, protocol: str = "pairs", n_pairs: int = 100, n_references: int | None = None,
                 shots: int | None = 1024, backend: str = "statevector", loader: str = "grover-rudolph",
                 qram_bits: int = 8, seed: int = 0):
        if protocol not in ("pairs", "superposition", "centroid"):
            raise ValueError("protocol must be 'pairs', 'superposition' or 'centroid'")
        if backend not in ("statevector", "aer"):
            raise ValueError("backend must be 'statevector' or 'aer'")
        if loader not in ("grover-rudolph", "qram-qrom", "qram-bb"):
            raise ValueError("loader must be grover-rudolph, qram-qrom or qram-bb")
        self.protocol, self.n_pairs, self.n_references = protocol, n_pairs, n_references
        self.shots, self.backend, self.loader, self.qram_bits, self.seed = shots, backend, loader, qram_bits, seed

    # fitting ------------------------------------------------------------------------------

    def fit(self, states: np.ndarray, labels: np.ndarray) -> "InterferenceClassifier":
        """Store the encoded training states (rows of unit norm) and their labels."""
        states = np.asarray(states, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)
        if not np.allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-8):
            raise ValueError("training states must have unit norm")
        rng = np.random.default_rng(self.seed)
        neg, pos = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        if self.protocol == "pairs":
            self.pairs_ = np.stack([rng.choice(neg, self.n_pairs, replace=len(neg) < self.n_pairs),
                                    rng.choice(pos, self.n_pairs, replace=len(pos) < self.n_pairs)], axis=1)
            self.references_ = states[self.pairs_]  # (n_pairs, 2, dim)
        elif self.protocol == "centroid":
            means = np.stack([states[labels == c].mean(0) for c in (0, 1)])
            self.references_ = (means / np.linalg.norm(means, axis=1, keepdims=True))[None]  # (1, 2, dim)
        else:
            if self.n_references is None:
                idx = np.arange(len(states))
            else:
                half = self.n_references // 2
                idx = np.concatenate([rng.choice(neg, half, replace=False), rng.choice(pos, half, replace=False)])
            self.reference_states_, self.reference_labels_ = states[idx], labels[idx]
            self.class_means_ = np.stack([states[idx][labels[idx] == c].mean(0) for c in (0, 1)])
        self.dim_ = states.shape[1]
        return self

    # prediction -----------------------------------------------------------------------------

    def exact_probabilities(self, test_states: np.ndarray) -> np.ndarray:
        """``[P(a=0,c=0), P(a=0,c=1), P(a=1)]`` per test sentence (and per pair)."""
        t = np.asarray(test_states, dtype=np.float64)
        if self.protocol in ("pairs", "centroid"):
            o = np.einsum("td,pcd->tpc", t, self.references_)          # overlaps (n_test, n_pairs, 2)
        else:
            o = (t @ self.class_means_.T)[:, None, :]                   # (n_test, 1, 2)
        p = (1.0 + o) / 4.0
        return np.concatenate([p, 1.0 - p.sum(-1, keepdims=True)], axis=-1)

    def predict_report(self, test_states: np.ndarray) -> PredictionReport:
        probs = self.exact_probabilities(test_states)                    # (n_test, n_circuits, 3)
        rng = np.random.default_rng(self.seed + 1)
        if self.backend == "aer":
            counts = self._aer_counts(np.asarray(test_states))
        elif self.shots is None:
            counts = None
        else:
            counts = rng.multinomial(self.shots, np.clip(probs, 0.0, None) / np.clip(probs, 0.0, None).sum(-1, keepdims=True))
        score = probs if counts is None else counts
        votes = np.where(np.isclose(score[..., 1], score[..., 0], atol=1e-12, rtol=0), -1,
                         (score[..., 1] > score[..., 0]).astype(int))
        if self.protocol == "pairs":
            positive, negative = (votes == 1).sum(1), (votes == 0).sum(1)
            labels = (positive > negative).astype(int)
            ties = positive == negative
        else:
            labels, ties = votes[:, 0].clip(min=0), votes[:, 0] == -1
        coin = rng.integers(0, 2, size=len(labels))
        labels = np.where(ties, coin, labels)
        return PredictionReport(
            labels=labels, ties=ties,
            acceptance=probs[..., :2].sum(-1).mean(1),
            margin=(probs[..., 1] - probs[..., 0]).mean(1),
            p_positive=(probs[..., 1] / probs[..., :2].sum(-1)).mean(1),
            pair_votes=votes if self.protocol in ("pairs", "centroid") else None,
        )

    def predict(self, test_states: np.ndarray) -> np.ndarray:
        return self.predict_report(test_states).labels

    # Qiskit Aer backend ---------------------------------------------------------------------

    def _aer_counts(self, test_states: np.ndarray) -> np.ndarray:
        """Simulate every circuit with Qiskit Aer; returns counts (n_test, n_circuits, 3)."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        if self.shots is None:
            raise ValueError("the aer backend needs a finite number of shots")
        simulator = AerSimulator(method="statevector")
        circuits = []
        for t in test_states:
            if self.protocol in ("pairs", "centroid"):
                for x0, x1 in self.references_:
                    if self.loader == "grover-rudolph":
                        circuits.append(pair_circuit(t, x0, x1))
                    else:
                        circuits.append(qram_pair_circuit(t, x0, x1, bits=self.qram_bits, query=self.loader[5:]))
            else:
                circuits.append(superposition_circuit(t, self.reference_states_, self.reference_labels_))
        compiled = transpile(circuits, simulator, optimization_level=0, seed_transpiler=self.seed)
        result = simulator.run(compiled, shots=self.shots, seed_simulator=self.seed).result()
        per_test = len(circuits) // len(test_states)
        counts = np.array([interpret_counts(result.get_counts(i)) for i in range(len(circuits))], dtype=float)
        return counts.reshape(len(test_states), per_test, 3)
