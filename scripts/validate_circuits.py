"""Validate the closed-form backend against Qiskit Aer and measure circuit resources.

1. Validation: on a subset of SST-2 test sentences, the pair circuit (Grover-Rudolph
   loaders, and explicit QRAM loaders for small n) and the superposition circuit are
   simulated with Qiskit Aer. The measured frequencies are compared with the exact
   probabilities used by the fast ``statevector`` backend (total-variation distance
   and a chi-square goodness-of-fit test), and the predicted labels are compared.
2. Resources: qubits, depth and CNOT count after transpilation to {u, cx} of the
   Grover-Rudolph loader, the pair circuit and the QRAM pair circuits, n = 1..8.

    python scripts/validate_circuits.py            # writes results/circuits/*.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa.data import load_sst2  # noqa: E402
from qsa.embeddings import EmbeddingCache, GloVeSIFEmbedder  # noqa: E402
from qsa.experiment import build_representation  # noqa: E402
from qsa.quantum import GroverRudolphLoader, InterferenceClassifier, pair_circuit, qram_pair_circuit  # noqa: E402
from qsa.quantum import quantized_state, resources  # noqa: E402
from qsa.quantum.circuits import pair_probabilities, superposition_circuit  # noqa: E402

LOGGER = logging.getLogger("qsa")


def validation(args) -> pd.DataFrame:
    from scipy.stats import chisquare

    data = load_sst2(args.data_dir)
    emb = EmbeddingCache(args.cache_dir).get_or_compute(GloVeSIFEmbedder(), data)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data.test), size=args.sentences, replace=False)
    rows = []
    settings = [("grover-rudolph", q) for q in (1, 2, 3)] + [("qram-qrom", q) for q in (1, 2)]
    for loader, q in settings:
        rep = build_representation("glove-sif", emb, data.train.labels, "pca", q)
        tests = rep.states["test"][idx]
        y = data.test.labels[idx]
        n_pairs = args.pairs if loader == "grover-rudolph" else max(1, args.pairs // 2)
        common = dict(protocol="pairs", n_pairs=n_pairs, shots=args.shots, seed=3, qram_bits=args.qram_bits)
        t0 = time.perf_counter()
        aer = InterferenceClassifier(**common, backend="aer", loader=loader).fit(rep.states["train"], data.train.labels)
        counts = aer._aer_counts(tests)                                   # (sentences, pairs, 3)
        elapsed = time.perf_counter() - t0
        fast = InterferenceClassifier(**common, backend="statevector").fit(rep.states["train"], data.train.labels)
        refs = aer.references_
        if loader.startswith("qram"):  # quantised angles: compare with the state the QRAM really loads
            def pair_probs(t, a, b):  # one memory per circuit: shared signed range
                signed = any(np.any(v < -1e-15) for v in (t, a, b))
                return pair_probabilities(*(quantized_state(v, args.qram_bits, signed) for v in (t, a, b)))

            probs = np.array([[pair_probs(t, a, b) for a, b in refs] for t in tests])
        else:
            probs = fast.exact_probabilities(tests)
        freq = counts / counts.sum(-1, keepdims=True)
        tvd = 0.5 * np.abs(freq - probs).sum(-1)
        pooled = chisquare(counts.reshape(-1, 3).sum(0), probs.reshape(-1, 3).sum(0) * args.shots)
        aer_votes = np.where(counts[..., 1] == counts[..., 0], -1, (counts[..., 1] > counts[..., 0]).astype(int))
        exact_votes = np.where(np.isclose(probs[..., 1], probs[..., 0]), -1, (probs[..., 1] > probs[..., 0]).astype(int))
        aer_labels = ((aer_votes == 1).sum(1) > (aer_votes == 0).sum(1)).astype(int)
        exact_labels = ((exact_votes == 1).sum(1) > (exact_votes == 0).sum(1)).astype(int)
        rows.append({
            "loader": loader, "qubits": q, "sentences": len(idx), "pairs": n_pairs, "shots": args.shots,
            "circuits": counts.shape[0] * counts.shape[1], "aer_seconds": round(elapsed, 1),
            "mean_tvd": float(tvd.mean()), "max_tvd": float(tvd.max()),
            "expected_tvd_shot_noise": float(np.mean(np.sqrt(probs * (1 - probs) / args.shots).sum(-1)) * np.sqrt(2 / np.pi) / 2),
            "pooled_chi2_pvalue": float(pooled.pvalue),
            "vote_agreement_with_exact": float((aer_votes == exact_votes).mean()),
            "label_agreement_with_exact": float((aer_labels == exact_labels).mean()),
            "aer_accuracy": float((aer_labels == y).mean()), "exact_accuracy": float((exact_labels == y).mean()),
        })
        LOGGER.info("validation %s n=%d: %s", loader, q, rows[-1])

    # superposition protocol (ideal QRAM oracle) on a few sentences
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector")
    for q in (1, 2, 3):
        rep = build_representation("glove-sif", emb, data.train.labels, "pca", q)
        clf = InterferenceClassifier("superposition", n_references=64, shots=args.shots, seed=3)
        clf.fit(rep.states["train"], data.train.labels)
        tests = rep.states["test"][idx[: args.sentences // 4]]
        probs = clf.exact_probabilities(tests)[:, 0, :]
        circuits = [superposition_circuit(t, clf.reference_states_, clf.reference_labels_) for t in tests]
        result = sim.run(transpile(circuits, sim, optimization_level=0), shots=args.shots, seed_simulator=1).result()
        from qsa.quantum.circuits import interpret_counts

        counts = np.array([interpret_counts(result.get_counts(i)) for i in range(len(circuits))], dtype=float)
        tvd = 0.5 * np.abs(counts / args.shots - probs).sum(-1)
        rows.append({"loader": "qram-oracle (superposition, 64 refs)", "qubits": q, "sentences": len(tests),
                     "pairs": 0, "shots": args.shots, "circuits": len(circuits), "mean_tvd": float(tvd.mean()),
                     "max_tvd": float(tvd.max()),
                     "pooled_chi2_pvalue": float(chisquare(counts.sum(0), probs.sum(0) * args.shots).pvalue)})
    return pd.DataFrame(rows)


def resource_table(max_qubits: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for n in range(1, max_qubits + 1):
        vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(3, 1 << n))]
        rows.append({"circuit": "grover-rudolph loader", "data_qubits": n,
                     **resources(GroverRudolphLoader().circuit(vecs[0]))})
        rows.append({"circuit": "pair circuit (controlled GR)", "data_qubits": n, **resources(pair_circuit(*vecs))})
        if n <= 5:
            rows.append({"circuit": "pair circuit (QRAM, QROM query, 8 bits)", "data_qubits": n,
                         **resources(qram_pair_circuit(*vecs, bits=8, query="qrom"))})
        if n <= 3:
            rows.append({"circuit": "pair circuit (QRAM, bucket brigade, 8 bits)", "data_qubits": n,
                         **resources(qram_pair_circuit(*vecs, bits=8, query="bb"))})
        LOGGER.info("resources n=%d done", n)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/sst2")
    parser.add_argument("--cache-dir", default="data/embeddings/cache")
    parser.add_argument("--sentences", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--qram-bits", type=int, default=8)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--output-dir", default="results/circuits")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resource_table(args.max_qubits).to_csv(out / "resources.csv", index=False)
    validation(args).to_csv(out / "validation.csv", index=False)
    print(pd.read_csv(out / "validation.csv").to_string())
    print(pd.read_csv(out / "resources.csv").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
