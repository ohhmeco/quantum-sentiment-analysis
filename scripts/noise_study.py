"""Hardware-noise study: the interference circuits on a simulated IBM Heron processor.

The pair circuit (Grover-Rudolph loaders) is transpiled for the fake backend
``FakeTorino`` (133-qubit IBM Heron r1 with calibrated gate, read-out and decoherence
errors) and simulated with its noise model in Qiskit Aer. For n = 1..3 data qubits it
compares, on the same test sentences and with the same shots:

* ``exact``      - infinite-shot accuracy (closed form);
* ``noiseless``  - finite shots, no hardware noise;
* ``noisy``      - finite shots, FakeTorino noise model;

for the ``centroid`` protocol (1 circuit per sentence) and the ``pairs`` protocol
(``--pairs`` circuits per sentence, majority vote), plus the transpiled depth and the
number of two-qubit gates. Needs ``pip install qiskit-ibm-runtime``.

    python scripts/noise_study.py --sentences 200 --pairs 10           # ~minutes per n on a laptop
    python scripts/noise_study.py --embedder '{"kind": "sbert", "model_name": "data/models/all-mpnet-base-v2"}'
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa.data import load_sst2  # noqa: E402
from qsa.embeddings import EmbeddingCache, build_embedder  # noqa: E402
from qsa.experiment import build_representation  # noqa: E402
from qsa.quantum import InterferenceClassifier, pair_circuit  # noqa: E402
from qsa.quantum.circuits import interpret_counts  # noqa: E402

LOGGER = logging.getLogger("qsa")


def labels_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote over circuits from counts (n_test, n_circuits, 3); ties -> -1."""
    votes = np.where(counts[..., 1] == counts[..., 0], -1, (counts[..., 1] > counts[..., 0]).astype(int))
    pos, neg = (votes == 1).sum(1), (votes == 0).sum(1)
    return np.where(pos == neg, -1, (pos > neg).astype(int)), votes


def accuracy(labels: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """Accuracy with ties (-1) broken by a fair coin."""
    labels = np.where(labels == -1, rng.integers(0, 2, len(labels)), labels)
    return float((labels == y).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", default='{"kind": "glove-sif"}', help="JSON spec, as in the configs")
    parser.add_argument("--compressor", default="pca-fisher")
    parser.add_argument("--qubits", default="1,2,3")
    parser.add_argument("--sentences", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data/sst2")
    parser.add_argument("--cache-dir", default="data/embeddings/cache")
    parser.add_argument("--output", default="results/noise/noise.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    backend = FakeTorino()
    noisy_sim = AerSimulator.from_backend(backend)
    data = load_sst2(args.data_dir)
    embedder = build_embedder(json.loads(args.embedder))
    emb = EmbeddingCache(args.cache_dir).get_or_compute(embedder, data)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(data.test), size=min(args.sentences, len(data.test)), replace=False)
    y = data.test.labels[idx]
    rows = []
    for q in [int(v) for v in args.qubits.split(",")]:
        rep = build_representation(embedder.name, emb, data.train.labels, args.compressor, q)
        tests = rep.states["test"][idx]
        for proto, kw in (("centroid", {}), ("pairs", {"n_pairs": args.pairs})):
            t0 = time.perf_counter()
            clf = InterferenceClassifier(proto, shots=args.shots, seed=args.seed, **kw)
            clf.fit(rep.states["train"], data.train.labels)
            exact = clf.exact_probabilities(tests)
            exact_labels, _ = labels_from_counts(exact)
            sampled = np.stack([np.stack([rng.multinomial(args.shots, p / p.sum()) for p in row]) for row in exact])
            noiseless_labels, noiseless_votes = labels_from_counts(sampled)
            circuits = [pair_circuit(t, x0, x1) for t in tests for x0, x1 in clf.references_]
            compiled = transpile(circuits, backend, optimization_level=3, seed_transpiler=7)
            result = noisy_sim.run(compiled, shots=args.shots, seed_simulator=args.seed).result()
            counts = np.array([interpret_counts(result.get_counts(i)) for i in range(len(circuits))], dtype=float)
            counts = counts.reshape(len(tests), -1, 3)
            noisy_labels, noisy_votes = labels_from_counts(counts)
            freq = counts / counts.sum(-1, keepdims=True)
            twoq = [sum(v for k, v in c.count_ops().items() if k in ("cz", "ecr", "cx")) for c in compiled]
            rows.append({
                "embedder": embedder.name, "compressor": args.compressor, "qubits": q, "protocol": proto,
                "circuits_per_sentence": exact.shape[1], "sentences": len(tests), "shots": args.shots,
                "exact_accuracy": accuracy(exact_labels, y, rng),
                "noiseless_accuracy": accuracy(noiseless_labels, y, rng),
                "noisy_accuracy": accuracy(noisy_labels, y, rng),
                "noisy_vs_noiseless_vote_agreement": float((noisy_votes == noiseless_votes).mean()),
                "mean_tvd_noisy_vs_exact": float((0.5 * np.abs(freq - exact).sum(-1)).mean()),
                "mean_acceptance_noisy": float(freq[..., :2].sum(-1).mean()),
                "mean_acceptance_exact": float(exact[..., :2].sum(-1).mean()),
                "mean_abs_margin_noisy": float(np.abs(freq[..., 1] - freq[..., 0]).mean()),
                "mean_abs_margin_exact": float(np.abs(exact[..., 1] - exact[..., 0]).mean()),
                "transpiled_depth": float(np.mean([c.depth() for c in compiled])),
                "two_qubit_gates": float(np.mean(twoq)),
                "seconds": round(time.perf_counter() - t0, 1),
            })
            LOGGER.info("n=%d %s: %s", q, proto, {k: rows[-1][k] for k in ("exact_accuracy", "noiseless_accuracy",
                                                                        "noisy_accuracy", "two_qubit_gates", "seconds")})
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
