"""Build the figures and LaTeX tables of the report from the result CSV files.

    python scripts/make_figures.py      # -> results/figures/, results/tables/

Optional inputs are used when present: results/bert/metrics.csv (BERT and SBERT
grid) for the embedder comparison, results/noise/*.csv for the hardware-noise table.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa import plots  # noqa: E402
from qsa.data import load_sst2  # noqa: E402
from qsa.embeddings import EmbeddingCache, GloVeSIFEmbedder  # noqa: E402
from qsa.experiment import build_representation  # noqa: E402
from qsa.quantum import InterferenceClassifier  # noqa: E402

EMBEDDER_NAMES = {"glove-sif": "GloVe-SIF", "bert-base-uncased-mean": "BERT (mean)",
                  "sbert-all-mpnet-base-v2": "SBERT (mpnet)"}


def margin_table(data_dir: str, cache_dir: str, compressor: str = "pca", z: float = 1.645) -> pd.DataFrame:
    """Decision margin of one circuit and the shots it needs, per encoded dimension.

    For a circuit with outcome probabilities p0 = P(a=0,c=0), p1 = P(a=0,c=1), the
    count difference after R shots has mean R(p1 - p0) and variance about
    R(p0 + p1); a decision correct with probability ~95% needs
    R >= z^2 (p0 + p1) / (p1 - p0)^2.
    """
    data = load_sst2(data_dir)
    emb = EmbeddingCache(cache_dir).get_or_compute(GloVeSIFEmbedder(), data)
    rows = []
    for q in range(1, 9):
        rep = build_representation("glove-sif", emb, data.train.labels, compressor, q)
        for proto, kw in (("superposition", {}), ("centroid", {}), ("pairs", {"n_pairs": 100})):
            clf = InterferenceClassifier(proto, shots=None, seed=0, **kw).fit(rep.states["train"], data.train.labels)
            p = clf.exact_probabilities(rep.states["test"])
            delta = np.abs(p[..., 1] - p[..., 0]).ravel()
            need = z**2 * (p[..., 0] + p[..., 1]).ravel() / np.maximum(delta, 1e-12) ** 2
            rows.append({"compressor": compressor, "protocol": proto, "qubits": q, "k": 1 << q,
                         "median_abs_margin": float(np.median(delta)), "mean_abs_margin": float(delta.mean()),
                         "median_shots_needed": float(np.median(need))})
    return pd.DataFrame(rows)


def _select(m: pd.DataFrame, model: str, protocol: str | None, shots: int | None) -> pd.DataFrame:
    d = m[m.model == model]
    if protocol is not None:
        s = d["shots"].fillna(-1)
        d = d[(d.protocol == protocol) & (s == shots)]
    return d


def _fmt(v: float) -> str:
    return "--" if pd.isna(v) else f"{v:.3f}"


def noise_table(noise: pd.DataFrame) -> str:
    """Compact table (fits one column): pairs uses `circuits_per_sentence` circuits, centroid one."""
    short = {"glove-sif": "GloVe", "sbert-all-mpnet-base-v2": "SBERT", "bert-base-uncased-mean": "BERT"}
    lines = [r"\begin{tabular}{@{}llcrrccc@{}}", r"\toprule",
             r"Emb. & Protocol & $n$ & 2q & Depth & Exact & Shots & Noisy \\", r"\midrule"]
    for _, r in noise.iterrows():
        lines.append(f"{short.get(r.embedder, r.embedder)} & {r.protocol} & {int(r.qubits)} & "
                     f"{r.two_qubit_gates:.0f} & {r.transpiled_depth:.0f} & "
                     f"{r.exact_accuracy:.3f} & {r.noiseless_accuracy:.3f} & {r.noisy_accuracy:.3f} \\\\".replace("0.", "."))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


ONE_QUBIT_ROWS = [
    ("Logistic regression (probe)", "logistic-regression", None, None),
    ("Nearest centroid", "nearest-centroid", None, None),
    ("Cosine 15-NN", "cosine-knn", None, None),
    ("Centroid pair, exact", "quantum-interference", "centroid", -1),
    ("Centroid pair, 1024 shots", "quantum-interference", "centroid", 1024),
    ("Pairs (100 circuits), 1024 shots", "quantum-interference", "pairs", 1024),
    ("Superposition, 1024 shots", "quantum-interference", "superposition", 1024),
]
SHORT = {"glove-sif": "GloVe-SIF", "bert-base-uncased-mean": "BERT", "sbert-all-mpnet-base-v2": "SBERT"}


def one_qubit_table(metrics: pd.DataFrame, compressors=("pca", "pca-fisher")) -> str:
    """Accuracy with ONE data qubit (k = 2), for every encoder and both PCA variants."""
    embs = [e for e in EMBEDDER_NAMES if e in set(metrics.embedder)]
    names = {"pca": "PCA", "pca-fisher": "+Fisher"}
    ncol = len(embs) * len(compressors)
    lines = [r"\begin{tabular}{@{}l" + "c" * ncol + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{len(compressors)}}}{{c}}{{{SHORT[e]}}}" for e in embs) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + len(compressors) * i}-{1 + len(compressors) * (i + 1)}}}"
                     for i in range(len(embs))),
             "Model & " + " & ".join(names[c] for _ in embs for c in compressors) + r" \\", r"\midrule"]
    m1 = metrics[metrics.qubits == 1]
    for label, model, proto, shots in ONE_QUBIT_ROWS:
        cells = []
        for e in embs:
            for c in compressors:
                d = _select(m1[(m1.embedder == e) & (m1.compressor == c)], model, proto, shots)
                cells.append(_fmt(d.accuracy.mean()))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    full = [_fmt(metrics[(metrics.embedder == e) & (metrics.compressor == "none")].accuracy.mean()) for e in embs]
    lines += [r"\midrule", "Full embedding, logistic regression & "
              + " & ".join(rf"\multicolumn{{{len(compressors)}}}{{c}}{{{f}}}" for f in full) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def resources_table(resources: pd.DataFrame, noise: pd.DataFrame | None, margins: pd.DataFrame,
                    metrics: pd.DataFrame, qubits=(1, 2, 3, 5, 8)) -> str:
    """Resource ledger of the centroid pair as a function of the data qubits."""
    pair = resources[resources.circuit == "pair circuit (controlled GR)"].set_index("data_qubits")
    mg = margins[(margins.compressor == "pca-fisher") & (margins.protocol == "centroid")].set_index("qubits")
    mgs = margins[(margins.compressor == "pca-fisher") & (margins.protocol == "superposition")].set_index("qubits")

    def acc(emb, q, proto="centroid", shots=1024):
        d = _select(metrics[(metrics.embedder == emb) & (metrics.compressor == "pca-fisher")
                            & (metrics.qubits == q)], "quantum-interference", proto, shots)
        return _fmt(d.accuracy.mean())

    def noisy(emb, q, col):
        if noise is None:
            return "--"
        d = noise[(noise.embedder == emb) & (noise.qubits == q) & (noise.protocol == "centroid")]
        if d.empty:
            return "--"
        v = d[col].iloc[0]
        return f"{v:.3f}" if col == "noisy_accuracy" else f"{v:,.0f}".replace(",", "{,}")

    rows = [
        ("Amplitudes $k=2^n$", [f"{1 << q}" for q in qubits]),
        ("Qubits of the pair circuit", [f"{int(pair.loc[q, 'qubits'])}" for q in qubits]),
        ("CNOTs (before transpilation)", [f"{int(pair.loc[q, 'cx']):,}".replace(",", "{,}") for q in qubits]),
        ("Two-qubit gates on IBM Heron", [noisy("glove-sif", q, "two_qubit_gates") for q in qubits]),
        ("Depth on IBM Heron", [noisy("glove-sif", q, "transpiled_depth") for q in qubits]),
        ("Shots for 95\\% correct decisions", [f"{mg.loc[q, 'median_shots_needed']:.0f}" for q in qubits]),
        ("\\quad same, superposition protocol", [f"{mgs.loc[q, 'median_shots_needed']:,.0f}".replace(",", "{,}")
                                                 for q in qubits]),
        ("Accuracy GloVe-SIF, 1024 shots", [acc("glove-sif", q) for q in qubits]),
        ("Accuracy SBERT, 1024 shots", [acc("sbert-all-mpnet-base-v2", q) for q in qubits]),
        ("Accuracy SBERT, with Heron noise$^*$", [noisy("sbert-all-mpnet-base-v2", q, "noisy_accuracy") for q in qubits]),
    ]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(qubits) + r"@{}}", r"\toprule",
             "Data qubits $n$ & " + " & ".join(str(q) for q in qubits) + r" \\", r"\midrule"]
    lines += [f"{label} & " + " & ".join(cells) + r" \\" for label, cells in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--data-dir", default="data/sst2")
    parser.add_argument("--cache-dir", default="data/embeddings/cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    res = Path(args.results)
    fig_dir = res / "figures"
    tables = res / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(res / "main" / "metrics.csv")
    retention = pd.read_csv(res / "main" / "retention.csv")
    plots.retention_figure(metrics, retention, fig_dir)
    margins = pd.concat([margin_table(args.data_dir, args.cache_dir, c) for c in ("pca", "pca-fisher")])
    margins.to_csv(res / "margins.csv", index=False)
    plots.margin_figure(margins[margins.compressor == "pca-fisher"], fig_dir)

    if (res / "bert" / "metrics.csv").is_file():
        allm = pd.concat([metrics, pd.read_csv(res / "bert" / "metrics.csv")])
        plots.embedders_figure(allm, fig_dir)
        (tables / "one_qubit_table.tex").write_text(one_qubit_table(allm))
    noise_files = sorted((res / "noise").glob("noise*.csv")) if (res / "noise").is_dir() else []
    noise = pd.concat(pd.read_csv(f) for f in noise_files) if noise_files else None
    if noise is not None:
        (tables / "noise_table.tex").write_text(noise_table(noise[noise.protocol == "centroid"]))
    allm = pd.concat([metrics] + ([pd.read_csv(res / "bert" / "metrics.csv")]
                                  if (res / "bert" / "metrics.csv").is_file() else []))
    (tables / "resources_table.tex").write_text(
        resources_table(pd.read_csv(res / "circuits" / "resources.csv"), noise, margins, allm))

    print(margins.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
