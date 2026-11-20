"""Error analysis: where and how the classifiers fail, sentence by sentence.

    python scripts/error_analysis.py                       # GloVe-SIF
    python scripts/error_analysis.py --embedders '[{"kind": "glove-sif"}, {"kind": "sbert", "model_name": "data/models/all-mpnet-base-v2"}]'

For one compressor (default PCA + Fisher ranking) and a few numbers of data qubits it
computes, on the whole test set, the predictions and the probabilities of

* logistic regression on the compressed vectors (the probe);
* the nearest centroid and the centroid-pair circuit (exact and 1024 shots);

and writes to ``results/errors/``:

* ``predictions_<embedder>_n<q>.csv`` - one row per test sentence;
* ``summary.csv`` - accuracy, log-loss and perplexity per model, overall and on
  linguistic subsets (negation, contrast, short and long sentences);
* ``confusion.csv`` - the confusion matrix of every model;
* ``examples_<embedder>_n<q>.md`` - the most confident mistakes and the sentences that
  every model gets wrong (candidates for hard cases or label noise).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa.baselines import LogisticProbe, NearestCentroid  # noqa: E402
from qsa.data import load_sst2  # noqa: E402
from qsa.embeddings import EmbeddingCache, build_embedder  # noqa: E402
from qsa.experiment import build_representation  # noqa: E402
from qsa.quantum import InterferenceClassifier  # noqa: E402

NEGATION = re.compile(r"\b(not|no|never|n't|nothing|neither|nor|none|nobody|without|hardly|barely)\b")
CONTRAST = re.compile(r"\b(but|yet|though|although|however|while|despite)\b")


def subsets(texts: list[str]) -> dict[str, np.ndarray]:
    lower = [t.lower() for t in texts]
    length = np.array([len(t.split()) for t in texts])
    short, long_ = np.quantile(length, [1 / 3, 2 / 3])
    return {
        "all": np.ones(len(texts), dtype=bool),
        "negation": np.array([bool(NEGATION.search(t)) for t in lower]),
        "contrast": np.array([bool(CONTRAST.search(t)) for t in lower]),
        "no negation or contrast": np.array([not (NEGATION.search(t) or CONTRAST.search(t)) for t in lower]),
        f"short (<= {int(short)} words)": length <= short,
        f"long (> {int(long_)} words)": length > long_,
    }


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def analyse(name: str, emb: dict, data, compressor: str, q: int, out: Path) -> tuple[list[dict], list[dict]]:
    y = {"train": data.train.labels, "dev": data.dev.labels, "test": data.test.labels}
    rep = build_representation(name, emb, y["train"], compressor, q)
    s, z = rep.states, rep.compressed
    preds: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}

    probe = LogisticProbe().fit(z["train"], y["train"], z["dev"], y["dev"])
    probs["logistic regression"] = probe.predict_proba(z["test"])
    preds["logistic regression"] = probe.predict(z["test"])
    exact = InterferenceClassifier("centroid", shots=None).fit(s["train"], y["train"])
    report = exact.predict_report(s["test"])
    probs["centroid pair"] = report.p_positive
    preds["centroid pair"] = report.labels
    shots = InterferenceClassifier("centroid", shots=1024, seed=0).fit(s["train"], y["train"])
    preds["centroid pair, 1024 shots"] = shots.predict(s["test"])
    preds["nearest centroid"] = NearestCentroid().fit(s["train"], y["train"]).predict(s["test"])

    texts, yt = data.test.texts, y["test"]
    table = pd.DataFrame({"sentence": texts, "label": yt})
    for m in preds:
        table[f"pred: {m}"] = preds[m]
    for m in probs:
        table[f"p(positive): {m}"] = probs[m]
    tag = f"{name}_n{q}"
    table.to_csv(out / f"predictions_{tag}.csv", index=False)

    summary, confusion = [], []
    groups = subsets(texts)
    for m, pr in preds.items():
        row = {"embedder": name, "compressor": compressor, "qubits": q, "model": m}
        for g, mask in groups.items():
            row[f"acc: {g}"] = float((pr[mask] == yt[mask]).mean())
        base = m.replace(", 1024 shots", "")
        if base in probs and base == m:
            ll = log_loss(yt, probs[m])
            row["log_loss"], row["perplexity"] = ll, float(np.exp(ll))
            row["mean |p - 1/2|"] = float(np.abs(probs[m] - 0.5).mean())
        summary.append(row)
        confusion.append({"embedder": name, "qubits": q, "model": m,
                          "TN": int(((pr == 0) & (yt == 0)).sum()), "FP": int(((pr == 1) & (yt == 0)).sum()),
                          "FN": int(((pr == 0) & (yt == 1)).sum()), "TP": int(((pr == 1) & (yt == 1)).sum())})
    summary.append({"embedder": name, "compressor": compressor, "qubits": q, "model": "(subset size)",
                    **{f"acc: {g}": int(mask.sum()) for g, mask in groups.items()}})

    exact_models = ["logistic regression", "nearest centroid", "centroid pair"]
    wrong_all = np.all([preds[m] != yt for m in exact_models], axis=0)
    lines = [f"# Errors: {name}, {compressor}, {q} data qubit(s)", "",
             f"Sentences wrong for all {len(exact_models)} models: {int(wrong_all.sum())} of {len(yt)} "
             f"({wrong_all.mean():.1%}).", "", "## Most confident mistakes of the logistic regression", ""]
    conf = np.abs(probs["logistic regression"] - 0.5)
    for i in np.argsort(-np.where(preds["logistic regression"] != yt, conf, -1))[:15]:
        lines.append(f"* [{'pos' if yt[i] else 'neg'}, p(pos) = {probs['logistic regression'][i]:.2f}] {texts[i]}")
    lines += ["", "## Mistakes of the centroid pair that the logistic regression gets right", ""]
    only = np.flatnonzero((preds["centroid pair"] != yt) & (preds["logistic regression"] == yt))
    for i in only[:15]:
        lines.append(f"* [{'pos' if yt[i] else 'neg'}] {texts[i]}")
    lines += ["", "## Wrong for every model (first 20)", ""]
    for i in np.flatnonzero(wrong_all)[:20]:
        lines.append(f"* [{'pos' if yt[i] else 'neg'}] {texts[i]}")
    (out / f"examples_{tag}.md").write_text("\n".join(lines) + "\n")
    return summary, confusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedders", default='[{"kind": "glove-sif"}]', help="JSON list of embedder specs")
    parser.add_argument("--compressor", default="pca-fisher")
    parser.add_argument("--qubits", default="1,3", help="comma-separated numbers of data qubits")
    parser.add_argument("--data-dir", default="data/sst2")
    parser.add_argument("--cache-dir", default="data/embeddings/cache")
    parser.add_argument("--output", default="results/errors")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load_sst2(args.data_dir)
    cache = EmbeddingCache(args.cache_dir)
    summary_file, confusion_file = out / "summary.csv", out / "confusion.csv"
    summary = pd.read_csv(summary_file).to_dict("records") if summary_file.is_file() else []
    confusion = pd.read_csv(confusion_file).to_dict("records") if confusion_file.is_file() else []
    for spec in json.loads(args.embedders):
        embedder = build_embedder(spec)
        emb = cache.get_or_compute(embedder, data)
        summary = [r for r in summary if r["embedder"] != embedder.name]
        confusion = [r for r in confusion if r["embedder"] != embedder.name]
        for q in (int(v) for v in args.qubits.split(",")):
            s, c = analyse(embedder.name, emb, data, args.compressor, q, out)
            summary += s
            confusion += c
            pd.DataFrame(summary).to_csv(summary_file, index=False)
            pd.DataFrame(confusion).to_csv(confusion_file, index=False)
            logging.info("%s, %d qubits done", embedder.name, q)
    print(pd.DataFrame(summary).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
