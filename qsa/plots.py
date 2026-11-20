"""Figures of the report (matplotlib, vector PDF + PNG).

Colour is assigned by entity in a fixed order (validated categorical palette, light
mode); every series also has its own marker and the figures carry legends, so
identity never relies on colour alone. Limits at infinite shots are drawn dashed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL_BLUE = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False, "savefig.dpi": 300,
})

COMPRESSORS = {"pca": ("PCA", SERIES[0], "o"), "pca-fisher": ("PCA + Fisher ranking", SERIES[3], "D"),
               "fisher": ("Fisher selection (raw coordinates)", SERIES[1], "s"),
               "random": ("Random subspace", SERIES[2], "^")}
EMBEDDERS = {"glove-sif": ("GloVe-SIF (300-d)", SERIES[0], "o"),
             "bert-base-uncased-mean": ("BERT, mean pooling (768-d)", SERIES[1], "s"),
             "sbert-all-mpnet-base-v2": ("Sentence-BERT mpnet (768-d)", SERIES[2], "^")}
QUBIT_TICKS = range(1, 9)
XLABEL = "data qubits n  (k = 2^n dimensions)"


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _mean_std(df: pd.DataFrame, by: list[str], col: str = "accuracy") -> pd.DataFrame:
    return df.groupby(by, dropna=False)[col].agg(["mean", "std"]).reset_index()


def _model_masks(m: pd.DataFrame) -> list[tuple[str, pd.Series, str, str, str]]:
    shots = m["shots"].fillna(-1)
    return [
        ("logistic regression (classical)", m.model == "logistic-regression", SERIES[0], "o", "-"),
        ("nearest centroid (classical)", m.model == "nearest-centroid", SERIES[1], "s", "-"),
        ("cosine 15-NN (classical)", m.model == "cosine-knn", SERIES[2], "D", "-"),
        ("quantum pairs, 1024 shots", (m.protocol == "pairs") & (shots == 1024), SERIES[3], "v", "-"),
        ("quantum centroid pair, 1024 shots", (m.protocol == "centroid") & (shots == 1024), SERIES[5], "P", "-"),
        ("quantum superposition, 1024 shots", (m.protocol == "superposition") & (shots == 1024), SERIES[4], "^", "-"),
        ("quantum superposition, exact", (m.protocol == "superposition") & (shots == -1), SERIES[4], "^", "--"),
    ]


def retention_figure(metrics: pd.DataFrame, retention: pd.DataFrame, out_dir: Path) -> None:
    full = metrics[(metrics.compressor == "none") & (metrics.model == "logistic-regression")].accuracy.mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    lr = metrics[(metrics.model == "logistic-regression") & (metrics.compressor != "none")]
    for comp, (label, color, marker) in COMPRESSORS.items():
        d = lr[lr.compressor == comp].sort_values("qubits")
        if d.empty:
            continue
        axes[0].plot(d.qubits, d.accuracy, color=color, marker=marker, label=label)
        r = retention[retention.compressor == comp].sort_values("qubits")
        axes[1].plot(r.qubits, r.neighbourhood_overlap, color=color, marker=marker, label=label)
    axes[0].axhline(full, color=INK, lw=1, ls=":", label=f"full embedding ({full:.3f})")
    axes[0].set(xlabel=XLABEL, ylabel="test accuracy", title="(a) Linear probe on compressed vectors")
    axes[1].set(xlabel=XLABEL, ylabel="10-NN overlap with full space", title="(b) Preserved semantic neighbourhoods",
                ylim=(0, 1))
    axes[0].legend(loc="lower right", fontsize=6.2)
    for ax in axes:
        ax.set_xticks(QUBIT_TICKS)
    fig.tight_layout()
    _save(fig, out_dir, "fig_retention")


def margin_figure(margins: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    styles = (("superposition", "superposition", SERIES[4], "^"), ("centroid", "centroid pair", SERIES[5], "P"),
              ("pairs", "single random pair", SERIES[3], "v"))
    for proto, label, color, marker in styles:
        d = margins[margins.protocol == proto].sort_values("k")
        if d.empty:
            continue
        slope = np.polyfit(np.log(d.k), np.log(d.median_abs_margin), 1)[0]
        axes[0].plot(d.k, d.median_abs_margin, color=color, marker=marker, label=f"{label} (slope {slope:.2f})")
        axes[1].plot(d.k, d.median_shots_needed, color=color, marker=marker, label=label)
    k = np.array(sorted(margins.k.unique()), dtype=float)
    ref = margins[margins.protocol == "superposition"].sort_values("k").median_abs_margin.iloc[0] * np.sqrt(k[0] / k)
    axes[0].plot(k, ref, color=MUTED, ls=":", lw=1, label="∝ 1/√k")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("encoded dimension k = 2^n")
    axes[0].set(ylabel="median |P(c=1) − P(c=0)|", title="(a) Decision margin of one circuit")
    axes[1].set(ylabel="shots for a 95% correct decision", title="(b) Measurement cost per circuit")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.2, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, out_dir, "fig_margin")


def embedders_figure(metrics: pd.DataFrame, out_dir: Path, compressor: str = "pca-fisher") -> None:
    """Probe and quantum (centroid pair, 1024 shots) accuracy for every embedder."""
    m = metrics[metrics.compressor.isin([compressor, "none"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for emb, (label, color, marker) in EMBEDDERS.items():
        e = m[m.embedder == emb]
        if e.empty:
            continue
        full = e[(e.compressor == "none")].accuracy.mean()
        lr = _mean_std(e[(e.model == "logistic-regression") & (e.compressor == compressor)], ["qubits"])
        axes[0].plot(lr.qubits, lr["mean"], color=color, marker=marker, label=label)
        axes[0].axhline(full, color=color, lw=0.8, ls=":")
        qc = e[(e.protocol == "centroid") & (e.shots == 1024)]
        d = _mean_std(qc, ["qubits"])
        axes[1].errorbar(d.qubits, d["mean"], yerr=d["std"].fillna(0), color=color, marker=marker, label=label,
                         capsize=1.5, elinewidth=0.8)
    axes[0].set(xlabel=XLABEL, ylabel="test accuracy", title="(a) Logistic probe (dotted: full embedding)",
                xticks=QUBIT_TICKS)
    axes[1].set(xlabel=XLABEL, title="(b) Quantum centroid pair, 1024 shots", xticks=QUBIT_TICKS)
    axes[1].legend(loc="lower right", fontsize=6.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig_embedders")
