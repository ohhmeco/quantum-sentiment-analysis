"""Data layer: loading SST-2 and drawing reproducible, class-balanced samples.

The repository ships the sentence-level binary SST-2 split used by SentEval
(train 6,920 / dev 872 / test 1,821 sentences), one ``sentence<TAB>label`` per
line, label 1 = positive. Any folder with ``train.tsv``/``dev.tsv``/``test.tsv``
in the same format, or ``train.csv``/``test.csv`` with ``text,label`` columns,
can be used instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["Split", "TextDataset", "load_sst2"]


@dataclass(frozen=True)
class Split:
    """One split of a binary text-classification dataset."""

    name: str
    texts: list[str]
    labels: np.ndarray  # int array with values in {0, 1}

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")
        if not set(np.unique(self.labels)) <= {0, 1}:
            raise ValueError("labels must be binary (0/1)")

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def class_counts(self) -> dict[int, int]:
        return {c: int((self.labels == c).sum()) for c in (0, 1)}

    def subset(self, indices: Sequence[int], name: str | None = None) -> "Split":
        idx = np.asarray(indices, dtype=int)
        return Split(name or self.name, [self.texts[i] for i in idx], self.labels[idx])


@dataclass(frozen=True)
class TextDataset:
    """A binary sentiment dataset with train / dev / test splits."""

    name: str
    train: Split
    dev: Split | None
    test: Split

    def describe(self) -> pd.DataFrame:
        rows = []
        for split in (self.train, self.dev, self.test):
            if split is None:
                continue
            lengths = [len(t.split()) for t in split.texts]
            rows.append({
                "split": split.name, "sentences": len(split), "positive": split.class_counts[1],
                "negative": split.class_counts[0], "mean_tokens": round(float(np.mean(lengths)), 2),
            })
        return pd.DataFrame(rows)


def _read_tsv(path: Path, name: str) -> Split:
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"], quoting=3, dtype=str)
    if str(df.iloc[0]["label"]).strip().lower() == "label":  # GLUE-style header
        df = df.iloc[1:]
    df = df.dropna()
    return Split(name, df["text"].astype(str).tolist(), df["label"].astype(int).to_numpy())


def _read_csv(path: Path, name: str) -> Split:
    df = pd.read_csv(path)
    if not {"text", "label"} <= set(df.columns):
        raise ValueError(f"{path} must contain 'text' and 'label' columns")
    return Split(name, df["text"].astype(str).tolist(), df["label"].astype(int).to_numpy())


def load_sst2(data_dir: str | Path = "data/sst2") -> TextDataset:
    """Load SST-2 (or any binary dataset in the same layout) from ``data_dir``."""
    data_dir = Path(data_dir)
    if (data_dir / "train.tsv").is_file():
        reader, ext = _read_tsv, "tsv"
    elif (data_dir / "train.csv").is_file():
        reader, ext = _read_csv, "csv"
    else:
        raise FileNotFoundError(f"{data_dir} must contain train.tsv/test.tsv or train.csv/test.csv")
    dev_path = data_dir / f"dev.{ext}"
    return TextDataset(
        name=data_dir.name,
        train=reader(data_dir / f"train.{ext}", "train"),
        dev=reader(dev_path, "dev") if dev_path.is_file() else None,
        test=reader(data_dir / f"test.{ext}", "test"),
    )


def balanced_indices(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``per_class`` examples of each class, drawn without replacement."""
    out = []
    for c in (0, 1):
        pool = np.flatnonzero(labels == c)
        if per_class > len(pool):
            raise ValueError(f"class {c} has only {len(pool)} examples (< {per_class})")
        out.append(rng.choice(pool, size=per_class, replace=False))
    return np.concatenate(out)
