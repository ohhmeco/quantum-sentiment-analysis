"""Sentence representations: every embedder maps a list of sentences to a dense matrix.

All embedders share the :class:`SentenceEmbedder` interface (``fit`` on the
training sentences only, then ``transform``). Pre-trained neural encoders have
no fitting step; corpus statistics (SIF word frequencies, TF-IDF vocabulary)
are always estimated on the training split, so no test information leaks into
the representation.

Available embedders
-------------------
* :class:`GloVeSIFEmbedder`  - GloVe word vectors averaged with SIF weights
  (Arora et al., 2017), optionally removing the common component. Works offline
  once the GloVe file has been downloaded (``scripts/download_glove.py``).
* :class:`TransformerEmbedder` - frozen BERT-like encoder from HuggingFace,
  mean pooling of the last hidden layer (or the [CLS] token).
* :class:`SentenceTransformerEmbedder` - Sentence-BERT models.

:class:`EmbeddingCache` stores computed matrices in ``.npz`` files, so the
expensive neural encoders run once (e.g. on a GPU machine) and every later
experiment only reads the cache.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
import gzip
import hashlib
import logging
from pathlib import Path
from typing import Callable

import numpy as np

LOGGER = logging.getLogger("qsa")

__all__ = [
    "SentenceEmbedder", "GloVeSIFEmbedder", "TransformerEmbedder", "SentenceTransformerEmbedder",
    "EmbeddingCache", "build_embedder", "tokenize",
]


def tokenize(text: str) -> list[str]:
    """SST-2 is already lower-cased and tokenised with spaces."""
    return text.lower().split()


class SentenceEmbedder(ABC):
    """Interface of every sentence embedder."""

    #: short identifier used in result tables and cache file names
    name: str = "embedder"

    def fit(self, texts: list[str]) -> "SentenceEmbedder":  # noqa: D401 - sklearn-like API
        """Estimate corpus statistics on the training sentences (default: nothing)."""
        return self

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def cache_key(self) -> str:
        """Identifier of the configuration, used to name cache files."""
        return self.name


# GloVe + SIF ----------------------------------------------------------------------------

def load_word_vectors(path: str | Path, vocabulary: set[str] | None = None) -> dict[str, np.ndarray]:
    """Read word vectors in (optionally gzipped) word2vec/GloVe text format.

    Only the words in ``vocabulary`` are kept, which keeps memory small.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `python scripts/download_glove.py --dim 300`")
    opener: Callable = gzip.open if path.suffix == ".gz" else open
    vectors: dict[str, np.ndarray] = {}
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().split()
        if len(first) > 2:  # no "<count> <dim>" header: the first line is a vector
            handle.seek(0)
        for line in handle:
            word, _, rest = line.rstrip().partition(" ")
            if vocabulary is None or word in vocabulary:
                vectors[word] = np.asarray(rest.split(), dtype=np.float32)
    LOGGER.info("loaded %d word vectors from %s", len(vectors), path.name)
    return vectors


class GloVeSIFEmbedder(SentenceEmbedder):
    """Smooth Inverse Frequency sentence embeddings (Arora, Liang & Ma, 2017).

    ``v(s) = sum_w a/(a + p(w)) * glove(w) / |s|``, with ``p(w)`` the unigram
    frequency estimated on the training sentences. If ``remove_common_component``
    is true, the projection on the first singular vector of the *training*
    embedding matrix is removed, as in the original method: this direction is
    shared by all sentences (syntax, frequent words) and carries little meaning.
    """

    def __init__(self, glove_path: str | Path = "data/embeddings/glove-wiki-gigaword-300.gz",
                 a: float = 1e-3, remove_common_component: bool = True):
        self.glove_path = Path(glove_path)
        self.a = a
        self.remove_common_component = remove_common_component
        self.name = "glove-sif" if remove_common_component else "glove-sif-nopc"
        self._vectors: dict[str, np.ndarray] | None = None
        self._freq: dict[str, float] = {}
        self._pc: np.ndarray | None = None
        self._vocabulary: set[str] | None = None

    def with_vocabulary(self, texts: list[str]) -> "GloVeSIFEmbedder":
        """Restrict loading to the words that occur in ``texts`` (all splits)."""
        self._vocabulary = {w for t in texts for w in tokenize(t)}
        return self

    def fit(self, texts: list[str]) -> "GloVeSIFEmbedder":
        if self._vectors is None:
            self._vectors = load_word_vectors(self.glove_path, self._vocabulary)
        counts = Counter(w for t in texts for w in tokenize(t))
        total = max(1, sum(counts.values()))
        self._freq = {w: c / total for w, c in counts.items()}
        self._pc = None
        if self.remove_common_component:
            x = self._weighted_average(texts)
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            self._pc = vt[0]
        return self

    def _weighted_average(self, texts: list[str]) -> np.ndarray:
        assert self._vectors is not None, "call fit() first"
        dim = len(next(iter(self._vectors.values())))
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            words = [w for w in tokenize(text) if w in self._vectors]
            if not words:
                continue
            weights = np.array([self.a / (self.a + self._freq.get(w, 0.0)) for w in words])
            out[i] = weights @ np.stack([self._vectors[w] for w in words]) / len(words)
        return out

    def transform(self, texts: list[str]) -> np.ndarray:
        x = self._weighted_average(texts)
        if self._pc is not None:
            x = x - np.outer(x @ self._pc, self._pc)
        return x

    @property
    def cache_key(self) -> str:
        return f"{self.name}-{self.glove_path.stem}-a{self.a:g}"


# Neural encoders ------------------------------------------------------------------------

class TransformerEmbedder(SentenceEmbedder):
    """Frozen HuggingFace encoder (default ``bert-base-uncased``), no fine-tuning."""

    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "mean",
                 max_length: int = 64, batch_size: int = 64, device: str | None = None):
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.model_name, self.pooling = model_name, pooling
        self.max_length, self.batch_size, self.device = max_length, batch_size, device
        self.name = f"{Path(model_name).name}-{pooling}"

    def transform(self, texts: list[str]) -> np.ndarray:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TransformerEmbedder needs `pip install torch transformers`") from exc
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True,
                                  max_length=self.max_length, return_tensors="pt").to(device)
                hidden = model(**batch).last_hidden_state
                if self.pooling == "cls":
                    emb = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                chunks.append(emb.float().cpu().numpy())
        return np.concatenate(chunks).astype(np.float64)


class SentenceTransformerEmbedder(SentenceEmbedder):
    """Sentence-BERT (Reimers & Gurevych, 2019), e.g. ``all-mpnet-base-v2``."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", batch_size: int = 128):
        self.model_name, self.batch_size = model_name, batch_size
        self.name = f"sbert-{Path(model_name).name}"

    def transform(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SentenceTransformerEmbedder needs `pip install sentence-transformers`") from exc
        model = SentenceTransformer(self.model_name)
        return np.asarray(model.encode(texts, batch_size=self.batch_size, show_progress_bar=False), dtype=np.float64)


def build_embedder(spec: dict) -> SentenceEmbedder:
    """Factory used by the experiment configuration files.

    ``spec`` is a dictionary such as ``{"kind": "glove-sif", "glove_path": ...}``.
    """
    spec = dict(spec)
    kind = spec.pop("kind")
    registry = {
        "glove-sif": GloVeSIFEmbedder,
        "transformer": TransformerEmbedder,
        "sbert": SentenceTransformerEmbedder,
    }
    if kind not in registry:
        raise ValueError(f"unknown embedder {kind!r}; choose among {sorted(registry)}")
    return registry[kind](**spec)


# Cache ----------------------------------------------------------------------------------

class EmbeddingCache:
    """Compute-once storage of the embedding matrices of all splits."""

    def __init__(self, directory: str | Path = "data/embeddings/cache"):
        self.directory = Path(directory)

    def path(self, embedder: SentenceEmbedder, dataset_name: str) -> Path:
        digest = hashlib.sha1(f"{embedder.cache_key}|{dataset_name}".encode()).hexdigest()[:10]
        return self.directory / f"{dataset_name}_{embedder.name}_{digest}.npz"

    def get_or_compute(self, embedder: SentenceEmbedder, dataset) -> dict[str, np.ndarray]:
        """Return ``{"train": X, "dev": X, "test": X}`` for a :class:`qsa.data.TextDataset`."""
        path = self.path(embedder, dataset.name)
        if path.is_file():
            LOGGER.info("embeddings read from cache %s", path)
            with np.load(path) as data:
                return {k: data[k] for k in data.files}
        LOGGER.info("computing %s embeddings for %s", embedder.name, dataset.name)
        if isinstance(embedder, GloVeSIFEmbedder):
            all_texts = dataset.train.texts + dataset.test.texts + (dataset.dev.texts if dataset.dev else [])
            embedder.with_vocabulary(all_texts)
        embedder.fit(dataset.train.texts)
        out = {"train": embedder.transform(dataset.train.texts), "test": embedder.transform(dataset.test.texts)}
        if dataset.dev is not None:
            out["dev"] = embedder.transform(dataset.dev.texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        return out
