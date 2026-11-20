"""Pre-compute and cache sentence embeddings (run once, e.g. on a machine with a GPU
and access to the HuggingFace hub). Every experiment afterwards reads the cache.

    python scripts/compute_embeddings.py --kind transformer --model bert-base-uncased
    python scripts/compute_embeddings.py --kind sbert --model sentence-transformers/all-mpnet-base-v2
    python scripts/compute_embeddings.py --kind glove-sif
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa.data import load_sst2  # noqa: E402
from qsa.embeddings import EmbeddingCache, build_embedder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("glove-sif", "transformer", "sbert"), required=True)
    parser.add_argument("--model", default=None, help="HuggingFace name or local folder (transformer/sbert)")
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--data-dir", default="data/sst2")
    parser.add_argument("--cache-dir", default="data/embeddings/cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    spec: dict = {"kind": args.kind}
    if args.kind == "transformer":
        spec.update(model_name=args.model or "bert-base-uncased", pooling=args.pooling)
    elif args.kind == "sbert":
        spec.update(model_name=args.model or "sentence-transformers/all-mpnet-base-v2")
    embedder = build_embedder(spec)
    data = load_sst2(args.data_dir)
    cache = EmbeddingCache(args.cache_dir)
    matrices = cache.get_or_compute(embedder, data)
    print({k: v.shape for k, v in matrices.items()}, "->", cache.path(embedder, data.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
