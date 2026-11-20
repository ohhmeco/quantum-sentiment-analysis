"""Download the two Transformer encoders into data/models/ (once, needs internet).

    python scripts/download_models.py

The experiment configurations read the models from these local folders, so that the
runs work offline and the embedder names in the results are stable.
"""
from __future__ import annotations

from pathlib import Path


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    out = Path("data/models")
    out.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained("bert-base-uncased").save_pretrained(out / "bert-base-uncased")
    AutoModel.from_pretrained("bert-base-uncased").save_pretrained(out / "bert-base-uncased")
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2").save(str(out / "all-mpnet-base-v2"))
    print(f"models saved in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
