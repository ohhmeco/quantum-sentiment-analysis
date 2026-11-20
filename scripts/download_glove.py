"""Download pre-trained GloVe vectors (Wikipedia + Gigaword, 400k words).

    python scripts/download_glove.py --dim 300

The file is taken from the GitHub releases of gensim-data and saved, unchanged,
in data/embeddings/glove-wiki-gigaword-<dim>.gz.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-{dim}/glove-wiki-gigaword-{dim}.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, choices=(50, 100, 200, 300), default=300)
    parser.add_argument("--dest", default="data/embeddings")
    args = parser.parse_args()
    out = Path(args.dest) / f"glove-wiki-gigaword-{args.dim}.gz"
    if out.is_file():
        print(f"{out} already present")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL.format(dim=args.dim)} -> {out}")
    urllib.request.urlretrieve(URL.format(dim=args.dim), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
