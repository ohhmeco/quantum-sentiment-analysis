"""Run an experiment described by a JSON configuration.

    python scripts/run_experiment.py --config configs/main.json
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa.experiment import ExperimentConfig, ExperimentRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON file in configs/")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    config = ExperimentConfig.from_json(args.config)
    ExperimentRunner(config).run()
    logging.getLogger("qsa").info("results written to %s", Path(config.output_dir) / config.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
