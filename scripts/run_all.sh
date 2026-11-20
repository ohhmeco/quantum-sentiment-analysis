#!/usr/bin/env bash
# Run every experiment of the report on a local machine, in order, one log per step.
#
#   bash scripts/run_all.sh            # from the repository root, with the venv active
#
# Needs the embedding caches (scripts/compute_embeddings.py) and the models in data/models/
# (scripts/download_models.py), see the README. A failed step is reported in logs/run_all.log and the next steps still run.
set -u
mkdir -p logs
SBERT='{"kind": "sbert", "model_name": "data/models/all-mpnet-base-v2"}'

step() {  # step <name> <command...>
  local name=$1; shift
  echo ">>> $name $(date +%T)"
  "$@" > "logs/$name.log" 2>&1 || echo "!!! $name FAILED (see logs/$name.log)"
}

{
  for cfg in main shots bert bert-shots; do
    step "$cfg" python scripts/run_experiment.py --config "configs/$cfg.json"
  done
  step circuits python scripts/validate_circuits.py
  step errors python scripts/error_analysis.py --qubits 1,3
  step noise-glove python scripts/noise_study.py --sentences 200 --pairs 10 --output results/noise/noise.csv
  step noise-sbert python scripts/noise_study.py --sentences 200 --pairs 10 --embedder "$SBERT" \
       --output results/noise/noise_sbert.csv
  step figures python scripts/make_figures.py
  echo ">>> DONE $(date +%T)"
} 2>&1 | tee logs/run_all.log
