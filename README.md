# How much sentiment fits in one qubit?

Final project for the *Natural Language Processing* course, Università degli Studi di Milano, A.Y. 2025/26 — Michele Bonomi.

**Report: [`report.pdf`](report.pdf)**

Sentence embeddings are compressed to 2ⁿ numbers, amplitude-encoded into n qubits and classified by the
interference circuit of Schuld et al. (2017) on SST-2. The project pushes the resources to the minimum and asks
how much binary sentiment can be read out of a **single data qubit**, at what cost, and what each additional
qubit buys.

```
sentence ─► sentence encoder ─► compression to k = 2ⁿ ─► amplitude encoding ─► interference classifier
            GloVe-SIF, BERT,      PCA, PCA + Fisher,      Grover–Rudolph          pairs / centroid pair /
            Sentence-BERT         Fisher, random                                  superposition
```

## Research question

*How much binary sentiment can be read out of a single data qubit, at what cost, and what does each additional qubit buy?*

1. **Compression:** how much sentiment and semantic-similarity information survives when an embedding is reduced to k = 2ⁿ numbers, and in particular to k = 2?
2. **Decision rule:** how does the quantum classifier compare with classical classifiers on the *same* compressed vectors?
3. **Cost:** qubits, gates, shots and robustness to hardware noise as a function of n.

## Main results

SST-2 test set (1,821 sentences), five seeds, PCA + Fisher ranking.

| One data qubit (k = 2) | GloVe-SIF | BERT (mean) | Sentence-BERT |
|---|---|---|---|
| Logistic regression (linear probe) | 0.708 | 0.699 | 0.846 |
| Nearest centroid | 0.700 | 0.710 | 0.842 |
| Quantum centroid pair, 1024 shots | 0.703 | 0.713 | 0.840 |
| *Logistic regression, full embedding* | *0.785* | *0.856* | *0.887* |

* **Choosing the two numbers is what matters.** Ranking principal directions by Fisher score instead of variance adds 6–8 points at one qubit; with Sentence-BERT one qubit keeps 95% of the full-embedding accuracy.
* **At one qubit every rule agrees**, quantum or classical, within about one point: two normalised numbers leave only an angle to threshold.
* **The single-qubit circuit is cheap.** 4 qubits and 19 CNOTs (26 two-qubit gates, depth 80 on an IBM Heron processor), about 11 shots for a 95%-reliable decision, no measurable loss under the `FakeTorino` noise model.
* **More qubits buy little.** At most +2 points (Sentence-BERT) or +5 (GloVe) for ~47·2ⁿ CNOTs, more shots and more noise; with all training data in superposition the margin shrinks as k^(−0.56) and accuracy falls as qubits are added.

## Repository layout

```
report.pdf                   the report
qsa/                         the Python package (all the logic)
  data.py                    SST-2 loading
  embeddings.py              GloVe-SIF, BERT (mean pooling), Sentence-BERT encoders + cache
  compression.py             PCA, PCA + Fisher ranking, Fisher selection, random subspace, normaliser
  baselines.py               logistic regression (probe), nearest centroid, cosine k-NN
  metrics.py                 accuracy / P / R / F1 / MCC, bootstrap intervals, retention metrics
  experiment.py              configuration-driven experiment runner
  plots.py                   figures of the report
  quantum/state_preparation.py   Grover–Rudolph loader (and QRAM loaders for the resource study)
  quantum/circuits.py        interference circuits, closed-form probabilities, resource counts
  quantum/classifier.py      InterferenceClassifier: pairs / centroid / superposition protocols
scripts/                     command-line entry points (one per step, see below)
configs/                     experiment definitions (JSON)
data/sst2/                   SST-2, sentence level (train / dev / test)
notebooks/demo.ipynb         walkthrough of the pipeline (demonstration only, executed)
results/                     raw results (CSV), figures and the logs of the reported runs
```

## Reproducing the results

Python 3.10–3.12. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_glove.py --dim 300            # GloVe vectors -> data/embeddings/
python scripts/download_models.py                      # BERT and Sentence-BERT -> data/models/
python scripts/compute_embeddings.py --kind glove-sif
python scripts/compute_embeddings.py --kind transformer --model data/models/bert-base-uncased
python scripts/compute_embeddings.py --kind sbert --model data/models/all-mpnet-base-v2

bash scripts/run_all.sh                                # all experiments + figures, logs in logs/
```

`run_all.sh` runs, in order:

| Step | Command | Output |
|---|---|---|
| GloVe-SIF grid (4 compressors, n = 1…8, all classifiers) | `scripts/run_experiment.py --config configs/main.json` | `results/main/` |
| Shot sweep | `scripts/run_experiment.py --config configs/shots.json` | `results/shots/` |
| BERT and Sentence-BERT grid, shot sweep | `configs/bert.json`, `configs/bert-shots.json` | `results/bert*/` |
| Qiskit Aer validation and gate counts | `scripts/validate_circuits.py` | `results/circuits/` |
| Error analysis | `scripts/error_analysis.py` | `results/errors/` |
| IBM Heron noise model | `scripts/noise_study.py` | `results/noise/` |
| Figures and LaTeX tables | `scripts/make_figures.py` | `results/figures/`, `results/tables/` |

The reported runs took about 70 minutes on a CPU-only virtual machine (logs in `results/logs/`). Every fitted
component (SIF weights, PCA, Fisher ranking, centroids) uses the training split only; regularisation is chosen on
dev; the test split is used once. A rerun on a second machine reproduced every exact result bit for bit.

## AI usage disclaimer

Parts of this project were developed with the assistance of Anthropic's Claude (Claude Opus): it helped refactor
the original experiment code into a modular package, write plotting code and experiment scripts, and structure
and draft the report. All experiments were run by the author on his own machine, every reported number was
generated by the code in this repository, and all text was reviewed and edited by the author, who takes full
responsibility for the content.

## License

MIT, see [`LICENSE`](LICENSE).
