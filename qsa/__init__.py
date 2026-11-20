"""qsa - Quantum Sentiment Analysis.

Hybrid pipeline for binary sentiment classification on SST-2:

    sentence -> pre-trained sentence embedding -> classical compression to 2^n
             -> amplitude encoding (Grover-Rudolph / QRAM) on n qubits
             -> interference-based (distance) quantum classifier

Layers: :mod:`qsa.data` (datasets), :mod:`qsa.embeddings` (representations),
:mod:`qsa.compression` (PCA / Fisher / random projection), :mod:`qsa.quantum`
(state preparation, circuits, classifier), :mod:`qsa.baselines`,
:mod:`qsa.metrics`, :mod:`qsa.experiment` and :mod:`qsa.plots`.
"""
__version__ = "1.0.0"
