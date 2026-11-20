"""Quantum layer: amplitude encoding (Grover-Rudolph, QRAM) and the interference classifier."""
from .circuits import pair_circuit, pair_probabilities, qram_pair_circuit, resources, superposition_circuit
from .classifier import InterferenceClassifier, PredictionReport
from .state_preparation import GroverRudolphLoader, QRAMLoader, quantized_state, tree_angles

__all__ = [
    "GroverRudolphLoader", "QRAMLoader", "quantized_state", "tree_angles",
    "pair_circuit", "qram_pair_circuit", "superposition_circuit", "pair_probabilities", "resources",
    "InterferenceClassifier", "PredictionReport",
]
