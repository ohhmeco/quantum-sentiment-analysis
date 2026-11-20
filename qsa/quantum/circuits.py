"""Quantum circuits of the interference (distance-based) classifier.

Pair circuit (Schuld, Fingerhuth & Petruccione 2017, two training vectors)
------------------------------------------------------------------------
Registers: ancilla ``a``, index ``m`` (which reference), data ``g`` (n qubits), label ``c``.

1. ``H(a) H(m)`` gives ``1/2 sum_{a,m} |a>|m>``;
2. controlled loaders write the test state ``t`` in the branch a = 0 and the
   reference ``x_m`` in the branch a = 1; ``CX(m, c)`` copies the class of x_m:
   ``1/2 sum_m (|0>|t> + |1>|x_m>) |m>|m>``;
3. a final ``H(a)`` interferes the two branches:
   ``1/(2 sqrt 2) sum_m (|0>(|t> + |x_m>) + |1>(|t> - |x_m>)) |m>|m>``.

Measuring ``a`` and ``c`` gives ``P(a=0, c=y) = ||t + x_y||^2 / 8 = (1 + <t|x_y>) / 4``
for real unit vectors. Post-selecting a = 0, the class whose reference has the
larger overlap with the test state (equivalently, the smaller Euclidean distance)
is measured more often.

Superposition circuit
---------------------
The same construction with *all* references indexed by a ``ceil(log2 R)``-qubit
register. With class-balanced index weights ``w_m^2 = 1/(2 R_c)``,
``P(a=0, c) = (1 + <t|mu_c>) / 4``, where ``mu_c`` is the mean of the reference
states of class c: in the infinite-shot limit the classifier is a nearest-centroid
rule on the unit sphere.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .state_preparation import BucketBrigadeTree, GroverRudolphLoader, QRAMLoader, QRAMMemory

__all__ = ["pair_circuit", "qram_pair_circuit", "superposition_circuit", "superposition_joint_state",
           "interpret_counts", "pair_probabilities"]


def pair_probabilities(t: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Exact ``[P(a=0,c=0), P(a=0,c=1), P(a=1)]`` of the pair circuit (broadcasts)."""
    o0, o1 = np.sum(t * x0, axis=-1), np.sum(t * x1, axis=-1)
    p00, p01 = (1 + o0) / 4, (1 + o1) / 4
    return np.stack([p00, p01, 1 - p00 - p01], axis=-1)


def pair_circuit(t: np.ndarray, x0: np.ndarray, x1: np.ndarray, measure: bool = True):
    """Pair circuit with Grover-Rudolph loaders controlled by (a, m)."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    loader = GroverRudolphLoader()
    g_t, g_0, g_1 = (loader.circuit(v, name).to_gate() for v, name in ((t, "Ut"), (x0, "U0"), (x1, "U1")))
    n = int(round(math.log2(len(t))))
    a, m, g, c = QuantumRegister(1, "a"), QuantumRegister(1, "m"), QuantumRegister(n, "g"), QuantumRegister(1, "c")
    qc = QuantumCircuit(a, m, g, c)
    qc.h(a[0])
    qc.h(m[0])
    qc.x(a[0])                                              # a = 0 branch: test state
    qc.append(g_t.control(1), [a[0], *g])
    qc.x(a[0])
    qc.x(m[0])                                              # a = 1, m = 0: negative reference
    qc.append(g_0.control(2), [a[0], m[0], *g])
    qc.x(m[0])
    qc.append(g_1.control(2), [a[0], m[0], *g])            # a = 1, m = 1: positive reference
    qc.cx(m[0], c[0])
    qc.h(a[0])
    if measure:
        out = ClassicalRegister(2, "out")
        qc.add_register(out)
        qc.measure(a[0], out[0])
        qc.measure(c[0], out[1])
    return qc


def qram_pair_circuit(t: np.ndarray, x0: np.ndarray, x1: np.ndarray, *, bits: int = 8, query: str = "qrom",
                      measure: bool = True):
    """Pair circuit whose single loader is a QRAM addressed by (a, m).

    Memory content: (a=0, m=*) -> t, (a=1, m=0) -> x0, (a=1, m=1) -> x1. The
    controlled loaders disappear: the address register selects the vector.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    states = [np.asarray(s, dtype=np.float64) for s in (t, x0, x1)]
    signed = any(np.any(s < -1e-15) for s in states)
    memory = QRAMMemory([states[0], states[0], states[1], states[2]], index_bits=2, bits=bits, signed=signed)
    loader = QRAMLoader(bits=bits, query=query)
    n = memory.n
    a, m, g, c = QuantumRegister(1, "a"), QuantumRegister(1, "m"), QuantumRegister(n, "g"), QuantumRegister(1, "c")
    angle = QuantumRegister(bits, "theta")
    registers = [a, m, g, c, angle]
    tree = None
    if query == "bb":
        tree = BucketBrigadeTree.allocate(loader.tree_depth(2, n))
        registers += [tree.routers, tree.wires]
    qc = QuantumCircuit(*registers)
    qc.h(a[0])
    qc.h(m[0])
    # address bits: a is the most significant, then m
    loader.prepare(qc, [a[0], m[0]], list(g), list(angle), memory, tree)
    qc.cx(m[0], c[0])
    qc.h(a[0])
    if measure:
        out = ClassicalRegister(2, "out")
        qc.add_register(out)
        qc.measure(a[0], out[0])
        qc.measure(c[0], out[1])
    return qc


def superposition_joint_state(t: np.ndarray, references: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """State right before the final Hadamard, with class-balanced index weights.

    Qubit order (little-endian): a, m[0..k-1], g[0..n-1], c.
    """
    references = np.asarray(references, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    r = len(references)
    k = max(1, int(math.ceil(math.log2(r))))
    n = int(round(math.log2(len(t))))
    counts = {c: int((labels == c).sum()) for c in (0, 1)}
    out = np.zeros(1 << (1 + k + n + 1))
    for idx, (x, y) in enumerate(zip(references, labels)):
        w = 1.0 / math.sqrt(2 * 2 * counts[int(y)])  # 1/sqrt(2) for a, sqrt(1/(2 R_c)) for m
        for gi in range(1 << n):
            base = (idx << 1) | (gi << (1 + k)) | (int(y) << (1 + k + n))
            out[base] += w * t[gi]
            out[base | 1] += w * x[gi]
    return out


def superposition_circuit(t: np.ndarray, references: np.ndarray, labels: np.ndarray):
    """Superposition protocol with an ideal QRAM oracle that prepares the joint state."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    state = superposition_joint_state(t, references, labels)
    nq = int(round(math.log2(len(state))))
    q = QuantumRegister(nq, "q")
    out = ClassicalRegister(2, "out")
    qc = QuantumCircuit(q, out)
    qc.initialize(state, q)
    qc.h(q[0])
    qc.measure(q[0], out[0])
    qc.measure(q[nq - 1], out[1])
    return qc


def interpret_counts(counts: dict[str, int]) -> tuple[int, int, int]:
    """Return (shots with a=0,c=0, shots with a=0,c=1, rejected shots)."""
    k0 = k1 = rejected = 0
    for bits, number in counts.items():
        value = int(bits.replace(" ", ""), 2)
        if value & 1:          # a = 1: rejected by post-selection
            rejected += number
        elif (value >> 1) & 1:  # a = 0, c = 1
            k1 += number
        else:
            k0 += number
    return k0, k1, rejected


def resources(circuit, basis_gates: Sequence[str] = ("cx", "u"), optimization_level: int = 1, seed: int = 0) -> dict:
    """Qubits, depth, CNOT count and size after transpilation to ``basis_gates``."""
    from qiskit import transpile

    tqc = transpile(circuit.remove_final_measurements(inplace=False), basis_gates=list(basis_gates),
                    optimization_level=optimization_level, seed_transpiler=seed)
    ops = tqc.count_ops()
    return {"qubits": circuit.num_qubits, "depth": tqc.depth(), "cx": int(ops.get("cx", 0)), "size": int(tqc.size())}
