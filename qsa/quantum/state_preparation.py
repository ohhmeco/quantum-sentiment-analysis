"""Amplitude encoding: Grover-Rudolph state preparation and its QRAM version.

Amplitude encoding maps a unit vector x in R^(2^n) to the n-qubit state
``|x> = sum_i x_i |i>``. Both loaders below use the *same* binary tree of angles
(Grover & Rudolph, 2002): at level l the qubit l is rotated by RY(theta_b),
conditioned on the l already prepared bits b, with
``cos(theta_b / 2) = ||x_{b0}|| / ||x_b||`` (norms of the two halves of the
sub-vector selected by the prefix b). At the last level the signed amplitudes
are used with ``atan2``, so arbitrary *real* vectors (also negative entries,
as produced by PCA) are prepared exactly.

* :class:`GroverRudolphLoader` synthesises every level as a uniformly controlled
  RY multiplexer (Gray-code construction of Mottonen et al., 2004):
  2^n - 1 RY and 2^n - 2 CNOT gates, no ancillas.
* :class:`QRAMLoader` follows Kerenidis & Prakash (2017): the angles are stored,
  quantised to ``bits`` bits, in a quantum memory; level l queries the memory
  with the prefix as address, applies a rotation controlled by the angle
  register and un-queries. The query is either a lookup table (QROM, depth O(N))
  or a bucket-brigade QRAM (Giovannetti, Lloyd & Maccone, 2008; O(N) qubits,
  depth O(log N) per query).

Qubit order follows Qiskit (little-endian): data[n-1] is the most significant
bit, i.e. the qubit prepared at level 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    "tree_angles", "state_from_tree", "GroverRudolphLoader", "QRAMMemory", "QRAMLoader",
    "quantized_state", "BucketBrigadeTree", "qrom_query", "bucket_brigade_query",
]


# Angle tree -------------------------------------------------------------------------------

def _check_state(state: np.ndarray) -> tuple[np.ndarray, int]:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    n = int(round(math.log2(len(state)))) if len(state) else 0
    if n < 1 or (1 << n) != len(state):
        raise ValueError("state length must be a power of two >= 2")
    if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-9):
        raise ValueError("state must have unit norm")
    return state, n


def tree_angles(state: np.ndarray) -> list[list[float]]:
    """Grover-Rudolph angles; ``levels[l][b]`` is the angle for prefix b (MSB first)."""
    state, n = _check_state(state)
    levels: list[list[float]] = []
    for level in range(n):
        width = 1 << (n - level)
        angles = []
        for prefix in range(1 << level):
            pos = prefix * width
            if width == 2:  # last level: signed single amplitudes
                left, right = float(state[pos]), float(state[pos + 1])
            else:
                left = float(np.linalg.norm(state[pos:pos + width // 2]))
                right = float(np.linalg.norm(state[pos + width // 2:pos + width]))
            angles.append(0.0 if math.hypot(left, right) <= 1e-15 else 2.0 * math.atan2(right, left))
        levels.append(angles)
    return levels


def state_from_tree(levels: list[list[float]]) -> np.ndarray:
    """Inverse of :func:`tree_angles`: the amplitudes produced by a tree of angles."""
    amps = np.ones(1)
    for angles in levels:
        theta = np.asarray(angles, dtype=np.float64)
        nxt = np.empty(2 * len(amps))
        nxt[0::2] = amps * np.cos(theta / 2.0)
        nxt[1::2] = amps * np.sin(theta / 2.0)
        amps = nxt
    return amps


def _gray(i: int) -> int:
    return i ^ (i >> 1)


# Grover-Rudolph with multiplexers -----------------------------------------------------------

class GroverRudolphLoader:
    """Grover-Rudolph state preparation with Gray-code RY multiplexers."""

    name = "grover-rudolph"

    def circuit(self, state: np.ndarray, label: str = "GR"):
        """Return a ``QuantumCircuit`` on n qubits with ``circuit|0> = |state>``."""
        from qiskit import QuantumCircuit

        state, n = _check_state(state)
        qc = QuantumCircuit(n, name=label)
        for level, alpha in enumerate(tree_angles(state)):
            target = n - 1 - level
            if level == 0:
                if abs(alpha[0]) > 1e-14:
                    qc.ry(alpha[0], target)
                continue
            size = 1 << level
            alpha = np.asarray(alpha)
            # The multiplexer applies RY(sum_i (-1)^{x . gray(i)} phi_i) for control state x,
            # hence phi = (1/2^l) * H_gray alpha (Walsh-Hadamard transform in Gray order).
            phi = np.array([
                sum((-1.0 if bin(x & _gray(i)).count("1") % 2 else 1.0) * alpha[x] for x in range(size)) / size
                for i in range(size)
            ])
            for i in range(size):
                if abs(phi[i]) > 1e-14:
                    qc.ry(float(phi[i]), target)
                changed = _gray(i) ^ _gray((i + 1) % size)
                # prefix bit j (LSB = 0) lives on data qubit n - level + j
                qc.cx(n - level + changed.bit_length() - 1, target)
        return qc

    @staticmethod
    def fidelity(circuit, state: np.ndarray) -> float:
        """|<state|circuit|0>|^2, computed with a statevector simulation."""
        from qiskit.quantum_info import Statevector

        prepared = Statevector.from_instruction(circuit).data
        return float(abs(np.vdot(np.asarray(state, dtype=complex), prepared)) ** 2)


# QRAM (Kerenidis-Prakash) -----------------------------------------------------------------

def _level_range(level: int, n_levels: int, signed: bool) -> float:
    """Inner levels store norms (theta in [0, pi]); the signed last level needs 4*pi."""
    return 4.0 * math.pi if (signed and level == n_levels - 1) else math.pi


def _quantize(theta: float, bits: int, span: float) -> int:
    levels = 1 << bits
    if span >= 4.0 * math.pi - 1e-12:
        return int(round((theta % (4.0 * math.pi)) / span * levels)) % levels
    return min(max(int(round(theta / span * levels)), 0), levels - 1)


def quantized_angle_table(state: np.ndarray, bits: int, signed: bool) -> list[list[int]]:
    levels = tree_angles(state)
    return [[_quantize(t, bits, _level_range(l, len(levels), signed)) for t in row] for l, row in enumerate(levels)]


def quantized_state(state: np.ndarray, bits: int | None, signed: bool | None = None) -> np.ndarray:
    """State produced by the QRAM loader when angles are stored with ``bits`` bits.

    ``bits=None`` means an ideal memory (exact angles). ``signed`` selects the angle
    range of the last level; it must match the memory the vector is stored in (a
    memory holding any vector with negative entries uses the signed range for all
    vectors). By default it is inferred from ``state``. The result equals the output
    of the explicit circuit, but is computed classically from the tree.
    """
    state, n = _check_state(state)
    if bits is None:
        return state.copy()
    if signed is None:
        signed = bool(np.any(state < -1e-15))
    table = quantized_angle_table(state, bits, signed)
    levels = [[_level_range(l, n, signed) * k / (1 << bits) for k in row] for l, row in enumerate(table)]
    out = state_from_tree(levels)
    return out / np.linalg.norm(out)


@dataclass
class QRAMMemory:
    """Quantised angle tables of several vectors; vector v and prefix p are stored at
    address ``v * 2^l + p`` of the level-l table."""

    vectors: list[np.ndarray]
    index_bits: int
    bits: int
    signed: bool
    tables: list[list[list[int]]] = field(init=False)
    n: int = field(init=False)

    def __post_init__(self) -> None:
        dims = {len(v) for v in self.vectors}
        if len(dims) != 1:
            raise ValueError("all vectors must have the same length")
        if len(self.vectors) > (1 << self.index_bits):
            raise ValueError("too many vectors for index_bits")
        self.n = int(round(math.log2(dims.pop())))
        self.tables = [quantized_angle_table(v, self.bits, self.signed) for v in self.vectors]

    def level_table(self, level: int) -> list[int]:
        width = 1 << level
        table = [0] * ((1 << self.index_bits) * width)
        for v, t in enumerate(self.tables):
            for p, k in enumerate(t[level]):
                table[v * width + p] = k
        return table

    def rotation_step(self, level: int) -> float:
        return _level_range(level, self.n, self.signed) / (1 << self.bits)

    @property
    def memory_cells(self) -> int:
        return len(self.vectors) * ((1 << self.n) - 1)


def qrom_query(qc: Any, addr: Sequence[Any], out: Sequence[Any], table: Sequence[int]) -> None:
    """Lookup-table query ``|addr>|y> -> |addr>|y XOR table[addr]>`` (addr[0] = MSB)."""
    k = len(addr)
    if len(table) != (1 << k):
        raise ValueError("table size must be 2^len(addr)")
    for value, data in enumerate(table):
        if data == 0:
            continue
        flips = [addr[i] for i in range(k) if not (value >> (k - 1 - i)) & 1]
        for q in flips:
            qc.x(q)
        for j, target in enumerate(out):
            if (data >> j) & 1:
                qc.mcx(list(addr), target) if k else qc.x(target)
        for q in flips:
            qc.x(q)


@dataclass
class BucketBrigadeTree:
    """Qubit bucket-brigade tree: one router per internal node and one wire per node
    (heap order, root = 1). Empty wires are |0>, so routers off the addressed path
    only swap |0> with |0>."""

    depth: int
    routers: Any
    wires: Any

    @classmethod
    def allocate(cls, depth: int) -> "BucketBrigadeTree":
        from qiskit import QuantumRegister

        return cls(depth, QuantumRegister((1 << depth) - 1, "router"), QuantumRegister((1 << (depth + 1)) - 1, "wire"))

    def r(self, v: int):
        return self.routers[v - 1]

    def w(self, v: int):
        return self.wires[v - 1]

    def _route_level(self, qc: Any, depth: int, down: bool) -> None:
        for v in range(1 << depth, 1 << (depth + 1)):
            r, w, left, right = self.r(v), self.w(v), self.w(2 * v), self.w(2 * v + 1)
            steps = [("R", right), ("L", left)] if down else [("L", left), ("R", right)]
            for side, child in steps:
                if side == "L":
                    qc.x(r)
                qc.cswap(r, w, child)
                if side == "L":
                    qc.x(r)

    def route_down(self, qc: Any, levels: int) -> None:
        for d in range(levels):
            self._route_level(qc, d, down=True)

    def route_up(self, qc: Any, levels: int) -> None:
        for d in reversed(range(levels)):
            self._route_level(qc, d, down=False)


def bucket_brigade_query(qc: Any, addr: Sequence[Any], out: Sequence[Any], table: Sequence[int],
                         tree: BucketBrigadeTree) -> None:
    """Bucket-brigade query ``|addr>|y> -> |addr>|y XOR table[addr]>``.

    The address bits are routed into the tree; for every output bit a bus qubit in
    |+> travels to the addressed leaf, picks up a Z phase if the stored bit is 1,
    travels back and is mapped to |y XOR bit> by a Hadamard; finally the address is
    un-routed, leaving the tree clean.
    """
    k = len(addr)
    if k > tree.depth or len(table) != (1 << k):
        raise ValueError("address longer than the tree, or wrong table size")

    def load() -> None:
        for d, q in enumerate(addr):
            qc.swap(q, tree.w(1))
            tree.route_down(qc, d)
            for v in range(1 << d, 1 << (d + 1)):
                qc.swap(tree.w(v), tree.r(v))

    def unload() -> None:
        for d in reversed(range(k)):
            for v in range(1 << d, 1 << (d + 1)):
                qc.swap(tree.w(v), tree.r(v))
            tree.route_up(qc, d)
            qc.swap(addr[d], tree.w(1))

    load()
    for j, bus in enumerate(out):
        marked = [value for value, data in enumerate(table) if (data >> j) & 1]
        if not marked:
            continue
        qc.h(bus)
        qc.swap(bus, tree.w(1))
        tree.route_down(qc, k)
        for value in marked:
            qc.z(tree.w((1 << k) + value))
        tree.route_up(qc, k)
        qc.swap(bus, tree.w(1))
        qc.h(bus)
    unload()


class QRAMLoader:
    """Grover-Rudolph with angles read from a quantum memory (Kerenidis-Prakash)."""

    def __init__(self, bits: int = 8, query: str = "qrom"):
        if query not in ("qrom", "bb"):
            raise ValueError("query must be 'qrom' or 'bb'")
        if not 1 <= bits <= 24:
            raise ValueError("bits must be in 1..24")
        self.bits, self.query = bits, query
        self.name = f"qram-{query}-b{bits}"

    def prepare(self, qc: Any, index: Sequence[Any], data: Sequence[Any], angle: Sequence[Any],
                memory: QRAMMemory, tree: BucketBrigadeTree | None = None) -> None:
        """Append the preparation of ``|v>|0>_data -> |v>|x_v>`` (index register v)."""
        n = len(data)
        for level in range(n):
            target = data[n - 1 - level]
            addr = list(index) + [data[n - 1 - i] for i in range(level)]
            table = memory.level_table(level)
            query = (lambda: qrom_query(qc, addr, angle, table)) if self.query == "qrom" else (
                lambda: bucket_brigade_query(qc, addr, angle, table, tree))
            query()
            step = memory.rotation_step(level)
            for j, q in enumerate(angle):  # RY(step * k), k read bit by bit
                qc.cry(step * (1 << j), q, target)
            query()  # the query is an XOR: applying it again un-computes the angle register

    def tree_depth(self, index_bits: int, n: int) -> int:
        return max(2, index_bits + n - 1)
