"""A minimal batched statevector simulator written in NumPy.

A state of ``n`` qubits is stored as a complex array of shape ``(B, 2, ..., 2)``:
axis 0 indexes a batch of independent circuits (used to evaluate all the
parameter-shifted circuits in one call) and axis ``w + 1`` is qubit ``w``.
Qubit 0 is the most significant bit, so ``|q0 q1 ... q_{n-1}>`` flattens to
index ``q0 * 2**(n-1) + ... + q_{n-1}`` (the same convention as PennyLane).

Rotations follow ``R_P(theta) = exp(-i theta P / 2)`` for ``P in {X, Y, Z}``.
"""

import numpy as np


def rx(theta):
    """RX(theta) for a scalar or a batch of angles; returns shape ``(..., 2, 2)``."""
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -1j * s], -1), np.stack([-1j * s, c], -1)], -2)


def ry(theta):
    """RY(theta) for a scalar or a batch of angles; returns shape ``(..., 2, 2)``."""
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def rz(theta):
    """RZ(theta) for a scalar or a batch of angles; returns shape ``(..., 2, 2)``."""
    theta = np.asarray(theta, dtype=float)
    e = np.exp(-0.5j * theta)
    zero = np.zeros_like(e)
    return np.stack([np.stack([e, zero], -1), np.stack([zero, e.conj()], -1)], -2)


ROTATIONS = {"RX": rx, "RY": ry, "RZ": rz}


def zero_state(n_qubits, batch=1):
    """``|0...0>`` repeated ``batch`` times, shape ``(batch, 2, ..., 2)``."""
    state = np.zeros((batch,) + (2,) * n_qubits, dtype=complex)
    state[(slice(None),) + (0,) * n_qubits] = 1.0
    return state


def apply_single_qubit(state, gate, wire):
    """Apply a ``(2, 2)`` or batched ``(B, 2, 2)`` gate to qubit ``wire``."""
    axis = wire + 1
    psi = np.moveaxis(state, axis, 1)
    shape = psi.shape
    psi = gate @ psi.reshape(shape[0], 2, -1)
    return np.moveaxis(psi.reshape(shape), 1, axis)


def apply_cnot(state, control, target):
    """Flip qubit ``target`` on the part of the state where ``control`` is 1."""
    assert control != target, "control and target must differ"
    state = state.copy()
    idx = [slice(None)] * state.ndim
    idx[control + 1] = 1
    idx = tuple(idx)
    # Removing the control axis shifts every later axis one place to the left
    target_axis = target + 1 if target < control else target
    state[idx] = np.flip(state[idx], axis=target_axis).copy()
    return state


def expval_z(state, wire):
    """``<Z_wire>`` for every state in the batch, shape ``(B,)``."""
    probs = np.abs(state) ** 2
    others = tuple(a for a in range(1, state.ndim) if a != wire + 1)
    p = probs.sum(axis=others)  # (B, 2): P(qubit = 0), P(qubit = 1)
    return p[:, 0] - p[:, 1]


class Circuit:
    """A fixed sequence of gates whose rotation angles are supplied at run time.

    Every rotation gate owns one *angle slot*. ``run(angles)`` takes one angle per
    slot (or a batch ``(B, n_slots)`` of them), so a circuit is a pure function
    ``angles -> state``. The autograd op in ``qnode.py`` differentiates that
    function with the parameter-shift rule, slot by slot.
    """

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.ops = []  # (name, wires, slot or None)
        self.n_slots = 0

    def _rotation(self, name, wire):
        assert 0 <= wire < self.n_qubits
        slot = self.n_slots
        self.ops.append((name, (wire,), slot))
        self.n_slots += 1
        return slot

    def rx(self, wire):
        return self._rotation("RX", wire)

    def ry(self, wire):
        return self._rotation("RY", wire)

    def rz(self, wire):
        return self._rotation("RZ", wire)

    def cnot(self, control, target):
        assert 0 <= control < self.n_qubits and 0 <= target < self.n_qubits
        self.ops.append(("CNOT", (control, target), None))

    def run(self, angles):
        """Final state for angles of shape ``(n_slots,)`` or ``(B, n_slots)``."""
        angles = np.atleast_2d(np.asarray(angles, dtype=float))
        assert angles.shape[1] == self.n_slots, f"expected {self.n_slots} angles"
        state = zero_state(self.n_qubits, batch=angles.shape[0])
        for name, wires, slot in self.ops:
            if name == "CNOT":
                state = apply_cnot(state, *wires)
            else:
                state = apply_single_qubit(state, ROTATIONS[name](angles[:, slot]), wires[0])
        return state

    def expval(self, angles, wires=(0,), shots=None, rng=None):
        """``<Z_w>`` for each ``w`` in ``wires``, shape ``(B, len(wires))``.

        With ``shots`` set, each expectation is estimated from that many simulated
        measurements instead of being computed exactly.
        """
        state = self.run(angles)
        z = np.stack([expval_z(state, w) for w in wires], axis=1)
        if shots is None:
            return z
        rng = np.random.default_rng() if rng is None else rng
        p0 = np.clip((1 + z) / 2, 0.0, 1.0)
        return 2 * rng.binomial(shots, p0) / shots - 1

    def draw(self, slot_labels=None):
        """Plain-text diagram, one line per qubit; gates on disjoint qubits share a column."""
        columns = []  # each column maps wire -> cell text
        free = [0] * self.n_qubits  # first column where each wire is free
        for name, wires, slot in self.ops:
            if name == "CNOT":
                span = range(min(wires), max(wires) + 1)
                cells = {w: "│" for w in span}
                cells.update({wires[0]: "●", wires[1]: "X"})
            else:
                span = wires
                label = slot_labels[slot] if slot_labels else f"θ{slot}"
                cells = {wires[0]: f"{name}({label})"}
            col = max(free[w] for w in span)
            while len(columns) <= col:
                columns.append({})
            columns[col].update(cells)
            for w in span:
                free[w] = col + 1
        lines = []
        for w in range(self.n_qubits):
            line = f"q{w}: "
            for cells in columns:
                width = max(len(c) for c in cells.values())
                line += "─" + cells.get(w, "").center(width, "─") + "─"
            lines.append(line)
        return "\n".join(lines)
