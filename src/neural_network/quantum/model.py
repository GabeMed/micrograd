"""A variational quantum classifier ("quantum neuron") trained through ``Value``."""

import numpy as np
from numpy import random

from ..network import Module
from ..value import Value
from .qnode import expval
from .simulator import Circuit


class QuantumNeuron(Module):
    """Data re-uploading variational circuit with an affine classical readout.

    Every layer ``l`` applies, to each qubit ``q``::

        RY(w[l, q] * x[q % n_inputs] + b[l, q])  ->  RZ(theta[l, q, 0])  ->  RY(theta[l, q, 1])

    followed by a CNOT chain ``0 -> 1 -> ... -> n-1``. The score is
    ``a * <Z_0> + c``, where ``a`` and ``c`` let ``<Z_0>`` in [-1, 1] reach the
    hinge-loss margin. The encoding weights ``w, b`` are classical ``Value``
    nodes; their gradients arrive through the parameter-shift rule.
    """

    def __init__(self, n_inputs=2, n_qubits=2, n_layers=3, shots=None, rng=None):
        self.n_inputs, self.n_qubits, self.n_layers = n_inputs, n_qubits, n_layers
        self.shots, self.rng = shots, rng
        self.circuit = Circuit(n_qubits)
        self.w, self.b, self.theta = [], [], []
        self._slots = []  # how to build each angle slot: ("enc", w, b, feature) or ("var", θ)
        self._labels = []
        for _ in range(n_layers):
            for q in range(n_qubits):
                w, b = Value(random.uniform(-1, 1)), Value(0.0)
                self.w.append(w)
                self.b.append(b)
                feature = q % n_inputs
                self.circuit.ry(q)
                self._slots.append(("enc", w, b, feature))
                self._labels.append(f"w·x{feature}+b")
                for gate in (self.circuit.rz, self.circuit.ry):
                    t = Value(random.uniform(0, 2 * np.pi))
                    self.theta.append(t)
                    gate(q)
                    self._slots.append(("var", t))
                    self._labels.append(f"θ{len(self.theta) - 1}")
            for q in range(n_qubits - 1):
                self.circuit.cnot(q, q + 1)
        self.a, self.c = Value(1.0), Value(0.0)

    def angles(self, x):
        out = []
        for slot in self._slots:
            if slot[0] == "enc":
                _, w, b, feature = slot
                out.append(w * x[feature] + b)
            else:
                out.append(slot[1])
        return out

    def expectation(self, x):
        """``<Z_0>`` as a ``Value``, before the classical readout."""
        return expval(self.circuit, self.angles(x), wires=(0,), shots=self.shots, rng=self.rng)[0]

    def __call__(self, x):
        return self.a * self.expectation(x) + self.c

    def parameters(self):
        return self.w + self.b + self.theta + [self.a, self.c]

    def draw(self):
        return self.circuit.draw(self._labels)

    def __repr__(self):
        return (
            f"QuantumNeuron(qubits={self.n_qubits}, layers={self.n_layers}, "
            f"params={len(self.parameters())})"
        )
