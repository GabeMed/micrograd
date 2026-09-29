"""Quantum variant: a NumPy statevector simulator and parameter-shift autograd op."""

from .model import QuantumNeuron
from .qnode import expval, parameter_shift_jacobian
from .simulator import Circuit

__all__ = ["Circuit", "QuantumNeuron", "expval", "parameter_shift_jacobian"]
