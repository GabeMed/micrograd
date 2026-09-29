"""A from-scratch scalar autograd engine with a classical MLP and a quantum variant."""

from .loss import accuracy, loss
from .network import NN, Layer, Module, Neuron
from .train import train
from .value import Value

__all__ = ["NN", "Layer", "Module", "Neuron", "Value", "accuracy", "loss", "train"]
