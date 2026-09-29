"""Sanity checks for the NumPy statevector simulator."""

import numpy as np
import pytest

from neural_network.quantum import Circuit
from neural_network.quantum.simulator import (
    apply_cnot,
    apply_single_qubit,
    expval_z,
    rx,
    ry,
    rz,
    zero_state,
)


def random_circuit(n_qubits, n_layers, seed=0):
    rng = np.random.default_rng(seed)
    c = Circuit(n_qubits)
    for _ in range(n_layers):
        for q in range(n_qubits):
            getattr(c, rng.choice(["rx", "ry", "rz"]))(q)
        if n_qubits > 1:
            a, b = rng.choice(n_qubits, size=2, replace=False)
            c.cnot(int(a), int(b))
    return c, rng.uniform(-np.pi, np.pi, size=c.n_slots)


def dense_unitary(circuit, angles):
    """Reference: build the full 2^n x 2^n unitary with Kronecker products."""
    n = circuit.n_qubits
    eye = np.eye(2)
    proj0, proj1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    x = np.array([[0, 1], [1, 0]])
    gates = {"RX": rx, "RY": ry, "RZ": rz}

    def kron(mats):
        out = np.array([[1.0]])
        for m in mats:
            out = np.kron(out, m)
        return out

    u = np.eye(2**n, dtype=complex)
    for name, wires, slot in circuit.ops:
        if name == "CNOT":
            c, t = wires
            a = kron([proj0 if w == c else eye for w in range(n)])
            b = kron([proj1 if w == c else x if w == t else eye for w in range(n)])
            g = a + b
        else:
            g = kron([gates[name](angles[slot]) if w == wires[0] else eye for w in range(n)])
        u = g @ u
    return u


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_state_is_normalised(n_qubits):
    circuit, angles = random_circuit(n_qubits, n_layers=5, seed=n_qubits)
    state = circuit.run(angles)
    assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_matches_dense_unitary(n_qubits):
    circuit, angles = random_circuit(n_qubits, n_layers=6, seed=10 + n_qubits)
    expected = dense_unitary(circuit, angles)[:, 0]  # U |0...0>
    np.testing.assert_allclose(circuit.run(angles).reshape(-1), expected, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, 2.0, np.pi, -1.1])
def test_single_qubit_expectations(theta):
    for gate, expected in (("rx", np.cos(theta)), ("ry", np.cos(theta)), ("rz", 1.0)):
        c = Circuit(1)
        getattr(c, gate)(0)
        assert c.expval([theta])[0, 0] == pytest.approx(expected, abs=1e-12)
    # RZ after RY changes the phase only, not <Z>
    c = Circuit(1)
    c.ry(0)
    c.rz(0)
    assert c.expval([theta, 0.9])[0, 0] == pytest.approx(np.cos(theta), abs=1e-12)


def test_rotation_amplitudes():
    theta = 0.7
    state = apply_single_qubit(zero_state(1), rx(theta), 0).reshape(-1)
    np.testing.assert_allclose(state, [np.cos(theta / 2), -1j * np.sin(theta / 2)], atol=1e-12)
    state = apply_single_qubit(zero_state(1), ry(theta), 0).reshape(-1)
    np.testing.assert_allclose(state, [np.cos(theta / 2), np.sin(theta / 2)], atol=1e-12)


def basis_state(bits):
    state = np.zeros((1,) + (2,) * len(bits), dtype=complex)
    state[(0,) + tuple(bits)] = 1.0
    return state


@pytest.mark.parametrize(
    "bits, expected",
    [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 1)), ((1, 1), (1, 0))],
)
def test_cnot_truth_table(bits, expected):
    out = apply_cnot(basis_state(bits), control=0, target=1)
    np.testing.assert_allclose(out, basis_state(expected))


def test_cnot_with_control_below_target_and_non_adjacent_wires():
    # control = qubit 2, target = qubit 0 on |0 1 1> -> |1 1 1>
    out = apply_cnot(basis_state((0, 1, 1)), control=2, target=0)
    np.testing.assert_allclose(out, basis_state((1, 1, 1)))
    # control 0 is |0>: nothing happens
    out = apply_cnot(basis_state((0, 1, 0)), control=0, target=2)
    np.testing.assert_allclose(out, basis_state((0, 1, 0)))


def test_bell_state():
    c = Circuit(2)
    c.ry(0)
    c.cnot(0, 1)
    state = c.run([np.pi / 2])
    np.testing.assert_allclose(np.abs(state.reshape(-1)) ** 2, [0.5, 0, 0, 0.5], atol=1e-12)
    np.testing.assert_allclose(c.expval([np.pi / 2], wires=(0, 1)), [[0.0, 0.0]], atol=1e-12)


def test_batched_run_equals_individual_runs():
    circuit, _ = random_circuit(3, n_layers=4, seed=7)
    batch = np.random.default_rng(1).uniform(-3, 3, size=(5, circuit.n_slots))
    together = circuit.expval(batch, wires=(0, 1, 2))
    one_by_one = np.concatenate([circuit.expval(row, wires=(0, 1, 2)) for row in batch])
    np.testing.assert_allclose(together, one_by_one, atol=1e-12)


def test_expval_z_of_basis_states():
    assert expval_z(basis_state((1, 0)), 0)[0] == pytest.approx(-1)
    assert expval_z(basis_state((1, 0)), 1)[0] == pytest.approx(1)


def test_shot_estimate_is_close_to_exact():
    c = Circuit(1)
    c.ry(0)
    theta, shots = 1.2, 20_000
    estimate = c.expval([theta], shots=shots, rng=np.random.default_rng(0))[0, 0]
    sigma = np.sqrt((1 - np.cos(theta) ** 2) / shots)
    assert abs(estimate - np.cos(theta)) < 5 * sigma


def test_draw_lists_every_qubit():
    circuit, _ = random_circuit(3, n_layers=2)
    lines = circuit.draw().splitlines()
    assert [line[:3] for line in lines] == ["q0:", "q1:", "q2:"]
