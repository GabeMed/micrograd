"""Parameter-shift gradients against finite differences (and PennyLane, if installed)."""

import numpy as np
import pytest

from neural_network import Value, loss
from neural_network.quantum import Circuit, QuantumNeuron, expval, parameter_shift_jacobian


def layered_circuit(n_qubits=3, n_layers=2):
    c = Circuit(n_qubits)
    for _ in range(n_layers):
        for q in range(n_qubits):
            c.rx(q)
            c.ry(q)
            c.rz(q)
        for q in range(n_qubits):
            c.cnot(q, (q + 1) % n_qubits)
    return c


def finite_difference_jacobian(circuit, theta, wires, h=1e-6):
    cols = []
    for j in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        cols.append((circuit.expval(up, wires)[0] - circuit.expval(down, wires)[0]) / (2 * h))
    return np.stack(cols, axis=1)


def test_parameter_shift_matches_finite_differences():
    circuit = layered_circuit()
    theta = np.random.default_rng(0).uniform(-np.pi, np.pi, size=circuit.n_slots)
    wires = (0, 1, 2)
    ps = parameter_shift_jacobian(circuit, theta, wires)
    fd = finite_difference_jacobian(circuit, theta, wires)
    assert ps.shape == (3, circuit.n_slots)
    np.testing.assert_allclose(ps, fd, atol=1e-8)


def test_gradient_flows_through_value_graph():
    """Classical preprocessing -> circuit -> classical postprocessing, with a shared angle."""
    circuit = Circuit(2)
    circuit.ry(0)  # slot 0: w * x + b
    circuit.rx(1)  # slot 1: theta
    circuit.cnot(0, 1)
    circuit.ry(1)  # slot 2: theta again (shared parameter)
    circuit.rz(0)  # slot 3: w * w

    def forward(w, b, theta):
        angles = [w * 0.7 + b, theta, theta, w * w]
        z0, z1 = expval(circuit, angles, wires=(0, 1))
        return (z0 * z1 + z0.tanh()) * 2 - z1

    leaves = [Value(0.4), Value(-0.3), Value(1.1)]
    out = forward(*leaves)
    out.backward_prop()

    h = 1e-6
    for i, leaf in enumerate(leaves):
        up = [Value(v.data + (h if k == i else 0)) for k, v in enumerate(leaves)]
        down = [Value(v.data - (h if k == i else 0)) for k, v in enumerate(leaves)]
        numeric = (forward(*up).data - forward(*down).data) / (2 * h)
        assert leaf.grad == pytest.approx(numeric, abs=1e-7)


def test_quantum_neuron_loss_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(8, 2))
    y = np.where(X[:, 0] > 0, 1, -1)
    np.random.seed(0)
    model = QuantumNeuron(n_layers=2)
    params = model.parameters()

    total, _ = loss(X, y, model, alpha=0.0)
    model.zero_grad()
    total.backward_prop()
    analytic = np.array([p.grad for p in params])

    h, numeric = 1e-6, []
    for p in params:
        p.data += h
        f_up = loss(X, y, model, alpha=0.0)[0].data
        p.data -= 2 * h
        f_down = loss(X, y, model, alpha=0.0)[0].data
        p.data += h
        numeric.append((f_up - f_down) / (2 * h))
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_jacobian_is_only_computed_on_backward():
    circuit = layered_circuit(2, 1)
    calls = []
    original = circuit.expval

    def counting_expval(angles, *args, **kwargs):
        calls.append(np.atleast_2d(angles).shape[0])
        return original(angles, *args, **kwargs)

    circuit.expval = counting_expval
    (z,) = expval(circuit, [Value(0.1 * i) for i in range(circuit.n_slots)])
    assert calls == [1]  # forward only
    (z * 3).backward_prop()
    assert calls == [1, 2 * circuit.n_slots]  # one batched call with all shifts


def test_shot_based_parameter_shift_is_unbiased():
    circuit = layered_circuit(2, 1)
    theta = np.random.default_rng(5).uniform(-np.pi, np.pi, size=circuit.n_slots)
    exact = parameter_shift_jacobian(circuit, theta)
    rng = np.random.default_rng(6)
    estimates = [parameter_shift_jacobian(circuit, theta, shots=500, rng=rng) for _ in range(200)]
    # std of one estimate <= sqrt(2 / 500) / 2; the mean of 200 has std <= 0.0023
    np.testing.assert_allclose(np.mean(estimates, axis=0), exact, atol=0.012)


def test_matches_pennylane():
    qml = pytest.importorskip("pennylane")
    pnp = pytest.importorskip("pennylane.numpy")

    circuit = layered_circuit(3, 2)
    theta = np.random.default_rng(2).uniform(-np.pi, np.pi, size=circuit.n_slots)
    dev = qml.device("default.qubit", wires=3)
    gates = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}

    @qml.qnode(dev, diff_method="parameter-shift")
    def reference(params):
        for name, wires, slot in circuit.ops:
            if name == "CNOT":
                qml.CNOT(wires=list(wires))
            else:
                gates[name](params[slot], wires=wires[0])
        return [qml.expval(qml.PauliZ(w)) for w in range(3)]

    params = pnp.array(theta, requires_grad=True)
    expected_values = np.array(reference(params))
    expected_jac = np.array(qml.jacobian(lambda p: qml.math.stack(reference(p)))(params))

    np.testing.assert_allclose(
        circuit.expval(theta, wires=(0, 1, 2))[0], expected_values, atol=1e-10
    )
    np.testing.assert_allclose(
        parameter_shift_jacobian(circuit, theta, wires=(0, 1, 2)), expected_jac, atol=1e-10
    )
