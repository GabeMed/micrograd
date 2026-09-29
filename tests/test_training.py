"""Short end-to-end training runs and command-line smoke tests."""

import matplotlib
import numpy as np
import pytest

from neural_network import NN, accuracy, loss, train
from neural_network.quantum import QuantumNeuron

matplotlib.use("Agg")


@pytest.fixture
def moons():
    from neural_network.experiments import make_data

    X, y, X_test, y_test = make_data(seed=0, n_train=60, n_test=100)
    return X, y, X_test, y_test


def test_mlp_learns_moons(moons):
    X, y, X_test, y_test = moons
    np.random.seed(0)
    model = NN(2, [8, 8, 1])
    history = train(model, X, y, steps=40, lr=1.0, alpha=1e-4)
    assert history["loss"][-1] < 0.5 * history["loss"][0]
    assert accuracy(X, y, model) >= 0.9
    assert accuracy(X_test, y_test, model) >= 0.85


def test_quantum_neuron_learns_moons(moons):
    X, y, X_test, y_test = moons
    np.random.seed(0)
    model = QuantumNeuron(n_layers=3)
    history = train(
        model, X, y, steps=40, lr=1.0, alpha=0.0, X_test=X_test, y_test=y_test, eval_every=20
    )
    assert history["loss"][-1] < 0.5 * history["loss"][0]
    assert history["test_step"] == [20, 40]
    assert accuracy(X, y, model) >= 0.85


def test_loss_with_minibatch():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([1, -1, 1, -1])
    np.random.seed(0)
    total, acc = loss(X, y, NN(2, [3, 1]), batch_size=2, rng=np.random.default_rng(0))
    assert total.data >= 0 and 0 <= acc <= 1


def test_cli_train_and_graph(tmp_path, capsys):
    from neural_network.__main__ import main

    main(["train", "mlp-small", "--steps", "3", "--plot", str(tmp_path / "boundary.png")])
    main(["graph", "--out", str(tmp_path)])
    main(["circuit"])
    assert "RY(w·x0+b)" in capsys.readouterr().out
    for name in ("boundary.png", "computation_graph.png", "hybrid_graph.png"):
        assert (tmp_path / name).stat().st_size > 1000


def test_compare_writes_table_and_plots(tmp_path):
    from neural_network.experiments import compare

    summary = compare(seeds=1, steps=2, out=tmp_path, kinds=("mlp-small", "vqc"))
    assert [row["kind"] for row in summary] == ["mlp-small", "vqc"]
    for name in ("results.md", "results.json", "decision_boundaries.png", "training_curves.png"):
        assert (tmp_path / name).exists()
