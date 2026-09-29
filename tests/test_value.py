"""Autograd gradients against central finite differences (and PyTorch, if installed)."""

import math

import numpy as np
import pytest

from neural_network import NN, Value, loss


def numeric_grad(f, xs, h=1e-6):
    """Central differences of the scalar f(list of floats) at xs."""
    grads = []
    for i in range(len(xs)):
        up, down = list(xs), list(xs)
        up[i] += h
        down[i] -= h
        grads.append((f(up) - f(down)) / (2 * h))
    return grads


def check(expr, xs, tol=1e-6):
    """Compare Value.backward_prop() with finite differences for expr(list of Values)."""
    vs = [Value(x) for x in xs]
    out = expr(vs)
    out.backward_prop()
    expected = numeric_grad(lambda ys: expr([Value(y) for y in ys]).data, xs)
    for v, g in zip(vs, expected):
        assert v.grad == pytest.approx(g, rel=tol, abs=tol)


@pytest.mark.parametrize(
    "expr",
    [
        lambda v: v[0] + v[1],
        lambda v: v[0] - v[1],
        lambda v: v[0] * v[1],
        lambda v: v[0] / v[1],
        lambda v: v[0] ** 3,
        lambda v: v[1] ** -1.5,
        lambda v: -v[0],
        lambda v: 2.0 + v[0],
        lambda v: 2.0 - v[0],
        lambda v: 3.0 * v[0],
        lambda v: 3.0 / v[1],
        lambda v: v[0].relu() + (-v[0]).relu(),
        lambda v: v[0].tanh(),
        lambda v: v[0].exp(),
        lambda v: v[1].log(),
        lambda v: v[0] * v[0],  # the same node on both sides of an op
    ],
)
def test_ops_match_finite_differences(expr):
    check(expr, [0.7, 1.9])


def test_composite_expression():
    def expr(v):
        a, b = v
        c = a * b + b**3
        d = (c * 2 + (b + a).relu()).tanh() + c.exp() * 0.01
        e = (d * d + 1).log() / (a * a + 1)
        return e - 3 * a

    check(expr, [-0.8, 0.45])


def test_deep_graph_does_not_hit_recursion_limit():
    x = Value(1.0)
    y = sum((x * 1.0 for _ in range(20_000)), Value(0.0))
    y.backward_prop()
    assert x.grad == pytest.approx(20_000)


def test_mlp_loss_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, -1)
    np.random.seed(1)
    model = NN(2, [4, 3, 1])
    params = model.parameters()
    # Nonzero biases keep every ReLU input away from the kink at exactly 0, where
    # the one-sided analytic derivative and the central difference legitimately differ
    for p in params:
        p.data += 0.05 * np.random.uniform(-1, 1)

    total, _ = loss(X, y, model, alpha=1e-2)
    model.zero_grad()
    total.backward_prop()
    analytic = [p.grad for p in params]

    base = [p.data for p in params]

    def f(values):
        for p, v in zip(params, values):
            p.data = v
        return loss(X, y, model, alpha=1e-2)[0].data

    numeric = numeric_grad(f, base)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_matches_pytorch():
    torch = pytest.importorskip("torch")

    def build(a, b, relu, tanh, exp, log):
        c = a * b + b**3
        d = tanh(c * 2 + relu(b + a)) + exp(c) * 0.01
        e = log(d * d + 1) / (a * a + 1)
        return e - 3 * a + 1 / b

    a, b = Value(-0.8), Value(0.45)
    out = build(a, b, Value.relu, Value.tanh, Value.exp, Value.log)
    out.backward_prop()

    ta = torch.tensor(-0.8, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(0.45, dtype=torch.float64, requires_grad=True)
    tout = build(ta, tb, torch.relu, torch.tanh, torch.exp, torch.log)
    tout.backward()

    assert out.data == pytest.approx(tout.item(), rel=1e-12)
    assert a.grad == pytest.approx(ta.grad.item(), rel=1e-10)
    assert b.grad == pytest.approx(tb.grad.item(), rel=1e-10)
    assert not math.isnan(a.grad)
