"""Quantum expectation values as nodes of the ``Value`` autograd graph.

For a rotation ``R_P(theta) = exp(-i theta P / 2)`` with a Pauli generator ``P``,
the expectation ``f(theta) = <psi(theta)| O |psi(theta)>`` is a sinusoid of period
``2 pi`` in each angle, so its derivative is exact at a finite shift:

    df/dtheta_j = [ f(theta + pi/2 e_j) - f(theta - pi/2 e_j) ] / 2

This is the parameter-shift rule. It needs only two extra circuit runs per angle
and, unlike finite differences, stays unbiased when ``f`` is estimated from shots.
"""

import numpy as np

from ..value import Value

SHIFT = np.pi / 2


def parameter_shift_jacobian(circuit, angles, wires=(0,), shots=None, rng=None):
    """Jacobian ``d<Z_w>/d angle_j`` with shape ``(len(wires), n_slots)``.

    All ``2 * n_slots`` shifted circuits are simulated in a single batched call.
    """
    theta = np.asarray(angles, dtype=float)
    n = theta.shape[0]
    shifts = SHIFT * np.eye(n)
    batch = np.concatenate([theta + shifts, theta - shifts])  # (2n, n)
    f = circuit.expval(batch, wires, shots=shots, rng=rng)  # (2n, len(wires))
    return 0.5 * (f[:n] - f[n:]).T


def expval(circuit, angles, wires=(0,), shots=None, rng=None):
    """Run ``circuit`` on ``angles`` and return ``[<Z_w> for w in wires]`` as Values.

    ``angles`` has one entry per angle slot of the circuit. Entries may be plain
    numbers or any ``Value`` (a trainable parameter, or ``w * x + b`` computed by
    classical nodes); the same ``Value`` may feed several slots. The backward pass
    computes the parameter-shift Jacobian once, lazily, and routes it into the
    angle nodes, so gradients continue through whatever classical graph produced
    them.
    """
    angles = [a if isinstance(a, Value) else Value(a) for a in angles]
    theta = np.array([a.data for a in angles], dtype=float)
    backward_rng = None
    if shots is not None:
        # The backward pass visits nodes in an order that depends on set iteration,
        # so each node gets its own generator, seeded here in forward order. This
        # keeps shot-based training reproducible for a fixed ``rng``.
        rng = np.random.default_rng() if rng is None else rng
        backward_rng = np.random.default_rng(rng.integers(2**63))
    z = circuit.expval(theta, wires, shots=shots, rng=rng)[0]

    cache = {}

    def jacobian():
        if "J" not in cache:
            cache["J"] = parameter_shift_jacobian(circuit, theta, wires, shots, backward_rng)
        return cache["J"]

    outs = []
    for k, w in enumerate(wires):
        out = Value(float(z[k]), tuple(angles), f"<Z{w}>")

        def _update_grad(k=k, out=out):
            row = jacobian()[k]
            for j, a in enumerate(angles):
                a.grad += row[j] * out.grad

        out._update_grad = _update_grad
        outs.append(out)
    return outs
