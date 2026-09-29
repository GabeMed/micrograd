"""Max-margin (SVM hinge) loss with L2 regularisation, shared by every model."""

import numpy as np

from .value import Value


def loss(X, y, model, batch_size=None, alpha=1e-4, rng=None):
    """Return ``(total_loss, accuracy)`` of ``model`` on ``(X, y)``.

    ``y`` holds labels in {-1, +1} and ``model(x)`` returns a scalar score ``Value``.
    ``alpha`` weights the L2 penalty on ``model.parameters()`` (0 disables it).
    """
    # By default the whole dataset is used, but a random mini-batch can be drawn
    if batch_size is None:
        Xb, yb = X, y
    else:
        rng = np.random.default_rng() if rng is None else rng
        ri = rng.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # Forward pass to get the scores
    scores = list(map(model, inputs))

    # SVM max-margin (hinge) loss
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))

    # L2 regularisation
    reg_loss = alpha * sum(p * p for p in model.parameters()) if alpha else 0.0
    total_loss = data_loss + reg_loss

    # Accuracy, as a diagnostic
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


def accuracy(X, y, model):
    """Fraction of points where ``sign(model(x))`` matches the label in {-1, +1}."""
    correct = [(yi > 0) == (model(list(xrow)).data > 0) for xrow, yi in zip(X, y)]
    return sum(correct) / len(correct)
