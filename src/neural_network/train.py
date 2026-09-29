"""Stochastic gradient descent with linear learning-rate decay."""

import time

from .loss import accuracy, loss


def train(
    model,
    X,
    y,
    steps=100,
    lr=1.0,
    lr_decay=0.9,
    alpha=1e-4,
    batch_size=None,
    X_test=None,
    y_test=None,
    eval_every=1,
    log_every=0,
    rng=None,
):
    """Train ``model`` in place with SGD on the hinge loss.

    The learning rate decays linearly from ``lr`` to ``lr * (1 - lr_decay)``.
    Returns a history dict with the per-step loss and train accuracy (measured on
    the forward pass of that step), the test accuracy every ``eval_every`` steps
    if a test set is given, and the wall-clock training time in seconds (test-set
    evaluation is excluded from the timing).
    """
    history = {"loss": [], "train_acc": [], "test_step": [], "test_acc": [], "train_time_s": 0.0}
    elapsed = 0.0
    for k in range(steps):
        t0 = time.perf_counter()

        # Forward
        total_loss, acc = loss(X, y, model, batch_size=batch_size, alpha=alpha, rng=rng)

        # Backward
        model.zero_grad()
        total_loss.backward_prop()

        # Update
        step_lr = lr * (1.0 - lr_decay * k / steps)
        for p in model.parameters():
            p.data -= step_lr * p.grad

        elapsed += time.perf_counter() - t0

        history["loss"].append(total_loss.data)
        history["train_acc"].append(acc)
        if X_test is not None and ((k + 1) % eval_every == 0 or k == steps - 1):
            history["test_step"].append(k + 1)
            history["test_acc"].append(accuracy(X_test, y_test, model))
        if log_every and (k % log_every == 0 or k == steps - 1):
            msg = f"step {k:4d}  loss {total_loss.data:.4f}  train acc {acc:.1%}"
            if history["test_acc"]:
                msg += f"  test acc {history['test_acc'][-1]:.1%}"
            print(msg)

    history["train_time_s"] = elapsed
    return history
