"""Reproduce the README results: train each model on make_moons, save plots and a table.

Run ``python -m neural_network --help`` for the command-line interface.
"""

import json
import platform
from pathlib import Path

import numpy as np

from .loss import accuracy
from .network import NN
from .quantum import QuantumNeuron
from .train import train
from .value import Value

# name, constructor(seed), L2 weight. Classical and quantum models share the loss,
# optimiser and schedule; L2 is off for the circuits because rotation angles are
# periodic, so shrinking them towards 0 has no meaning.
MODELS = {
    "mlp": ("MLP 2-16-16-1", lambda seed: NN(2, [16, 16, 1]), 1e-4),
    "mlp-small": ("MLP 2-6-1", lambda seed: NN(2, [6, 1]), 1e-4),
    "vqc": ("VQC 2 qubits x 3 layers, exact", lambda seed: QuantumNeuron(n_layers=3), 0.0),
    "vqc-shots": (
        "VQC 2 qubits x 3 layers, 1000 shots",
        lambda seed: QuantumNeuron(n_layers=3, shots=1000, rng=np.random.default_rng(seed)),
        0.0,
    ),
}

# Palette (light surface): categorical slots in fixed order, recessive chrome
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"mlp": "#2a78d6", "mlp-small": "#eb6834", "vqc": "#1baf7a", "vqc-shots": "#eda100"}
CLASS_COLORS = ("#2a78d6", "#eb6834")  # label -1, label +1


def make_data(seed, n_train=100, n_test=500, noise=0.1):
    """make_moons train/test sets, standardised with the train statistics, labels in {-1, +1}."""
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n_train, noise=noise, random_state=seed)
    Xt, yt = make_moons(n_samples=n_test, noise=noise, random_state=seed + 1000)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sd, y * 2 - 1, (Xt - mu) / sd, yt * 2 - 1


def run(kind, seed=0, steps=200, lr=1.0, eval_every=10, log_every=0):
    """Train one model on one seed and return the model and its metrics."""
    name, build, alpha = MODELS[kind]
    X, y, X_test, y_test = make_data(seed)
    np.random.seed(seed)  # parameter initialisation uses NumPy's global RNG
    model = build(seed)
    history = train(
        model,
        X,
        y,
        steps=steps,
        lr=lr,
        alpha=alpha,
        X_test=X_test,
        y_test=y_test,
        eval_every=eval_every,
        log_every=log_every,
    )
    result = {
        "kind": kind,
        "name": name,
        "seed": seed,
        "params": len(model.parameters()),
        "steps": steps,
        "train_acc": accuracy(X, y, model),
        "test_acc": accuracy(X_test, y_test, model),
        "final_loss": history["loss"][-1],
        "train_time_s": history["train_time_s"],
        "circuit_runs_per_step": (
            len(X) * (1 + 2 * model.circuit.n_slots) if isinstance(model, QuantumNeuron) else 0
        ),
    }
    return model, history, result, (X, y)


def summarize(results):
    """Mean and standard deviation over seeds, per model."""
    rows = []
    for kind in dict.fromkeys(r["kind"] for r in results):
        rs = [r for r in results if r["kind"] == kind]
        row = {
            "kind": kind,
            "name": rs[0]["name"],
            "params": rs[0]["params"],
            "seeds": len(rs),
            "steps": rs[0]["steps"],
            "circuit_runs_per_step": rs[0]["circuit_runs_per_step"],
        }
        for key in ("train_acc", "test_acc", "train_time_s"):
            vals = np.array([r[key] for r in rs])
            row[key] = [float(vals.mean()), float(vals.std())]
        rows.append(row)
    return rows


def markdown_table(summary):
    lines = [
        "| Model | Params | Train acc | Test acc | Train time (s) | Circuit runs / step |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        (tr, trs), (te, tes), (t, ts) = r["train_acc"], r["test_acc"], r["train_time_s"]
        runs = f"{r['circuit_runs_per_step']:,}" if r["circuit_runs_per_step"] else "-"
        lines.append(
            f"| {r['name']} | {r['params']} | {tr:.1%} ± {trs:.1%} | {te:.1%} ± {tes:.1%} "
            f"| {t:.1f} ± {ts:.1f} | {runs} |"
        )
    return "\n".join(lines)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_decision_boundaries(panels, path, resolution=70):
    """``panels``: list of (title, model, X, y). One decision-region plot per model."""
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    fig = Figure(figsize=(3.3 * len(panels), 3.5), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    tint = ListedColormap(["#dbe8f8", "#fbe1d6"])  # light tints of the two class colours
    for ax, (title, model, X, y) in zip(axes, panels):
        x0 = np.linspace(X[:, 0].min() - 0.6, X[:, 0].max() + 0.6, resolution)
        x1 = np.linspace(X[:, 1].min() - 0.6, X[:, 1].max() + 0.6, resolution)
        g0, g1 = np.meshgrid(x0, x1)
        scores = np.array([model([a, b]).data for a, b in zip(g0.ravel(), g1.ravel())])
        scores = scores.reshape(g0.shape)
        _style(ax)
        ax.grid(False)
        ax.contourf(g0, g1, scores > 0, levels=[-0.5, 0.5, 1.5], cmap=tint)
        ax.contour(g0, g1, scores, levels=[0], colors=INK_2, linewidths=1.0)
        for label, color in zip((-1, 1), CLASS_COLORS):
            m = y == label
            ax.scatter(
                X[m, 0],
                X[m, 1],
                s=22,
                c=color,
                edgecolors=SURFACE,
                linewidths=0.8,
                label=f"y = {label:+d}",
                zorder=3,
            )
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("x1 (standardised)", fontsize=8, color=INK_2)
        ax.set_ylabel("x2 (standardised)", fontsize=8, color=INK_2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    return path


def plot_training_curves(histories, path):
    """``histories``: dict kind -> history for one seed. Loss and test accuracy vs step."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.0, 3.3), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    ax_loss, ax_acc = fig.subplots(1, 2)
    for kind, h in histories.items():
        style = "--" if kind == "vqc-shots" else "-"
        name = MODELS[kind][0]
        steps = np.arange(1, len(h["loss"]) + 1)
        ax_loss.plot(steps, h["loss"], style, color=SERIES[kind], lw=1.5, label=name)
        ax_acc.plot(h["test_step"], h["test_acc"], style, color=SERIES[kind], lw=1.5, label=name)
    for ax, title, ylabel in (
        (ax_loss, "Training loss (hinge + L2)", "loss"),
        (ax_acc, "Test accuracy (500 held-out points)", "accuracy"),
    ):
        _style(ax)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("SGD step", fontsize=8, color=INK_2)
        ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
    # The hinge loss reaches exactly 0 once every margin is met, so a log axis cannot
    # show it; symlog is linear below 1e-3 and logarithmic above.
    ax_loss.set_yscale("symlog", linthresh=1e-3)
    ax_loss.set_ylim(bottom=0)
    ax_acc.set_ylim(0.85, 1.005)
    ax_acc.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_acc.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    return path


def compare(seeds=5, steps=200, out="assets", kinds=tuple(MODELS)):
    """Train every model on every seed; write results.json, results.md and the plots."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results, first = [], {}
    for kind in kinds:
        for seed in range(seeds):
            model, history, result, data = run(kind, seed=seed, steps=steps)
            results.append(result)
            print(
                f"{result['name']:<38} seed {seed}  train {result['train_acc']:.1%}  "
                f"test {result['test_acc']:.1%}  {result['train_time_s']:.1f}s",
                flush=True,
            )
            if seed == 0:
                first[kind] = (model, history, data)

    summary = summarize(results)
    table = markdown_table(summary)
    meta = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    curves = {
        k: {key: v[1][key] for key in ("loss", "test_step", "test_acc")} for k, v in first.items()
    }
    (out / "results.json").write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "runs": results, "seed0_curves": curves}, indent=2
        )
        + "\n"
    )
    (out / "results.md").write_text(table + "\n")
    print("\n" + table)

    panels = []
    for kind in ("mlp", "mlp-small", "vqc"):
        if kind in first:
            model, _, (X, y) = first[kind]
            test = next(r for r in results if r["kind"] == kind and r["seed"] == 0)["test_acc"]
            panels.append(
                (
                    f"{MODELS[kind][0]}\n{len(model.parameters())} params, test acc {test:.1%}",
                    model,
                    X,
                    y,
                )
            )
    if panels:
        plot_decision_boundaries(panels, out / "decision_boundaries.png")
    plot_training_curves({k: v[1] for k, v in first.items()}, out / "training_curves.png")
    return summary


def draw_graphs(out="assets"):
    """Save the computation graph of one ReLU neuron and of a one-qubit hybrid model."""
    from .network import Neuron
    from .quantum.qnode import expval
    from .visualize import draw_matplotlib

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # A single neuron from network.py: relu(w1*x1 + w2*x2 + b)
    n = Neuron(2)
    n.b.data = 0.5
    for i, (w, value) in enumerate(zip(n.w, (0.9, -0.4)), 1):
        w.data, w.label = value, f"w{i}"
    n.b.label = "b"
    x = [Value(1.0, label="x1"), Value(-2.0, label="x2")]
    y = n(x)
    y.label = "out"
    y.backward_prop()
    draw_matplotlib(y, out / "computation_graph.png", title="One ReLU neuron after backward_prop()")

    # One qubit, one layer: a * <Z>(RY(w*x + b), RZ(θ0), RY(θ1)) + c
    np.random.seed(0)
    q = QuantumNeuron(n_inputs=1, n_qubits=1, n_layers=1)
    q.w[0].label, q.b[0].label, q.a.label, q.c.label = "w", "b", "a", "c"
    q.theta[0].label, q.theta[1].label = "θ0 (RZ)", "θ1 (RY)"
    angles = q.angles([Value(0.8, label="x")])
    angles[0].label = "w·x + b (RY)"
    z = expval(q.circuit, angles)[0]
    z.label = "<Z0>"
    score = q.a * z + q.c
    score.label = "score"
    score.backward_prop()
    draw_matplotlib(
        score,
        out / "hybrid_graph.png",
        highlight_ops=("<Z",),
        title="Hybrid graph: the shaded op is the circuit, differentiated by parameter shift",
    )
    return out
