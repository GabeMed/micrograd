"""Command-line entry point: ``python -m neural_network {train,compare,graph,circuit}``."""

import argparse

from .experiments import MODELS, compare, draw_graphs, plot_decision_boundaries, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m neural_network")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train one model on make_moons")
    p.add_argument("model", choices=list(MODELS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--plot", help="save a decision-boundary PNG to this path")

    p = sub.add_parser("compare", help="train every model over several seeds, save plots/table")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--out", default="assets")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))

    p = sub.add_parser("graph", help="save computation-graph images")
    p.add_argument("--out", default="assets")

    sub.add_parser("circuit", help="print the variational circuit")

    args = parser.parse_args(argv)
    if args.command == "train":
        model, _, result, (X, y) = run(
            args.model, seed=args.seed, steps=args.steps, lr=args.lr, log_every=20
        )
        print(model)
        print({k: result[k] for k in ("params", "train_acc", "test_acc", "train_time_s")})
        if args.plot:
            plot_decision_boundaries([(MODELS[args.model][0], model, X, y)], args.plot)
    elif args.command == "compare":
        compare(seeds=args.seeds, steps=args.steps, out=args.out, kinds=args.models)
    elif args.command == "graph":
        print(f"saved graphs to {draw_graphs(args.out)}/")
    elif args.command == "circuit":
        from .quantum import QuantumNeuron

        print(QuantumNeuron(n_layers=3).draw())


if __name__ == "__main__":
    main()
