"""Split-MNIST: do local learning rules forget differently from backprop?

Settings are fixed by ``docs/preregistration.md`` and are identical for both rules.
Neither rule gets its own tuning pass.

Two protocols, both reported (see Amendment 1 in the pre-registration)::

    uv run python experiments/run_continual_comparison.py --protocol task
    uv run python experiments/run_continual_comparison.py --protocol domain

``task`` is the originally pre-registered setting and produces a floor effect — almost no
forgetting for either rule, so it cannot separate them. ``domain`` shares a single output head
across the five tasks, which makes them interfere and is where the comparison has signal.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import average_accuracy, average_forgetting, per_task_forgetting
from forgetlab.train import accuracy, train

RULES = ["backprop", "chl"]
SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
RESULTS = Path("results")


def build(
    seed: int, trunk: list[int], gamma: float, protocol: str,
    untied: bool = False, flat_scale: bool = False,
) -> list[LayeredNet]:
    """Return the net used for each task, under the chosen protocol.

    ``task``   — task-incremental: one private head per task, shared trunk. Task identity
                 is available at test time.
    ``domain`` — domain-incremental: a *single* net, so every task writes to the same
                 output units and they interfere directly. No task identity at test time.
    """
    if protocol == "task":
        return LayeredNet.multi_head(trunk, head_size=2, n_heads=5, gamma=gamma, seed=seed)
    if protocol == "domain":
        shared = LayeredNet(list(trunk) + [2], gamma=gamma, tied=not untied, seed=seed)
        if flat_scale:
            # Control arm for the untied comparison: the *tied* rule with the
            # gamma^(k-L) rescaling switched off.  Isolates the factor itself, so a
            # difference in the untied arm cannot be blamed on the missing factor.
            shared.lr_scale = lambda k: 1.0
        return [shared] * 5
    raise ValueError(f"unknown protocol {protocol!r}")


def accuracy_matrix(nets, tasks, rule, args, seed) -> list[list[float]]:
    """``R[i][j]`` = accuracy on task ``j`` after finishing training on task ``i``."""
    kwargs = {} if rule == "backprop" else SETTLE
    R = []
    for i, task in enumerate(tasks):
        train(
            nets[i],
            rule,
            task.x_train,
            task.y_train,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            rule_kwargs=kwargs,
            n_classes=2,
        )
        R.append([accuracy(nets[j], t.x_test, t.y_test) for j, t in enumerate(tasks)])
    return R


def joint_upper_bound(tasks, rule, args, seed) -> float:
    """Same architecture, tasks interleaved i.i.d. instead of sequentially.

    Not a competitor — a ceiling. It shows how much of any accuracy drop is caused by the
    *sequence*, as opposed to the capacity of the model or the difficulty of the tasks.
    """
    nets = build(seed, args.trunk, args.gamma, args.protocol, args.untied, args.flat_scale)
    kwargs = {} if rule == "backprop" else SETTLE
    gen = torch.Generator().manual_seed(seed)
    steps_per_task = (len(tasks[0].x_train) // args.batch_size) * args.epochs

    for _ in range(steps_per_task * len(tasks)):
        t = int(torch.randint(len(tasks), (1,), generator=gen))
        task = tasks[t]
        idx = torch.randperm(len(task.x_train), generator=gen)[: args.batch_size]
        train(
            nets[t],
            rule,
            task.x_train[idx],
            task.y_train[idx],
            lr=args.lr,
            epochs=1,
            batch_size=args.batch_size,
            seed=seed,
            rule_kwargs=kwargs,
            n_classes=2,
        )
    return sum(accuracy(nets[j], t.x_test, t.y_test) for j, t in enumerate(tasks)) / len(tasks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-per-task", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--trunk", type=int, nargs="+", default=[784, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--protocol", choices=["task", "domain"], default="domain")
    p.add_argument("--rules", nargs="+", choices=RULES, default=RULES,
                   help="which rules to run; backprop is gamma-independent, so a "
                        "gamma sweep only needs chl")
    p.add_argument("--skip-joint", action="store_true")
    p.add_argument("--untied", action="store_true",
                   help="untied-feedback CHL: fixed random feedback matrices instead of "
                        "W^T (exploratory; see docs/exploratory-untied.md)")
    p.add_argument("--flat-scale", action="store_true",
                   help="tied CHL with the gamma^(k-L) rescaling off -- the control arm "
                        "that isolates the factor itself")
    args = p.parse_args()
    if args.untied or args.flat_scale:
        if args.protocol != "domain" or list(args.rules) != ["chl"]:
            raise SystemExit("--untied/--flat-scale: domain protocol and --rules chl only")
    if args.untied and args.flat_scale:
        raise SystemExit("--untied and --flat-scale are separate arms; pick one")

    torch.set_default_dtype(torch.float32)
    RESULTS.mkdir(exist_ok=True)
    # The pre-registered run is gamma=0.1 and owns the plain filenames; anything else
    # is an exploratory sweep and must not overwrite it.
    suffix = args.protocol
    if args.gamma != 0.1:
        suffix += f"_gamma{args.gamma:g}"
    if list(args.rules) != RULES:
        suffix += "_" + "-".join(args.rules)
    if args.untied:
        suffix += "_untied"
    if args.flat_scale:
        suffix += "_flatscale"

    tasks = load_split_mnist(train_per_task=args.train_per_task)
    print(f"Split-MNIST, {args.protocol}-incremental. Settings from docs/preregistration.md.")
    print(f"tasks: {[t.name for t in tasks]}")
    print(f"train per task: {[len(t.x_train) for t in tasks]}")
    print(f"lr={args.lr} epochs={args.epochs} batch={args.batch_size} "
          f"gamma={args.gamma} trunk={args.trunk} seeds={args.seeds}\n")

    rows, matrices = [], {}
    for rule in args.rules:
        accs, forgets, per_task, joints = [], [], [], []
        for seed in args.seeds:
            start = time.time()
            R = accuracy_matrix(
                build(seed, args.trunk, args.gamma, args.protocol, args.untied, args.flat_scale),
                tasks, rule, args, seed,
            )
            matrices[(rule, seed)] = R
            accs.append(average_accuracy(R))
            forgets.append(average_forgetting(R))
            per_task.append(per_task_forgetting(R))
            if not args.skip_joint:
                joints.append(joint_upper_bound(tasks, rule, args, seed))
            print(f"  {rule:>8} seed {seed}: ACC {accs[-1]*100:5.2f}%  "
                  f"forgetting {forgets[-1]*100:5.2f}pp  ({time.time()-start:.0f}s)")

        rows.append({
            "rule": rule,
            "acc_mean": sum(accs) / len(accs),
            "acc_std": _std(accs),
            "forgetting_mean": sum(forgets) / len(forgets),
            "forgetting_std": _std(forgets),
            "joint_mean": (sum(joints) / len(joints)) if joints else float("nan"),
            "per_task_forgetting": [
                sum(c) / len(c) for c in zip(*per_task)
            ],
        })

    _write_csv(rows, suffix)
    _plot(matrices, args, suffix)
    _print_table(rows, tasks)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _write_csv(rows, suffix: str) -> None:
    path = RESULTS / f"results_{suffix}.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rule", "acc_mean", "acc_std", "forgetting_mean", "forgetting_std",
            "joint_mean", "per_task_forgetting",
        ])
        for r in rows:
            writer.writerow([
                r["rule"], f"{r['acc_mean']:.4f}", f"{r['acc_std']:.4f}",
                f"{r['forgetting_mean']:.4f}", f"{r['forgetting_std']:.4f}",
                f"{r['joint_mean']:.4f}",
                ";".join(f"{v:.4f}" for v in r["per_task_forgetting"]),
            ])
    print(f"\nwrote {path}")


def _plot(matrices, args, suffix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    stages = range(1, 6)

    # The whole point of the result is that the two curves land on top of each other, so
    # they must stay individually visible: a wide translucent line under a thinner dashed
    # one, rather than two opaque lines where only the last drawn is seen.
    style = {
        "backprop": dict(linestyle="-", marker="o", linewidth=5, alpha=0.35),
        "chl": dict(linestyle=":", marker="^", linewidth=2, markersize=6),
    }

    for rule in args.rules:
        seen, first = [], []
        for stage in range(5):
            per_seed_seen, per_seed_first = [], []
            for seed in args.seeds:
                R = matrices[(rule, seed)]
                per_seed_seen.append(sum(R[stage][: stage + 1]) / (stage + 1))
                per_seed_first.append(R[stage][0])
            seen.append(sum(per_seed_seen) / len(per_seed_seen))
            first.append(sum(per_seed_first) / len(per_seed_first))
        left.plot(stages, [v * 100 for v in seen], label=rule, **style[rule])
        right.plot(stages, [v * 100 for v in first], label=rule, **style[rule])

    left.set_xlabel("tasks trained")
    left.set_ylabel("mean accuracy over tasks seen (%)")
    left.set_title("Average accuracy")
    right.set_xlabel("tasks trained")
    right.set_ylabel("accuracy on task 1 (0v1) (%)")
    right.set_title("Retention of the first task")
    fig.suptitle(
        f"Split-MNIST, {args.protocol}-incremental — 3 seeds, identical settings",
        fontsize=12,
    )
    for ax in (left, right):
        ax.set_xticks(list(stages))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = RESULTS / f"forgetting_curves_{suffix}.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def _print_table(rows, tasks) -> None:
    print("\n" + "=" * 72)
    print(f"{'rule':>9} | {'ACC':>14} | {'forgetting':>15} | {'joint (ceiling)':>15}")
    print("-" * 72)
    for r in rows:
        print(f"{r['rule']:>9} | {r['acc_mean']*100:6.2f}% ±{r['acc_std']*100:4.2f} | "
              f"{r['forgetting_mean']*100:7.2f}pp ±{r['forgetting_std']*100:4.2f} | "
              f"{r['joint_mean']*100:13.2f}%")
    print("=" * 72)
    print("\nper-task forgetting (percentage points):")
    header = "  ".join(f"{t.name:>6}" for t in tasks[:-1])
    print(f"{'':>9}   {header}")
    for r in rows:
        cells = "  ".join(f"{v*100:6.2f}" for v in r["per_task_forgetting"])
        print(f"{r['rule']:>9}   {cells}")


if __name__ == "__main__":
    main()
