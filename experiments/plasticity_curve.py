"""Does untied CHL forget less than its plasticity alone explains?

Design and prediction are in docs/exploratory-plasticity.md, committed before this ran.

Throttling the tied rule's hidden-layer update traces how much forgetting a
backprop-like rule sheds per unit of trunk movement given up. The untied arm is then
placed on the same axes: on the curve means plasticity explains it, below the curve
means something in the rule survives the confound.

Every point reports attainment too, because a throttled arm that stops learning would
show low forgetting for a trivial reason and bend the curve (trap 1).

Run with::

    uv run python experiments/plasticity_curve.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import average_accuracy, average_forgetting
from forgetlab.train import accuracy, train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=2)
SEEDS = [0, 1, 2, 3, 4]
# 0.05 and 0.02 were added after the first pass: the untied arm's trunk movement
# (2.28%) fell below the original lowest point, which would have forced the
# comparison to extrapolate. These bracket it instead.
THROTTLES = [1.0, 0.6, 0.35, 0.2, 0.1, 0.05, 0.02]


def throttled_scale(net: LayeredNet, s: float):
    """gamma^(k-L) * s below the output, 1.0 at the output."""
    g, L = net.gamma, net.L
    return lambda k: (g ** (k - L)) * s if k < L else 1.0


def run(tasks, seed: int, untied: bool, throttle: float | None):
    """One sequence. Returns (forgetting, ACC, attainment, trunk movement), all in %."""
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=seed)
    if throttle is not None:
        net.lr_scale = throttled_scale(net, throttle)
    w0 = net.W[0].detach().clone()

    R, attained = [], []
    for i, task in enumerate(tasks):
        train(net, "chl", task.x_train, task.y_train, seed=seed, rule_kwargs=SETTLE, **SHARED)
        row = [accuracy(net, t.x_test, t.y_test) for t in tasks]
        attained.append(row[i])
        R.append(row)

    moved = float((net.W[0].detach() - w0).norm() / w0.norm()) * 100
    # attainment over the tasks that can be forgotten, matching the forgetting metric
    att = sum(attained[:-1]) / (len(attained) - 1) * 100
    return average_forgetting(R) * 100, average_accuracy(R) * 100, att, moved


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def report(label: str, rows):
    forg = [r[0] for r in rows]
    print(f"{label:>28} | {mean([r[3] for r in rows]):>7.2f}% | "
          f"{mean(forg):>6.2f} ±{std(forg):>4.2f} | "
          f"{mean([r[1] for r in rows]):>7.2f}% | {mean([r[2] for r in rows]):>7.2f}%")


def main() -> None:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)
    print(f"domain-incremental Split-MNIST, {len(SEEDS)} seeds "
          f"(docs/exploratory-plasticity.md)\n")
    print(f"{'arm':>28} | {'trunk':>8} | {'forgetting':>12} | {'ACC':>8} | {'attain':>8}")
    print("-" * 78)

    print("  --- the curve: tied CHL, hidden update throttled ---")
    for s in THROTTLES:
        rows = [run(tasks, seed, untied=False, throttle=s) for seed in SEEDS]
        report(f"tied, s = {s}", rows)

    print("  --- placed against the curve ---")
    rows = [run(tasks, seed, untied=True, throttle=None) for seed in SEEDS]
    report("UNTIED (the rule)", rows)

    # s = 0.1 IS the flat-scale control: gamma^(k-L) * 0.1 = 10 * 0.1 = 1.0 at the hidden
    # layer, i.e. no per-layer rescaling. It is a point on the curve, not a separate arm.
    print("\n  note: the s = 0.1 row above is the flat-scale control from")
    print("  docs/exploratory-untied.md -- the throttle that cancels the factor exactly.")


if __name__ == "__main__":
    main()
