"""Does the untied effect survive a wider output head?

Design and prediction: docs/exploratory-class-incremental.md, committed before this ran.

Every result in this project so far was measured inside a two-unit output head that
accounts for roughly 86% of all its forgetting. This widens the head to ten units, one per
digit, with true digit labels and ten-way evaluation, and changes nothing else. If the
untied rule's advantage survives, it is a phenomenon; if it disappears, it was a property
of the bottleneck.

Trunk movement and per-task attainment are recorded for every arm: the first because the
plasticity confound follows us here, the second because class-incremental is the hardest
of the three settings and an arm that cannot fit its tasks would show low forgetting for
the trivial reason.

Run with::

    uv run python experiments/class_incremental.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import average_accuracy, average_forgetting
from forgetlab.train import accuracy, train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=10)
SEEDS = [0, 1, 2, 3, 4]


def run(tasks, seed: int, rule: str, untied: bool, throttle: float | None):
    net = LayeredNet([784, 256, 10], gamma=0.1, tied=not untied, seed=seed)
    if throttle is not None:
        g, L = net.gamma, net.L
        net.lr_scale = lambda k: (g ** (k - L)) * throttle if k < L else 1.0
    w0 = net.W[0].detach().clone()

    kwargs = {} if rule == "backprop" else SETTLE
    R, attained = [], []
    for i, task in enumerate(tasks):
        train(net, rule, task.x_train, task.y_train, seed=seed, rule_kwargs=kwargs, **SHARED)
        row = [accuracy(net, t.x_test, t.y_test) for t in tasks]
        attained.append(row[i])
        R.append(row)

    moved = float((net.W[0].detach() - w0).norm() / w0.norm()) * 100
    return (average_forgetting(R) * 100, average_accuracy(R) * 100,
            sum(attained[:-1]) / (len(attained) - 1) * 100, moved)


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main() -> None:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000, binary_labels=False)
    print("Class-incremental Split-MNIST: ten output units, ten-way evaluation.")
    print(f"{len(SEEDS)} seeds. Chance is 10%, not 50%.\n")
    print(f"{'arm':>26} | {'forgetting':>13} | {'final ACC':>10} | "
          f"{'attainment':>11} | {'trunk':>7}")
    print("-" * 82)
    for label, rule, untied, throttle in [
        ("backprop", "backprop", False, None),
        ("tied CHL", "chl", False, None),
        ("tied CHL, throttled 0.05", "chl", False, 0.05),
        ("UNTIED CHL", "chl", True, None),
    ]:
        rows = [run(tasks, s, rule, untied, throttle) for s in SEEDS]
        f = [r[0] for r in rows]
        print(f"{label:>26} | {mean(f):>7.2f} ±{std(f):>4.2f} | "
              f"{mean([r[1] for r in rows]):>9.2f}% | {mean([r[2] for r in rows]):>10.2f}% | "
              f"{mean([r[3] for r in rows]):>6.2f}%")


if __name__ == "__main__":
    main()
