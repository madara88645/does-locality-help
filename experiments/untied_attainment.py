"""Trap-1 guard: did each arm actually LEARN each task before forgetting it?

Forgetting is peak-minus-final, so an arm that attains a lower peak shows less
forgetting for a mechanical reason.  This recomputes the accuracy matrix for all three
CHL arms with the frozen settings and reports attainment R[j][j] next to forgetting.
Tied CHL reproduces the frozen run exactly (same seeds, deterministic).

Run with::

    uv run python experiments/untied_attainment.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.train import accuracy, train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=2)
SEEDS = [0, 1, 2]


def matrix(untied: bool, flat_scale: bool, seed: int) -> list[list[float]]:
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=seed)
    if flat_scale:
        net.lr_scale = lambda k: 1.0
    R = []
    for task in TASKS:
        train(net, "chl", task.x_train, task.y_train, seed=seed, rule_kwargs=SETTLE, **SHARED)
        R.append([accuracy(net, t.x_test, t.y_test) for t in TASKS])
    return R


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    TASKS = load_split_mnist(train_per_task=1000)
    print(f"{'arm':>12} | {'attainment R[j][j], mean first-4':>32} | {'per task':>34}")
    print("-" * 90)
    for label, untied, flat in [("tied", False, False), ("flat-scale", False, True), ("untied", True, False)]:
        per_seed = []
        per_task = [0.0] * len(TASKS)
        for seed in SEEDS:
            R = matrix(untied, flat, seed)
            att = [R[j][j] for j in range(len(TASKS))]
            per_seed.append(sum(att[:-1]) / (len(att) - 1))
            for j, a in enumerate(att):
                per_task[j] += a / len(SEEDS)
        mean = sum(per_seed) / len(per_seed)
        cells = "  ".join(f"{v*100:5.1f}" for v in per_task)
        print(f"{label:>12} | {mean*100:>31.2f}% | {cells:>34}")
