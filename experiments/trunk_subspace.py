"""Do the tasks reuse the same trunk directions, or each carve out new ones?

Fourth attempt at the residual. The three before it (damage_direction, trunk_interference,
and the head/trunk split) either failed or located the effect without explaining it, and
the effect itself survived 15 seeds, so there is something to find.

The hypothesis, written before running, and structural rather than guessed:

    Feedback into the hidden layer is (output state) @ (feedback matrix). The output layer
    is two units wide, so that contribution can only ever occupy a subspace of dimension
    at most 2 -- the row space of the feedback matrix.

    Untied: the feedback matrix is B, drawn once and never updated. That subspace is
    FIXED for the whole run, so every task pushes the trunk within the same two
    directions.

    Tied: the feedback matrix is W_1, which trains. The subspace ROTATES as learning
    proceeds, so each task can push the trunk somewhere the previous ones did not reach.

    Prediction: the untied arm's per-task trunk changes should share a subspace far more
    than the tied arm's. If they do, the mechanism for the residual is that untied tasks
    reuse the same ground while tied tasks keep breaking new ground.

Measured per rule:

    stable rank   ||A||_F^2 / ||A||_2^2 for each task's trunk change. Near 1 means the
                  change is essentially one direction; larger means it is spread out.
    overlap       mean |cos| between the leading directions of different tasks' changes.
                  Near 1 means every task pushed the same way; near 0 means each found
                  its own.

Run with::

    uv run python experiments/trunk_subspace.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.train import train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=2)
SEEDS = [0, 1, 2]


def per_task_changes(tasks, seed: int, untied: bool) -> list[torch.Tensor]:
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=seed)
    out = []
    for task in tasks:
        before = net.W[0].detach().clone()
        train(net, "chl", task.x_train, task.y_train, seed=seed,
              rule_kwargs=SETTLE, **SHARED)
        out.append(net.W[0].detach() - before)
    return out


def summarise(ds: list[torch.Tensor]) -> tuple[float, float]:
    ranks, leads = [], []
    for d in ds:
        u, s, _ = torch.linalg.svd(d.double(), full_matrices=False)
        ranks.append(float((s ** 2).sum() / s[0] ** 2))
        leads.append(u[:, 0])
    pairs = [
        abs(float(leads[i] @ leads[j]))
        for i in range(len(leads)) for j in range(i + 1, len(leads))
    ]
    return sum(ranks) / len(ranks), sum(pairs) / len(pairs)


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)
    print("Do the five tasks share trunk directions, or each take their own?")
    print(f"{len(SEEDS)} seeds, domain-incremental Split-MNIST\n")
    print(f"{'rule':>10} | {'stable rank':>12} | {'shared direction':>17}")
    print("-" * 46)
    for label, untied in [("tied", False), ("UNTIED", True)]:
        rows = [summarise(per_task_changes(tasks, s, untied)) for s in SEEDS]
        print(f"{label:>10} | {mean([r[0] for r in rows]):>12.2f} | "
              f"{mean([r[1] for r in rows]):>17.4f}")


if __name__ == "__main__":
    main()
