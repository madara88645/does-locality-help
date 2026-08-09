"""Why is per-task forgetting so uneven — 89 points for 4v5, 16 points for 6v7?

The average-forgetting number hides the answer. This script trains the domain-incremental
sequence and then asks, for every original digit, which class the final network assigns it.

The result is that the network does not retain a weakened version of each old task. It has
collapsed onto the *last* task's decision rule, and an old task looks "remembered" only when
its own label assignment happens to agree with that rule.

Run with::

    uv run python experiments/analyse_forgetting.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import TASK_PAIRS, load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.train import accuracy, train


def main() -> None:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)

    net = LayeredNet([784, 256, 2], gamma=0.1, seed=0)
    for task in tasks:
        train(
            net, "backprop", task.x_train, task.y_train,
            lr=0.05, epochs=10, batch_size=32, seed=0, n_classes=2,
        )

    print("After the full sequence, which class does the final network give each digit?")
    print(f"last task trained: {tasks[-1].name}\n")
    print(f"{'task':>6} {'digit':>6} {'true label':>11} {'assigned':>9} {'agrees':>8} {'recall':>8}")
    print("-" * 56)

    for i, (first, second) in enumerate(TASK_PAIRS):
        task = tasks[i]
        with torch.no_grad():
            predicted = net.feedforward(task.x_test)[-1].argmax(dim=1)
        for label, digit in ((0, first), (1, second)):
            mask = task.y_test == label
            assigned = int((predicted[mask] == 1).float().mean().round())
            recall = float((predicted[mask] == label).float().mean())
            print(f"{task.name:>6} {digit:>6} {label:>11} {assigned:>9} "
                  f"{'yes' if assigned == label else 'NO':>8} {recall * 100:7.1f}%")
        print(f"{'':>6} {'':>6} {'':>11} {'':>9} {'task:':>8} "
              f"{accuracy(net, task.x_test, task.y_test) * 100:7.1f}%")

    print(
        "\n'agrees' is whether the class the final network assigns to that digit happens to\n"
        "match the label that digit had in its own task. Tasks where both digits agree keep\n"
        "a high score; tasks where both disagree fall below chance. That is the whole of the\n"
        "per-task spread — not some tasks being more robust than others."
    )


if __name__ == "__main__":
    main()
