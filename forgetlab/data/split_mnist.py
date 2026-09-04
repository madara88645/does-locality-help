"""Split-MNIST: five sequential binary tasks — 0v1, 2v3, 4v5, 6v7, 8v9.

**Task-incremental** protocol: the task identity is given at test time and each task has its
own output head. This is the easiest of van de Ven & Tolias's three continual-learning
settings, and it is chosen deliberately. PC and CHL are documented to degrade with depth
and difficulty, so the experiment stays where the field says these rules actually work —
otherwise a null result would be uninformative, since it could just mean the rules broke.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from forgetlab.data.mnist import load_mnist

TASK_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


@dataclass
class Task:
    """One binary task. Labels are remapped to 0/1 for that task's own head."""

    index: int
    digits: tuple[int, int]
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def name(self) -> str:
        return f"{self.digits[0]}v{self.digits[1]}"


def _select(x, y, digits, limit, seed, binary_labels=True):
    mask = (y == digits[0]) | (y == digits[1])
    xs, ys = x[mask], y[mask]
    # Binary: each task remaps its two digits to 0/1 for its own head, which is what the
    # task- and domain-incremental protocols need.  Class-incremental instead keeps the
    # true digit, so all ten output units mean the same thing for every task.
    ys = (ys == digits[1]).long() if binary_labels else ys.long()
    if limit is not None and limit < len(xs):
        idx = torch.randperm(len(xs), generator=torch.Generator().manual_seed(seed))[:limit]
        xs, ys = xs[idx], ys[idx]
    return xs, ys


def load_split_mnist(
    root: str = "data",
    train_per_task: int | None = 1000,
    test_per_task: int | None = None,
    seed: int = 0,
    binary_labels: bool = True,
) -> list[Task]:
    """Build the five tasks. ``train_per_task`` caps the training set of each task.

    ``binary_labels=False`` keeps the true digit as the label, for the class-incremental
    protocol where one ten-unit head is shared and every unit means the same digit to
    every task.
    """
    x_train, y_train, x_test, y_test = load_mnist(root)
    tasks = []
    for i, digits in enumerate(TASK_PAIRS):
        xtr, ytr = _select(x_train, y_train, digits, train_per_task, seed + i, binary_labels)
        xte, yte = _select(x_test, y_test, digits, test_per_task, seed + 100 + i, binary_labels)
        tasks.append(Task(i, digits, xtr, ytr, xte, yte))
    return tasks
