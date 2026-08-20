"""MNIST, flattened and subsampled.

Kept small on purpose. The expensive part of PC and CHL is the inference loop run for
every batch, not the model size, so the practical lever on runtime is the number of
examples, not the width of the network.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms

DEFAULT_ROOT = Path("data")


def _subsample(x: torch.Tensor, y: torch.Tensor, n: int | None, seed: int):
    if n is None or n >= len(x):
        return x, y
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=gen)[:n]
    return x[idx], y[idx]


def load_mnist(
    root: str | Path = DEFAULT_ROOT,
    train_size: int | None = None,
    test_size: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x_train, y_train, x_test, y_test)`` with images flattened to 784 floats.

    Pixels are scaled to [0, 1] and left unnormalised — with a tanh hidden layer and a
    linear output that is enough, and it keeps one fewer arbitrary constant in the setup.
    """
    root = Path(root)
    to_tensor = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=to_tensor)
    test = datasets.MNIST(root, train=False, download=True, transform=to_tensor)

    x_train = train.data.reshape(len(train), -1).to(torch.get_default_dtype()) / 255.0
    x_test = test.data.reshape(len(test), -1).to(torch.get_default_dtype()) / 255.0

    x_train, y_train = _subsample(x_train, train.targets, train_size, seed)
    x_test, y_test = _subsample(x_test, test.targets, test_size, seed + 1)
    return x_train, y_train, x_test, y_test


def one_hot(y: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    """One-hot targets for the squared-error objective the equivalence is stated against."""
    out = torch.zeros(len(y), n_classes, dtype=torch.get_default_dtype())
    out[torch.arange(len(y)), y] = 1.0
    return out
