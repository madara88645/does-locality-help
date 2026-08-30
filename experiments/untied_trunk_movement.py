"""Trap-2 guard for the untied comparison: how much does the shared trunk actually move?

The flat-scale control's hidden updates are ~gamma x smaller than tied CHL's, so if it
shows less forgetting, the boring explanation is an under-trained trunk (less learning ->
less interference), not credit assignment.  This measures trunk movement over the full
five-task sequence, per arm, seed 0 -- the number the pre-registration says must be
checked before any forgetting difference is interpreted.

Run with::

    uv run python experiments/untied_trunk_movement.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.train import train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, seed=0, n_classes=2)


def trunk_movement(untied: bool, flat_scale: bool) -> tuple[float, float]:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=0)
    if flat_scale:
        net.lr_scale = lambda k: 1.0
    w0 = net.W[0].detach().clone()
    for task in tasks:
        train(net, "chl", task.x_train, task.y_train, rule_kwargs=SETTLE, **SHARED)
    moved = float((net.W[0].detach() - w0).norm() / w0.norm()) * 100
    final = float(net.W[0].detach().norm())
    return moved, final


if __name__ == "__main__":
    print(f"{'arm':>24} | {'trunk moved over 5 tasks':>25} | {'final ||W0||':>12}")
    print("-" * 70)
    for label, untied, flat in [
        ("tied (frozen reference)", False, False),
        ("flat-scale control", False, True),
        ("untied", True, False),
    ]:
        moved, final = trunk_movement(untied, flat)
        print(f"{label:>24} | {moved:>24.2f}% | {final:>12.2f}")
