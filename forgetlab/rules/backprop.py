"""Backpropagation baseline — the ground truth every other rule is measured against.

Uses PyTorch autograd on purpose.  This is the only rule in the project allowed to,
because it is the reference, not the object of study.
"""

from __future__ import annotations

import torch

from forgetlab.layers import LayeredNet


def squared_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Half squared error, averaged over the batch.

    This is the cost Xie & Seung's equivalence is stated against, given *linear* output
    units.  With sigmoid outputs the equivalence is to cross-entropy instead (Eq. 3.20).
    """
    return 0.5 * ((output - target) ** 2).sum(dim=1).mean()


def backprop_updates(
    net: LayeredNet,
    x0: torch.Tensor,
    target: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(dW, db)`` — the weight changes to **add**, i.e. the descent direction.

    These are ``-dLoss/dW``, so they are directly comparable to what ``chl_updates``
    returns.  Multiply by a learning rate and add.
    """
    for p in list(net.W) + list(net.b):
        if p.grad is not None:
            p.grad = None

    output = net.feedforward(x0)[-1]
    loss = squared_error(output, target)
    loss.backward()

    dW = [(-w.grad).detach().clone() for w in net.W]
    db = [(-b.grad).detach().clone() for b in net.b]

    for p in list(net.W) + list(net.b):
        p.grad = None
    return dW, db
