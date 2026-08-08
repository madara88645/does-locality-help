"""Plain SGD on top of whichever learning rule is selected.

Every rule returns ``(dW, db)`` in the same sign convention — the change to **add** — so
the training loop below is identical for backprop, PC and CHL. That is deliberate: any
difference in the results has to come from the rule, not from the optimiser around it.
"""

from __future__ import annotations

import torch

from forgetlab.data import one_hot
from forgetlab.layers import LayeredNet
from forgetlab.rules import RULES


@torch.no_grad()
def accuracy(net: LayeredNet, x: torch.Tensor, y: torch.Tensor, chunk: int = 2048) -> float:
    """Top-1 accuracy from the plain feedforward pass.

    Inference is always feedforward, for every rule. The relaxation is part of *learning*,
    not of prediction.
    """
    correct = 0
    for start in range(0, len(x), chunk):
        logits = net.feedforward(x[start : start + chunk])[-1]
        correct += int((logits.argmax(dim=1) == y[start : start + chunk]).sum())
    return correct / len(x)


def train(
    net: LayeredNet,
    rule: str,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.05,
    epochs: int = 5,
    batch_size: int = 64,
    seed: int = 0,
    rule_kwargs: dict | None = None,
    n_classes: int = 10,
    log_every: int | None = None,
) -> list[float]:
    """Train in place. Returns the mean squared-error loss per epoch."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}, expected one of {sorted(RULES)}")
    update_fn = RULES[rule]
    rule_kwargs = dict(rule_kwargs or {})
    gen = torch.Generator().manual_seed(seed)

    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        total, seen = 0.0, 0
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            xb, yb = x[idx], one_hot(y[idx], n_classes)

            dW, db = update_fn(net, xb, yb, **rule_kwargs)
            with torch.no_grad():
                for w, d in zip(net.W, dW):
                    w += lr * d
                for b, d in zip(net.b, db):
                    b += lr * d

            with torch.no_grad():
                out = net.feedforward(xb)[-1]
                total += float(0.5 * ((out - yb) ** 2).sum(dim=1).sum())
                seen += len(idx)

        history.append(total / seen)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch + 1:>2}  loss {history[-1]:.4f}")
    return history
