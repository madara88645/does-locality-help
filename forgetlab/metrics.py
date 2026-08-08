"""Metrics for comparing a local learning rule's update against backprop's.

Cosine similarity is reported **per layer**, never as a single aggregate number.
*What Accuracy and Gradient Cosine Miss* (arXiv:2606.21126) shows that an aggregate
cosine suffers "aggregation collapse": it masks layerwise heterogeneity when credit
concentrates at one end of the network, and gave no signal of failure in any case the
authors audited.  Since CHL's entire depth story is the ``gamma^(k-L)`` factor, an
averaged cosine would hide exactly the effect this project exists to measure.
"""

from __future__ import annotations

import torch


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors, flattened."""
    av, bv = a.flatten(), b.flatten()
    denom = av.norm() * bv.norm()
    if denom == 0:
        return float("nan")
    return float((av @ bv) / denom)


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error ``||a - b|| / ||b||``, with ``b`` the reference."""
    ref = b.norm()
    if ref == 0:
        return float("nan")
    return float((a - b).norm() / ref)


def per_layer_cosine(
    updates: list[torch.Tensor], reference: list[torch.Tensor]
) -> list[float]:
    """Cosine similarity to the reference update, one number per layer."""
    if len(updates) != len(reference):
        raise ValueError("layer count mismatch")
    return [cosine(u, r) for u, r in zip(updates, reference)]


def per_layer_relative_error(
    updates: list[torch.Tensor], reference: list[torch.Tensor]
) -> list[float]:
    """Relative L2 error against the reference update, one number per layer."""
    if len(updates) != len(reference):
        raise ValueError("layer count mismatch")
    return [relative_error(u, r) for u, r in zip(updates, reference)]


# ------------------------------------------------------- continual learning metrics
#
# The standard pair from Lopez-Paz & Ranzato's GEM paper, computed from the accuracy
# matrix ``R``, where ``R[i][j]`` is the accuracy on task ``j`` after finishing training
# on task ``i``.  Only the lower triangle plus the diagonal is meaningful here, since we
# never evaluate a task before its head has been trained.


def average_accuracy(R: list[list[float]]) -> float:
    """Mean accuracy over all tasks once the whole sequence has been trained (ACC)."""
    return sum(R[-1]) / len(R[-1])


def average_forgetting(R: list[list[float]]) -> float:
    """Mean drop from each task's own peak to its final accuracy.

    Positive means forgetting.  The last task is excluded — it has had no opportunity to
    be forgotten, so including it would dilute the number with a guaranteed zero.
    """
    n = len(R)
    if n < 2:
        return 0.0
    drops = [R[j][j] - R[-1][j] for j in range(n - 1)]
    return sum(drops) / len(drops)


def per_task_forgetting(R: list[list[float]]) -> list[float]:
    """The individual drops behind :func:`average_forgetting`, for the results table."""
    return [R[j][j] - R[-1][j] for j in range(len(R) - 1)]
