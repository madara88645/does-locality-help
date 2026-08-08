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
