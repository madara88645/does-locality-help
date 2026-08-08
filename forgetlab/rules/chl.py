"""Contrastive Hebbian Learning — Xie & Seung (2003), Eq. 2.8.

    dW_k  =  eta * gamma^(k-L) * ( x_hat_k x_hat_{k-1}^T  -  x_check_k x_check_{k-1}^T )

``x_hat`` is the clamped state (output held at the target), ``x_check`` the free state.
Each term is a plain Hebbian correlation — the product of the activities at the two ends
of a connection.  The rule is the *difference* of two such correlations: be more like the
state you were in when you were right, less like the state you were in when you were wrong.

No autograd anywhere in this file.  CHL does not compute a derivative, it measures one:
clamp the output, let the network settle, and see how far each unit moved.  That is why
the feedback must be weak — a derivative is the response to an *infinitesimal* nudge, and
a large nudge mixes in higher-order terms that are not the derivative.
"""

from __future__ import annotations

import torch

from forgetlab.layers import LayeredNet


@torch.no_grad()
def chl_updates(
    net: LayeredNet,
    x0: torch.Tensor,
    target: torch.Tensor,
    n_steps: int = 100,
    dt: float = 0.5,
    tol: float = 1e-9,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(dW, db)`` — the weight changes to **add**.

    Sign convention matches :func:`forgetlab.rules.backprop.backprop_updates`, so the two
    can be compared directly.  As ``net.gamma -> 0`` these should align with backprop's
    descent direction (Xie & Seung Eq. 3.19).
    """
    clamped, free = net.settle_both_phases(x0, target, n_steps=n_steps, dt=dt, tol=tol)
    batch = x0.shape[0]

    dW, db = [], []
    for k in range(1, net.L + 1):
        scale = net.lr_scale(k)
        hebb_clamped = clamped[k].T @ clamped[k - 1]
        hebb_free = free[k].T @ free[k - 1]
        dW.append(scale * (hebb_clamped - hebb_free) / batch)
        db.append(scale * (clamped[k] - free[k]).sum(dim=0) / batch)
    return dW, db
