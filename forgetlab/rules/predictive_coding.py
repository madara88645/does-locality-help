"""Predictive coding — Whittington & Bogacz (2017), fixed-prediction-assumption variant.

Each layer holds a *value* node ``x_k`` and an *error* node ``eps_k``.  Layer ``k-1`` sends
down a prediction of layer ``k``'s activity,

    mu_k   = f_k( W_k x_{k-1} + b_k )
    eps_k  = x_k - mu_k

and the network minimises the total squared prediction error ``F = 1/2 * sum_k ||eps_k||^2``
by settling the value nodes.  Only the mismatch travels up; only the prediction travels
down.  That is the whole mechanism the book chapter describes.

**The fixed prediction assumption.**  During inference the predictions ``mu_k`` (and the
slopes ``f_k'``) are held at their feedforward values instead of being recomputed as the
value nodes move.  With that assumption the settled error nodes satisfy

    eps_k = W_{k+1}^T ( f_{k+1}' * eps_{k+1} )

which is *exactly* backpropagation's backward recursion — so the resulting weight update is
not an approximation of the gradient, it is the gradient.  This is why the PC test asserts
exact equality to float tolerance while the CHL test can only assert an O(gamma) rate.

Strict / full-equilibrium PC — recomputing the predictions every step — is deliberately out
of scope.  Rosenbaum (arXiv:2106.13082) showed it fails to converge on deeper networks, and
debugging that is not what this project is for.  Pass ``fixed_prediction=False`` to see the
difference, but the experiments use the default.
"""

from __future__ import annotations

import torch

from forgetlab.layers import LayeredNet


@torch.no_grad()
def pc_updates(
    net: LayeredNet,
    x0: torch.Tensor,
    target: torch.Tensor,
    n_steps: int = 200,
    dt: float = 0.5,
    tol: float = 1e-12,
    fixed_prediction: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return ``(dW, db)`` — the weight changes to **add**.

    Sign convention matches :func:`forgetlab.rules.backprop.backprop_updates`.
    """
    feedforward = net.feedforward(x0)
    batch = x0.shape[0]
    layers = range(1, net.L + 1)

    # Predictions and slopes, frozen at their feedforward values.
    mu = {k: feedforward[k] for k in layers}
    slope = {k: net.act_deriv(k, feedforward[k]) for k in layers}

    # Value nodes start at the feedforward pass, so every error node starts at zero.
    # Clamping the output is what injects the only non-zero error into the system.
    x = [t.clone() for t in feedforward]
    x[net.L] = target

    for _ in range(n_steps):
        if not fixed_prediction:
            mu = {k: net.act(k, x[k - 1] @ net.W[k - 1].T + net.b[k - 1]) for k in layers}
            slope = {k: net.act_deriv(k, mu[k]) for k in layers}

        eps = {k: x[k] - mu[k] for k in layers}

        delta = 0.0
        for k in range(1, net.L):
            top_down = (slope[k + 1] * eps[k + 1]) @ net.W[k]
            step = dt * (-eps[k] + top_down)
            x[k] = x[k] + step
            delta = max(delta, step.abs().max().item())
        if delta < tol or net.L == 1:
            break

    eps = {k: x[k] - mu[k] for k in layers}

    dW, db = [], []
    for k in layers:
        signal = slope[k] * eps[k]
        dW.append(signal.T @ feedforward[k - 1] / batch)
        db.append(signal.sum(dim=0) / batch)
    return dW, db
