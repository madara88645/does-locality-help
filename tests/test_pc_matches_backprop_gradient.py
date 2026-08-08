"""The exact-equivalence anchor.

Under the fixed prediction assumption, predictive coding's settled error nodes satisfy
backpropagation's own backward recursion, so the resulting weight update is not an
approximation of the gradient — it *is* the gradient.  That makes this the one place in
the project where "textbook exact" is actually checkable, and it is the anchor that proves
the rest of the implementation (the shared network, the activation derivatives, the sign
conventions) is not silently wrong.

Contrast with CHL, whose equivalence is only O(gamma): see
``test_chl_matches_backprop_gradient.py``.
"""

from __future__ import annotations

import pytest
import torch

from forgetlab.layers import LayeredNet
from forgetlab.metrics import per_layer_relative_error
from forgetlab.rules.backprop import backprop_updates
from forgetlab.rules.predictive_coding import pc_updates

BATCH = 8
SETTLE = dict(n_steps=5000, dt=0.5, tol=1e-15)


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _batch(sizes, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(BATCH, sizes[0], generator=gen)
    target = torch.randn(BATCH, sizes[-1], generator=gen)
    return x0, target


@pytest.mark.parametrize("sizes", [[6, 5, 4, 3], [7, 6, 5, 4, 3, 2]])
def test_pc_equals_backprop_to_float_precision(sizes):
    """Exact, not approximate — the difference should be machine epsilon, not a tolerance."""
    x0, target = _batch(sizes)
    net = LayeredNet(sizes, gamma=0.1, seed=1)

    reference, reference_b = backprop_updates(net, x0, target)
    update, update_b = pc_updates(net, x0, target, **SETTLE)

    for layer, (u, r) in enumerate(zip(update, reference), start=1):
        assert torch.allclose(u, r, atol=1e-12, rtol=0), (
            f"layer {layer}: max abs difference {float((u - r).abs().max()):.3e}"
        )
    for layer, (u, r) in enumerate(zip(update_b, reference_b), start=1):
        assert torch.allclose(u, r, atol=1e-12, rtol=0), f"bias {layer} differs"


def test_pc_is_independent_of_gamma():
    """PC never uses the feedback connections, so gamma must not touch its update.

    Guards against accidentally coupling the two rules through the shared network.
    """
    sizes = [6, 5, 4, 3]
    x0, target = _batch(sizes)
    updates = []
    for gamma in (0.0, 0.1, 0.5):
        net = LayeredNet(sizes, gamma=gamma, seed=1)
        updates.append(pc_updates(net, x0, target, **SETTLE)[0])
    for later in updates[1:]:
        for a, b in zip(updates[0], later):
            assert torch.allclose(a, b, atol=1e-14, rtol=0)


def test_fixed_prediction_assumption_is_load_bearing():
    """Turning the assumption off must change the answer materially.

    If this test passes trivially, the flag is a no-op and the "exact" result above would
    be exact for the wrong reason.  Measured: relative error jumps from ~1e-15 to ~0.4.
    """
    sizes = [6, 5, 4, 3]
    x0, target = _batch(sizes)
    net = LayeredNet(sizes, gamma=0.1, seed=1)

    reference, _ = backprop_updates(net, x0, target)
    strict, _ = pc_updates(net, x0, target, fixed_prediction=False, **SETTLE)

    errors = per_layer_relative_error(strict, reference)
    assert max(errors) > 0.1, (
        "dropping the fixed prediction assumption should visibly change the update; "
        f"largest relative error was only {max(errors):.3e}"
    )
