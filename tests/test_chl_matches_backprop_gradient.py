"""The equivalence anchor: does CHL actually reproduce backprop as gamma -> 0?

Xie & Seung (2003) Eq. 3.19 claims the CHL update equals backprop's up to O(gamma).
That is a stronger and more falsifiable claim than "the updates look similar", so these
tests check the *rate*, not just the closeness.

Three independent checks are needed, because each is blind to a different failure:

1. per-layer cosine     — checks direction. Blind to any per-layer rescaling.
2. per-layer norm ratio — checks the gamma^(k-L) factor. Blind to direction.
3. O(gamma) error rate  — checks the theorem itself, not merely proximity.

Check 1 alone would pass even with the gamma^(k-L) factor deleted, because cosine is
invariant to positive scaling. Check 2 alone would pass for an update pointing the wrong
way with the right magnitude. This mirrors the aggregation-collapse result of
arXiv:2606.21126 from the opposite side: aggregate cosine hides layerwise *direction*
error, per-layer cosine hides cross-layer *scale* error.
"""

from __future__ import annotations

import pytest
import torch

from forgetlab.layers import LayeredNet
from forgetlab.metrics import per_layer_cosine, per_layer_relative_error
from forgetlab.rules.backprop import backprop_updates
from forgetlab.rules.chl import chl_updates

SIZES = [6, 5, 4, 3]
BATCH = 8
SETTLE = dict(n_steps=4000, dt=0.5, tol=1e-13)


@pytest.fixture(autouse=True)
def _float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _batch():
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(BATCH, SIZES[0], generator=gen)
    target = torch.randn(BATCH, SIZES[-1], generator=gen)
    return x0, target


def _compare(gamma: float):
    x0, target = _batch()
    net = LayeredNet(SIZES, gamma=gamma, seed=1)
    reference, _ = backprop_updates(net, x0, target)
    update, _ = chl_updates(net, x0, target, **SETTLE)
    return net, update, reference


def test_gamma_zero_relaxes_to_the_feedforward_pass():
    """With no feedback there is nothing to settle: the fixed point is the forward pass."""
    x0, _ = _batch()
    net = LayeredNet(SIZES, gamma=0.0, seed=1)
    forward = net.feedforward(x0)
    settled, _ = net.relax(x0, **SETTLE)
    for expected, actual in zip(forward, settled):
        assert torch.allclose(expected, actual, atol=1e-10)


def test_direction_matches_backprop_at_small_gamma():
    """Every layer's update points essentially the same way as backprop's."""
    _, update, reference = _compare(gamma=0.01)
    for layer, cos in enumerate(per_layer_cosine(update, reference), start=1):
        assert cos > 0.999, f"layer {layer} cosine {cos:.6f} too low"


def test_error_shrinks_linearly_with_gamma():
    """The theorem says the error is O(gamma), so halving gamma must halve the error.

    This is the test that distinguishes "correct" from "merely close". A wrong
    implementation can sit near backprop by accident; matching the *rate* is much harder
    to fake.
    """
    errors = {}
    for gamma in (0.02, 0.01, 0.005):
        _, update, reference = _compare(gamma)
        errors[gamma] = per_layer_relative_error(update, reference)

    for layer in range(len(SIZES) - 1):
        for coarse, fine in ((0.02, 0.01), (0.01, 0.005)):
            ratio = errors[coarse][layer] / errors[fine][layer]
            assert 1.8 < ratio < 2.2, (
                f"layer {layer + 1}: halving gamma {coarse}->{fine} changed the error "
                f"by {ratio:.3f}x, expected ~2x for an O(gamma) result"
            )


def test_layerwise_scale_factor_is_load_bearing():
    """Regression test for the easiest silent bug in this project.

    The gamma^(k-L) factor in Eq. 2.8 cancels the exponential attenuation of the clamping
    signal (Eq. 3.3). Delete it and the code still runs, still produces plausible plots,
    and every per-layer cosine is *unchanged* — but layers far from the output are updated
    exponentially too weakly. Only the magnitudes reveal it.
    """
    gamma = 0.01
    net, update, reference = _compare(gamma)

    ratios = [float(u.norm() / r.norm()) for u, r in zip(update, reference)]
    for layer, ratio in enumerate(ratios, start=1):
        assert 0.95 < ratio < 1.05, (
            f"layer {layer} update magnitude is {ratio:.4f}x backprop's; "
            "the gamma^(k-L) factor is probably wrong"
        )

    without_factor = [
        u / net.lr_scale(k) for k, u in zip(range(1, net.L + 1), update)
    ]
    assert per_layer_cosine(without_factor, reference) == pytest.approx(
        per_layer_cosine(update, reference), abs=1e-12
    ), "per-layer cosine should be blind to the factor — that is why check 2 exists"

    broken = [float(u.norm() / r.norm()) for u, r in zip(without_factor, reference)]
    assert broken[0] < 0.1, (
        "dropping the factor should starve the layer furthest from the output; "
        f"got ratio {broken[0]:.4f}"
    )
