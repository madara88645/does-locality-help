"""The untied rule's anchors: it must be genuinely different, and still sane.

The tied rule's whole point is *matching* backprop; the untied rule's whole point is NOT
matching it while remaining a working learning rule.  So the assertions are inverted:
where test_chl_matches_backprop_gradient.py requires alignment, this file requires
misalignment -- plus the sanity properties that stop "different" from meaning "broken".
"""

from __future__ import annotations

import pytest
import torch

from forgetlab.layers import LayeredNet
from forgetlab.metrics import per_layer_cosine
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


def test_untying_breaks_the_equivalence_and_tying_keeps_it():
    """The one claim the untied rule rests on, checked in both directions at once."""
    x0, target = _batch()
    for tied, hidden_should_align in ((True, True), (False, False)):
        net = LayeredNet(SIZES, gamma=0.1, tied=tied, seed=1)
        ref, _ = backprop_updates(net, x0, target)
        upd, _ = chl_updates(net, x0, target, **SETTLE)
        cos = per_layer_cosine(upd, ref)
        # The equivalence is O(gamma), so at gamma = 0.1 "aligned" means ~0.9+, not
        # 1 - epsilon; the gamma -> 0 *rate* is what test_chl_matches_backprop_gradient
        # checks.  The gap is the claim: tied stays above 0.9 in every layer, untied
        # falls below 0.5 in every layer that receives feedback.  The output layer
        # receives no feedback directly, but its update still contains the settled
        # *presynaptic* activity, which the feedback path shapes -- so it degrades too,
        # just less than any layer the feedback reaches.  (On the shallow experiment
        # architecture the effect is negligible; on this deeper toy net it is not.)
        if hidden_should_align:
            assert all(c > 0.9 for c in cos), cos
        else:
            assert all(abs(c) < 0.5 for c in cos[:-1]), cos
            assert cos[-1] > max(abs(c) for c in cos[:-1]), cos


def test_untied_settling_still_reaches_a_fixed_point():
    """Tied weights guarantee convergence via symmetry; untied has no such theorem, so
    convergence is asserted, not assumed."""
    x0, target = _batch()
    net = LayeredNet(SIZES, gamma=0.1, tied=False, seed=1)
    for tgt in (None, target):
        _, steps = net.relax(x0, target=tgt, **SETTLE)
        assert steps < SETTLE["n_steps"]


def test_untied_rule_has_no_per_layer_rescaling():
    """gamma^(k-L) is derived for the transposed path; the untied rule is defined
    without it (and with it, training diverges -- see untied_prototype_checks.py)."""
    tied = LayeredNet(SIZES, gamma=0.1, tied=True, seed=1)
    untied = LayeredNet(SIZES, gamma=0.1, tied=False, seed=1)
    for k in range(1, len(SIZES)):
        assert tied.lr_scale(k) == pytest.approx(0.1 ** (k - tied.L))
        assert untied.lr_scale(k) == 1.0


def test_feedback_matrices_are_fixed_and_out_of_autograd():
    """B is part of the rule's definition, not a trainable parameter."""
    net = LayeredNet(SIZES, gamma=0.1, tied=False, seed=1)
    assert all(not b.requires_grad for b in net.B)
    # same seed, same W as the tied net: untying must not change the forward init
    tied = LayeredNet(SIZES, gamma=0.1, tied=True, seed=1)
    for w_u, w_t in zip(net.W, tied.W):
        assert torch.equal(w_u, w_t)


def test_multi_head_refuses_untied():
    with pytest.raises(NotImplementedError):
        LayeredNet.multi_head([6, 5], head_size=3, n_heads=2, tied=False)
