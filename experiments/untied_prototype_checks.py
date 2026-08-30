"""Prototype checks for untied feedback — is this a usable rule at all?

The frozen comparison in the repository cannot separate "locality does not help" from
"this rule was not different enough from backprop to tell" (see ``docs/limits.md``).
Untying the feedback weights is the proposed fix: the top-down path gets its own fixed
random matrices, so the update is no longer engineered to match the backprop gradient.

Before any of that is worth pre-registering, four things have to hold. This script checks
them and prints the evidence.

    1. Untying actually breaks the equivalence.  Otherwise it changes nothing.
    2. The relaxation still converges.  Tied weights make the dynamics symmetric; that
       guarantee is gone, so settling has to be verified rather than assumed.
    3. The rule still learns.  A network that never learned a task cannot forget it —
       the same floor effect that made the task-incremental protocol uninformative.
    4. It is not learning for a boring reason.  ~84% on this setup is reachable with a
       frozen random hidden layer and a trained output layer, which would be a
       random-features model rather than feedback alignment.

Naming — this is NOT feedback alignment, and must not be labelled as such.

    Lillicrap et al. (2016), *Random synaptic feedback weights support error
    backpropagation for deep learning*, Nature Communications 7:13276, replace ``W^T``
    in **backprop's backward pass** with a fixed random ``B`` and change nothing else.
    There is no relaxation, no gamma, and no per-layer scaling. What this file does is
    different: it changes the *dynamics of a settling network*. Calling it feedback
    alignment invites the obvious objection that it is not. Call it **untied-feedback
    CHL**, and cite Lillicrap as prior evidence that a random feedback path does not
    destroy learning — not as the algorithm being implemented.

    (Recorded 2026-08-23 after reading the wrong Lillicrap 2016: the same author
    published *Continuous control with deep reinforcement learning* — DDPG — the same
    year. Cite by title, not by author-year.)

That paper also settles, provisionally, the open question below. Feedback alignment has
no per-layer compensation factor at all, because it has no attenuation to compensate. So
there is no precedent for untied CHL inheriting ``gamma^(k-L)``, which was derived for the
transposed path. Dropping it looks like *defining a different rule* rather than tuning
one — but that argument has not been checked against the CHL equations themselves, and
must be before anything here is pre-registered.

Run with::

    uv run python experiments/untied_prototype_checks.py
"""

from __future__ import annotations

import torch

from forgetlab.data import load_mnist
from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import per_layer_cosine
from forgetlab.rules.backprop import backprop_updates
from forgetlab.rules.chl import chl_updates
from forgetlab.train import accuracy, train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.1, epochs=8, batch_size=64, seed=0)


def check_1_equivalence_is_broken() -> None:
    """Per-layer alignment of the CHL update against backprop's, tied vs untied."""
    print("\n1. Does untying break the equivalence to backprop?\n")
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)
    x0 = tasks[0].x_train[:32]
    y = torch.nn.functional.one_hot(tasks[0].y_train[:32].long(), 2).to(torch.float32)

    print(f"   {'':>8} {'gamma':>6} | {'cosine hidden':>14} {'cosine output':>14}")
    for tied in (True, False):
        for g in (0.1, 0.5):
            net = LayeredNet([784, 256, 2], gamma=g, tied=tied, seed=0)
            ref, _ = backprop_updates(net, x0, y)
            upd, _ = chl_updates(net, x0, y, **SETTLE)
            cos = per_layer_cosine(upd, ref)
            print(f"   {'tied' if tied else 'UNTIED':>8} {g:>6} | {cos[0]:>14.5f} {cos[1]:>14.5f}")
    print("\n   The output layer is clamped straight to the target, so no feedback enters")
    print("   its own state -- but its update still contains the settled *presynaptic*")
    print("   activity, which the feedback path shapes.  On this shallow architecture that")
    print("   indirect effect is negligible (cosine ~0.999); on deeper nets it is not")
    print("   (see tests/test_untied_feedback.py).")


def check_2_settling_still_converges() -> None:
    """Tied weights make the relaxation symmetric; untied has no such guarantee."""
    print("\n2. Does the relaxation still reach a fixed point?\n")
    torch.set_default_dtype(torch.float64)
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(8, 784, generator=gen)
    target = torch.randn(8, 10, generator=gen)
    n = 2000

    print(f"   {'':>8} {'gamma':>6} | {'free':>18} {'clamped':>18}")
    for tied in (True, False):
        for g in (0.1, 0.5):
            net = LayeredNet([784, 128, 10], gamma=g, tied=tied, seed=1)
            _, free = net.relax(x0, target=None, n_steps=n, dt=0.5, tol=1e-10)
            _, clamped = net.relax(x0, target=target, n_steps=n, dt=0.5, tol=1e-10)
            mark = lambda s: f"{s} steps" + ("" if s < n else "  <-- NO FIXED POINT")
            print(f"   {'tied' if tied else 'UNTIED':>8} {g:>6} | {mark(free):>18} {mark(clamped):>18}")


def _train_once(tied: bool, lr_scale_mode: str) -> tuple[float, float]:
    net = LayeredNet([784, 128, 10], gamma=0.1, tied=tied, seed=1)
    w0 = net.W[0].detach().clone()
    # NOTE: since the lr_scale change in layers.py, an untied net returns scale 1.0 by
    # *definition*.  "on" therefore has to force the tied-style factor back, to reproduce
    # the divergence that motivated that definition.
    if lr_scale_mode == "on":
        net.lr_scale = lambda k, g=net.gamma, L=net.L: g ** (k - L)
    elif lr_scale_mode == "off":
        net.lr_scale = lambda k: 1.0
    elif lr_scale_mode == "freeze_hidden":
        net.lr_scale = lambda k, L=net.L: 0.0 if k < L else 1.0
    x_tr, y_tr, x_te, y_te = _DATA
    train(net, "chl", x_tr, y_tr, rule_kwargs=SETTLE, **SHARED)
    moved = float((net.W[0].detach() - w0).norm() / w0.norm()) * 100
    return accuracy(net, x_te, y_te) * 100, moved


def checks_3_and_4_learning() -> None:
    """Does it learn, and is the hidden layer doing any of the work?"""
    print("\n3+4. Does it learn, and is the hidden layer contributing?\n")
    torch.set_default_dtype(torch.float32)

    rows = [
        ("tied, gamma^(k-L) on", True, "default"),
        ("UNTIED, gamma^(k-L) FORCED on", False, "on"),
        ("UNTIED (factor off, the rule)", False, "default"),
        ("UNTIED, hidden layer FROZEN", False, "freeze_hidden"),
    ]
    print(f"   {'':>30} | {'accuracy':>9} | {'hidden layer moved':>19}")
    for label, tied, mode in rows:
        acc, moved = _train_once(tied, mode)
        shown = "diverged (NaN)" if moved != moved else f"{moved:.2f}%"
        print(f"   {label:>30} | {acc:>8.2f}% | {shown:>18}")

    print("\n   The gamma^(k-L) factor compensates an attenuation specific to the")
    print("   *transposed* feedback path. Untied, it amplifies a nearly-orthogonal update")
    print("   tenfold; the weights blow up to NaN and accuracy sits at chance. Switching")
    print("   it off recovers training, and the hidden layer earns about the same as it")
    print("   does tied -- roughly 3 points over the frozen-hidden control, which both")
    print("   rules have to clear. Whether switching it off is the *principled* fix or")
    print("   merely the one that works is NOT settled here, and must be before any of")
    print("   this is pre-registered.")


if __name__ == "__main__":
    check_1_equivalence_is_broken()
    check_2_settling_still_converges()
    torch.set_default_dtype(torch.float32)
    _DATA = load_mnist(train_size=6000, test_size=2000, seed=0)
    checks_3_and_4_learning()
