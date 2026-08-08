"""The fuse box: do these rules actually *train*, not just produce a correct update?

The equivalence tests prove the update rule matches the theory on one batch. That is a
different claim from "the network learns", and conflating the two is how a project like
this discovers in week 4 that CHL never converged.

Needs the MNIST download, so it is skipped when the data is not present. The equivalence
anchors have no such dependency and always run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from forgetlab.layers import LayeredNet
from forgetlab.train import accuracy, train

DATA = Path("data")
pytestmark = pytest.mark.skipif(
    not (DATA / "MNIST").exists(),
    reason="MNIST not downloaded; run an experiment script once to fetch it",
)

SIZES = [784, 128, 10]
SHARED = dict(lr=0.1, epochs=8, batch_size=64, seed=0)
SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)


@pytest.fixture(scope="module")
def data():
    from forgetlab.data import load_mnist

    torch.set_default_dtype(torch.float32)
    return load_mnist(train_size=6000, test_size=2000, seed=0)


def _run(rule, data):
    x_train, y_train, x_test, y_test = data
    net = LayeredNet(SIZES, gamma=0.1, seed=1)
    train(
        net,
        rule,
        x_train,
        y_train,
        rule_kwargs=({} if rule == "backprop" else SETTLE),
        **SHARED,
    )
    return accuracy(net, x_test, y_test)


@pytest.mark.parametrize("rule", ["backprop", "pc", "chl"])
def test_rule_learns(rule, data):
    """Well clear of the 10% chance floor, under a budget shared by all three rules."""
    assert _run(rule, data) > 0.80


def test_pc_tracks_backprop_through_a_whole_training_run(data):
    """Exactness is not just a one-batch property.

    The fixed-prediction assumption makes PC's update *identical* to backprop's, so the two
    should follow the same trajectory for an entire run, not merely start at the same place.
    """
    assert abs(_run("pc", data) - _run("backprop", data)) < 0.02


def test_chl_stays_close_to_backprop(data):
    """CHL is only O(gamma)-equivalent, so it may drift — but not far, and not downward
    into a different regime."""
    assert abs(_run("chl", data) - _run("backprop", data)) < 0.05
