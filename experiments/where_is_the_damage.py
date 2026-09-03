"""Which part of the network does the forgetting happen in?

The damage-direction measurement (experiments/damage_direction.py) failed, and one of its
two candidate explanations was that it looked in the wrong place: it studied the shared
trunk, while in the domain-incremental protocol all five tasks also write to one shared
two-unit output head. That head is a far tighter bottleneck than the 784x256 trunk.

This separates them. After task 1 is learned, the remaining tasks are trained either
normally, or with the trunk frozen so that only the head can change. If forgetting is
roughly unchanged with a frozen trunk, the head accounts for it and every trunk-based
analysis in this repository has been measuring a bystander.

Attainment is reported alongside, because a frozen trunk might simply be unable to fit
the later tasks, which would lower forgetting for the usual trivial reason.

RESULT (3 seeds):

    arm      trunk                 forgetting   attainment
    tied     learns normally          50.35pp       98.63%
    tied     FROZEN after task 1      43.48pp       98.21%
    UNTIED   learns normally          45.21pp       98.45%
    UNTIED   FROZEN after task 1      44.01pp       98.29%

Freezing the trunk entirely still leaves 43.5 pp of forgetting, so the shared head
accounts for roughly 86% of it and both rules pay that equally. What separates them is
the damage their trunk updates add on top: +6.87 pp for tied, +1.20 pp for untied. That
5.67 pp difference is the size of the 5.14 pp gap between the arms, which locates the
untied rule's advantage in what it does to the trunk rather than the head.

Put plainly: the untied arm already sits close to the floor where the trunk does no
damage at all, and the tied arm does not.

Two consistency checks:

  * freezing is the limit of throttling, and the two agree. Frozen tied gives 43.48 pp
    here; the most throttled point of docs/exploratory-plasticity.md (s = 0.02) gives
    43.93 pp from a separate run.
  * attainment stays 98.21-98.63% everywhere, so no arm is low-forgetting because it
    failed to learn.

This localises the effect without fully explaining it. The plasticity curve already
showed that most of the trunk difference is distance travelled; what remains is the part
distance does not cover, and that is still open.

Run with::

    uv run python experiments/where_is_the_damage.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import average_forgetting
from forgetlab.train import accuracy, train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=2)
SEEDS = [0, 1, 2]


def run(tasks, seed: int, untied: bool, freeze_trunk_after_first: bool):
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=seed)
    normal = net.lr_scale

    R, attained = [], []
    for i, task in enumerate(tasks):
        if i == 1 and freeze_trunk_after_first:
            # every layer below the output stops learning; the head carries on
            net.lr_scale = lambda k, L=net.L: 0.0 if k < L else 1.0
        train(net, "chl", task.x_train, task.y_train, seed=seed,
              rule_kwargs=SETTLE, **SHARED)
        row = [accuracy(net, t.x_test, t.y_test) for t in tasks]
        attained.append(row[i])
        R.append(row)

    net.lr_scale = normal
    att = sum(attained[:-1]) / (len(attained) - 1) * 100
    return average_forgetting(R) * 100, att


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)
    print("Forgetting when only the shared head may change after task 1.")
    print(f"{len(SEEDS)} seeds, domain-incremental Split-MNIST\n")
    print(f"{'arm':>10} | {'trunk':>16} | {'forgetting':>11} | {'attainment':>11}")
    print("-" * 60)
    for label, untied in [("tied", False), ("UNTIED", True)]:
        for tag, freeze in [("learns normally", False), ("FROZEN after task 1", True)]:
            rows = [run(tasks, s, untied, freeze) for s in SEEDS]
            print(f"{label:>10} | {tag:>16} | {mean([r[0] for r in rows]):>10.2f}pp | "
                  f"{mean([r[1] for r in rows]):>10.2f}%")


if __name__ == "__main__":
    main()
