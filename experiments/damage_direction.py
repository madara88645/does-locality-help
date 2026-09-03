"""Trap 2 from docs/exploratory-plasticity.md: is the untied rule's movement less harmful?

The plasticity curve matched arms on *how far* the shared trunk moves, which is blind to
*where* it moves. Two rules can travel the same distance and do different amounts of
damage, depending on whether they move along directions the earlier task depended on.

Measured here, per arm:

    d           ||W_after_all - W_after_task1||  -- how far the trunk moved (relative)
    cos         cosine of that movement against grad L1, the direction that most
                increases task 1's loss
    projection  <grad L1, movement>  -- the first-order damage prediction, which is
                distance and direction combined
    actual      the measured drop on task 1

If the untied arm shows a similar distance but a lower cosine, it moves as much but aims
less at what task 1 needed, and that is a mechanism for the residual the curve could not
explain. If the cosine matches too, the residual is not about direction.

RESULT (3 seeds), and it refutes the measurement rather than answering the question:

    arm                        distance   cos(move, harm)   damage   task 1 lost
    tied (standard)             10.08%           0.1268     0.0172       49.80pp
    tied, throttled s = 0.05     2.07%           0.2863     0.0081       52.31pp
    UNTIED                       2.67%           0.0405     0.0008       54.96pp

The untied arm does move in a far less harmful direction by this metric -- a third of
tied's cosine, and a twentieth of its first-order damage. But it loses the MOST on task 1,
5 pp more than the arm whose movement is supposedly 20x more damaging. The prediction is
not merely weak, it is inverted, so this quantity does not measure what it was built to
measure. Two candidate reasons, neither tested here:

  * the first-order approximation is meaningless at this scale. These are 2-10% weight
    changes producing ~50 pp accuracy drops, nowhere near the regime where a gradient
    dot-product predicts a loss change.
  * it looks only at the trunk. In the domain-incremental protocol every task writes to
    the same output head, and that collision may be where the damage happens, in which
    case trunk direction was never the right place to look.

Kept because a negative that closes off a line of reasoning is worth the same as a
positive that opens one, and because the second candidate above, if true, would also
qualify how docs/exploratory-plasticity.md should be read.

Run with::

    uv run python experiments/damage_direction.py
"""

from __future__ import annotations

import torch

from forgetlab.data import one_hot
from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import cosine
from forgetlab.rules.backprop import backprop_updates
from forgetlab.train import accuracy, train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=2)
SEEDS = [0, 1, 2]


def harm_direction(net: LayeredNet, task) -> torch.Tensor:
    """The trunk direction that most increases task 1's loss, at the current weights.

    backprop_updates returns the descent direction (-grad), so the harmful direction is
    its negation. Averaged over a fixed slice of the task's own training data.
    """
    x = task.x_train[:512]
    y = one_hot(task.y_train[:512], 2)
    dW, _ = backprop_updates(net, x, y)
    return -dW[0].detach().clone()          # layer 0 = the shared trunk


def run(tasks, seed: int, untied: bool, throttle: float | None):
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=seed)
    if throttle is not None:
        g, L = net.gamma, net.L
        net.lr_scale = lambda k: (g ** (k - L)) * throttle if k < L else 1.0

    train(net, "chl", tasks[0].x_train, tasks[0].y_train, seed=seed,
          rule_kwargs=SETTLE, **SHARED)
    w_after_1 = net.W[0].detach().clone()
    acc_after_1 = accuracy(net, tasks[0].x_test, tasks[0].y_test)
    harm = harm_direction(net, tasks[0])

    for task in tasks[1:]:
        train(net, "chl", task.x_train, task.y_train, seed=seed,
              rule_kwargs=SETTLE, **SHARED)

    move = net.W[0].detach() - w_after_1
    return (
        float(move.norm() / w_after_1.norm()) * 100,        # distance, %
        cosine(move, harm),                                  # direction
        float((move * harm).sum()),                          # first-order damage
        (acc_after_1 - accuracy(net, tasks[0].x_test, tasks[0].y_test)) * 100,
    )


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)
    print("Movement after task 1, and how much of it points at task 1's loss.")
    print(f"{len(SEEDS)} seeds, domain-incremental Split-MNIST\n")
    print(f"{'arm':>26} | {'distance':>9} | {'cos(move, harm)':>16} | "
          f"{'damage':>9} | {'task 1 lost':>12}")
    print("-" * 88)
    for label, untied, throttle in [
        ("tied (standard)", False, None),
        ("tied, throttled s = 0.05", False, 0.05),
        ("UNTIED", True, None),
    ]:
        rows = [run(tasks, s, untied, throttle) for s in SEEDS]
        print(f"{label:>26} | {mean([r[0] for r in rows]):>8.2f}% | "
              f"{mean([r[1] for r in rows]):>16.4f} | {mean([r[2] for r in rows]):>9.4f} | "
              f"{mean([r[3] for r in rows]):>11.2f}pp")


if __name__ == "__main__":
    main()
