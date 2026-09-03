"""Do successive tasks push the shared trunk in directions that conflict?

docs/exploratory-plasticity.md located the untied rule's advantage in the trunk, and
showed that most of it is simply distance travelled. What distance does not cover is
still open, and this asks the obvious next question: not how far each task moves the
trunk, but whether the tasks move it in directions that collide.

Per rule, the trunk is snapshotted before and after every task, giving one displacement
per task. Two summaries:

    pairwise cosine   mean cosine between the displacements of different tasks.
                      near 0 means tasks use unrelated directions and can coexist;
                      positive means they pull the same way; negative means later tasks
                      partly undo earlier ones.

    path / net        total distance walked, divided by the straight-line distance from
                      start to finish. 1.0 means every task pushed the same way and
                      nothing was wasted; large means the trunk wandered back and forth,
                      which is movement spent cancelling itself.

RESULT (3 seeds), and it does not separate the rules:

    rule     pairwise cosine   path / net
    tied             -0.1347         3.36
    UNTIED           -0.1687         4.01

Both rules push the trunk in mildly conflicting directions, and both spend most of their
movement cancelling itself -- the trunk walks three to four times further than it ends up
from where it started. The untied rule is very slightly worse on both counts, not better,
so this is not where its advantage comes from.

Third failed explanation for the residual, after the harmful-direction measurement and
this one; the head-vs-trunk split (experiments/where_is_the_damage.py) located the effect
but did not explain it. At that point the honest hypothesis is that the residual may not
be a real effect: it was about two seed standard deviations on 5 seeds, and three
independent attempts to find a mechanism have found nothing. More seeds is the way to
settle that, and it is the next thing to run.

Run with::

    uv run python experiments/trunk_interference.py
"""

from __future__ import annotations

import torch

from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import cosine
from forgetlab.train import train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=2)
SEEDS = [0, 1, 2]


def displacements(tasks, seed: int, untied: bool) -> list[torch.Tensor]:
    """One trunk displacement per task, in order."""
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=seed)
    out = []
    for task in tasks:
        before = net.W[0].detach().clone()
        train(net, "chl", task.x_train, task.y_train, seed=seed,
              rule_kwargs=SETTLE, **SHARED)
        out.append(net.W[0].detach() - before)
    return out


def summarise(ds: list[torch.Tensor]) -> tuple[float, float]:
    pairs = [cosine(ds[i], ds[j]) for i in range(len(ds)) for j in range(i + 1, len(ds))]
    path = sum(float(d.norm()) for d in ds)
    net_move = float(sum(ds).norm())
    return sum(pairs) / len(pairs), path / net_move


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    torch.set_default_dtype(torch.float32)
    tasks = load_split_mnist(train_per_task=1000)
    print("How the five tasks push the shared trunk, relative to each other.")
    print(f"{len(SEEDS)} seeds, domain-incremental Split-MNIST\n")
    print(f"{'rule':>10} | {'pairwise cosine':>16} | {'path / net':>11}")
    print("-" * 44)
    for label, untied in [("tied", False), ("UNTIED", True)]:
        rows = [summarise(displacements(tasks, s, untied)) for s in SEEDS]
        print(f"{label:>10} | {mean([r[0] for r in rows]):>16.4f} | "
              f"{mean([r[1] for r in rows]):>11.2f}")


if __name__ == "__main__":
    main()
