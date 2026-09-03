"""Untying changes two things at once. Which one produces the residual?

trunk_subspace.py found that the untied arm's five tasks push the trunk along one shared
direction (0.995) where the tied arm's spread out (0.887), and traced that to B never
updating while W_1 rotates. That was offered as the mechanism for the residual. This tests
it, by separating the two things untying does at the same time:

    tied          feedback is W_1 itself, which trains
    frozen-copy   feedback is a copy of W_1 taken at initialisation and then frozen --
                  FIXED but not independent of the forward path
    untied        feedback is B, drawn separately and frozen -- fixed AND unrelated

Trunk movement is recorded too, because it is the confound that has already swallowed one
result in this project and the raw forgetting numbers cannot be read without it.

RESULT (15 seeds), against the plasticity curve of docs/exploratory-plasticity.md:

    arm            forgetting   trunk moved   shared dir   attain   curve says   residual
    tied              50.89pp         9.90%       0.8874   98.53%           --         --
    frozen-copy       46.32pp         1.95%       0.9949   98.56%      46.73pp     0.41pp
    untied            45.00pp         2.24%       0.9950   98.50%      47.37pp     2.37pp

**Freezing the feedback does not produce the residual.** The frozen-copy arm forgets
4.6 pp less than tied, and essentially all of it is explained by moving the trunk less: it
lands on the plasticity curve, 0.41 pp off, down from 0.97 pp at 5 seeds and heading
toward zero. The untied arm's 2.37 pp residual meanwhile does not move.

**So the subspace story explains the wrong thing.** Both frozen arms collapse their tasks
onto one shared trunk direction, at 0.9949 and 0.9950 -- indistinguishable -- yet only one
of them has a residual. Sharing a direction accounts for why the trunk moves less, which
is already counted as plasticity; it does not account for what is left over.

What remains is that the residual needs B to be *unrelated to the forward path*, not
merely fixed. That is the fifth hypothesis this project has spent on the residual, and the
first four are recorded as failures in damage_direction.py, trunk_interference.py,
where_is_the_damage.py and trunk_subspace.py.
"""
import torch
from torch import nn
from forgetlab.data.split_mnist import load_split_mnist
from forgetlab.layers import LayeredNet
from forgetlab.metrics import average_forgetting
from forgetlab.train import accuracy, train

SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
SHARED = dict(lr=0.05, epochs=10, batch_size=32, n_classes=2)
SEEDS = list(range(15))

def run(tasks, seed, mode):
    untied = mode in ("untied", "frozen-copy")
    net = LayeredNet([784, 256, 2], gamma=0.1, tied=not untied, seed=seed)
    if mode == "frozen-copy":
        for k in range(net.L):
            net.B[k] = nn.Parameter(net.W[k].detach().clone(), requires_grad=False)
    w0 = net.W[0].detach().clone()
    R, att, leads = [], [], []
    for i, task in enumerate(tasks):
        before = net.W[0].detach().clone()
        train(net, "chl", task.x_train, task.y_train, seed=seed, rule_kwargs=SETTLE, **SHARED)
        u, _, _ = torch.linalg.svd((net.W[0].detach() - before).double(), full_matrices=False)
        leads.append(u[:, 0])
        row = [accuracy(net, t.x_test, t.y_test) for t in tasks]
        att.append(row[i]); R.append(row)
    share = [abs(float(leads[i] @ leads[j])) for i in range(5) for j in range(i+1, 5)]
    moved = float((net.W[0].detach() - w0).norm() / w0.norm()) * 100
    return average_forgetting(R)*100, moved, sum(share)/len(share), sum(att[:-1])/4*100

torch.set_default_dtype(torch.float32)
tasks = load_split_mnist(train_per_task=1000)
m = lambda xs: sum(xs)/len(xs)
print(f"{'arm':>14} | {'forgetting':>11} | {'trunk moved':>12} | {'shared dir':>11} | {'attain':>8}")
print("-" * 68)
for mode in ("tied", "frozen-copy", "untied"):
    rows = [run(tasks, s, mode) for s in SEEDS]
    print(f"{mode:>14} | {m([r[0] for r in rows]):>10.2f}pp | {m([r[1] for r in rows]):>11.2f}% | "
          f"{m([r[2] for r in rows]):>11.4f} | {m([r[3] for r in rows]):>7.2f}%")
