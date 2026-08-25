# Exploratory: does feedback strength change how much CHL forgets?

**This is exploratory, not confirmatory.** The pre-registered comparison in
[`preregistration.md`](preregistration.md) is finished, published and frozen; this asks a
new question of the same code. It is written and committed *before* the runs, so the
prediction below is on record ahead of the numbers — but the honest label for anything
found here is "worth testing properly", not "shown".

## Question

CHL's `gamma` scales the feedback connections: the network runs forward at full strength
and backward at `gamma * W^T`. It is the channel through which clamping the output reaches
the hidden layer at all — at `gamma = 0` the two phases would settle identically and the
weight update would be exactly zero.

The frozen result says CHL forgets as much as backprop at `gamma = 0.1`. Does that hold
across feedback strengths, or is 0.1 a coincidence?

## Prediction, recorded before running

**Mehmet, 2026-08-23: forgetting does not change with gamma.**

## Fixed

Everything from the main pre-registration is unchanged — architecture `784 -> 256 -> 2`,
domain-incremental protocol, 1000 images per task, `lr = 0.05`, 10 epochs, batch 32, seeds
0/1/2, 64 settling steps, no replay or regularisation. Only `gamma` varies.

| | |
|---|---|
| gamma | 0.01, 0.1, 0.3, 0.5 |
| rules | CHL only |
| reference | the frozen backprop row: 57.90% accuracy, 50.59 pp forgetting |

`gamma = 0.1` is the already-published run and is not re-run; its numbers are reused.

Backprop is not swept. It has no `gamma` — the parameter does not exist in its update
rule — so a sweep would spend minutes reproducing one number four times.

## The confound this has to survive

**Low forgetting is not automatically good news.** A network that never learned a task
cannot forget it. If some `gamma` leaves CHL unable to fit the tasks in the first place,
forgetting collapses toward zero for a trivial reason — the same floor effect that made the
task-incremental protocol uninformative (Amendment 1).

So forgetting is never read on its own. Every row reports final accuracy and the
joint-training ceiling next to it, and any `gamma` where CHL fails to learn is reported as
**a failure to learn**, not as low forgetting.

`gamma = 0.5` is strong feedback and may not settle at all. If it diverges, that is
reported as a result, not dropped.

## What could be claimed

**Could be claimed:** on this architecture and protocol, over this range of feedback
strengths, CHL's forgetting did / did not move.

**Could not be claimed:** anything about why, anything outside this range, anything about
depth or other datasets, and — because this analysis was chosen after the main result was
known — anything with the evidential weight of the pre-registered comparison. A difference
found here is a hypothesis for a future pre-registered run.

---

# Result

| γ | final accuracy | forgetting | joint ceiling |
|---|---|---|---|
| 0.01 | 57.89% ±0.42 | 50.60 pp ±0.66 | 89.80% |
| 0.1 *(the frozen run)* | 58.03% ±0.36 | 50.35 pp ±0.75 | 89.84% |
| 0.3 | 58.19% ±0.45 | 49.93 pp ±1.01 | 89.80% |
| 0.5 | 58.34% ±0.74 | 49.39 pp ±1.41 | 90.01% |
| backprop *(reference, no γ)* | 57.90% ±0.42 | 50.59 pp ±0.67 | 89.75% |

**The prediction holds: forgetting does not track γ.**

`γ = 0.5` did not diverge — that pre-registered risk did not materialise. The joint ceiling
is flat across the whole sweep (89.8–90.0%), so the confound named above is ruled out: no γ
left CHL unable to learn, and no change in forgetting is explained by a change in capacity.

## Why the monotone means are not a trend

The means fall monotonically, 50.60 → 49.39, which is tempting to read as a real effect.
Per seed it does not survive — the seeds are shared across γ, so these rows are paired:

| seed | γ=0.01 | γ=0.1 | γ=0.3 | γ=0.5 | change |
|---|---|---|---|---|---|
| 0 | 50.96 | 50.87 | 51.02 | 51.01 | **+0.05** |
| 1 | 49.83 | 49.49 | 49.02 | 48.67 | −1.16 |
| 2 | 51.00 | 50.70 | 49.75 | 48.48 | −2.52 |

One of three seeds shows no effect at all across a 50× change in γ. The spread between
seeds (+0.05 to −2.52) is larger than the effect in the mean, so the monotone means are
carried by two seeds out of three. On three seeds that is not a demonstrable effect.

Stated plainly: **γ was varied 50-fold and forgetting moved less than the random choice of
initial weights moves it.**

## The finding this sweep did not produce, and why

The sweep was run partly on the expectation that a large γ would push CHL meaningfully away
from backprop, since the equivalence holds only as γ → 0. Measurement refutes that: at
γ = 0.5 the CHL update is still within 2.5% of backprop's in magnitude and above 0.99
cosine in direction (see the measured table in [`limits.md`](limits.md)).

So this sweep does not widen the comparison's scope. Over the whole range tested, CHL
remains a small perturbation of backprop, and the null result inherits the limitation
recorded in `limits.md`: it cannot separate "locality does not help" from "this rule was not
different enough from backprop to tell".
