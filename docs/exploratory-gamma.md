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
