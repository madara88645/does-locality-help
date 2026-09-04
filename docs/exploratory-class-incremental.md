# Exploratory: does the untied effect survive a wider output head?

**Written before the runs.** This is the experiment that decides whether the project has
somewhere left to go.

## The problem this addresses

Everything measured so far happens inside a **two-unit output head**. Freezing the shared
trunk entirely still leaves 43.5 pp of forgetting
(`experiments/where_is_the_damage.py`), so roughly 86% of all forgetting in this project
comes from five tasks overwriting each other in two output units. That is an extreme
bottleneck.

The untied rule's advantage — 2.38 pp below a plasticity-matched control, real at 15 seeds
([`exploratory-plasticity.md`](exploratory-plasticity.md)) — was measured entirely inside
that bottleneck. So the open question is not the mechanism any more. It is prior to that:

> Is this a phenomenon, or an artifact of a two-unit head?

## Design

**Class-incremental** — the third setting in van de Ven & Tolias's taxonomy, and the one
this repository has so far declared out of scope. The output head is widened to **ten
units, one per digit**, and labels are the true digits rather than each task's own 0/1
remapping. Tasks still arrive as digit pairs in the same order, and task identity is still
withheld at test time, so evaluation is a full ten-way choice.

This relieves the head bottleneck without touching anything else: different tasks now
write to different output units, so they can no longer overwrite each other by
construction the way two shared units force them to.

| | |
|---|---|
| architecture | `784 → 256 → 10` (only the head width changes) |
| labels | true digits, 0–9 |
| evaluation | ten-way argmax, no task identity |
| arms | backprop, tied CHL, untied CHL, and tied CHL throttled to match untied's trunk movement |
| seeds | 0–4 |
| everything else | frozen: lr 0.05, 10 epochs, batch 32, γ = 0.1, 1000 images/task, same task order |

## Prediction, recorded before running

**Claude's prediction: 40% that the untied advantage survives.** The two-unit bottleneck is
doing most of the work in every measurement so far, and an effect measured inside it has a
fair chance of being a property of it. Mehmet has not committed a prediction.

## Named traps

1. **The opposite floor effect.** Class-incremental is the hardest of the three settings.
   If every arm collapses toward 10% chance accuracy, there is again nothing to separate
   them — the mirror image of the task-incremental floor that forced Amendment 1. Guard:
   per-task attainment is reported, and any arm that cannot fit its tasks in the first
   place is reported as failing to learn rather than as low-forgetting.
2. **The plasticity confound follows us here.** Trunk movement is recorded for every arm,
   and no forgetting difference is read without it.
3. **Five seeds.** Enough to see whether an effect of this size is present at all; not
   enough to size it precisely.

## What either outcome means

**If the effect survives:** it is not an artifact of the two-unit head, and the project has
a real phenomenon worth pursuing and writing up properly.

**If it dies:** the untied advantage was a property of an extreme bottleneck. That is a
clean, honest close — "we checked whether it generalises and it did not" is a much stronger
ending than leaving the question unasked.

Either way this is the last planned experiment.
