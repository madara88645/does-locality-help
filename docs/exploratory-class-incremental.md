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

---

# Result

5 seeds. Chance is 10%, not 50%.

| arm | forgetting | final accuracy | attainment | trunk moved |
|---|---|---|---|---|
| backprop | 98.74 ±0.15 | 19.34% | 98.74% | 11.28% |
| tied CHL | 98.66 ±0.28 | 19.33% | 98.66% | 10.80% |
| tied CHL, throttled 0.05 | 98.65 ±0.04 | 19.30% | 98.65% | 1.59% |
| untied CHL | 98.52 ±0.08 | 19.27% | 98.53% | 2.98% |

**Trap 1 fired, in its harshest form.** Every arm learns every task — attainment is 98.5%
or better throughout — and then forgets essentially all of it. Forgetting is ~98.7 pp out
of a possible ~98.7. Final accuracy sits at 19.3%, which is what a network collapsed
entirely onto the last task scores when averaged over five tasks: near-perfect on 8v9,
near-zero on the rest.

So this is a **ceiling effect**, and it is the exact mirror of the floor effect that forced
Amendment 1. Task-incremental left nothing to forget; class-incremental leaves nothing to
retain. Neither has room to separate learning rules. The four arms span 0.22 pp against a
possible ~99, which is saturation, not similarity.

## What this settles

**The prediction was 40% that the effect survives. It did not — but not because it was
refuted.** The test saturated before it could measure anything, so the honest verdict is
that this protocol cannot answer the question at this scale, not that the answer is no.

What that leaves is a sharp constraint on the finding:

> The untied rule's advantage exists in domain-incremental Split-MNIST, which is the only
> one of the three standard protocols with measurement room at this architecture and
> budget. The two neighbouring settings saturate in opposite directions.

An effect that can only be seen through one window is not thereby false. It is, however,
unresolvable from inside this project: relieving the ceiling would need more data, more
capacity and a retuned budget, which breaks the frozen identical-tuning-budget discipline
that makes the original comparison trustworthy. That is a different experiment, and a
bigger one.

## Why this closes the project

The question this repository set out to ask has been answered as far as this setup can
answer it, and the answer is documented along with the reasons it cannot go further. Three
protocols, two of them saturated; one real effect confined to the third; five hypotheses
for its mechanism, four eliminated and the fifth explaining the wrong quantity.

Leaving it here with that stated is a better ending than either pretending the finding
generalises or quietly not checking.
