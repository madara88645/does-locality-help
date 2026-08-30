# Exploratory: untied feedback — a local rule that is not backprop in disguise

**Exploratory, not confirmatory.** The pre-registered result is frozen; this is the
follow-up that [`limits.md`](limits.md) says would make the comparison a real test.
Committed before the continual-learning runs, so the hypothesis below is on record ahead
of the numbers.

## Question

The frozen null result cannot separate "locality does not help against forgetting" from
"the chosen local rule was not different enough from backprop to tell": across every γ
swept, tied CHL's update stays within ~2.5% of backprop's in magnitude and above 0.99
cosine in direction.

**Untied-feedback CHL** breaks that closeness. The top-down path gets its own fixed random
matrices `B` — drawn once from the same distribution as `W`, never updated — instead of the
forward weights' transpose. Measured on the experiment's architecture, the hidden-layer
update's cosine against backprop falls from 0.99998 to **0.0248**
(`experiments/untied_prototype_checks.py`); the output layer — clamped straight to the
target, no feedback in its update, under 0.3% of the parameters — stays at ~0.999 either
way. Where the parameters live, this rule genuinely computes something other than the
backprop gradient. Does it forget differently?

**Naming.** This is *not* feedback alignment. Lillicrap et al. 2016 (*Random synaptic
feedback weights support error backpropagation for deep learning*, Nat Commun 7:13276)
replace `W^T` inside backprop's backward pass and change nothing else. Here the *dynamics
of a settling network* change instead. Lillicrap is cited as prior evidence that a random
feedback path does not destroy learning — not as the algorithm implemented.

## The rule, and one definitional choice made before these runs

Identical to tied CHL — same two-phase settling, same clamped-minus-free Hebbian update —
except:

1. feedback travels down `γ·B` instead of `γ·Wᵀ`;
2. **no `γ^(k−L)` per-layer rescaling.** That factor is derived (Eq. 3.3) to cancel an
   attenuation specific to the *transposed* path. Inheriting it anyway multiplies a
   nearly-orthogonal hidden update tenfold and the weights diverge to NaN — measured, in
   `untied_prototype_checks.py`. Random-feedback learning carries no such factor either
   (it has nothing to compensate). The untied rule is therefore *defined* without it.

Disclosed: choice 2 was informed by single-task prototype runs (untied reaches 84.15% vs
tied 84.40% on quick-check MNIST, and clears a frozen-hidden-layer control by ~3 points, so
the hidden layer genuinely learns). It was fixed before any continual run.

Because untying and dropping the factor change **two** things at once, a **control arm**
runs alongside: *tied* CHL with the factor also off (`--flat-scale`). If the control forgets
like tied CHL, a difference in the untied arm cannot be blamed on the missing factor.

## Design

Everything from the frozen pre-registration is unchanged: domain-incremental Split-MNIST,
`784 → 256 → 2`, lr 0.05, 10 epochs, batch 32, 1000 images/task, γ = 0.1, seeds 0/1/2,
64 settling steps, identical tuning budget (none of these arms gets its own tuning pass).

| arm | feedback | γ^(k−L) | command |
|---|---|---|---|
| untied CHL | random fixed `B` | off (by definition) | `--untied` |
| control: flat-scale CHL | `Wᵀ` (tied) | off | `--flat-scale` |
| reference: tied CHL | `Wᵀ` | on | *(frozen result)* |
| reference: backprop | — | — | *(frozen result)* |

Same seeds ⇒ identical `W` initialisation across arms (`B` is drawn after `W`, so untying
does not perturb the forward init — asserted in `tests/test_untied_feedback.py`).

The task-incremental protocol is not run: sharing untied `B` across heads is unimplemented
(`multi_head` refuses), and that protocol has no signal anyway (Amendment 1).

## Hypothesis, recorded before running

**Working hypothesis: forgetting stays at ~50 pp in both arms.** Reasoning: the collapse
analysis shows forgetting here is *total overwriting* of a shared trunk — whatever rule
succeeds in learning task N by moving the same shared weights will overwrite task N−1,
however credit is assigned. This is a hypothesis drawn from that analysis and the
random-feedback literature, not a pre-registered confirmatory prediction.

## Named traps, before the numbers

1. **An arm that fails to learn shows low forgetting trivially.** Guard: per-task
   attainment (`R[j][j]`) and the joint-training ceiling, reported per arm.
2. **The flat-scale control may under-train the trunk** (hidden updates are ~γ× smaller),
   which would *reduce interference* for a capacity-ish reason, not a credit-assignment
   one. Guard: trunk weight movement is checked before any such difference is interpreted.
3. **Untied variance may be larger** (a random `B` per seed is part of the rule). 3 seeds;
   spreads reported next to every mean.

## What could / could not be claimed

**Could:** on this architecture and protocol, a local rule measurably *different* from
backprop (cosine 0.025) forgot this much — which the frozen comparison could not test.

**Could not:** anything about depth, other datasets, other protocols, or local rules in
general; nor confirmatory weight — the analysis was chosen after the main result was known.

---

# Result

| arm | attainment (first 4) | final accuracy | forgetting | joint ceiling | trunk moved |
|---|---|---|---|---|---|
| backprop *(frozen)* | — | 57.90% ±0.42 | 50.59 pp ±0.67 | 89.75% | — |
| tied CHL *(frozen)* | 98.63% | 58.03% ±0.36 | 50.35 pp ±0.75 | 89.84% | 9.89% |
| flat-scale control | 98.62% | 59.42% ±0.69 | 48.49 pp ±0.92 | 84.63% | 2.90% |
| **untied CHL** | 98.45% | **61.79% ±0.87** | **45.21 pp ±1.17** | 87.50% | 2.43% |

*(attainment = accuracy on each task right after training it, mean over the first four —
`experiments/untied_attainment.py`; trunk movement = relative change of the shared 784→256
weights over the full sequence, seed 0 — `experiments/untied_trunk_movement.py`.)*

**The working hypothesis missed.** Forgetting did not stay at ~50 pp: the untied arm
forgot 45.2 pp — about 5 pp less than tied CHL and backprop — and ended the sequence with
the highest final accuracy of any arm, despite the lowest-but-one joint ceiling.

## The traps, checked

**Trap 1 — cleared.** Attainment is essentially identical across arms (98.45% vs 98.63%).
The untied arm learned each task as well as the others before moving on, so the reduction
is not a lower-peak artifact.

**Trap 2 — fired, and it caps the claim.** Both new arms move the shared trunk 3–4×
less than tied CHL (2.4% / 2.9% vs 9.9%). Less trunk plasticity is a boring, sufficient
mechanism for less interference, and it is confounded with everything the tied-vs-new
comparison shows. Two observations survive it, cautiously:

- The control and the untied arm move the trunk almost equally (2.9% vs 2.4%), yet differ
  by 3.3 pp of forgetting and 2.4 pp of final accuracy — so reduced plasticity alone does
  not account for the gap *between the two new arms*.
- The untied arm pairs less forgetting with *more* final accuracy, not with less learning.

**Trap 3 — as expected.** Seed spreads grew (±1.17 vs ±0.75), and everything above rests
on 3 seeds.

## Honest reading

A local rule that measurably computes something other than the backprop gradient
(hidden-update cosine 0.025) forgot somewhat less than backprop on this benchmark — but the
effect is one-tenth of the total forgetting, it travels together with a large reduction in
trunk plasticity, and the analysis was chosen after the frozen result was known.

This is **a hypothesis generated, not a finding established**: "untied-feedback CHL trades
trunk plasticity for retention at equal attainment" is now precise enough to pre-register
properly — more seeds, a plasticity-matched control (e.g. tied CHL with the hidden learning
rate scaled to match trunk movement), and the prediction written down first.

What it already does establish: the frozen comparison's blind spot was real. Change the
rule into something genuinely different from backprop and the forgetting numbers move —
so "CHL forgets like backprop" was indeed a statement about closeness to backprop, not
about locality.
