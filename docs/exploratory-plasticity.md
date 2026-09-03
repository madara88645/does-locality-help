# Exploratory: does untied CHL forget less than its plasticity explains?

**Exploratory, and written before the runs.** The untied result in
[`exploratory-untied.md`](exploratory-untied.md) is confounded: the untied rule forgot
5.4 pp less than backprop, but it also moved the shared trunk 3–4× less over the sequence
(2.43% against tied CHL's 9.89%). Less movement is a sufficient, uninteresting explanation
for less interference, and one matched control cannot separate the two.

## The design: a curve, not a single control

Throttle the *tied* rule — the one already known to track backprop — by scaling its
hidden-layer update by a factor `s`, and read off `(trunk movement, forgetting)` at
several values of `s`. That traces how much forgetting a backprop-like rule sheds per
unit of plasticity given up. Then place the untied arm on the same axes.

- untied lands **on** the curve → plasticity explains the whole effect, nothing left over
- untied lands **below** it (forgets less than its own movement predicts) → something in
  the rule, beyond how much it moves the weights, is doing work

`s` scales only layers below the output: `lr_scale(k) = γ^(k−L) · s` for `k < L`, and
`1.0` at `k = L`. The output layer is untouched so every arm can still fit each task.

| | |
|---|---|
| throttle values `s` | 1.0, 0.6, 0.35, 0.2, 0.1 |
| arms on the curve | tied CHL at each `s` |
| arms placed against it | untied CHL, flat-scale control, backprop |
| seeds | 0, 1, 2, 3, 4 (up from 3) |
| everything else | frozen: domain-incremental, 784→256→2, lr 0.05, 10 epochs, batch 32, γ = 0.1, 1000 images/task |

Per arm this records trunk movement, forgetting, **and attainment** — the accuracy on each
task right after training it.

## Prediction, recorded before running

**Claude's prediction: untied lands slightly below the curve, and most of the 5.4 pp gap
is explained by plasticity.** Reasoning: the flat-scale control already moves the trunk
almost as little (2.90%) and still forgets 3.3 pp more than untied, so something survives
— but two arms at one point each is thin evidence, and the plasticity effect is likely to
be the larger term. Mehmet has not committed a prediction on this one.

## Named traps

1. **Throttling can stop the network learning.** A rule that never fit a task cannot
   forget it, which would drag low-`s` points down for a trivial reason and bend the
   curve. Guard: attainment is reported for every point, and any arm that fails to reach
   the ~98% the other arms manage is excluded from the curve rather than fitted.
2. **Trunk movement is a crude proxy.** It is the relative L2 change of the shared
   weights, blind to direction: two rules can move the same distance to different places.
   The curve constrains the plasticity explanation; it does not fully characterise it.
3. **Five points, five seeds.** The curve is descriptive, not a fitted model, and no
   claim rests on interpolating between its points.

## What could / could not be claimed

**Could:** on this architecture and protocol, the untied rule's forgetting is / is not
lower than a backprop-like rule matched to the same trunk movement.

**Could not:** that plasticity and credit assignment have been fully separated (trap 2),
anything outside this range of `s`, or confirmatory weight — this analysis was chosen
after the untied result was known.

---

# Result

5 seeds, domain-incremental Split-MNIST, everything else frozen.

| arm | trunk moved | forgetting | ACC | attainment |
|---|---|---|---|---|
| tied, s = 1.0 *(the standard rule)* | 9.89% | 50.47 ±0.76 | 57.88% | 98.63% |
| tied, s = 0.6 | 7.11% | 50.71 ±0.73 | 57.69% | 98.67% |
| tied, s = 0.35 | 5.30% | 50.60 ±0.74 | 57.80% | 98.69% |
| tied, s = 0.2 | 4.04% | 49.83 ±0.70 | 58.40% | 98.67% |
| tied, s = 0.1 *(= flat-scale control)* | 2.88% | 48.36 ±0.86 | 59.53% | 98.62% |
| **UNTIED** | **2.28%** | **44.77 ±1.04** | **62.16%** | **98.47%** |
| tied, s = 0.05 | 1.96% | 46.44 ±1.16 | 61.00% | 98.57% |
| tied, s = 0.02 | 1.05% | 43.93 ±1.32 | 62.82% | 98.43% |

Rows are ordered by trunk movement, so the untied arm sits where it belongs on the curve.
`s = 0.05` and `s = 0.02` were added after the first pass, because the untied arm moved
less than the original lowest point and the comparison would otherwise have had to
extrapolate. They bracket it.

**Trap 1 cleared.** Attainment stays in 98.43–98.69% across every arm. No throttle setting
stopped the network fitting its tasks, so nothing on this curve is low-forgetting for the
trivial reason.

## Reading the curve

**The curve is not a line.** From 9.89% down to 5.30% trunk movement — a 1.9× reduction —
forgetting does not move at all (50.47, 50.71, 50.60). Only below roughly 5% does it start
to fall, and then it falls steeply. So "moving the shared weights less causes less
forgetting" is false over most of the range and true only at low plasticity. That was not
obvious before measuring, and it is the reason a single matched control could not have
settled this.

**The untied arm sits below the curve.** Interpolating between its two bracketing points
(1.96% → 46.44 and 2.88% → 48.36) puts the curve at **47.11** at the untied arm's own
2.28% trunk movement. It measured 44.77, i.e. **2.34 pp lower than its plasticity
predicts**.

Splitting the original gap against the standard rule:

```
tied, s = 1.0                     50.47
curve at untied's trunk movement  47.11     <- 3.36 pp explained by plasticity  (59%)
UNTIED, measured                  44.77     <- 2.34 pp left over               (41%)
```

**The pre-registered prediction held.** It said the untied arm would land slightly below
the curve with most of the gap explained by plasticity: 59% explained, 2.34 pp residual.

## The deflating part, stated plainly

A tied rule throttled far enough reaches the untied arm's numbers without any untying at
all. At `s = 0.02` it forgets 43.93 pp (below untied's 44.77) at 62.82% ACC (above untied's
62.16%). Whatever the untied rule buys, **turning plasticity down buys the same thing and
more.** The untied rule is not a floor, and it is not the only route to this behaviour.

So the honest summary is narrower than the untied write-up's original framing: most of the
untied arm's advantage is plasticity, a small residual survives matching, and a simpler
intervention on the ordinary rule reaches the same place.

## What this does and does not settle

**Settles:** a single plasticity-matched control was not enough, and the curve shows why —
the relationship is flat over most of its range. Plasticity accounts for the majority of
the untied effect. The remaining 2.34 pp was two seed standard deviations on 5 seeds, and
was called suggestive rather than established. It has since been re-measured at 15 seeds
and did not move (2.38 pp), so it is real — see the section below.

**Does not settle:** trap 2 stands. Trunk movement is a direction-blind proxy, so matching
on it does not fully match "how much the rule disturbs what earlier tasks needed". A rule
could move the same distance in a less destructive direction, and this experiment cannot
see that. Testing it needs a measure of *where* the trunk moved, not just how far.

---

# Follow-up: where the damage happens

Trap 2 above said trunk movement is blind to direction. The first attempt to measure
direction failed outright (`experiments/damage_direction.py`): the untied arm moves in a
far less harmful direction by that metric and then loses *more* on task 1, so the
prediction was inverted rather than weak. One of its two candidate explanations was that
the measurement looked at the trunk while the protocol collides every task in one shared
two-unit output head.

That is now separated (`experiments/where_is_the_damage.py`). After task 1, the remaining
tasks are trained either normally or with the trunk frozen so only the head can change:

| arm | trunk | forgetting | attainment |
|---|---|---|---|
| tied | learns normally | 50.35 pp | 98.63% |
| tied | **frozen after task 1** | 43.48 pp | 98.21% |
| untied | learns normally | 45.21 pp | 98.45% |
| untied | **frozen after task 1** | 44.01 pp | 98.29% |

**The shared head accounts for most of the forgetting, and both rules pay it equally.**
With the trunk frozen outright, 43.5 pp still goes. That is roughly 86% of the total, and
neither rule can avoid it: five tasks writing to two output units overwrite each other
regardless of how credit was assigned.

**The rules differ in the damage their trunk updates add on top.** Tied adds 6.87 pp,
untied adds 1.20 pp. The 5.67 pp difference is the size of the 5.14 pp gap between the
arms. The untied arm's advantage is therefore located: it is in what the rule does to the
trunk, not the head, and the untied arm already sits close to the floor where the trunk
does no damage at all.

Two checks. Freezing is the limit of throttling and the two agree across separate runs:
frozen tied gives 43.48 pp here, the most throttled point above (`s = 0.02`) gives
43.93 pp. And attainment stays 98.21–98.63% throughout, so nothing here is low-forgetting
because it failed to learn.

This locates the effect without finishing the explanation. The curve above already showed
that most of the trunk difference is distance travelled. What distance does not cover is
still open.

---

# The residual is real: 15 seeds

Three attempts to find a mechanism behind the 2.34 pp residual failed in a row:

- `experiments/damage_direction.py` — whether the untied rule moves in a less harmful
  direction. Its prediction came out **inverted**, so the measurement was invalid.
- `experiments/where_is_the_damage.py` — the head/trunk split. This one **worked**, and
  located the difference in the trunk rather than the shared head, but locating is not
  explaining.
- `experiments/trunk_interference.py` — whether successive tasks push the trunk in
  colliding directions. Both rules came out the same, the untied one marginally worse.

At that point the simplest remaining hypothesis was that there was no effect to explain,
and the residual was sampling noise on 5 seeds. That is testable: noise shrinks with more
seeds. It was tested (`experiments/residual_more_seeds.py`).

| arm | trunk moved | forgetting | seeds |
|---|---|---|---|
| **UNTIED** | 2.24% | **45.00 ±0.82** | 15 |
| tied, s = 0.05 | 1.93% | 46.69 ±0.87 | 15 |
| tied, s = 0.1 | 2.84% | 48.68 ±0.65 | 15 |

Interpolating between the two tied points, which bracket the untied arm's trunk movement:

```
curve at 2.24%    47.38
untied measured   45.00
residual          +2.38 pp     (2.34 pp at 5 seeds)
```

**The residual did not shrink.** Tripling the seeds moved it by 0.04 pp while the
per-seed spread stayed near 0.8, so the mean now carries roughly 0.21 pp of standard
error against a 2.38 pp gap. The noise hypothesis is refuted.

So the position is: **a real effect with an unknown mechanism.** The untied rule forgets
about 2.4 pp less than a backprop-like rule matched to the same trunk movement, that
difference lives in the trunk rather than the shared head, and neither the direction it
moves nor how its tasks' movements interact accounts for it. Three plausible explanations
have been eliminated, which narrows the search without ending it.

Trap 2 from the original design still constrains the reading: trunk movement is a
direction-blind proxy, so "matched on plasticity" means matched on distance, and a
mechanism operating in some other property of the movement would be invisible to this
comparison. That is now the most likely place for the answer to be hiding.
