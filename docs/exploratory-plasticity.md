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
