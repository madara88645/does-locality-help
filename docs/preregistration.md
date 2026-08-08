# Pre-registration

Written and committed **before** the continual-learning experiment is run. The point is to
fix the analysis in advance, because the standard way a continual-learning comparison gets
quietly rigged is by tuning one method harder than the others and reporting the result as
if the budget had been equal.

## Question

On task-incremental Split-MNIST, do the local learning rules (predictive coding,
contrastive Hebbian learning) forget differently from backpropagation?

This is a **replication-and-extension** question, not a new-method question. No claim of
superiority is being tested. A null result — all three forget the same amount — is a valid
and publishable outcome, and is the outcome currently considered most likely.

## Fixed setup

| | |
|---|---|
| Architecture | shared trunk `784 → 256`, one private head `256 → 2` per task |
| Hidden activation | `tanh` |
| Output | linear (required for the squared-error equivalence; see `docs/limits.md`) |
| Objective | ½‖output − onehot‖², plain SGD, no momentum |
| Tasks | 0v1, 2v3, 4v5, 6v7, 8v9, trained strictly in that order |
| Protocol | task-incremental — task identity given at test time, separate head per task |
| Train set | 1000 images per task |
| Test set | full MNIST test split for the two digits of each task |
| γ (CHL) | 0.1 |
| Inference steps (PC, CHL) | 64, `dt = 0.5`, `tol = 1e-8` |
| Seeds | 3 (0, 1, 2) |
| No | replay, rehearsal, EWC/SI, regularisation, momentum, LR schedule, early stopping |

## Tuning budget — identical for all three rules

**One shared learning rate and epoch count, chosen once, applied to all three rules
unchanged.** No per-rule tuning. If a rule performs badly under the shared setting, that is
reported as a result, not fixed by giving that rule its own hyperparameters.

The shared values are chosen by the single-task sanity check below, using **backprop only**,
before any continual-learning run. PC and CHL never get a tuning pass.

Known constraint found while calibrating: `lr = 0.2` with a 512-unit hidden layer **diverges**
(9.8% test accuracy, i.e. chance). The shared learning rate must be verified non-divergent
on the sanity check before the experiment runs.

## Single-task sanity bar — corrected

The project plan originally set this bar at "≈98%, the published MNIST MLP ballpark". **That
figure is wrong for this setup and has been corrected.**

Those numbers come from cross-entropy + softmax + ReLU + a modern optimiser. The equivalence
theorem this project reproduces requires **squared error with linear output units**, and
plain SGD on that objective plateaus lower. Measured ceiling with backprop on full MNIST:

| hidden | epochs | lr | output | test accuracy |
|---|---|---|---|---|
| 128 | 20 | 0.1 | linear | 94.07% |
| 256 | 40 | 0.1 | linear | **94.96%** |
| 256 | 40 | 0.1 | sigmoid | 94.94% |
| 512 | 60 | 0.2 | linear | 9.80% (diverged) |

So the sanity bar is: **each rule reaches ≈94–95% on full single-task MNIST, and the three
agree with each other**, not "each rule reaches 98%". Holding to the original figure would
have made a correctly-implemented CHL look broken.

Measured, full MNIST, 20 epochs, identical settings: backprop 94.07%, PC 94.07%,
CHL 94.30%. PC matches backprop exactly, as the fixed-prediction assumption requires.

## Metrics

Computed from the accuracy matrix `R`, where `R[i][j]` is accuracy on task `j` after
finishing training on task `i`:

- **ACC** — mean of `R[last]`, average accuracy over all five tasks at the end.
- **Forgetting (BWT)** — mean over `j < last` of `R[j][j] − R[last][j]`. Positive means
  forgetting. The final task is excluded, since it cannot yet have been forgotten.
- **Joint-training upper bound** — the same architecture trained on all five tasks shuffled
  i.i.d., as a reference row. Not a competitor; a ceiling.

Reported per rule, mean ± standard deviation over the 3 seeds, plus per-task forgetting.

## What will be claimed, and what will not

**Can be claimed:** on this architecture, on Split-MNIST, in the task-incremental setting,
with an identical tuning budget, rule X forgot *this much*.

**Will not be claimed:** that predictive coding or CHL forgets less than backprop *in
general*; that local rules are more robust to catastrophic forgetting; any statement about
depth, other datasets, class-incremental settings, or language models. The literature is
explicit that PC and CHL degrade with depth, and this experiment is deliberately run in the
shallow regime where they are documented to work — which is exactly why it cannot support a
general claim.

## Amendment 1 — adding a second protocol, before the comparison was run

The task-incremental protocol above was run first, with backprop, and produced a **floor
effect**: average forgetting of 0.07 percentage points, with all three rules at 98.36%
accuracy. There is nothing to compare.

The cause is structural, not a bug. Each task has its own head, and the shared trunk learns
generic stroke features that transfer between digit pairs, so the tasks barely interfere.
This is consistent with van de Ven & Tolias's own point that task-incremental Split-MNIST is
close to solved; catastrophic forgetting lives in the harder settings.

A null result from this protocol would be *uninformative* — null because the benchmark has
no signal, not because the rules behave alike. So a second protocol is added:

**Domain-incremental** — a *single shared output head* for all five tasks. Each task still
maps its two digits to labels 0/1, but now every task writes to the same output units, so
they interfere directly. Task identity is not given at test time.

Measured with backprop, same settings, before any rule comparison:

| protocol | ACC | forgetting |
|---|---|---|
| task-incremental (5 heads) | 98.36% | 0.07 pp |
| domain-incremental (1 shared head) | 57.64% | **51.00 pp** |

Depth is unchanged, so this stays in the shallow regime where PC and CHL are documented to
work — the reason the original protocol was chosen. Only the interference between tasks
increases.

**Both protocols are reported.** The task-incremental floor effect is a result in its own
right and is not dropped. The domain-incremental protocol is where the comparison between
rules can actually carry information. Everything else — architecture, learning rate, epochs,
seeds, the identical-tuning-budget rule — is unchanged.

This amendment was written and committed before the three-rule comparison was run. The
`--protocol` flag in `experiments/run_continual_comparison.py` selects between them.

## Stopping rule

The results table is generated once, from `experiments/run_continual_comparison.py`, with
the settings above. If a bug is found afterwards, the fix and the re-run are recorded in the
commit history rather than silently replacing the numbers.
