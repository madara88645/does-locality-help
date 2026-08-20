# ForgetLab

A from-scratch implementation of **Contrastive Hebbian Learning** (Xie & Seung, 2003) in
PyTorch, validated against the published theorem that relates it to backpropagation, and
then used to run one small, controlled continual-learning comparison against backprop on
Split-MNIST.

## Result

On domain-incremental Split-MNIST, with an identical tuning budget and 3 seeds:

| rule | final accuracy | forgetting | joint-training ceiling |
|---|---|---|---|
| backprop | 57.90% ±0.42 | 50.59 pp ±0.67 | 89.75% |
| contrastive Hebbian | 58.03% ±0.36 | **50.35 pp ±0.75** | 89.84% |

![forgetting curves](results/forgetting_curves_domain.png)

**The local learning rule forgets just as much as backpropagation.** With 51 percentage
points of forgetting available to separate them, CHL differs from backprop by 0.24 pp
against a seed-to-seed standard deviation of 0.7 — indistinguishable.

One thing this result is *not*:

- **This does not show that the brain's mechanisms do not help.** What was swapped is the
  *credit assignment* mechanism, holding everything else fixed. Replay, neuromodulation,
  sparsity and structural plasticity — all things brains have and this experiment does not —
  are untouched. The claim is narrow: locality of the learning rule, on its own, buys
  nothing against catastrophic forgetting here.

The originally pre-registered **task-incremental** protocol is also reported, and produced a
floor effect: forgetting of −0.03 pp ±0.22, with both rules at 98.4% and the joint
ceiling (98.15%) no better than sequential training. Separate heads plus transferable stroke
features mean the tasks barely interfere, so that protocol cannot separate the rules at all.
It is kept because "this benchmark has no signal" is itself worth recording. See Amendment 1
in [`docs/preregistration.md`](docs/preregistration.md).

### Why per-task forgetting is so uneven

Forgetting is far from uniform across tasks — 89.5 pp for 4v5 but only 15.9 pp for 6v7 —
and the average hides why. `experiments/analyse_forgetting.py` asks which class the *final*
network assigns to each original digit:

| task | digits | class assigned by the final net | own labels | task accuracy |
|---|---|---|---|---|
| 4v5 | 4, 5 | 1, 0 | 0, 1 | **9.2%** (below chance) |
| 0v1 | 0, 1 | 0, 0 | 0, 1 | 48.8% |
| 2v3 | 2, 3 | 0, 0 | 0, 1 | 50.1% |
| 6v7 | 6, 7 | 0, 1 | 0, 1 | **82.9%** |

The network is not retaining a weakened version of each old task. It has collapsed onto the
last task's decision rule — "does this look more like an 8 or a 9?" — and an old task scores
well only when its own label assignment happens to *agree* with that rule. 6 resembles 8 and
7 resembles 9, and both agree, so 6v7 survives. 4 resembles 9 and 5 resembles 8, and both
disagree, so 4v5 lands below chance: the network is confidently, systematically inverted.

So the 89.5-vs-15.9 spread is not some tasks being more robust than others. Every task was
overwritten equally; the spread is coincidental label alignment with whatever was trained
last. Forgetting here is total overwriting plus luck, not graded decay — which also means
per-task forgetting numbers should not be read as a memory-strength ranking.

### Scope of the claim

On this architecture, on Split-MNIST, in these two settings, with an identical tuning
budget, these two rules forgot this much. Nothing about depth, other datasets,
class-incremental settings, or language models. CHL is documented to degrade with
depth, and this experiment deliberately stays in the shallow regime where it works — which
is exactly why it cannot support a general claim.

### Reproduce

```bash
uv sync --extra dev
uv run python experiments/run_continual_comparison.py --protocol domain
uv run python experiments/run_continual_comparison.py --protocol task
```

CPU only, a few minutes total.

## What this is / is not

- **Is:** a minimal, tested, readable implementation of one local learning rule, plus a
  small pre-registered replication of a specific literature claim.
- **Is not:** a new method, a SOTA result, or a general-purpose library. There is no claim
  here that biologically-motivated learning rules are better than backprop at anything.

## Why it exists

Accounts of the brain stress that cortex learns **continuously and
locally** from an ongoing sensory stream, while a language model is trained **offline and
globally** by backpropagation and cannot absorb a new example without disturbing its
weights. That contrast is usually stated qualitatively. This repository turns one narrow
version of it into something measurable: implement the local rules properly, verify they
do what their theorems say, then check whether they actually forget differently.

## The narrow claim

A minimal, tested, PyTorch-native implementation of Contrastive Hebbian Learning,
validated against its published equivalence theorem, used to run a controlled
continual-learning comparison against backprop.

Existing CHL code is embedded in large cognitive architectures
([Leabra/GeneRec](https://github.com/emer/leabra),
[PsyNeuLink](https://princetonuniversity.github.io/PsyNeuLink/ContrastiveHebbianMechanism.html)),
hardware-specific ([Vivilux](https://github.com/NeuroSumbaD/Vivilux)), a single-paper
research script ([Dual-Propagation](https://github.com/Rasmuskh/Dual-Propagation)), or dead
single-file demos. None of them run this comparison.

That is the whole claim. It is *not* "no CHL implementation exists" — that is false, and
the repos above are why.

## Prior work

| Project | What it is | Why this repo is not it |
|---|---|---|
| [emer/leabra](https://github.com/emer/leabra) | O'Reilly's GeneRec; its symmetric-midpoint variant *is* CHL. The canonical implementation, taught for 25+ years via [compcogneuro.org](https://compcogneuro.org). | Full cognitive architecture in Go; CHL is buried inside it rather than isolated and testable. |
| [PsyNeuLink](https://princetonuniversity.github.io/PsyNeuLink/ContrastiveHebbianMechanism.html) | Princeton's cognitive-modelling toolkit, actively maintained, ships a first-class `ContrastiveHebbianMechanism`. | Modelling framework, not a PyTorch trainer; no equivalence-to-backprop test suite. |
| [Vivilux](https://github.com/NeuroSumbaD/Vivilux) | CHL on simulated photonic hardware. Actively maintained. | Hardware-specific. |
| [Dual-Propagation](https://github.com/Rasmuskh/Dual-Propagation) | An accelerated CHL variant using dyadic neurons (JAX/Julia). | A faster formulation. This repo implements the original 2003 equations deliberately, because the point is to test *that* theorem — pedagogical clarity over speed. |
| [Lillicrap et al. 2016](https://www.nature.com/articles/ncomms13276) | Feedback Alignment: random fixed feedback weights still support learning, because the forward weights align to them. | A different algorithm proving a different claim. "Still learns" is not "equals the backprop gradient", so the weights here stay tied. |

## Honest statement of the theorem's cost

Xie & Seung's equivalence is not free. It requires **infinitesimally weak feedback**
(`γ → 0`) and, as a direct consequence, **per-layer learning rates that grow exponentially
with distance from the output** (the `γ^(k-L)` factor in Eq. 2.8). Both are biologically
awkward and are fair criticisms of the result. They are stated here rather than buried.

See [`docs/limits.md`](docs/limits.md) for exactly which theorem needs which limit, and why
CHL's `γ` is not Equilibrium Propagation's `β`.

## Layout

```
docs/limits.md      which theorem needs which limit and which assumption
forgetlab/layers.py the shared layered network with tied, weak feedback
forgetlab/rules/    backprop (reference), CHL
forgetlab/metrics.py per-layer gradient alignment
tests/              the equivalence anchors
```

## Install

```bash
uv sync --extra dev
uv run pytest
```

CPU only. No GPU is required at any point.

## License

MIT
