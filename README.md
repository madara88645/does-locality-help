# ForgetLab

From-scratch implementations of **Contrastive Hebbian Learning** (Xie & Seung, 2003) and
**Predictive Coding** (Whittington & Bogacz, 2017) in PyTorch, validated against the
published theorems that relate them to backpropagation, and then used to run one small,
controlled continual-learning comparison against backprop on Split-MNIST.

> **Status: work in progress.** The implementation and equivalence tests come first; the
> continual-learning results are not in yet. This README will not claim a result before
> there is one.

## What this is / is not

- **Is:** a minimal, tested, readable implementation of two local learning rules, plus a
  small pre-registered replication of a specific literature claim.
- **Is not:** a new method, a SOTA result, or a general-purpose library. There is no claim
  here that biologically-motivated learning rules are better than backprop at anything.

## Why it exists

Predictive coding accounts of the brain stress that cortex learns **continuously and
locally** from an ongoing sensory stream, while a language model is trained **offline and
globally** by backpropagation and cannot absorb a new example without disturbing its
weights. That contrast is usually stated qualitatively. This repository turns one narrow
version of it into something measurable: implement the local rules properly, verify they
do what their theorems say, then check whether they actually forget differently.

## The narrow claim

A minimal, tested, PyTorch-native implementation of Contrastive Hebbian Learning and
Predictive Coding, validated against their published equivalence theorems, used to run a
controlled task-incremental continual-learning comparison against backprop.

Existing CHL code is embedded in large cognitive architectures
([Leabra/GeneRec](https://github.com/emer/leabra),
[PsyNeuLink](https://princetonuniversity.github.io/PsyNeuLink/ContrastiveHebbianMechanism.html)),
hardware-specific ([Vivilux](https://github.com/NeuroSumbaD/Vivilux)), a single-paper
research script ([Dual-Propagation](https://github.com/Rasmuskh/Dual-Propagation)), or dead
single-file demos. Existing PC libraries are JAX-based
([pcx](https://github.com/liukidar/pcx), [jpc](https://github.com/thebuckleylab/jpc)) or
`nn.Sequential`-only ([Torch2PC](https://github.com/RobertRosenbaum/Torch2PC)). None of them
run this comparison.

That is the whole claim. It is *not* "no CHL implementation exists" — that is false, and
the repos above are why.

## Prior work

| Project | What it is | Why this repo is not it |
|---|---|---|
| [emer/leabra](https://github.com/emer/leabra) | O'Reilly's GeneRec; its symmetric-midpoint variant *is* CHL. The canonical implementation, taught for 25+ years via [compcogneuro.org](https://compcogneuro.org). | Full cognitive architecture in Go; CHL is buried inside it rather than isolated and testable. |
| [PsyNeuLink](https://princetonuniversity.github.io/PsyNeuLink/ContrastiveHebbianMechanism.html) | Princeton's cognitive-modelling toolkit, actively maintained, ships a first-class `ContrastiveHebbianMechanism`. | Modelling framework, not a PyTorch trainer; no equivalence-to-backprop test suite. |
| [Vivilux](https://github.com/NeuroSumbaD/Vivilux) | CHL on simulated photonic hardware. Actively maintained. | Hardware-specific. |
| [Dual-Propagation](https://github.com/Rasmuskh/Dual-Propagation) | An accelerated CHL variant using dyadic neurons (JAX/Julia). | A faster formulation. This repo implements the original 2003 equations deliberately, because the point is to test *that* theorem — pedagogical clarity over speed. |
| [Bogacz-Group/PredictiveCoding](https://github.com/Bogacz-Group/PredictiveCoding) | The Oxford lab's PC library + tutorial notebooks. Best place to learn PC. | Reference for correctness here, not a dependency. |
| [Torch2PC](https://github.com/RobertRosenbaum/Torch2PC) | PyTorch PC, companion to Rosenbaum's divergence analysis. | `nn.Sequential`-only, stale since Feb 2024, no CHL. |
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
forgetlab/rules/    backprop (reference), predictive coding, CHL
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
