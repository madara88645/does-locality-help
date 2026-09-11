# Does locality help?

**Question:** does a local learning rule forget less than backpropagation when a neural network learns tasks sequentially?

**Status:** completed study, with a preregistered comparison and exploratory follow-ups.

## Short answer

Not decisively. In a preregistered comparison on Split-MNIST (handwritten digits learned in successive pairs), tied Contrastive Hebbian Learning (CHL) and backpropagation showed similar measured forgetting: **50.35 ± 0.75** and **50.59 ± 0.67 percentage points**, respectively. These are means and seed-to-seed standard deviations across three seeds, not confidence intervals. Because tied CHL’s update is almost the same as backpropagation’s in this setup, the comparison does not isolate locality or establish general equivalence.

With independent fixed feedback, the update became different and forgetting decreased. Reducing an ordinary tied rule’s shared-weight updates also reduced forgetting; strong throttling reached comparable or better measured results. An exploratory residual was estimated between measured control settings. This interpolation departed from the original analysis plan, and its mechanism and paired uncertainty remain unresolved here. No statistical-significance or causal-percentage claim is made in this summary. This is not evidence that locality itself solves catastrophic forgetting.

There is no universal or new-method claim here. The result is specific to this shallow network, Split-MNIST, training budget, and protocol.

## What was tested

- A from-scratch PyTorch implementation of CHL, validated against its backpropagation-equivalence limit.
- A preregistered domain-incremental comparison with tied feedback.
- Exploratory untied-feedback and plasticity controls, including throttled tied rules.
- A later class-incremental experiment with a ten-unit output head. All arms learned the tasks, then forgot almost all earlier tasks, leaving too little measurement room to distinguish them.

The [registration](docs/preregistration.md), [implementation limits](docs/limits.md), [plasticity controls](docs/exploratory-plasticity.md) and [class-incremental test](docs/exploratory-class-incremental.md) retain the details. The former README is preserved as [research history](docs/research-history.md), including some stronger original wording. This summary takes precedence for current scope; the historical notes are not a fresh statistical audit.

## Reproduce

This project uses [`uv`](https://docs.astral.sh/uv/) and runs on CPU:

```bash
uv sync --extra dev
uv run pytest
uv run python experiments/run_continual_comparison.py --protocol domain
uv run python experiments/run_continual_comparison.py --protocol task
uv run python experiments/analyse_forgetting.py
```

The commands above are reproduction instructions, not claims that they were freshly run for this summary. The repository is released under the [MIT License](LICENSE).

This was AI-assisted research: language models supported implementation, analysis and writing. Detailed records, including failed hypotheses and limitations, are retained for inspection.
