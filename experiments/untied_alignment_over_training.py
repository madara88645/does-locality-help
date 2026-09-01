"""Does the untied rule align with backprop during training, as the README claims?

The repository explains why untied-feedback CHL still learns by saying the forward
weights come to align with the random feedback matrices -- the feedback-alignment
mechanism (Lillicrap et al. 2016). That explanation has never been measured here. It is
asserted in prose and used to reassure the reader, which is exactly the kind of claim
this project has twice caught itself getting wrong.

Two quantities are tracked over training, on the same net:

  (a) cos(W_k, B_k)          -- do the forward weights rotate toward the fixed random
                                feedback matrices at all?
  (b) cos(dW_chl, dW_bp)     -- does the update the rule actually delivers come to agree
                                with the true gradient? This is the claim that matters:
                                feedback alignment says the delivered signal ends up
                                within 90 degrees of the gradient, which is what makes
                                descent possible.

If (b) stays flat near zero, the README's explanation is wrong and must be removed: the
rule would be learning for some reason we have not identified.

Run with::

    uv run python experiments/untied_alignment_over_training.py
"""

from __future__ import annotations

import torch

from forgetlab.data import load_mnist, one_hot
from forgetlab.layers import LayeredNet
from forgetlab.metrics import cosine, per_layer_cosine
from forgetlab.rules.backprop import backprop_updates
from forgetlab.rules.chl import chl_updates
from forgetlab.train import accuracy, train

SIZES = [784, 128, 10]
SETTLE = dict(n_steps=64, dt=0.5, tol=1e-8)
EPOCHS = 8
SHARED = dict(lr=0.1, batch_size=64, seed=0)


def probe(net: LayeredNet, xb: torch.Tensor, yb: torch.Tensor) -> tuple[float, float]:
    """Return (weight-vs-feedback cosine, delivered-update-vs-gradient cosine)."""
    # Feedback only enters layers below the output; with L=2 that is layer 1 only.
    w_vs_b = cosine(net.W[1].detach(), net.B[1].detach()) if not net.tied else 1.0
    ref, _ = backprop_updates(net, xb, yb)
    upd, _ = chl_updates(net, xb, yb, **SETTLE)
    return w_vs_b, per_layer_cosine(upd, ref)[0]      # [0] = the hidden layer


def main() -> None:
    torch.set_default_dtype(torch.float32)
    x_tr, y_tr, x_te, y_te = load_mnist(train_size=6000, test_size=2000, seed=0)
    # A fixed probe batch, so every measurement is comparable across epochs.
    xb, yb = x_tr[:64], one_hot(y_tr[:64], 10)

    net = LayeredNet(SIZES, gamma=0.1, tied=False, seed=1)
    net.lr_scale = lambda k: 1.0                       # the untied rule's definition

    print("untied-feedback CHL, single-task MNIST, 8 epochs\n")
    print(f"{'epoch':>6} | {'cos(W, B)':>10} | {'cos(update, gradient)':>22} | {'test acc':>9}")
    print("-" * 60)

    w_vs_b, upd_vs_grad = probe(net, xb, yb)
    print(f"{0:>6} | {w_vs_b:>10.4f} | {upd_vs_grad:>22.4f} | {accuracy(net, x_te, y_te)*100:>8.2f}%")

    for epoch in range(1, EPOCHS + 1):
        train(net, "chl", x_tr, y_tr, epochs=1, rule_kwargs=SETTLE, **SHARED)
        w_vs_b, upd_vs_grad = probe(net, xb, yb)
        print(f"{epoch:>6} | {w_vs_b:>10.4f} | {upd_vs_grad:>22.4f} | "
              f"{accuracy(net, x_te, y_te)*100:>8.2f}%")

    print("\nReading it: column 2 asks whether W rotated toward B; column 3 asks whether")
    print("the delivered update came to agree with the gradient. Feedback alignment")
    print("predicts column 3 rises well clear of zero. A flat column 3 refutes the")
    print("README's stated explanation for why this rule learns.")


if __name__ == "__main__":
    main()
