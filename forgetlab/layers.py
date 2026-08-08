"""The shared layered network that all three learning rules train.

Layer indexing follows Xie & Seung (2003) so the code can be read against the paper:
layer 0 is the input, layer L is the output, and weight matrix ``W_k`` maps layer
``k-1`` to layer ``k`` for ``k = 1 … L``.  In Python the same matrix is ``W[k-1]``.

There is exactly **one** weight matrix per layer.  It is used forward as ``W_k`` and,
during relaxation, backward as ``gamma * W_k^T``.  The weights are *tied*: that is not
an implementation convenience, it is what the equivalence proof requires.  Untying them
gives Feedback Alignment, a different algorithm proving a different claim.
See ``docs/limits.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

ACTIVATIONS = {
    "tanh": (torch.tanh, lambda y: 1.0 - y * y),
    "sigmoid": (torch.sigmoid, lambda y: y * (1.0 - y)),
    "identity": (lambda z: z, lambda y: torch.ones_like(y)),
}


class LayeredNet(nn.Module):
    """A multilayer perceptron with weak, tied feedback connections.

    The fixed point of the relaxation dynamics (Xie & Seung Eq. 2.5) is

        x_k = f_k( W_k x_{k-1}  +  gamma * W_{k+1}^T x_{k+1}  +  b_k )

    with ``x_{L+1} = 0`` and ``W_{L+1} = 0``.  Setting ``gamma = 0`` removes the feedback
    entirely and the fixed point degenerates to the ordinary feedforward pass, which is a
    useful sanity check.

    Args:
        sizes: layer widths, e.g. ``[784, 256, 10]`` gives ``L = 2`` weight matrices.
        gamma: feedback strength.  The equivalence to backprop holds as ``gamma -> 0``.
        hidden_activation: transfer function for layers ``1 … L-1``.  Must be monotonically
            increasing (a condition of the proof).
        linear_output: if True, layer ``L`` is linear.  Required for equivalence to
            backprop on squared error.  With a sigmoid output the equivalence is instead to
            backprop on cross-entropy (Xie & Seung Eq. 3.20).
    """

    def __init__(
        self,
        sizes: list[int],
        gamma: float = 0.1,
        hidden_activation: str = "tanh",
        linear_output: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if len(sizes) < 2:
            raise ValueError("need at least an input and an output layer")
        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {hidden_activation!r}")

        self.sizes = list(sizes)
        self.L = len(sizes) - 1
        self.gamma = float(gamma)
        self.hidden_activation = hidden_activation
        self.linear_output = linear_output

        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)

        weights, biases = [], []
        for i in range(self.L):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            bound = 1.0 / math.sqrt(fan_in)
            w = torch.empty(fan_out, fan_in).uniform_(-bound, bound, generator=gen)
            weights.append(nn.Parameter(w))
            biases.append(nn.Parameter(torch.zeros(fan_out)))
        self.W = nn.ParameterList(weights)
        self.b = nn.ParameterList(biases)

    # ------------------------------------------------------------------ helpers

    def _act(self, k: int, z: torch.Tensor) -> torch.Tensor:
        """Transfer function of layer ``k`` (1-indexed, paper convention)."""
        if k == self.L and self.linear_output:
            return z
        return ACTIVATIONS[self.hidden_activation][0](z)

    def lr_scale(self, k: int) -> float:
        """The per-layer factor ``gamma^(k-L)`` from the CHL update rule (Eq. 2.8).

        Clamping the output perturbs layer ``k`` by only ``gamma^(L-k)`` (Eq. 3.3), because
        the feedback is weak and the disturbance is attenuated once per layer it travels.
        This factor cancels that attenuation exactly.  It grows *exponentially* with
        distance from the output: 1, 1/gamma, 1/gamma^2, …

        Dropping it is the single most likely silent bug in this project — the code still
        runs, and deep layers are simply updated exponentially too weakly.
        """
        return self.gamma ** (k - self.L)

    # ------------------------------------------------------------- forward pass

    def feedforward(self, x0: torch.Tensor) -> list[torch.Tensor]:
        """Ordinary MLP forward pass, ignoring feedback.

        Returns ``[x_0, x_1, …, x_L]``.  This is also the natural starting point for
        relaxation: Xie & Seung Eq. 3.1 shows the free state differs from it by O(gamma).
        """
        xs = [x0]
        for i in range(self.L):
            z = xs[-1] @ self.W[i].T + self.b[i]
            xs.append(self._act(i + 1, z))
        return xs

    # -------------------------------------------------------------- relaxation

    @torch.no_grad()
    def relax(
        self,
        x0: torch.Tensor,
        target: torch.Tensor | None = None,
        init: list[torch.Tensor] | None = None,
        n_steps: int = 50,
        dt: float = 0.5,
        tol: float = 1e-7,
    ) -> tuple[list[torch.Tensor], int]:
        """Settle the network to a fixed point of Eq. 2.5.

        The input ``x_0`` is always held fixed.  If ``target`` is given the output layer is
        **hard-clamped** to it — this is the clamped phase.  Note that CHL hard-clamps; it
        is Equilibrium Propagation that applies a weak nudge instead (``docs/limits.md``).

        Deliberately runs without autograd.  CHL does not compute a gradient, it *measures*
        one by nudging the system and watching what moves.

        Args:
            init: starting state.  Pass the clamped state here when settling the free phase,
                per Xie & Seung §4: settle the clamped phase first, then run the free phase
                **without resetting the hidden units**, so that the free state stays in the
                basin of attraction of the clamped one and the contrastive cost stays
                non-negative.
            n_steps: maximum Euler steps.
            dt: Euler step size.  ``dt = 1`` is plain fixed-point iteration; smaller is
                slower but more stable.
            tol: stop once the largest state change falls below this.

        Returns:
            ``(states, steps_taken)`` where ``states`` is ``[x_0, …, x_L]``.
        """
        xs = [t.clone() for t in (init if init is not None else self.feedforward(x0))]
        xs[0] = x0
        if target is not None:
            xs[self.L] = target

        free_top = self.L - 1 if target is not None else self.L
        steps = 0
        for steps in range(1, n_steps + 1):
            drive = []
            for k in range(1, free_top + 1):
                z = xs[k - 1] @ self.W[k - 1].T + self.b[k - 1]
                if k < self.L:
                    z = z + self.gamma * (xs[k + 1] @ self.W[k])
                drive.append(self._act(k, z))

            delta = 0.0
            for k in range(1, free_top + 1):
                new = xs[k] + dt * (drive[k - 1] - xs[k])
                delta = max(delta, (new - xs[k]).abs().max().item())
                xs[k] = new
            if delta < tol:
                break
        return xs, steps

    def settle_both_phases(
        self,
        x0: torch.Tensor,
        target: torch.Tensor,
        n_steps: int = 50,
        dt: float = 0.5,
        tol: float = 1e-7,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the clamped phase, then the free phase seeded from it.

        Returns ``(clamped, free)`` — ``x_hat`` and ``x_check`` in the paper's notation.
        The ordering is prescribed by Xie & Seung §4 and is not interchangeable.
        """
        clamped, _ = self.relax(x0, target=target, n_steps=n_steps, dt=dt, tol=tol)
        free, _ = self.relax(x0, target=None, init=clamped, n_steps=n_steps, dt=dt, tol=tol)
        return clamped, free
