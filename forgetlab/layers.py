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
        tied: bool = True,
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
        self.tied = bool(tied)

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

        # Untied feedback: the top-down path gets its own fixed random matrices instead of
        # the forward weights' transpose.  Backprop's chain rule *requires* the transpose,
        # so this is exactly the assumption that makes CHL approximate backprop -- breaking
        # it is the point.  See Lillicrap et al. 2016 for why learning survives it.
        #
        # B is drawn from the same distribution as W so the feedback magnitude is
        # comparable; a different scale would confound the untying with a change in gamma.
        # It is never updated: the training loop iterates over W and b only, and
        # requires_grad=False keeps it out of autograd in backprop_updates too.
        if not self.tied:
            feedback = []
            for i in range(self.L):
                fan_in, fan_out = sizes[i], sizes[i + 1]
                bound = 1.0 / math.sqrt(fan_in)
                m = torch.empty(fan_out, fan_in).uniform_(-bound, bound, generator=gen)
                feedback.append(nn.Parameter(m, requires_grad=False))
            self.B = nn.ParameterList(feedback)

    # ------------------------------------------------------------------ helpers

    def feedback(self, k: int) -> torch.Tensor:
        """The matrix the top-down signal travels down, from layer ``k+1`` to layer ``k``.

        Tied (the default, and what every published result in this repo used): the forward
        weights themselves.  Untied: fixed random weights that ``W`` knows nothing about.
        """
        return self.W[k] if self.tied else self.B[k]

    def act(self, k: int, z: torch.Tensor) -> torch.Tensor:
        """Transfer function of layer ``k`` (1-indexed, paper convention)."""
        if k == self.L and self.linear_output:
            return z
        return ACTIVATIONS[self.hidden_activation][0](z)

    def act_deriv(self, k: int, y: torch.Tensor) -> torch.Tensor:
        """``f_k'`` of layer ``k``, expressed in terms of that layer's *output* ``y``.

        Both supported transfer functions have a derivative that is cheap to write this
        way (``tanh' = 1 - y^2``, ``sigmoid' = y(1 - y)``), which avoids having to keep the
        pre-activations around.
        """
        if k == self.L and self.linear_output:
            return torch.ones_like(y)
        return ACTIVATIONS[self.hidden_activation][1](y)

    def lr_scale(self, k: int) -> float:
        """The per-layer factor ``gamma^(k-L)`` from the CHL update rule (Eq. 2.8).

        Clamping the output perturbs layer ``k`` by only ``gamma^(L-k)`` (Eq. 3.3), because
        the feedback is weak and the disturbance is attenuated once per layer it travels.
        This factor cancels that attenuation exactly.  It grows *exponentially* with
        distance from the output: 1, 1/gamma, 1/gamma^2, …

        Dropping it is the single most likely silent bug in this project — the code still
        runs, and deep layers are simply updated exponentially too weakly.
        """
        if not self.tied:
            # The factor above compensates an attenuation specific to the *transposed*
            # feedback path (Eq. 3.3 is derived from W^T).  With untied random feedback
            # that derivation does not apply, and inheriting the factor anyway amplifies
            # a nearly-orthogonal update ~1/gamma-fold per layer -- measured, the weights
            # diverge to NaN (experiments/untied_prototype_checks.py).  The untied rule is
            # therefore *defined* without per-layer rescaling, matching the convention of
            # random-feedback learning (Lillicrap et al. 2016, "Random synaptic feedback
            # weights support error backpropagation for deep learning", Nat Commun
            # 7:13276), which has no such factor because it has nothing to compensate.
            return 1.0
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
            xs.append(self.act(i + 1, z))
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

        # x_0 is clamped, so layer 1's bottom-up drive never changes. On MNIST that is the
        # 784x256 matmul — by far the dominant cost — so hoisting it out of the loop is
        # worth more than every other optimisation here combined.
        bottom_up_1 = x0 @ self.W[0].T + self.b[0]

        steps = 0
        for steps in range(1, n_steps + 1):
            drive = []
            for k in range(1, free_top + 1):
                if k == 1:
                    z = bottom_up_1
                else:
                    z = xs[k - 1] @ self.W[k - 1].T + self.b[k - 1]
                if k < self.L:
                    z = z + self.gamma * (xs[k + 1] @ self.feedback(k))
                drive.append(self.act(k, z))

            delta = 0.0
            for k in range(1, free_top + 1):
                new = xs[k] + dt * (drive[k - 1] - xs[k])
                delta = max(delta, (new - xs[k]).abs().max().item())
                xs[k] = new
            if delta < tol:
                break
        return xs, steps

    @classmethod
    def multi_head(
        cls,
        trunk: list[int],
        head_size: int,
        n_heads: int,
        **kwargs,
    ) -> list["LayeredNet"]:
        """Build ``n_heads`` networks that **share every trunk parameter** but have their
        own output layer — the standard task-incremental setup.

        Sharing is by object identity, not by copying: each returned net holds the very
        same ``Parameter`` objects for the trunk, so the in-place updates the training loop
        applies to one net are immediately visible to all the others. That is what makes
        the trunk accumulate across tasks while each head stays private.

        Args:
            trunk: shared layer widths, e.g. ``[784, 256]``.
            head_size: width of each private output layer.
            n_heads: one per task.
        """
        sizes = list(trunk) + [head_size]
        if not kwargs.get("tied", True):
            raise NotImplementedError(
                "multi_head does not share untied feedback matrices across heads yet; "
                "the untied prototype runs on the domain-incremental protocol, which uses "
                "a single shared net"
            )
        base_seed = kwargs.pop("seed", None)
        nets = [
            cls(sizes, seed=None if base_seed is None else base_seed + i, **kwargs)
            for i in range(n_heads)
        ]
        for net in nets[1:]:
            for i in range(len(trunk) - 1):
                net.W[i] = nets[0].W[i]
                net.b[i] = nets[0].b[i]
        return nets

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
