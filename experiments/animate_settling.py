"""Animate the two settlings that contrastive Hebbian learning subtracts.

CHL never computes a gradient. It measures one, by watching a network settle twice and
taking the difference. That "measure by settling twice" idea is the one piece of this
project that is easy to state in words and hard to actually picture, so this script
draws it directly from the project's own relaxation code rather than describing it.

What gets animated, on a network small enough to look at unit by unit:

  * FREE PHASE   -- ``net.relax(x0, target=None)``: the input is held fixed, everything
                    else (hidden units, output units) settles to wherever the current
                    weights take it.
  * CLAMPED PHASE -- ``net.relax(x0, target=target)``: the input *and* the output are
                    both held fixed at the target; only the hidden units are free to
                    settle, pulled by the ``gamma * W^T`` feedback term in
                    ``forgetlab/layers.py``'s Eq. 2.5.
  * DIFFERENCE    -- ``clamped - free`` at matching settling steps. This is not a
                    metaphor: for the output layer with a single example it is exactly
                    ``db`` from ``forgetlab/rules/chl.py`` (``scale * (clamped - free)``,
                    and ``scale = gamma**(L-L) = 1`` at the output layer). Watching it
                    stop moving *is* watching the weight update converge.

Both phases are settled independently from the same starting state (hidden and output
units at zero) so they can be plotted on a shared step axis and visibly pull apart --
this is a deliberate change from ``LayeredNet.settle_both_phases``, which chains the free
phase *after* the clamped one (seeded from the clamped fixed point, per Xie & Seung Sec.
4) because that keeps the contrastive cost well-behaved during real training. Here we
only want to look at the two fixed points, not train anything, so starting both from the
same point is more honest to *this script's* purpose: it is what lets a single frame show
"the gap between these two curves" instead of "one curve, then a second curve pinned to
where the first one ended up". Every number drawn still comes from real, unmodified calls
to ``LayeredNet.relax`` -- see the sanity check in ``main()``, which confirms that taking
that relaxation one Euler step at a time (to keep every intermediate state for the
animation) reproduces exactly what a single multi-step ``relax()`` call returns.

The feedback strength ``gamma`` is set higher here (0.3) than the near-zero values used
where this repo checks CHL against backprop (``docs/limits.md``) -- purely so the
settling motion is visible on a human timescale. Equivalence to backprop is not the point
of this picture; watching two states relax and pull apart is.

Run with::

    uv run python experiments/animate_settling.py

Writes ``results/settling.gif``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from forgetlab.layers import LayeredNet

# --------------------------------------------------------------------------- simulation
#
# Deliberately tiny: 3 input units (not drawn -- they never move, they're the clamp),
# 4 hidden units, 2 output units. Small enough that every line in the plot is a specific,
# nameable unit, not a blur.

SIZES = [3, 4, 2]
GAMMA = 0.3
DT = 0.5
N_SETTLE_STEPS = 30  # empirically settled well before this -- see the printed deltas
SEED = 0

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "settling.gif"

OUT_COLORS = ["#1f6feb", "#cf222e"]  # one per output unit, reused across every panel
HIDDEN_COLOR = "#8b949e"
DIFF_HIDDEN_COLOR = "#2f9e44"


def build_example() -> tuple[LayeredNet, torch.Tensor, torch.Tensor]:
    """A single (batch=1) synthetic example on a fresh, untrained network.

    Untrained on purpose: the whole point of the picture is that the free settling does
    *not* land on the target yet -- that gap is what training is for.
    """
    torch.set_default_dtype(torch.float32)
    gen = torch.Generator().manual_seed(SEED)
    x0 = torch.randn(1, SIZES[0], generator=gen)
    target = torch.randn(1, SIZES[-1], generator=gen)
    net = LayeredNet(SIZES, gamma=GAMMA, seed=SEED)
    return net, x0, target


def _zero_init(x0: torch.Tensor) -> list[torch.Tensor]:
    """Hidden and output units start at zero; only the input is the real state."""
    return [x0.clone()] + [torch.zeros(1, s) for s in SIZES[1:]]


def settle_trajectory(
    net: LayeredNet,
    x0: torch.Tensor,
    target: torch.Tensor | None,
    n_steps: int,
) -> list[list[torch.Tensor]]:
    """Every intermediate state ``net.relax`` visits on its way to a fixed point.

    ``LayeredNet.relax`` only returns the *final* state. To animate the settling we need
    each step along the way, so this calls it repeatedly with ``n_steps=1`` and feeds the
    result back in as ``init``. Because the Euler update inside ``relax`` depends only on
    the current state (not on how it got there), this reproduces exactly the trajectory a
    single ``n_steps=n_steps`` call would take -- confirmed by the assertion in ``main()``.
    """
    state = _zero_init(x0)
    history = [state]
    for _ in range(n_steps):
        state, _ = net.relax(x0, target=target, init=state, n_steps=1, dt=DT, tol=0.0)
        history.append(state)
    return history


def _stack(history: list[list[torch.Tensor]], layer: int) -> np.ndarray:
    """Stack one layer's state across every step into a ``(steps+1, width)`` array."""
    return torch.cat([state[layer] for state in history], dim=0).numpy()


# --------------------------------------------------------------------------- rendering


def _pad(lo: float, hi: float, frac: float = 0.10) -> tuple[float, float]:
    span = hi - lo if hi > lo else 1.0
    return lo - frac * span, hi + frac * span


def render_frames(
    free_hist: list[list[torch.Tensor]],
    clamped_hist: list[list[torch.Tensor]],
    target: torch.Tensor,
) -> list[Image.Image]:
    n = len(free_hist) - 1
    free_out, clamped_out = _stack(free_hist, -1), _stack(clamped_hist, -1)
    free_hid, clamped_hid = _stack(free_hist, 1), _stack(clamped_hist, 1)
    diff_out = clamped_out - free_out
    diff_hid_norm = np.linalg.norm(clamped_hid - free_hid, axis=1)
    target_np = target.numpy().flatten()
    n_hidden, n_out = free_hid.shape[1], free_out.shape[1]

    out_lo, out_hi = _pad(
        min(free_out.min(), clamped_out.min(), target_np.min()),
        max(free_out.max(), clamped_out.max(), target_np.max()),
    )
    hid_lo, hid_hi = _pad(
        min(free_hid.min(), clamped_hid.min()), max(free_hid.max(), clamped_hid.max())
    )
    diff_lo, diff_hi = _pad(
        min(diff_out.min(), diff_hid_norm.min(), 0.0),
        max(diff_out.max(), diff_hid_norm.max()),
    )

    # Hold a few frames at the start (so the shared zero starting point registers) and
    # several at the end (so the stabilised difference -- the punchline -- has time to
    # read) before the loop cuts back to frame zero.
    schedule = [0] * 4 + list(range(1, n + 1)) + [n] * 9

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#57606a",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), dpi=100)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.15, wspace=0.32)

    # Created once and mutated with `set_text` every frame: figure-level text (unlike an
    # axis's own artists) survives `ax.clear()`, so re-calling `fig.suptitle`/`fig.text`
    # inside the loop would pile up a new label on top of the last one every frame.
    fig.suptitle(
        "Contrastive Hebbian learning settles the network twice — "
        "the difference between the two settlings is the weight update",
        fontsize=10.5, y=0.965,
    )
    footer = fig.text(0.5, 0.015, "", ha="center", fontsize=7.5, color="#57606a")

    frames: list[Image.Image] = []
    for i in schedule:
        for ax in axes:
            ax.clear()
        x = np.arange(0, i + 1)

        # -- panel 1: free phase ------------------------------------------------
        ax = axes[0]
        for h in range(n_hidden):
            ax.plot(
                x, free_hid[: i + 1, h], color=HIDDEN_COLOR, lw=1.0, alpha=0.65,
                label="hidden units" if h == 0 else None,
            )
        for o in range(n_out):
            ax.axhline(target_np[o], color=OUT_COLORS[o], lw=1.0, ls=":", alpha=0.5)
            ax.plot(
                x, free_out[: i + 1, o], color=OUT_COLORS[o], lw=2.3,
                label=f"output {o + 1}",
            )
            ax.plot(x[-1], free_out[i, o], "o", color=OUT_COLORS[o], ms=5, zorder=5)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(min(out_lo, hid_lo), max(out_hi, hid_hi))
        ax.set_title("free phase")
        ax.set_xlabel("settling step")
        ax.text(
            0.5, 1.16, "output left alone · dotted line = target it hasn't reached",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6, color="#57606a",
        )
        ax.legend(loc="lower right", frameon=False, ncols=1)

        # -- panel 2: clamped phase ----------------------------------------------
        ax = axes[1]
        for h in range(n_hidden):
            ax.plot(x, clamped_hid[: i + 1, h], color=HIDDEN_COLOR, lw=1.0, alpha=0.65)
        for o in range(n_out):
            ax.plot(x, clamped_out[: i + 1, o], color=OUT_COLORS[o], lw=2.3)
            ax.plot(x[-1], clamped_out[i, o], "o", color=OUT_COLORS[o], ms=5, zorder=5)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(min(out_lo, hid_lo), max(out_hi, hid_hi))
        ax.set_title("clamped phase")
        ax.set_xlabel("settling step")
        ax.text(
            0.5, 1.16, "output pinned to target · only hidden units move",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6, color="#57606a",
        )

        # -- panel 3: the difference ----------------------------------------------
        ax = axes[2]
        ax.axhline(0.0, color="#8b949e", lw=0.8)
        ax.plot(
            x, diff_hid_norm[: i + 1], color=DIFF_HIDDEN_COLOR, lw=1.4, ls="--",
            label="‖hidden diff‖",
        )
        for o in range(n_out):
            ax.plot(
                x, diff_out[: i + 1, o], color=OUT_COLORS[o], lw=2.3,
                label=f"output {o + 1} diff",
            )
            ax.plot(x[-1], diff_out[i, o], "o", color=OUT_COLORS[o], ms=5, zorder=5)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(diff_lo, diff_hi)
        ax.set_title("clamped − free", color="#1a7f37")
        ax.set_xlabel("settling step")
        ax.text(
            0.5, 1.16, "the learning signal: CHL's ΔW, Δb (Eq. 2.8)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3,
            color="#1a7f37", fontweight="bold",
        )
        ax.legend(loc="upper right", frameon=False, ncols=1)

        footer.set_text(
            f"step {i}/{n}   ·   single synthetic example, batch=1   ·   "
            f"γ={GAMMA}   ·   LayeredNet.relax()"
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))

    plt.close(fig)
    return frames


def save_gif(frames: list[Image.Image], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Match the hold/transition split used to build `schedule` in render_frames so the
    # start and end of the loop linger a little longer than the settling itself.
    n_hold_start, n_hold_end = 4, 9
    n_transition = len(frames) - n_hold_start - n_hold_end
    durations = (
        [140] * n_hold_start + [85] * n_transition + [140] * n_hold_end
    )

    quantized = [
        f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
        for f in frames
    ]
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    net, x0, target = build_example()

    # Sanity check: the step-chained trajectory used for the animation must land on the
    # exact same fixed point a single direct `relax(n_steps=N)` call reaches. If this
    # ever fails, the frames below are not a faithful trajectory of the real dynamics and
    # the animation must not be trusted.
    direct, _ = net.relax(
        x0, target=None, init=_zero_init(x0), n_steps=N_SETTLE_STEPS, dt=DT, tol=0.0
    )
    chained = settle_trajectory(net, x0, None, N_SETTLE_STEPS)[-1]
    for a, b in zip(direct, chained):
        assert torch.allclose(a, b, atol=1e-6), (
            "step-chained relaxation diverged from a direct relax() call -- "
            "the animation would not reflect the real dynamics"
        )
    print("sanity check passed: chained single-steps == one direct relax() call")

    free_hist = settle_trajectory(net, x0, None, N_SETTLE_STEPS)
    clamped_hist = settle_trajectory(net, x0, target, N_SETTLE_STEPS)

    final_diff = free_hist[-1][-1] - clamped_hist[-1][-1]
    print(f"free output settled to   {free_hist[-1][-1].tolist()}")
    print(f"clamped output (target)  {clamped_hist[-1][-1].tolist()}")
    print(f"stabilised output diff   {(-final_diff).tolist()}")

    frames = render_frames(free_hist, clamped_hist, target)
    save_gif(frames, OUT_PATH)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.1f} KiB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
