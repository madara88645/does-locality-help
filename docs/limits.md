# Which limit does which theorem need?

This document exists because the two best-known "local learning rule ≈ backpropagation"
theorems are routinely conflated. They both involve a small parameter going to zero, but
**the small parameter is not the same thing in the two papers**, and getting this wrong
means implementing the wrong algorithm under the right name.

Everything below is read off the primary sources, with equation numbers.

---

## 1. Xie & Seung 2003 — Contrastive Hebbian Learning

> Xie, X. & Seung, H.S. (2003). *Equivalence of Backpropagation and Contrastive Hebbian
> Learning in a Layered Network.* Neural Computation 15(2), 441–454.

### The network

A multilayer perceptron with `L+1` layers (layer 0 = input, layer L = output), feedforward
weights `W_k` from layer `k-1` to layer `k`, **plus feedback connections that are the
transpose of the feedforward weights scaled by a factor γ**:

```
feedback from layer k back to layer k-1  =  γ · W_kᵀ
```

So the weights are **tied**: there is exactly one weight matrix per layer, used forward as
`W_k` and backward as `γ W_kᵀ`. This is not an implementation convenience — the proof
depends on it.

### What γ is

**γ scales the strength of the feedback connections.** It is *not* a nudge on the output.
The paper's own section heading is "Equivalence in the Limit of **Weak Feedback**", and it
states the regime as `γ ≪ 1`, with the closing result holding **as γ → 0**.

### The two phases

- **Free phase** — input `x_0` held fixed, output free. Run the dynamics (Eq. 2.5)

  ```
  dx_k/dt + x_k  =  f_k( W_k x_{k-1}  +  γ W_{k+1}ᵀ x_{k+1}  +  b_k )
  ```

  to a fixed point. The result is the free state `x̌_k`.

- **Clamped phase** — output layer **hard-clamped** at the desired value `d` (`x_L = d`),
  same dynamics run on `k = 1 … L-1` to a fixed point. The result is the clamped state `x̂_k`.

There is **no soft/weak clamping and no β anywhere in this paper.** The output clamp is hard.

### The weight update (Eq. 2.8)

```
ΔW_k  =  η · γ^(k-L) · ( x̂_k x̂_{k-1}ᵀ  −  x̌_k x̌_{k-1}ᵀ )
```

Clamped correlation minus free correlation — the "contrastive" part. Note the prefactor.

### ⚠️ The layer-wise learning-rate factor `γ^(k-L)` — the part that silently breaks implementations

Since `k ≤ L` and `γ ≪ 1`, the factor `γ^(k-L) = (1/γ)^(L-k)` **grows exponentially with
distance from the output layer**:

| layer | prefactor | with γ = 0.1 |
|---|---|---|
| `k = L` (output) | `γ⁰ = 1` | 1 |
| `k = L-1` | `γ⁻¹ = 1/γ` | 10 |
| `k = L-2` | `γ⁻² = 1/γ²` | 100 |

**Why it is there.** Step 3 of the proof (Eq. 3.3) shows that the difference between the
clamped and free states decays exponentially as you move away from the output:

```
δx_k  =  γ^(L-k) · y_k  +  O( γ^(L-k+1) )
```

where `y_k` is exactly backpropagation's error signal at layer `k` (Eq. 2.3). Because the
feedback is weak, the clamping signal is attenuated by one factor of γ per layer it travels.
The `γ^(k-L)` prefactor in the update rule **exactly cancels that attenuation** (Eq. 3.16 → 3.19),
leaving

```
ΔW_k  =  η · y_k · x̌_{k-1}ᵀ  +  O(γ)
```

which is backpropagation's update (Eq. 2.4).

**Implementation consequence:** if you use one uniform learning rate for all layers, CHL will
**not** match backprop. Layers far from the output will be updated exponentially too weakly,
the equivalence test will fail, and the failure will look like "CHL just doesn't work" rather
than "I dropped a term". This is the single most likely silent bug in this project.

This is also the honest weakness of the theorem, and it must be stated in the README: the
equivalence is bought with **infinitesimal feedback weights and exponentially growing
learning rates for remote layers**. That is biologically awkward and it is a fair criticism
of the result — not something to hide.

### Other conditions

- **Linear output units** are required for equivalence to squared-error backprop.
  (Remark, p. 448: with *sigmoid* output units, CHL is instead equivalent to backprop on the
  **cross-entropy** cost `−dᵀlog(x_L) − (1−d)ᵀlog(1−x_L)`, Eq. 3.20. Worth noting — this is
  the same cross-entropy-as-surprise object the book chapter discusses.)
- Transfer functions `f_k` monotonically increasing.
- For nonlinear output units the CHL and backprop update directions are no longer identical,
  but the paper states they stay **within 90 degrees** of each other.

### Phase ordering (Section 4) — a real implementation detail

CHL performs gradient descent on the contrastive function `C(W) = E(x̂) − E(x̌)`, the
difference of the network's Lyapunov function between the clamped and free phases. `C(W) ≥ 0`
is only guaranteed if `x̂` is the global minimum, or, when it is only a local minimum, if the
free state lies in its basin of attraction. The paper's recommended strategy:

> settle the **clamped** phase first, then run the **free** phase **without resetting the
> hidden neurons**.

Do this. Running the two phases from independent random initialisations is a different
algorithm with no guarantee that the cost function is even non-negative.

---

## 2. Scellier & Bengio 2017 — Equilibrium Propagation

> Scellier, B. & Bengio, Y. (2017). *Equilibrium Propagation: Bridging the Gap Between
> Energy-Based Models and Backpropagation.* arXiv:1602.05179.

### What β is

A **total energy function** is formed by adding the cost to the internal energy (Eq. 20):

```
F(θ, v, β, s)  :=  E(θ, v, s)  +  β · C(θ, v, s)
```

β is called the **influence parameter**. It scales the **cost term** — i.e. how strongly the
output units are pulled toward the target. `C = ½‖y − ŷ‖²`.

### The two phases

- **Free phase**: `β = 0`. Output units are entirely free. Settles to free fixed point `s⁰`.
- **Weakly clamped phase**: `β` set to a small positive value. The external force on the
  output unit is `β(y_i − ŷ_i)` (Eq. 9) — a **nudge**, not a clamp. Settles to `s^β`.

The paper's own words: the output units are *slightly nudged* toward their target. The output
is **never hard-clamped**. This is the core difference from Xie & Seung.

### The limit and the update (Theorem 1, Eq. 22; update Eq. 10)

```
∂J/∂θ  =  lim_{β→0}  (1/β) · ( ∂F/∂θ(β, s^β)  −  ∂F/∂θ(0, s⁰) )

ΔW_ij  ∝  (1/β) · ( ρ(u_i^β) ρ(u_j^β)  −  ρ(u_i⁰) ρ(u_j⁰) )
```

Note: **one single `1/β` prefactor for the whole network** — not a per-layer, depth-dependent
factor. This is a concrete, checkable difference from CHL's `γ^(k-L)`.

Weights are symmetric (`W_ij = W_ji`, used to derive Eq. 8; "the case of tied weights studied
here", Eq. 13).

The paper explicitly distinguishes itself: *"our learning rule is different from the Boltzmann
machine learning rule and the contrastive Hebbian learning rule."*

---

## 3. The answer, side by side

| | **Xie & Seung 2003 (CHL)** | **Scellier & Bengio 2017 (EqProp)** |
|---|---|---|
| small parameter | **γ** | **β** |
| what it scales | strength of the **feedback connections** (`γ W_kᵀ`) | strength of the **nudge** on the output (`βC` in the energy) |
| output in 2nd phase | **hard-clamped** at target | **weakly nudged** toward target |
| limit for equivalence | γ → 0 | β → 0 |
| learning-rate structure | **per-layer** `γ^(k-L)`, exponential in depth | **single** `1/β` for the whole network |
| weight structure | feedback = tied transpose, scaled by γ | symmetric, `W_ij = W_ji` |
| equivalent to BP on | squared error (linear outputs) / cross-entropy (sigmoid outputs) | the cost `C` defining `F` |

**Conclusion for this project.** The framing "CHL converges to the backprop gradient as γ→0"
is **correct** for Xie & Seung 2003 — γ is the feedback strength, and the limit is genuinely
required. It is *not* a mistaken import of EqProp's β. But the two must not be described
interchangeably: they are different small parameters attached to different mechanisms, and
CHL additionally carries a depth-dependent learning-rate factor that EqProp does not.

Movellan (1990) is sometimes cited for a CHL/backprop equivalence with no small parameter —
that result is for networks with **a single layer only**. For a layered network with hidden
layers, the weak-feedback limit is what makes the equivalence go through.

---

## 4. What this fixes in the implementation

1. Tie the weights. One `W_k` per layer, used as `W_k` forward and `γ W_kᵀ` backward.
2. Apply the **per-layer** `γ^(k-L)` factor in the CHL update. Not a uniform learning rate.
3. Hard-clamp the output in the clamped phase. Do not implement a soft nudge and call it CHL.
4. Settle the **clamped phase first**, then the free phase **without resetting hidden units**.
5. Use **linear** output units for the equivalence test against squared-error backprop.
6. State the theorem's cost honestly in the README: infinitesimal feedback plus exponentially
   growing per-layer learning rates.
7. Report gradient cosine similarity **per layer**, never as a single aggregate number.
   *What Accuracy and Gradient Cosine Miss: Evaluating Feedback Alignment via Scale Stability,
   Reference Validity, and Depth Utility* (arXiv:2606.21126, June 2026) shows
   that aggregate cosine suffers "aggregation collapse" — it masks layerwise heterogeneity
   when credit concentrates at one end of the network, and gives no signal of failure in any
   case they audited. Given that CHL's whole depth story is the `γ^(k-L)` factor, a single
   averaged cosine would hide exactly the effect this project is trying to measure.
8. But per-layer cosine is **not sufficient on its own** — see below.

### Measured: three checks are needed, not one

Cosine similarity is invariant to positive scaling, so a per-layer cosine cannot see a
per-layer rescaling. Deleting the `γ^(k-L)` factor entirely leaves every per-layer cosine
**bit-for-bit unchanged**. Measured on a `[6,5,4,3]` net at γ = 0.05:

| check | correct | `γ^(k-L)` factor deleted |
|---|---|---|
| per-layer cosine | 0.99975, 0.99993, 0.99081 | 0.99975, 0.99993, 0.99081 |
| global (concatenated) cosine | 0.9971 | **0.5822** |
| per-layer norm ratio vs backprop | 1.009, 1.000, 1.021 | **0.0025, 0.0500, 1.021** |

The broken norm ratios are exactly `γ², γ¹, γ⁰` — the attenuation of Eq. 3.3 left
uncancelled, recovered numerically.

This is the mirror image of arXiv:2606.21126's result. That paper shows an *aggregate*
cosine hides layerwise **direction** error; the table above shows a *per-layer* cosine
hides cross-layer **scale** error. Neither is sufficient alone. `tests/` therefore checks
three independent things:

1. **per-layer cosine** — direction (blind to per-layer scale)
2. **per-layer norm ratio** — the `γ^(k-L)` factor (blind to direction)
3. **O(γ) error rate** — the theorem itself: halving γ must halve the relative error

Check 3 is the strongest. An implementation can sit near backprop by accident; reproducing
the predicted *rate* is much harder to fake. Measured relative error, layer 1:
0.0096 → 0.0048 → 0.0024 for γ = 0.02 → 0.01 → 0.005. Exactly linear.

### ⚠️ Measured: how close is CHL to backprop at the γ this project actually uses?

The section above shows the equivalence holds *in the limit*. That is not the same question
as: at the γ the experiments actually ran with, how different is the update?

Measured on the experiment's own architecture (`784 → 256 → 2`), its settling settings
(64 steps, `dt = 0.5`, `tol = 1e-8`) and a real Split-MNIST batch — not the tiny float64
net the equivalence tests use:

| γ | per-layer cosine | per-layer norm ratio |
|---|---|---|
| 0.01 | 1.0000, 1.0000 | 0.9995, 0.9997 |
| 0.1 *(used in every reported run)* | 1.0000, 0.9996 | 0.9956, 0.9976 |
| 0.3 | 0.9997, 0.9968 | 0.9861, 0.9938 |
| 0.5 | 0.9990, 0.9910 | 0.9754, 0.9913 |

**This is a limitation of the whole comparison, and it is stated here rather than left for
a reviewer to find.**

Across the entire range swept — a 50× change in feedback strength — CHL's update stays
within about 2.5% of backprop's in magnitude and above 0.99 cosine in direction. The
continual-learning result has to be read against that: two rules computing nearly the same
update forgot nearly the same amount. The measured update deviation (1–2.5%) and the
measured forgetting difference (0–1 pp) are consistent with each other, but neither is
evidence about locality.

So the honest scope of the null result is narrower than "a local learning rule forgets as
much as backprop". It is:

> A local rule that is *constructed to approximate the backprop gradient*, and measurably
> does so to within a few percent, also forgets like backprop.

That is a much weaker statement, and close to a corollary of the equivalence theorem rather
than an independent finding. The experiment cannot separate "locality does not help" from
"this rule was not different enough from backprop to tell".

**What would make it a real test.** A local rule whose update is *not* engineered to match
backprop. The obvious candidate is already discussed in the next section: untie the feedback
weights. Feedback alignment still learns, is still local, and its updates genuinely differ
from the backprop gradient — so a forgetting comparison against backprop would carry
information that this one does not. Everything else in the repository (protocols, metrics,
seeds, pre-registration discipline) transfers unchanged.

---

## 5. Why tied weights, and what happens without them

The feedback path must be the transpose of the forward path because that is what backprop's
chain rule does: the error is propagated backward by multiplying by `W_kᵀ` (Eq. 2.3). CHL's
feedback connections are `γ W_kᵀ`, so the disturbance created by clamping travels back along
exactly the same route the signal came forward. Untie them and the settling measures the
sensitivity of a *different* network.

Note, however, that untying does not destroy learning — it changes the algorithm:

> Lillicrap, T., Cownden, D., Tweed, D. & Akerman, C. (2016). *Random synaptic feedback
> weights support error backpropagation for deep learning.* Nature Communications 7:13276.

**Feedback Alignment** replaces `W_kᵀ` with a fixed random matrix and still learns, because
the forward weights come to *align* with the random feedback — the network adapts to whatever
feedback channel it is given. This is a genuinely surprising result and worth citing in the
README, but it is a **different algorithm proving a different claim**: "still learns" is not
"equals the backprop gradient". This project reproduces an exact equivalence theorem, so the
weights stay tied. Feedback Alignment is a v0.2 comparison at most, not a variant of CHL.
