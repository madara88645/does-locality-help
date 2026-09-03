"""Is the residual real, or noise? The same three points at 15 seeds instead of 5.

docs/exploratory-plasticity.md found the untied arm sitting 2.34 pp below the plasticity
curve at its own trunk movement, on 5 seeds -- about two seed standard deviations, which
it called suggestive rather than established. Three separate attempts to find a mechanism
behind it then failed (damage_direction.py, trunk_interference.py, and the head/trunk
split in where_is_the_damage.py, which located the effect without explaining it). The
leading hypothesis became that there was no effect to explain.

RESULT (15 seeds): the residual does not shrink.

    arm               trunk    forgetting     n
    UNTIED            2.24%    45.00 ±0.82    15
    tied, s = 0.05    1.93%    46.69 ±0.87    15
    tied, s = 0.10    2.84%    48.68 ±0.65    15

    curve at 2.24%    47.38
    untied measured   45.00
    residual          +2.38 pp

Tripling the seeds moved the estimate from 2.34 pp to 2.38 pp while the per-seed spread
stayed near 0.8. Noise does not behave like that: with n = 15 the mean carries about
0.21 pp of standard error, and the gap is an order of magnitude larger. The hypothesis
that the residual was sampling noise is refuted.

So the effect is real and its mechanism is unknown -- a more useful state than either
half alone, and the honest one to record.

The untied arm's trunk movement is bracketed by the two tied points, so the curve is
interpolated between adjacent measurements rather than extrapolated. The curve is
non-linear across its full range, but linear interpolation between two neighbouring
points 0.91 percentage points apart is defensible.
"""
import importlib.util, torch
spec = importlib.util.spec_from_file_location("pc", "experiments/plasticity_curve.py")
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)

torch.set_default_dtype(torch.float32)
from forgetlab.data.split_mnist import load_split_mnist
tasks = load_split_mnist(train_per_task=1000)
SEEDS = list(range(15))

print(f"{'arm':>22} | {'trunk':>7} | {'forgetting':>14} | {'n'}")
print("-" * 58)
res = {}
for label, untied, thr in [("UNTIED", True, None),
                           ("tied, s = 0.05", False, 0.05),
                           ("tied, s = 0.1", False, 0.1)]:
    rows = [pc.run(tasks, s, untied=untied, throttle=thr) for s in SEEDS]
    f = [r[0] for r in rows]; t = pc.mean([r[3] for r in rows])
    res[label] = (t, pc.mean(f), pc.std(f))
    print(f"{label:>22} | {t:>6.2f}% | {pc.mean(f):>6.2f} ±{pc.std(f):>4.2f} | {len(SEEDS)}")

# interpolate the curve at the untied arm's trunk movement
(tu, fu, su) = res["UNTIED"]
(t5, f5, _) = res["tied, s = 0.05"]
(t1, f1, _) = res["tied, s = 0.1"]
slope = (f1 - f5) / (t1 - t5)
pred = f5 + (tu - t5) * slope
print(f"\ncurve at untied's trunk movement ({tu:.2f}%): {pred:.2f}")
print(f"untied measured:                        {fu:.2f}")
print(f"residual: {pred - fu:+.2f} pp   (untied seed SD = {su:.2f}, so {(pred-fu)/su:.1f} SD)")
