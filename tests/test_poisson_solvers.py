"""Checks on the three intensity updates in Cost_Spatial.

The thesis gives two ways to recover I from G (Section 6.5): the iterative
Richardson step (Eq. 6.61) and the exact frequency-domain solution
(Eq. 6.64-6.65). We had only the first, which is why our reconstructions were
edge maps: its convergence factor per spatial frequency is (1 - delta_GI|k|^2),
so at delta_GI = 0.05 and 75 iterations nothing below |k|^2 ~ 0.27 forms.

    python test_poisson_solvers.py

1. correctness — on a synthetic image with a known gradient, the exact solvers
   must reproduce it (up to the additive constant the Poisson problem leaves
   free); the iterative one must not, at the step count we actually use.
2. residual   — the exact solvers must attain a lower ||grad I - G||^2 than the
   iterative update, since they minimise that exact objective in closed form.
3. spectrum   — the exact solvers must carry more low-frequency energy, which
   is the visible difference between a photograph and an emboss.
"""
import sys
import numpy as np

from interacting_maps.network_dissertation import Cost_Spatial, Quantity

H, W = 180, 240
FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))
    if not ok:
        FAILURES.append(name)


def forward_grad(I):
    """The same forward differences Cost_Spatial uses."""
    gx = np.zeros_like(I); gx[:, :-1] = I[:, 1:] - I[:, :-1]
    gy = np.zeros_like(I); gy[:-1, :] = I[1:, :] - I[:-1, :]
    return np.stack([gx, gy], axis=-1)


def make_cost(poisson, I0, G):
    q = {'I': Quantity((H, W), 'I'), 'G': Quantity((H, W, 2), 'G')}
    q['I'].value = I0.copy()
    q['G'].value = G.copy()
    return Cost_Spatial(q, delta_IG=0.0, delta_GI=1.0, poisson=poisson), q


def run(poisson, I0, G, n_iters):
    cost, q = make_cost(poisson, I0, G)
    for _ in range(n_iters):
        q['I'].reset_gradient(); q['G'].reset_gradient()
        cost.compute_and_send_gradients()
        q['I'].update(1.0)                      # delta_GI carries the rate
    return q['I'].value


def residual(I, G):
    return float(np.mean((forward_grad(I) - G) ** 2))


def low_freq_share(I, radius=10):
    F = np.abs(np.fft.fftshift(np.fft.fft2(I - I.mean())))
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - H // 2, xx - W // 2)
    return float(F[r < radius].sum() / F.sum())


# a scene with structure at every scale: broad ramp + waves + a sharp edge
yy, xx = np.mgrid[0:H, 0:W].astype(float)
yy /= H - 1.0
xx /= W - 1.0
truth = (2.0 * xx                                            # low frequency
         + np.sin(6 * np.pi * yy) * np.cos(4 * np.pi * xx)
         + (np.hypot(yy - 0.5, xx - 0.5) < 0.2) * 1.5)       # sharp edge
truth -= truth.mean()
G_true = forward_grad(truth)
I0 = np.random.default_rng(0).standard_normal((H, W)) * 0.01

# Each solver at the rate it is actually valid at: the Richardson step of
# Eq. 6.61 is only stable for delta_GI < ~1/4 (the 5-point Laplacian has
# spectral radius 8), and 0.05 is the value the pipeline runs. The exact
# solvers have no such limit, so they take delta_GI = 1 and land in one step.
RATE = {'iterative': 0.05, 'fft': 1.0, 'dct': 1.0}

print(__doc__.strip().splitlines()[0])
print(f"\nsynthetic scene {H}x{W}, exact gradient, 75 iterations, "
      f"delta_GI = 0.05 iterative / 1.0 exact\n")

print('1. can each solver recover a known image from its exact gradient?')
res = {}
for mode in ('iterative', 'fft', 'dct'):
    try:
        cost, q = make_cost(mode, I0, G_true)
        cost.delta_GI = RATE[mode]
        for _ in range(75):
            q['I'].reset_gradient()
            cost.compute_and_send_gradients()
            q['I'].update(1.0)
        I = q['I'].value
    except ImportError as e:                    # dct needs scipy
        print(f"  [skip] {mode}: {e}")
        continue
    I = I - I.mean()                            # the free additive constant
    rms = float(np.sqrt(np.mean((I - truth) ** 2))) / truth.std()
    res[mode] = (I, rms)
    print(f"       {mode:<9} relative RMS vs truth = {rms:.4f}")

# The periodic transform cannot represent a scene whose left and right edges
# differ (here a ramp), so 'fft' pays a wrap-around low-frequency error even
# though it exactly minimises its own periodic objective. 'dct' assumes
# reflecting boundaries, which is what the zero-gradient border convention in
# Cost_Spatial actually implies, and is exact.
check('dct recovers the image exactly', res['dct'][1] < 1e-6,
      f"relative RMS = {res['dct'][1]:.2e}")
check('fft recovers it up to the periodic wrap artefact',
      res['fft'][1] < 0.8, f"relative RMS = {res['fft'][1]:.3f}")
check('both exact solvers beat the iterative update',
      max(res['fft'][1], res['dct'][1]) < res['iterative'][1])
check('iterative leaves the image unrecovered in 75 iterations',
      res['iterative'][1] > 0.5, f"relative RMS = {res['iterative'][1]:.3f}")

print('\n2. does each solver minimise ||grad I - G||^2 ?')
for m, (I, _) in res.items():
    print(f"       {m:<9} residual = {residual(I, G_true):.3e}")
check('exact residual below iterative',
      all(residual(res[m][0], G_true) < residual(res['iterative'][0], G_true)
          for m in res if m != 'iterative'))

print('\n3. low-frequency share of |FFT| (what makes it look like a photo)')
print(f"       {'truth':<9} {low_freq_share(truth):.3f}")
for m, (I, _) in res.items():
    print(f"       {m:<9} {low_freq_share(I):.3f}")
check('exact solvers carry more low frequency',
      all(low_freq_share(res[m][0]) > low_freq_share(res['iterative'][0])
          for m in res if m != 'iterative'))

print('\n4. blending: delta_GI must interpolate, not jump')
half = run('fft', I0, G_true, 1)
cost, q = make_cost('fft', I0, G_true)
cost.delta_GI = 0.5
q['I'].reset_gradient(); cost.compute_and_send_gradients(); q['I'].update(1.0)
star = run('fft', I0, G_true, 1)
check('delta_GI=0.5 lands halfway to the exact solution',
      np.allclose(q['I'].value, 0.5 * I0 + 0.5 * star, atol=1e-9))
check('the exact solve preserves the mean of I (gauge)',
      abs(star.mean() - I0.mean()) < 1e-9, f"{star.mean():.2e} vs {I0.mean():.2e}")

print()
if FAILURES:
    print(f"FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print('all checks passed')
