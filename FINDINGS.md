# InteractingMaps — Validation Findings Log

> Running log of experimental findings on the validation pipeline. Each entry
> is dated and links to the test script that produced the result. Use this
> as the single source of truth when arguing about what does / doesn't move
> the needle on `validation.py`'s accuracy.

---

## Table of contents

1. [TL;DR — what we know so far](#1-tldr--what-we-know-so-far)
2. [F1 — Sensor size mismatch (264×221 vs 240×180)](#f1--sensor-size-mismatch-264221-vs-240180)
3. [F2 — Stability value clipping (I, G, F, R bounds) is inert](#f2--stability-value-clipping-i-g-f-r-bounds-is-inert)
4. [F3 — Event-frame `clip_value` and `normalise` sweep](#f3--event-frame-clip_value-and-normalise-sweep)
5. [F4 — Per-frame inference collapses to a near-fixed point](#f4--per-frame-inference-collapses-to-a-near-fixed-point)
6. [F5 — The β-scale ambiguity is the dominant systematic error](#f5--the-β-scale-ambiguity-is-the-dominant-systematic-error)
7. [Open questions / next experiments](#open-questions--next-experiments)

---

## 1. TL;DR — what we know so far

Three sources of suspicion have been **ruled out** as the cause of the large
validation error in `validation.py`:

| # | Suspect | Verdict | Evidence |
|---|---|---|---|
| F1 | Wrong sensor size (loader builds 264×221 frames) | **Innocent** | Cancels at the kinematics fixed point; identical R after correction |
| F2 | Internal value clips `I±10, G±5, F±10, R±1` | **Innocent** | Zero saturation hits across both datasets; results bit-identical with clips disabled |
| F3 | `clip_value` / `normalise` in `events_to_vframe` | **Production setting is already optimal** | Sweep optimum at `clip=10, normalise=True`; flipping `normalise=False` is worse |

The validation error has **structural** causes (F4, F5):

- **F4** Under per-frame GT reseeding, the inner loop destroys the seed and
  collapses to a near-constant `ω ≈ const` regardless of the true ω that
  frame.  This puts the bug squarely in the per-frame inference, not in
  carry-over dynamics or initialisation.
- **F5** The β-scale ambiguity is finding a small-‖F‖ fixed point with
  normalised V → |R_est| ≈ 0.50 rad/s vs GT ≈ 1.17 rad/s (≈ 2.3× under-scale
  on poster_rotation).  Un-normalising V fixes the magnitude (|R_est| ≈
  1.18 rad/s) but worsens the direction because the per-pixel OFCE gradient
  clip `max_grad=5.0` and the step size `δ_VFG=0.20` are tuned for V ∈
  [−1, +1].

> **Bottom line:** the next experiment to run is a δ_FR sweep (see
> [§Open questions](#open-questions--next-experiments)).  Changes to data
> loading, V framing, or value clipping will not close the validation gap.

---

## F1 — Sensor size mismatch (264×221 vs 240×180)

### Observation
`data_loader.EventFrameSequence` originally derived `(H, W)` by doubling the
principal point:
```python
self.H = int(round(self.calib.cy * 2))   # → 221
self.W = int(round(self.calib.cx * 2))   # → 264
```
The DAVIS240C sensor is physically **240 × 180**, and the principal point
`(cx, cy) ≈ (132.2, 110.7)` is offset from the geometric centre, so this
fabricated ≈ 26 % of phantom pixels.

### Hypothesis (initially advanced)
Phantom pixels would inflate `M = Σ Cᵀ C` (built over all H × W pixels) but
not contribute to `v = Σ Cᵀ F` (no events fire there), so
`R_target = M⁻¹·v` would be **under-estimated** in proportion to the dead-pixel
area.

### Why the hypothesis is wrong (post-hoc analysis)
In dead pixels:
- `V = 0`, `G ≈ 0` (no input, no spatial structure).
- → OFCE exerts no pressure on F (`∂C_OFCE/∂F = 2(V + F·G)·G ≈ 0`).
- → Kinematics freely pulls `F → C·R` there.

So at the converged state, dead-region `F = C·R`. Plug into the kinematics
closed-form:
```
v       = v_real + M_dead · R
R_target = (M_real + M_dead)⁻¹ · v
        ⇒ at fixed point: R = M_real⁻¹ · v_real
```
Dead pixels **cancel out**. The fixed point of R is unchanged.

### Action taken
Applied Option A in `data_loader.py`:
```python
self.H = 180     # DAVIS240C hardcoded
self.W = 240
```
Reason: hygiene (events at sensor edges no longer silently dropped; diagnostics
more meaningful), even though it does not move R.

### Result confirmed
User reported no significant change in `validation.py` after the fix.
Consistent with the analysis above.

---

## F2 — Stability value clipping (I, G, F, R bounds) is inert

### What is being tested
After every Phase-2 update inside `InteractingMapsThesis.step()`:
```python
I = clip(I, ±10);  G = clip(G, ±5);  F = clip(F, ±10);  R = clip(R, ±1)
```
Hypothesis: these bounds might silently clip legitimate values and bias the
estimate.

### Test script
`test_clipping_effect.py` — runs validation under two configurations:
- **A** Default clips
- **B** Loose clips (everything ±1e6, effectively disabled)

Both with `RESEED_R_FROM_GT = True` (per-frame oracle init). Counts how many
pixels actually hit the clip limit per iteration.

### Results — `shapes_rotation`
```
RUN A (default clips)  →  mean err 25.18 deg/s
RUN B (loose clips)    →  mean err 25.18 deg/s
Total saturation hits over the whole run (BOTH runs):
    I: 0   G: 0   F: 0   R: 0
```
Bit-identical to displayed precision. Zero saturation hits across
25 frames × 75 inner iters × ~43 k pixels.

### Results — `poster_rotation`
```
RUN A (default clips)  →  mean err 45.53 deg/s
RUN B (loose clips)    →  mean err 45.53 deg/s
Δ mean err             =  +0.0000
Max |Δ per-frame|      =  0.0000
Saturation hits        =  0 / 0 / 0 / 0
```
Same verdict.

### Conclusion
**Stability value clipping is provably innocent** on both datasets. The
quantities never approach their bounds, on any iteration, for any frame.
Removing them would change nothing.

This **does not** rule out the per-pixel **gradient** clip
`max_grad = 5.0` in `Cost_OFCE` — that one operates on a different quantity
and is not measured here. See F5.

---

## F3 — Event-frame `clip_value` and `normalise` sweep

### What is being tested
`events_to_vframe()` has two parameters that scale V:
- `clip_value` — bound on signed event count before division.
- `normalise=True/False` — whether to divide by `clip_value` afterwards.

Production setting in `validation.py`: `clip_value = 10.0, normalise = True`,
giving V ∈ [−1, +1].

### Test script
`test_clip_value_effect.py` — sweeps `clip_value ∈ {1, 3, 10, 30, 100, ∞}` ×
`normalise ∈ {True, False}` on `poster_rotation`. Reports V statistics and
mean angular error vs GT under per-frame reseed.

### Results — `poster_rotation`
```
GT mean |omega|         : 1.167 rad/s
Expected omega (config) : 1.122 rad/s

   clip   norm |  mean|V|   max|V|   sat% | mean err  med err | mean |R_est|
                                              (deg/s)  (deg/s)      (rad/s)
   ---------------------------------------------------------------------------
      1   True |   0.5106    1.000 100.00 |   108.95   106.69 |       0.7803
      3   True |   0.2527    1.000  11.09 |   102.32   100.88 |       0.6815
     10   True |   0.0768    0.700   0.00 |    91.73    90.48 |       0.5023   ← best
     30   True |   0.0256    0.233   0.00 |    95.96    96.89 |       0.5666
    100   True |   0.0077    0.070   0.00 |    95.93    92.69 |       0.5713
    inf   True |   0.0000    0.000   0.00 |    94.68    93.34 |       0.5459
   ---------------------------------------------------------------------------
      1  False |   0.5106    1.000 100.00 |   108.95   106.69 |       0.7803
      3  False |   0.7581    3.000  11.09 |   134.65   136.43 |       1.2293
     10  False |   0.7679    7.000   0.00 |   131.72   134.39 |       1.1756
     30  False |   0.7679    7.000   0.00 |   131.72   134.39 |       1.1756
    100  False |   0.7679    7.000   0.00 |   131.72   134.39 |       1.1756
    inf  False |   0.7679    7.000   0.00 |   131.72   134.39 |       1.1756
```

### Observations

**1. The data's natural saturation point is 7.**
`max|V|` plateaus at 7.0 once `clip_value ≥ 7`. No pixel on poster_rotation
ever sees more than 7 net events in a 20 ms window. So `clip_value ∈ {10, 30,
100, ∞}` (un-normalised) all produce **bit-identical** V frames.

**2. Production default is at the sweep optimum (`norm=True`).**
clip = 10, norm = True → mean err 91.73 deg/s (best). All other normalised
settings are between 92 and 109 deg/s. The optimum is shallow — none of the
no-saturation settings (clip ∈ {10, 30, 100, ∞}) differ by more than 5 deg/s.

**3. Two regimes for normalised V:**
- **clip ≤ 3 (saturation regime)**: V loses dynamic range; OFCE can no longer
  distinguish strong edges from weak ones; error rises by ~10–17 deg/s.
- **clip ≥ 10 (scaling regime)**: V is in [−1, +1] but with peak shrinking as
  clip grows. Error stabilises at ~92–96 deg/s. The network is effectively
  **scale-invariant** here (β-ambiguity), and `|R_est|` settles at
  ≈ 0.55 rad/s — a systematic **2.3× under-scale vs GT** (1.17 rad/s).

**4. The normalise flag is a sharp trade-off:**
- `norm=True, clip=10`: V ∈ [−0.7, 0.7]. **|R_est| ≈ 0.50** (under-scale 2.3×), err 91.7 deg/s.
- `norm=False, clip=10`: V ∈ [−7, +7]. **|R_est| ≈ 1.18** (matches GT magnitude!) but err 131.7 deg/s.

The un-normalised case fixes the magnitude collapse (|R_est| ≈ |ω_GT|) but
worsens the direction.

### Why un-normalised V is worse (despite matching magnitude)
Two compounding effects when V is 10× larger:

1. **Per-pixel `max_grad = 5.0` clip in `Cost_OFCE` starts biting.**
   The OFCE gradient `2·(V + F·G)·G` easily exceeds 5 when V ≈ 7 and G is O(1),
   so the per-pixel clamp truncates the message and throws away angular
   information.
2. **`δ_VFG = 0.20` over-steps.**
   The effective Lipschitz constant of OFCE is 10× larger; the relaxation
   oscillates instead of converging.

### Conclusion
- `clip_value` is **not** the cause of the validation gap. Optimal value
  (`10`) is already in use.
- `normalise = True` is the correct choice. Flipping to `False` recovers
  magnitude but breaks direction more severely.
- Neither setting closes the gap. Error floor is ~92 deg/s vs a GT signal of
  only ~67 deg/s (i.e. error > signal).

---

## F4 — Per-frame inference collapses to a near-fixed point

### Observation
With `RESEED_R_FROM_GT = True` (every frame R is overwritten with `ω_GT · Δt`
before `net.step()`), the estimate **does not track the seed**.

#### `shapes_rotation`
```
Est ωy  clusters in [-0.45, -0.34]   (sd ≈ 0.025)
GT  ωy  varies over [-1.13, -0.33]
Est ωx, Est ωz ≈ 0
```

#### `poster_rotation`
```
Est ωx  clusters in [+0.394, +0.432] (sd ≈ 0.008)
GT  ωx  varies over [+0.469, +1.312]
Est ωy, Est ωz ≈ 0
```

The estimator output **barely moves** even though GT swings by an order of
magnitude. This is a textbook scale + direction collapse.

### Why this happens (mechanistically)
Inside one frame's 75 inner iterations:
1. We seed `R = ω_GT · Δt` and `F ≈ (1-f_decay)·F_prev + f_decay·C·R_GT`.
2. Each inner iteration blends R with `δ_FR = 0.80` toward the closed-form
   `R_target = M⁻¹·Σ Cᵀ F`.
3. After 75 iterations, the original seed has decayed by `(1 − 0.80)^75 =
   0.20^75 ≈ 10⁻⁵²`. R is **entirely determined by the closed form**.
4. The closed form evaluates to a near-constant whose direction depends on
   which axis V's structure projects most strongly onto via C_mat — **not** on
   the input GT.

### Conclusion
Per-frame inference is destroying the seed. The bug is **inside** `step()`,
**not** in carry-over dynamics, **not** in initialisation, **not** in the
sensor model.

---

## F5 — The β-scale ambiguity is the dominant systematic error

### Background (theoretical)
The three IVM constraints are invariant under
```
G → β·G    I → β·I    F → F/β    R → R/β    for any β ≠ 0
```
There is no information in V that fixes β. Practically, the network can
settle at any scale that satisfies OFCE.

### Empirical signature
From F3:

| V scale | Result |
|---|---|
| Normalised V (peak ≈ 0.7) | β > 1: G inflated, F deflated → R collapses to ≈ 0.5 rad/s |
| Un-normalised V (peak ≈ 7) | β ≈ 1: F restored to physical scale → R ≈ 1.18 rad/s ≈ GT |

The network's **fixed-point scale tracks the V scale**, but neither value is
correct in direction (F5 manifests as a direction error too because the
OFCE-driven F is not pure rotational flow when β ≠ 1).

### Mitigations currently in place
`F_DECAY = 0.5` and `G_DECAY = 0.7` (config) re-anchor F to `C·R` and shrink
G between frames so the OFCE can rebuild G at the correct scale. **These do
not help when R itself is wrong** — they only break a stale β-lock from a
previous frame.

### Conclusion
The β-ambiguity is **the** dominant systematic error in `validation.py`.
F1–F3 results all reinforce this: changing the sensor size, the value clips,
or the V framing leaves the β-locked fixed point in place.

---

## Open questions / next experiments

In rough order of expected impact:

### A. δ_FR sweep (highest expected impact)

The current `δ_FR = 0.80` over 75 inner iterations makes R follow the
closed-form exactly — F4 shows this destroys any seeded value of R.

**Hypothesis:** smaller `δ_FR` (e.g. 0.05–0.20) lets the GT seed survive the
inner loop, and the closed-form acts more like a regulariser than a
dictator.

**Setup:** extend `test_clipping_effect.py` infrastructure to sweep
`δ_FR ∈ {0.05, 0.10, 0.20, 0.40, 0.80}` on `poster_rotation` with
`RESEED_R_FROM_GT = True`. Report mean err and |R_est|.

**Predicted result:** error drops sharply as δ_FR → 0.1; magnitude collapse
disappears because R retains its GT seed.

### B. δ_VFG / max_grad interaction

`δ_VFG = 0.20` and the per-pixel `max_grad = 5.0` clip in `Cost_OFCE` jointly
determine how aggressively OFCE pulls F. F3 showed un-normalised V triggers
the clip — but does the clip bite even on normalised V at the iteration where
F is near zero (large `error = V + F·G`)?

**Setup:** instrument `Cost_OFCE` to count per-iteration gradient clip hits
(analogous to F2's value-clip counter).

**Predicted result:** clip is hit during the first 5–10 inner iterations even
on normalised V, biasing the initial OFCE direction.

### C. Scale gauge fixing

A principled fix for the β-ambiguity is to add a soft constraint that pins
`‖F‖ ≈ ‖C·R_seed‖` for the first N iterations. Not currently in the network.

**Setup:** add a new `Cost_ScaleGauge` that contributes `λ·(‖F‖² − ‖C·R‖²)`
during a warmup window; disable after iteration ≈ 10.

### D. OFCE gradient direction-only mode

When OFCE's gradient is large, normalise it to unit length per pixel before
applying. Preserves direction information while removing magnitude
dependence — a soft alternative to `max_grad` clipping.

---

## Appendix — index of test scripts

| Script | Tests |
|---|---|
| `test_clipping_effect.py` | F2: value clip activity & error delta with disabled clips |
| `test_clip_value_effect.py` | F3: clip_value × normalise sweep, V statistics, |R_est| collapse |
| `validation.py` (`RESEED_R_FROM_GT = True`) | F4: per-frame reseed unmasks the collapse-to-constant behaviour |

All three share the per-frame reseed pattern: GT ω from quaternion differencing
in `validation.gt_omega_body`, converted to rad/frame via `ω · Δt`. They are
the canonical evaluation harness for inner-loop diagnostics.
