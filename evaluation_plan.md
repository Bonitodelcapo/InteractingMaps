# Interacting Maps — Evaluation Plan

> A six-part evaluation of joint angular-velocity and scene estimation from a
> rotating event camera, organised around the one property that governs every
> result: the scale ambiguity inherent in the constraint set.
>
> Written as methodology prose (no code references) for the project report.

---

## Aims

Three questions, in order of importance:

1. **How accurately does each variant recover angular velocity**, and how does
   that accuracy hold up over the length of a sequence?
2. **Which parameters actually matter**, and how sensitive is the result to each?
3. **How does angular-velocity accuracy relate to reconstruction quality** — the
   intensity and gradient maps?

The third motivates most of the design below, because measurement suggests the
two are not merely uncorrelated but actively traded against each other.

---

## Foundation: the ambiguity that organises everything

The network is driven by a single input — the temporal derivative of
log-intensity accumulated from events — and infers four quantities tied by three
constraints: brightness conservation, the gradient definition, and the
rotational-flow relation.

For any non-zero β, scaling intensity and gradient **up** by β while scaling flow
and angular velocity **down** by β satisfies all three constraints *identically*:

```
I → βI      G → βG      F → F/β      ω → ω/β
```

Brightness conservation constrains only the **product** of flow and gradient.
There is **one degree of freedom and two claimants**:

| Spend the freedom on… | Result |
|---|---|
| the image | intensity looks right; angular velocity is off by β |
| the angular velocity | an external anchor fixes ω; intensity and gradient follow |

The five variants differ only in what supplies the missing absolute reference:
nothing (two pure-vision baselines), an inertial rotation-rate measurement, or a
purely visual criterion — maximising motion-compensated event-image sharpness,
which is *not* invariant to the scaling above and so pins the scale without an
extra sensor.

---

## Measurement decisions

### Three numbers, always reported together

A single scalar hides the mechanism: it cannot separate an estimate pointing the
wrong way from one pointing the right way at the wrong magnitude — different
causes, different fixes.

| Metric | Meaning |
|---|---|
| **Magnitude error** | Norm of the difference from the reference (deg/s). The headline. |
| **Direction error** | Angle between the vectors. Isolates the rotation *axis*. |
| **Scale ratio** | Reference magnitude / estimated magnitude. The ambiguity, measured. Unity is correct. |

> **Why this is not optional.** In a recorded single-frame convergence trace,
> total error fell 33.2 → 22.5 °/s over ten iterations while direction error
> *rose* 9.8° → 10.8°. Optimising the scalar alone would have scored that as
> unambiguous improvement.

### Reference sources have different noise floors

| Source | How obtained | Consequence |
|---|---|---|
| Direct rate | Published per-instant by the synthetic renderer | No differentiation, no sensor noise — the clean case |
| Pose differencing | Motion-capture orientations differenced over one frame | Drift-free but finite-difference noise at 20 ms |
| Gyroscope | Inertial sensor | Circular for the inertially-anchored variant; last resort |

Gyroscope-vs-reference disagreement is itself reported, as a **lower bound on
measurable error** for the real datasets. Comparing that floor to the synthetic
one is what makes the synthetic numbers worth quoting.

### Segments: data property vs experiment choice

A segment is an interval over which rotation is approximately constant — a
property of the **recording**. How finely it is sampled and how much is used are
choices of the **experiment**.

> **Resolved tension — fixed 3 s windows vs constant rotation.**
> Measured constant-rotation windows are **0.35–1.3 s** (real) and **2.4 s**
> (synthetic). A uniform 3 s window would run 2–6× past the interval it claims to
> hold constant — the premise false by construction.
>
> **Resolution: two tiers.** Short validated windows wherever the *scale ratio*
> is interpreted (it is only meaningful against a steady reference magnitude).
> Long windows with *genuinely varying* rotation for tracking and drift, on
> synthetic data where the per-instant reference is exact — a harder and more
> honest test than holding the motion still.

The original instinct is retained: whenever accumulation-window length is varied,
**physical duration is held fixed** by adjusting the frame count. Otherwise a
shorter window simultaneously shortens the observed span and reduces events per
frame, and the two effects cannot be separated.

---

## Part 1 — Qualitative survey of the recovered maps

One figure per dataset and segment: each variant on a row; columns are the
accumulated event input, recovered intensity, gradient magnitude, flow field, and
estimated rotation against the reference.

Gradient and flow must share a **common colour scale across rows**. Normalising
each panel independently would make every model look equally structured and would
destroy exactly the comparison the figure exists to support.

*Cost: minutes — derived from Part 3.*

---

## Part 2 — Convergence dynamics within a single frame

An animation of the relaxation for one frame at a segment centre, rendering
intensity, gradient, flow and rotation at every iteration, from a freshly
initialised network so the maps are seen **forming** rather than inherited.

Two requirements make it informative rather than decorative: colour scales are
**fixed in advance** from a preliminary pass (so the animation shows the maps
developing, not the normalisation fluctuating), and it is accompanied by a
**quantitative trace** — rotation error and constraint residual vs iteration.

Run on synthetic data with an exact per-instant reference, this is also the
clearest illustration of how intensity and gradient emerge from events alone.

> **Diagnostic value.** For the variant whose rotation is driven purely by image
> sharpness, rotation is updated by *no other mechanism*. A trace showing
> rotation frozen across iterations is proof the sharpness term never engaged — a
> failure that is otherwise silent, since the remaining terms still move the maps
> and produce plausible output.

*Cost: ~1–2 min per model.*

---

## Part 3 — Main comparison across models, datasets and segments

Every model on every segment of every dataset, under **identical settings** —
same accumulation window, iteration count, relaxation weights, distortion
handling. Only model, dataset and segment vary.

Deliberately **no per-cell tuning**: best-known parameters exist for one dataset
and three of five models, so applying them would tune a fraction of the table and
leave the rest at defaults — unfair in a direction that is hard to reason about.
A uniform, stated configuration is worth more than a partially optimised one.

Results are aggregated **across segments** per dataset and model — mean, standard
deviation, median, min and max — so spread is visible and one favourable segment
cannot carry a claim. Frame counts are clamped per segment so no run extends past
its validated interval; shorter segments are recorded as such, never padded.

*Cost: ~2–3 h — the long pole.*

---

## Part 4 — Parameter sensitivity, one factor at a time

Holding Part 3's configuration fixed, each parameter is varied alone while
everything else stays at default. A full grid search is explicitly not attempted:
exponentially more expensive, and it answers a question not being asked.

Four groups:

- **Iteration count** — has the relaxation converged, or is the default arbitrary?
- **The five relaxation weights** — one per constraint direction; how strongly
  each constraint pulls.
- **Accumulation window length** — the one parameter with direct physical meaning.
- **Anchor strength** — how hard the external reference pulls the rotation
  estimate; the parameter most directly implicated in the ambiguity.

Angular-velocity error is the objective. Direction error and scale ratio are
recorded at every point, for the reason given above. Parameters that do not apply
to a variant are skipped and reported as skipped, never silently defaulted.

> **Scope.** Two datasets — one with a noisy pose-derived reference, one with an
> exact synthetic reference. Not redundancy: it tests whether the sensitivity
> ranking is a property of the method or an artefact of reference noise.

Stated limitation: one-factor-at-a-time cannot reveal **interactions**. It
measures sensitivity around one operating point, and that is the claim made.

*Cost: ~3 h.*

---

## Part 5 — Coupling between rotation accuracy and reconstruction quality

Better rotation estimates appear to coincide with worse-looking intensity maps.
Measured across the five variants on one segment:

| Variant | Rotation error | Scale ratio | Gradient, interior | Border / interior |
|---|---:|---:|---:|---:|
| Pure vision A | 47.5 | 1.71 | 0.041 | 1.32 |
| Pure vision B | 35.8 | 1.42 | 0.036 | 0.91 |
| Inertial anchor | 23.8 | 1.19 | 0.073 | 3.01 |
| Sharpness anchor | 20.9 | 1.15 | 0.080 | 2.93 |
| Sharpness in-loop | 21.6 | 0.89 | 0.165 | 2.24 |

Rotation error correlates with the border/interior ratio at **−0.80**. The effect
**splits into two phenomena, and only one is a defect** — a distinction the
evaluation must make explicitly, or tuning effort is spent on something that
cannot be tuned.

### The magnitude change is the ambiguity, not damage

Gradient magnitude tracks the scale ratio almost perfectly and inversely —
precisely the transformation above. One degree of freedom; anchoring spends it on
rotation, so intensity and gradient land at whatever scale brightness
conservation then demands. **No choice of relaxation weights recovers both.**
Under any scale-invariant view the structure is largely preserved. This is a
*prediction of the theory*, not a failure — and framing it so is a result.

### The border concentration is a genuine artefact

A pure scale change multiplies interior and border equally, leaving their ratio
untouched. That ratio moves by more than 3×, so something structural is
happening. Likely mechanism: rotational flow grows with distance from the optical
centre, so a strong anchor forces large peripheral flow; brightness conservation
then demands a correspondingly large gradient there, but the periphery is
event-starved and the gradient is driven to extremes. Worth confirming.

### What a reconstruction-quality metric must satisfy

*(Specification only — not implemented.)*

- **Invariant to intensity scale and offset.** A metric sensitive to those simply
  re-measures the scale ratio under another name. Normalised correlation against
  a reference image, or structural similarity after standardisation, qualifies.
- **Report border/interior ratio separately** — that component is structural, and
  a scale-invariant metric will not capture it.
- **Score flow directly** where per-pixel reference flow exists (synthetic data).

### Establishing direction, not just correlation

The above is correlational, and causality is genuinely ambiguous: poor
reconstruction could cause poor rotation rather than result from it. The clean
test is an **injection sweep** — force rotation to the known true value scaled by
a controlled error factor, sweep that factor, measure reconstruction quality
against it. Because rotation is imposed rather than inferred, the causal
direction is unambiguous. Requires exact per-instant reference, so synthetic only.
Proposed as the centrepiece of the synthetic section.

---

## Part 6 — Supporting analyses

All derived from per-frame records already produced by Part 3 — no new runs.

- **Error and accumulated drift vs time.** Integral of rotation error over the
  segment, in degrees. Separates a variant that tracks (slow, near-linear
  accumulation) from one that diverges (curves upward).
- **Decomposition into direction and scale.** The quantitative statement of the
  central claim: flow direction is recovered well, absolute scale poorly unless
  anchored.
- **Accuracy vs computational cost.** The sharpness-anchored variants are several
  times more expensive per frame; make the trade-off explicit.
- **Error vs rotation rate.** Segments span ~0.35–2.8 rad/s, enough range to show
  whether accuracy degrades at speed.
- **Reference noise floor.** Gyroscope-vs-reference disagreement per dataset.

*Cost: minutes.*

---

## What this evaluation cannot show

- **Real-data accuracy is floored by the reference.** Pose-differenced rotation at
  20 ms carries real noise; differences below that floor cannot be attributed to
  the models.
- **Few segments per dataset.** Genuine constant-rotation intervals are scarce
  (~5 per recording), so per-dataset statistics rest on a small sample — hence
  reporting spread, not means alone.
- **One scene per dataset.** The synthetic pair is the exception and is
  deliberately exploited: both sequences share an identical camera trajectory
  rendered through different scenes, making them a controlled test of scene
  content at fixed motion. Nothing comparable exists for the real recordings.
- **Sensitivity is local.** One-factor-at-a-time varies around a single operating
  point and cannot detect parameter interactions.
- **Pure rotation is assumed throughout.** The flow model admits no translation.

---

## Dependencies and sequencing

Part 3 is the long pole and everything cheap depends on it, so it runs first.
Parts 1 and 6 are *derived* from its artefacts, so an interrupted run still yields
them for whatever completed. Part 4 needs Part 3's configuration fixed, so it
follows. Parts 2 and 5 are independent and can run at any point.

Within Part 3, synthetic datasets run before real ones: cheapest, and cleanest
reference — so if anything is cut short, the most defensible numbers are already
in hand.
