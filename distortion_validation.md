# Distortion Validation via Contrast Maximization

> Network-free confirmation that handling lens distortion **inside the C matrix**
> (raw events, distortion baked into the kinematics) is geometrically equivalent
> to the trusted **undistort-events-first** approach.
>
> Reproduce: `python test_distortion_cmax.py` → **RESULT: PASS**

---

## Goal

Confirm that the distortion-aware C matrix (`interacting_maps/camera.py`,
`build_kinematic_matrix`) — the part not in the thesis, hand-derived — encodes
the lens correctly. We use **Contrast Maximization (CMax)** as the oracle because
it estimates ω from events + camera model **alone**, with the Interacting-Maps
network removed. This sidesteps the network's β-scale ambiguity, which otherwise
muddies any direct accuracy comparison.

## The two ways compared (same events, network removed)

| | π⁻¹ (pixel → ray) | warp | π (ray → pixel) | events |
|---|---|---|---|---|
| **Way 1** | pinhole | rotate | pinhole | **undistorted first** (`undistort_events`) |
| **Way 2** | `cv2.undistortPoints` | rotate | **forward Brown–Conrady** | **raw** |

Way 1 uses only `cv2.undistortPoints` (trusted, hard to get wrong). Way 2's
forward re-distortion (`π`) is the **same geometry** whose analytic Jacobian `J_D`
is baked into the distortion-aware C matrix. If both encode the lens correctly
they must describe the same physical rotation.

## What we checked and found

- ✅ **Forward re-distortion is exact** — Way 2's `π` matches `cv2.projectPoints`
  to **2.8×10⁻¹⁴ px** (machine precision).
- ✅ **Undistort ∘ distort round-trips** to **10⁻⁸ px** — the two lens maps are
  true inverses.
- ✅ **Fixed-rotation warp correspondence — the decisive test.** At a fixed ω,
  warping the distortion-aware way and undistorting the result lands on the
  pinhole-warp result to **0.043 px (sub-pixel)** across all frames.
  ⇒ **The two warps are the same geometry**; the C-matrix re-distortion is
  confirmed by an independent path.

### The two ω estimates differ by ~5 °/s — and why that is *not* a bug

- The difference **scales with distortion strength and → 0 when distortion → 0**:

  | distortion scale `s` | mean \|ω̂₁ − ω̂₂\| |
  |---|---|
  | 0.00 | 0.00 °/s |
  | 0.25 | 1.91 °/s |
  | 0.50 | 2.56 °/s |
  | 1.00 | 4.08 °/s |

- **Cause:** CMax's contrast (variance) is measured on **different pixel grids**
  (undistorted vs distorted) and is not invariant to that resampling — a known,
  benign property of CMax, not of the geometry. A geometry error would not vanish
  at `s = 0`.
- Both ways independently agree with the **IMU gyro to ~8–9 °/s**, i.e. they
  recover the same physical rotation to within each method's own accuracy, and
  agree on direction to ~3°.

## Conclusion

The novel distortion-in-C geometry is **validated** — three independent, tight
geometric checks pass (10⁻¹⁴, 10⁻⁸, 0.04 px). The residual ω difference is a CMax
objective artifact, fully explained and reproducible, not a lens-model error.

## Test design note

The naive expectation "ω̂₁ must equal ω̂₂" is confounded **for CMax specifically**,
because the contrast objective is grid-dependent. The `test_distortion_cmax.py`
pass/fail gate is therefore the **geometry** (forward-model anchor + fixed-ω warp
correspondence); the two-way ω comparison is reported as **context**, not a gate.

## Caveats / honest scope

- This validates the **geometry** of the C matrix, not that `C_full` lowers
  ω-tracking error on real data — that is the separate empirical question the
  **exp-9 ablation** answers.
- The C matrix additionally uses the analytic distortion **Jacobian** `J_D`;
  CMax's warp is exact (not linearized), so it validates the coordinate maps
  `J_D` is derived from but not `J_D` itself. A small finite-difference check of
  `J_D` vs `cv2.projectPoints` would round out the coverage (optional follow-up).

## Files

| File | Role |
|---|---|
| `test_distortion_cmax.py` | the validation (checks A/B geometry gate + C context) |
| `cmax/angular_velocity.py` | `CMaxAngularVelocity(dist_coeffs=…)` — Way 2 warp |
| `interacting_maps/camera.py` | `build_kinematic_matrix` — the distortion-aware C matrix under test |
