# Contrast Maximization (CMax) — Design, Findings & Contribution

> Working reference for the `Cmax` branch: replacing the IMU with a self-contained
> event-based angular-velocity estimator (Contrast Maximization) inside the
> Interacting-Maps message passing.

---

## SUMMARY (read this first)

**Goal.** The pure-vision network drifts because of the β-scale ambiguity; it only
works when an **absolute ω is injected into the update loop every iteration**
(today: the IMU gyro). **CMax supplies that same absolute ω from the events alone**
— its contrast objective is *not* β-invariant, so it fixes the gauge without any
IMU. Deliverable: **joint scene interpretation** (I, G, F *and* ω), with CMax as
the anchor.

**Status.**
| Step | What | State |
|---|---|---|
| **1a** | Standalone CMax front-end (`cmax/angular_velocity.py`) | ✅ done, validated |
| **1b (V1)** | Feed CMax ω into the message passing (`model='thesis_cmax'`) | ✅ implemented, small-sample OK |
| **2 (V2)** | CMax as the in-loop R-update rule (`model='thesis_cmax_v2'`) | ✅ implemented, small-sample OK |

**Key results (poster seg_C).**
- **1a:** CMax agrees with the *independent* gyro to **8.4 °/s** and with smoothed
  GT to **13.5 °/s** (gyro: 8.6) → **CMax is an IMU-quality, sensor-free ω source.**
- **1b (V1):** `thesis_cmax` (IMU-free in-loop) tracks GT **comparably to `thesis_imu`**
  on a 5-frame sanity (9.7 vs 12.4 °/s smoothed GT). *Small sample — verify at scale.*
- **2 (V2):** `thesis_cmax_v2` (R driven by 1 CMax step/iter, CMax-only) reaches the
  **standalone CMax quality** — vs IMU **8.2 °/s**, vs smoothed GT 14.2 (5-frame sanity),
  *better* than V1 vs IMU because kinematics no longer pulls R. **`lr` is sensitive:**
  stable ~`[1e-5, 1e-4]`, `≳5e-4` diverges (gradient ∝ event-count²).
- The raw **20 ms GT is noisy** (quaternion differencing); score against **smoothed
  GT (~100 ms)** or the gyro, not the 20 ms GT.
- **Analytic-gradient + CG** is implemented and correct (FD-validated ~1e-4) but in
  numpy is **slower** than Nelder-Mead with no accuracy gain → Nelder-Mead default.

**Locked design choices.** Linear warp (bearing-vector `p+(ω·dt)×p`) · variance
objective · reference time = window midpoint · warm-start = previous ω · **fixed-time
window = one frame** · bilinear voting (polarity) · Nelder-Mead default · body-frame
ω in rad/s. For `thesis_cmax` (1b): **B1** event plumbing (slice events in the cmax
path, leave `EventFrameSequence` alone) · **frame-0 init from gyro** only · reuse
`Cost_IMU` as the anchor with target = ω_cmax.

**Files.** `cmax/angular_velocity.py` (estimator), `cmax/__init__.py`, this doc,
`test_cmax_frontend.py` (1a check), `evaluation.py` (`model='thesis_cmax'`).

---

## 1. Why CMax — the diagnosis

Pure vision (`cook`, `thesis`) **drifts** (direction error 12°→85° over a segment).
Root cause: the **β-scale ambiguity** — OFCE `V+F·G=0`, spatial `G=∇I`, kinematics
`F=C·R` are all invariant to `(G→βG, F→F/β, R→R/β)`, so vision alone cannot fix ω's
scale or sign.

Empirically (Gallego meeting): the network is good **only when the IMU gyro is fed
into the loop every iteration**; init-only or absent → diverges.

**CMax is the sensor-free replacement.** It warps events under a motion hypothesis,
builds the Image of Warped Events (IWE), and maximizes its contrast (variance). The
IWE is sharpest at the *true* ω (wrong magnitude → blur → lower contrast), so the
objective is **not β-invariant** — it supplies the absolute scale + direction the
OFCE lacks, from events alone.

### Integration plan
- **V1 (loose):** full CMax per frame → `ω_cmax` fed as the `R` anchor (replaces the
  IMU). = `thesis_imu` with the anchor source swapped. **← done (1a+1b).**
- **V2 (tight):** one CMax gradient step = the per-iteration `R` update inside the
  loop; **R driven by CMax only** (kinematics only propagates R→F). **← done.**

---

## 2. Step 1a — standalone CMax front-end

### Reference (`tub-rip/cmax_slam`, `src/frontend/`)
| File | Contribution |
|---|---|
| `ang_vel_estimator.cpp` | warm-start = previous ω; ref time = window midpoint; event-count packets |
| `local_image_warped_events.cpp` | linear warp `p+(ω·dt)×p`; bilinear voting |
| `local_focus_funcs.cpp` | variance objective (maximized); analytic gradient |
| `local_optim_contrast_gsl.cpp` | GSL conjugate-gradient |

### Design decisions (with rationale)
| Decision | Choice | Why |
|---|---|---|
| Warp | **Linear** `p_rot = p + (ω·dt)×p`, reproject | user "only linear"; matches Gallego; keeps reprojection (more accurate than pure 2-D `x'=x+dt·B(x)ω`, which is reserved for V2) |
| Objective | **Variance** of IWE, maximized | Gallego default; well-behaved |
| Reference time | window **midpoint** | Gallego |
| Warm-start | **previous ω** (zeros on frame 0) | non-convex objective; good init matters |
| Window | **fixed-TIME = one frame (20 ms)** — *adaptation* vs Gallego's event-count | clean frame-synchronous coupling with the message passing |
| Accumulation | bilinear voting, **polarity** weight (count optional) | Gallego; count vs polarity negligible (§4) |
| Smoothing | Gaussian blur σ=1 px (Nelder-Mead path) | smooths landscape for derivative-free search |
| Optimizer | **Nelder-Mead** default; analytic-grad+CG optional | Nelder-Mead faster+robust in numpy (§4) |
| Coords / units | undistorted pinhole bearings; **rad/s, body frame** | consistent with pipeline, GT, network |

### Parameters (`CMaxAngularVelocity`)
| Param | Default | Meaning |
|---|---|---|
| `H,W,fx,fy,cx,cy` | — | sensor size + intrinsics |
| `use_polarity` | `True` | signed polarity vs event count in the IWE |
| `blur_sigma` | `1.0` | IWE blur (px) before variance (Nelder-Mead path) |
| `optimizer` | `'Nelder-Mead'` | SciPy method (forced `'CG'` if analytic gradient on) |
| `use_analytic_gradient` | `False` | analytic-gradient + CG on the unblurred variance |

`estimate(events, t_ref=None, omega_init=None) → ω (rad/s)`.
`events=(N,4)[t,x,y,pol]` undistorted; `t_ref` default = midpoint; `omega_init` = warm start.

### Analytic gradient (optional path)
Exact for the discrete IWE (scatter bilinear-kernel derivatives into 3 ∂I/∂ω images):
```
dVar/dω_j = (2/P) Σ_u I_u · D_j(u),   D_j = ∂I_u/∂ω_j,   Σ_u D_j = 0
Jₑ = ∂(x',y')/∂ω = P_proj(p_rot) · (−dt·[p]×)     (2×3 per event)
```
Validated vs finite differences: **max rel err ~1e-4**.

---

## 3. Step 1b — CMax anchor in the message passing (`thesis_cmax`)

`thesis_cmax` = `thesis_imu`, but the per-frame `R` anchor is a **full CMax solve on
the events** instead of the gyro. **No network change** — `Cost_IMU` is a generic
"pull R toward an external ω"; we just feed it `ω_cmax`.

### How it is wired (`evaluation.py`)
- **RunConfig:** new `model='thesis_cmax'` → `use_thesis=True, use_imu=True`
  (Cost_IMU is the anchor), `use_cmax=True`.
- **B1 event plumbing:** in `experiment_tracking`, load raw events once
  (`load_events_fast + undistort_events`) and slice per window `[t_lo,t_hi)`;
  `EventFrameSequence` is untouched (double event-load accepted).
- **Per frame:** `ω_cmax = cmax_est.estimate(win, t_ref=midpoint, omega_init=prev)`,
  warm-started across frames, passed as `omega_imu` to `net.step`.
- **Init:** `initial_R` still from the **gyro at frame 0** only (decision C — revisit
  later with CMax-init). Everything else in the loop is IMU-free.

### Information flow (vs current)
Same as the `thesis_imu` diagram, except the two ω sources change:
`Cost_IMU.target = ω_cmax·dt` (in-loop, every iteration) and — for now — frame-0
`initial_R` still from the gyro. All else (C matrix, two-phase message passing,
recurrent R, GT scoring) is identical.

### Small-sample result (poster seg_C, 5 frames — sanity only)
| Model | vs GT(smooth) | vs IMU | vs GT(20 ms) |
|---|---|---|---|
| `thesis_imu` | 12.36 | 16.08 | 31.22 |
| **`thesis_cmax`** | **9.71** | 13.25 | 29.34 |
`thesis_cmax` runs, is IMU-free in-loop, and tracks GT comparably to `thesis_imu`.
**Verify at scale on the workstation** (more frames, all poster segments).

---

## 3b. Step 2 — CMax as the in-loop R update (`thesis_cmax_v2`, V2)

CMax is no longer a separate solve; **one CMax gradient-ascent step on the IWE
variance is the per-iteration update rule for R**, interleaved with the other
message-passing updates. R is driven by **CMax only**.

### What changes in the network (`network_dissertation.py`)
- **`Cost_Kinematics(update_r=False)`** — updates **F only** (`F ← C·R`); no longer
  touches R.
- **`Cost_CMax`** (new) — per MP iteration: `ω = R/dt`; call the validated
  `_contrast_and_grad` → `∂Var/∂ω`; chain-rule `∂Var/∂R = (∂Var/∂ω)/dt`; take one
  **ascent** step `R ← R + λ·∂Var/∂R` (implemented as `add_gradient(−λ·∂Var/∂R)`
  since Phase-2 descends).
- **`enable_cmax_r_update(estimator, lr)`** — flips the network into V2: sets
  `cost_kin.update_r=False`, appends `Cost_CMax`. `Cost_IMU` becomes inert (no ω_imu).
- **`step(V, …, events=…)`** — the raw window events are handed to `Cost_CMax`
  (warped to the window midpoint) once per frame.

### Wiring (`evaluation.py`)
`model='thesis_cmax_v2'` → `use_thesis=True, use_imu=False, use_cmax=True,
use_cmax_v2=True`. `make_network` builds a `CMaxAngularVelocity` and calls
`enable_cmax_r_update(est, lr=rc.cmax_lr)`. The loop passes the per-window events to
`net.step(V, events=win)`. Frame-0 init still from the gyro (decision C).

### The `lr` (ascent step) — sensitive
`cmax_lr` default **1e-4**. Stable range ~`[1e-5, 1e-4]` (result plateaus — R reaches
the CMax optimum, `lr` only sets convergence speed); `≳5e-4` **diverges**. The
gradient magnitude scales with **event-count²**, so denser streams need a smaller
`lr`. **Recommended follow-up:** normalize the step (`λ·grad/‖grad‖`, λ in rad/s) or
scale `lr ∝ 1/N²` for robustness across datasets.

### Small-sample result (poster seg_C, 5 frames)
| lr | vs GT(smooth) | vs IMU |
|---|---|---|
| 1e-5 | 14.36 | 8.27 |
| 3e-5 | 14.23 | 8.23 |
| 1e-4 | 14.22 | 8.22 |
| 5e-4 | 49.98 | 46.32 (diverging) |

At a stable `lr`, V2 reaches **standalone-CMax quality** (vs IMU ~8.2) — *better than
V1 vs IMU* because kinematics no longer competes for R. **Verify at scale.**

---

## 4. Findings (verification detail)

Test: `test_cmax_frontend.py` — poster seg_C, 25 frames × 20 ms, warm-started.

- **CMax ≈ IMU quality.** CMax vs IMU **8.4 °/s**; vs smoothed GT **13.5** (IMU 8.6);
  vs 20 ms GT 24.2 (IMU 20.4). Two independent sensors agreeing to 8.4 °/s ⇒ CMax is
  a valid IMU replacement.
- **20 ms GT is noisy.** Both estimators' error ~halves against smoothed GT
  (CMax 24→13.5, IMU 20→8.6) ⇒ score against smoothed GT / gyro.
- **Polarity vs count: negligible** (24.2 vs 24.1).
- **Nelder-Mead beats analytic+CG in numpy:** 8.39 vs 9.44 °/s, **20.6 s vs 74.8 s**
  (12 `np.add.at` scatters/eval + rougher unblurred landscape). Would flip with a
  `np.bincount` scatter or a compiled backend.

---

## 5. Open questions / next steps

1. **Verify V1 & V2 at scale** (workstation): all poster segments, more frames, vs smoothed GT.
2. **V2 `lr` robustness** — normalize the ascent step (`λ·grad/‖grad‖`) or scale
   `lr ∝ 1/N²` so it's not event-count-dependent. (Current: fixed `lr`, data-tuned.)
3. **Window sweep** — CMax over a wider *centered* span `[t_mid ± {10,20,40} ms]`
   (sharper contrast peak, but ω must stay ~constant across it). Main accuracy lever.
4. **CMax-init** — replace frame-0 gyro init with CMax(frame 0) for a fully IMU-free pipeline.
5. **`bincount`** scatter — speed up the analytic gradient (helps V2, which calls it
   75×/frame, and the analytic+CG front-end path).

---

## 6. Files

| File | Purpose |
|---|---|
| `cmax/angular_velocity.py` | `CMaxAngularVelocity` — the front-end estimator |
| `cmax/__init__.py` | exports `CMaxAngularVelocity` |
| `cmax/cmax.md` | this document |
| `test_cmax_frontend.py` (root) | Step-1a verification vs GT & IMU |
| `evaluation.py` | `model='thesis_cmax'` (V1), `'thesis_cmax_v2'` (V2) |
| `interacting_maps/network_dissertation.py` | `Cost_CMax`, `Cost_Kinematics(update_r)`, `enable_cmax_r_update`, `step(events=…)` (V2) |
