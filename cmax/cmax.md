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
| **1b** | Feed CMax ω into the message passing (`model='thesis_cmax'`) | ✅ implemented, small-sample OK |
| 2 (V2) | CMax as the in-loop R-update rule | ⏳ later |

**Key results (poster seg_C).**
- **1a:** CMax agrees with the *independent* gyro to **8.4 °/s** and with smoothed
  GT to **13.5 °/s** (gyro: 8.6) → **CMax is an IMU-quality, sensor-free ω source.**
- **1b:** `thesis_cmax` (IMU-free in-loop) tracks GT **comparably to `thesis_imu`**
  on a 5-frame sanity (9.7 vs 12.4 °/s smoothed GT). *Small sample — verify at scale.*
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
  loop; **R driven by CMax only** (kinematics only propagates R→F). ⏳ later.

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

1. **Verify 1b at scale** (workstation): all poster segments, more frames, vs smoothed GT.
2. **Window sweep** — CMax over a wider *centered* span `[t_mid ± {10,20,40} ms]`
   (sharper contrast peak, but ω must stay ~constant across it). Main accuracy lever.
3. **CMax-init** — replace frame-0 gyro init with CMax(frame 0) for a fully IMU-free pipeline.
4. **`bincount`** scatter — only if the analytic+CG path needs to be competitive.
5. **V2** — CMax as the in-loop R-update (1 step/iter, CMax-only on R, likely 2-D `B(x)ω` warp).

---

## 6. Files

| File | Purpose |
|---|---|
| `cmax/angular_velocity.py` | `CMaxAngularVelocity` — the front-end estimator |
| `cmax/__init__.py` | exports `CMaxAngularVelocity` |
| `cmax/cmax.md` | this document |
| `test_cmax_frontend.py` (root) | Step-1a verification vs GT & IMU |
| `evaluation.py` | `model='thesis_cmax'` (Step 1b integration) |
