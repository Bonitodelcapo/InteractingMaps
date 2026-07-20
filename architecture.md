# InteractingMaps — Architecture & End-to-End Reference

> How this repository works, from an `events.txt` file to an angular-velocity
> plot. Covers both networks (Cook 2011 and Martel 2019 thesis, incl. the IMU
> sensor-fusion variant) and the evaluation harness. This is the `main`-branch
> reference; use it to answer "where does X come from?" and "what does this step
> actually compute?".

---

## Table of contents

1. [What the system does](#1-what-the-system-does)
2. [The model: quantities and constraints](#2-the-model-quantities-and-constraints)
3. [The three model variants](#3-the-three-model-variants)
4. [End-to-end pipeline (data → metric)](#4-end-to-end-pipeline-data--metric)
5. [Repository map](#5-repository-map)
6. [Module-by-module](#6-module-by-module)
   - [6.1 config.py](#61-configpy)
   - [6.2 data_loader.py](#62-data_loaderpy)
   - [6.3 interacting_maps/camera.py](#63-interacting_mapscamerapy)
   - [6.4 interacting_maps/network.py — Cook 2011](#64-interacting_mapsnetworkpy--cook-2011)
   - [6.5 interacting_maps/network_dissertation.py — Martel 2019](#65-interacting_mapsnetwork_dissertationpy--martel-2019)
   - [6.6 evaluation.py](#66-evaluationpy)
   - [6.7 find_segments.py & demo.py](#67-find_segmentspy--demopy)
7. [Cross-cutting concerns](#7-cross-cutting-concerns)

---

## 1. What the system does

An event camera (DVS) reports, per pixel, the **sign of the temporal change of
log-intensity** as a stream of asynchronous events. Binned over a short window
this gives a map `V ≈ ∂I/∂t` — the *only* input. From this single quantity the
network jointly infers:

- `I` — the grayscale image (light intensity),
- `G` — its spatial gradient `∇I`,
- `F` — the optical flow,
- `R` (= Ω) — the camera's 3-DoF angular velocity.

There is **no feed-forward path**: none of I, G, F, R can be computed directly
from V. Instead the maps are tied by three mutual constraints and refined by
iterative relaxation until they are mutually consistent. The headline output we
evaluate is `R` (angular velocity).

---

## 2. The model: quantities and constraints

| Map | Shape | Meaning | Role |
|-----|-------|---------|------|
| `V` | `(H, W)` | temporal intensity derivative | **input** (from events) |
| `I` | `(H, W)` or `(H+1, W+1)` | light intensity | inferred |
| `G` | `(H, W, 2)` | spatial gradient `∇I` | inferred |
| `F` | `(H, W, 2)` | optical flow (px/frame) | inferred |
| `R` | `(3,)` | angular velocity (rad/frame) | inferred (the answer) |
| `C` | `(H, W, 2, 3)` | kinematic matrix | constant (from calib) |

Three constraints (both papers):

1. **OFCE** (optical-flow constraint): `V + F·G = 0`
   Brightness conservation — `∂I/∂t = −∇I·F`.
2. **Spatial**: `G = ∇I` (forward differences).
3. **Kinematics**: `F = C·R`
   Pure-rotation optic flow. `C[x,y]` is a 2×3 matrix depending only on pixel
   position + intrinsics (thesis Eq. 6.37/6.38).

Each is turned into a squared-residual **cost**; inference minimizes their sum
by nudging each quantity toward satisfying the costs it participates in.

**β-scale ambiguity (fundamental).** For any β≠0, `(G→βG, I→βI, F→F/β, R→R/β)`
satisfies all three constraints identically. V alone cannot fix the scale (or
sign). This is not a bug — both papers state it — and it is the root cause of
the pure-vision accuracy ceiling (see [§7](#7-cross-cutting-concerns)).

---

## 3. The three model variants

Selected by the `model` string in the evaluation harness:

| Variant | Class | Update scheme | Extra |
|---|---|---|---|
| `cook` | `InteractingMaps` | **Gauss-Seidel** (sequential; each sub-update sees the latest values) | — |
| `thesis` | `InteractingMapsThesis` | **Jacobi** two-phase (all gradients from one snapshot, then all update) | — |
| `thesis_imu` | `InteractingMapsThesis` | Jacobi two-phase | + `Cost_IMU`: a 4th cost pulling `R` toward the gyroscope reading (thesis §6.8.3 sensor fusion) |

`thesis_imu` is the thesis's own fix for the β-ambiguity: the IMU gyro supplies
an **absolute** angular-velocity reference that anchors the scale/sign, which
pure vision lacks.

---

## 4. End-to-end pipeline (data → metric)

```
 events.txt ─┐
 calib.txt  ─┤   EventFrameSequence
             │   • load_events_fast (bulk read, t_start..t_start+n·dt)
             │   • per frame k:  mask events in [t_lo,t_hi)
             │                   undistort_events (cv2, plumb-bob)
             │                   events_to_vframe (signed count, clip, /clip)
             └──►  yields (V_k, t_mid_k)
                                     │
                                     ▼
             make_network(cook | thesis | thesis_imu)
                • build_kinematic_matrix C  (from intrinsics)
                • initial_R = gyro(t_start)·dt      (warm start)
                • initialize_from_rotation → F = C·R
                                     │
   imu.txt ─► gyro(t_lo,t_hi) ──► net.step(V_k, n_iters=75, omega_imu=gyro)
                                     │   (omega_imu only used by thesis_imu)
                                     ▼
                              omega_est = net.R / dt        (rad/frame → rad/s)
                                     │
 groundtruth.txt ─► gt_omega_body(t_lo,t_hi)  (R1ᵀR2, body frame) ── SCORING ref
                                     ▼
              compute_metrics(omega_est, omega_ref)
                • err  = ‖ω_est − ω_ref‖ (deg/s)
                • dir  = angle(ω_est, ω_ref) (deg)
                • β    = ‖ω_ref‖ / ‖ω_est‖
                                     ▼
                   tracking.csv · summary.json · tracking_plot.png
```

**Key roles of the sensors** (important, easy to conflate):
- `imu.txt` gyroscope → **model input** for `thesis_imu` *and* the warm-start
  `initial_R`.
- `groundtruth.txt` (Vicon) → **independent scoring reference** (added so the
  IMU variant is not graded against its own input; toggle `SCORE_AGAINST`).

---

## 5. Repository map

```
InteractingMaps/
├── config.py                 (1) datasets/segments + model hyper-parameters
├── data_loader.py            (2) events → V frames (+ cv2 undistortion)
├── interacting_maps/
│   ├── camera.py             (3) calibration + kinematic matrix C
│   ├── network.py            (4) InteractingMaps  (Cook, Gauss-Seidel)
│   └── network_dissertation.py (5) InteractingMapsThesis (+ Cost_IMU)
├── evaluation.py             (6) RunConfig + experiments 1–8 + metrics
├── find_segments.py          (7) discover constant-ω segments from imu.txt
├── demo.py                       live animated demo
├── data/<dataset>/
│   ├── events.txt  calib.txt  imu.txt  groundtruth.txt  images.txt  images/
└── results/                      per-run outputs + parameter_grid_*.csv
```

---

## 6. Module-by-module

### 6.1 `config.py`

Pure data + a couple of helpers.

- **`DATASET_SEGMENTS`** — per dataset, a list of hand-picked segments (from
  `find_segments.py`). Each has `id`, `t_start`, `frame_duration`, `n_frames`,
  `initial_R` (usually `None` → derived from IMU), `sensor_size`.
- **`DATASET_CONFIGS`** — `{name: segments[0]}`, a backward-compatible "first
  segment" shortcut.
- **`THESIS_PARAMS`** / **`COOK_PARAMS`** — the relaxation step sizes `delta_*`
  per relation (`delta_VFG` OFCE, `delta_IG`/`delta_GI` spatial, `delta_RF`/
  `delta_FR` kinematics, `delta_IMU` fusion strength).
- **`ITERS_PER_FRAME = 75`** — inner iterations per frame (thesis uses 50–75).
- **`get_dataset_paths(name)`** — resolves the 5 data files.
- **`get_initial_R_from_imu(name)`** — warm-start `R = gyro(t_start)·dt`.

> Note: `sensor_size` is authoritative (240×180 for the DAVIS240C). It is *not*
> derived from `2·(cx,cy)` because the principal point is offset from center.

### 6.2 `data_loader.py`

Turns events into V frames.

- **`CameraCalibration`** — parses `calib.txt` = `fx fy cx cy k1 k2 p1 p2 k3`.
  `.dist` holds the 5 distortion coefficients (now actually used, see below).
- **`load_events_fast(path, t_start, duration)`** — scans to the first row with
  `t ≥ t_start`, then bulk `np.loadtxt` (cap **30M** rows), masks to
  `t < t_start+duration`. Returns `(N,4) = [t, x, y, pol]`.
- **`undistort_events(events, calib)`** — **`cv2.undistortPoints`** with the
  plumb-bob model `[k1,k2,p1,p2,k3]`, keeping points in pixel space (`P=K`).
  This is how distortion is handled on `main`: the *event coordinates* are
  corrected before binning (rather than modifying `C`). Called **unconditionally
  every frame** in `EventFrameSequence.__iter__` (the only skip is the empty-frame
  `len(events)==0` guard) — so **distortion IS applied** on `main`. It is applied
  **exactly once**, at the event-coordinate level; `camera.py` (which builds `C`)
  does **not** re-apply it, so there is no double-undistortion.
  Caveat: `cv2.undistortPoints` returns sub-pixel float coordinates, but
  `events_to_vframe` casts `x,y` back to `int32` when binning — so the correction
  is re-quantized to the pixel grid. It still moves events across pixel boundaries
  (largest effect at the periphery, `k1≈−0.37` for the DAVIS240C), just at pixel
  resolution.
- **`events_to_vframe(events, H, W, clip_value, normalise)`** — signed-count
  image: `+1` per ON, `−1` per OFF via `np.add.at`; clip to `±clip_value`;
  divide by `clip_value` → `V ∈ [−1, 1]`. This is the standard *event-frame /
  2D histogram* representation (Maqueda 2018; Gallego 2020 survey).
- **`EventFrameSequence`** — iterator yielding `(V, t_mid)` for `n_frames`
  windows of width `frame_duration` from `t_start`. Loads all needed events
  once in the constructor; `__iter__` slices per window, undistorts, bins.

### 6.3 `interacting_maps/camera.py`

- **`compute_calibration(H,W,fx,fy,cx,cy) → (H,W,3)`** — per-pixel unit ray
  direction `normalize((u−cx)/fx, (v−cy)/fy, 1)`. (Legacy general-calibration
  form used by Cook.)
- **`build_kinematic_matrix(H,W,fx,fy,cx,cy) → (H,W,2,3)`** — the matrix `C`
  such that `F = C·R` reproduces rotational flow (thesis Eq. 6.38, skew=0):
  ```
  x' = (u−cx)/fx,  y' = (v−cy)/fy
  F_u = fx·[ x'y'·ωx − (x'²+1)·ωy + y'·ωz ]
  F_v = fy·[ (y'²+1)·ωx − x'y'·ωy − x'·ωz ]
  ```
  Used by **both** networks. Pure **pinhole** — it takes no distortion
  coefficients, because the events feeding `V`/`G`/`F` were already undistorted
  upstream in `data_loader.undistort_events`. Distortion is therefore applied
  once (at the data level), not here.

### 6.4 `interacting_maps/network.py` — Cook 2011

`InteractingMaps` — sequential (Gauss-Seidel). One `step(V, n_iters)` runs, per
iteration, in this (re-tuned) order:

```
update_R_from_FC()   # R ← blend toward M⁻¹·ΣCᵀF     (closed-form least squares)
update_F_from_VG(V)  # OFCE gradient on F
update_G_from_VF(V)  # OFCE gradient on G
update_G_from_I()    # G ← blend toward ∇I
update_I_from_G()    # I ← I − δ·div(G−∇I)           (discrete Poisson step)
update_F_from_RC()   # F ← blend toward C·R
# then value clips on F, G, I
```

- `I` is `(H+1, W+1)` so forward differences give a full `(H,W)` gradient.
- R update uses precomputed `M = ΣCᵀC`, `M⁻¹` (normal equations).
- Diagnostics: `residual_VFG`, `residual_GI`.

### 6.5 `interacting_maps/network_dissertation.py` — Martel 2019

Energy-based message passing, faithful to **Algorithm 6.5**.

**Building blocks**
- `Quantity` — holds `value` + `gradient_accumulator`; `update(lr)` does
  `value -= lr · accumulator`.
- `Cost.compute_and_send_gradients()` — reads current quantity values, adds
  each cost's contribution (already scaled by its `delta_*`) to the targets.

**Costs**
- `Cost_OFCE` — `∂/∂F = 2(V+F·G)G`, `∂/∂G = 2(V+F·G)F`; per-pixel gradient clip
  `max_grad=5.0` bounds cubic blow-up.
- `Cost_Spatial` — `G ← ∇I` blend; `I` gets the discrete **negative divergence**
  of `(G−∇I)`.
- `Cost_Kinematics` — `F ← C·R` blend; `R ← M⁻¹·ΣCᵀF` (precomputed `M⁻¹`).
- `Cost_IMU` — `R ← R − δ_IMU·(R − ω_imu·dt)`; only active when `omega_imu` is
  supplied (i.e. `thesis_imu`). Thesis §6.8.3.

**`step(V, n_iters, omega_imu=None)`** — per iteration:
1. Phase 1: zero all accumulators; every cost computes gradients from the
   *current* snapshot (skips `Cost_IMU` if no `omega_imu`).
2. Phase 2: all quantities `update(1.0)` simultaneously (Jacobi).
3. Stability clips: `I∈[−10,10]`, `G∈[−5,5]`, `F∈[−10,10]`, `R∈[−1,1]`.

**`initialize_from_rotation(R_init)`** — sets `R=R_init` and `F = C·R_init` (a
mutually consistent start; essential for Jacobi, else kinematics crushes R to 0
before OFCE builds structure), plus tiny noise in `I` to seed `∇I`.

### 6.6 `evaluation.py`

The harness. `RunConfig` bundles dataset/segment/model/params and derives
`initial_R` from the IMU. Model built by `make_network`.

**Reference & scoring** (the sensor split):
- `get_gyro_for_frame(imu, t_lo, t_hi)` — averages gyro columns → **model input**.
- `gt_omega_body(gt, t_lo, t_hi)` — `dR = R1ᵀR2` (body frame) from two
  `groundtruth.txt` quaternions → **scoring reference**.
- `get_reference_omega(...)` + `SCORE_AGAINST` — chooses Vicon (default) vs gyro.
- `compute_metrics(ω_est, ω_ref)` → `(err_deg_s, dir_err_deg, beta)`.

**R lifecycle — fully recurrent, no per-frame oracle.** In `main`, `R` is set
**once** before the frame loop, from the IMU gyro at `t_start`:
`net.R = rc.initial_R` (Cook) / `net.initialize_from_rotation(rc.initial_R)`
(thesis), where `initial_R = gyro(t_start)·dt`. Inside the loop `R` is **never
overwritten** — it evolves through `net.step(...)` and carries into the next
frame. So every run is the standard *recurrent* run.

> There is **no `RESEED_R_FROM_GT`** in `main`. That flag (a `gianluca`-branch
> `validation.py` construct) hard-overwrote `net.R = ω_GT·dt` at the start of
> **every** frame — a "perfect oracle" that isolates single-frame inference from
> carry-over dynamics. It does not exist here; grepping the tree finds no
> reference. The closest analogue on `main` is `thesis_imu`'s `Cost_IMU`, but
> that is a **soft** per-frame pull toward the *gyro* (weighted by `δ_IMU`, other
> costs still contribute), not a hard overwrite with the *Vicon GT*. Consequence:
> pure-vision drift over long segments is fully unmasked — nothing re-anchors R
> mid-run except (for `thesis_imu`) the soft IMU cost.

**Experiments** (`--exp N`):
1. Single-frame convergence — maps at iteration checkpoints (visual).
2. **Multi-frame tracking** — the core: per-frame ω_est vs ω_ref → `tracking.csv`,
   `summary.json`, `tracking_plot.png` (+ optional 3-col video frames).
3. Parameter influence — iterations & frame-duration sweeps for one frame.
4. Qualitative video (alias of Exp 2 with frames).
5. Assemble MP4s from saved frames (batch).
6. Basin of attraction — vary `initial_R` distance from GT; does it converge?
7. Full evaluation — all datasets × models × segments → `full_evaluation.csv`.
8. **Parameter grid** — sweeps `frame_duration × n_frames × n_iters × delta_FR ×
   delta_IMU` (thesis_imu only for `delta_IMU`) → `parameter_grid_<dataset>.csv`.

Exp 2/7/8 all funnel through `experiment_tracking`, so any scoring change lives
in one place.

### 6.7 `find_segments.py` & `demo.py`

- **`find_segments.py`** — slides a window over `imu.txt`, flags intervals of
  low gyro-std (≈ constant ω) and prints a `DATASET_SEGMENTS` block to paste
  into `config.py`.
- **`demo.py`** — live matplotlib animation (V, I, |G|, F, R, residuals) for a
  single dataset; qualitative sanity check.

---

## 7. Cross-cutting concerns

**β-scale ambiguity → drift.** With no absolute reference, pure vision (`cook`,
`thesis`) settles at an arbitrary scale/sign and the ω *axis* drifts over time
(direction error grows with track length). `thesis_imu` fixes it by anchoring R
to the gyro (thesis: the IMU "totally removes color flips").

**Two ground truths, different jobs.**
`imu.txt` gyro = *measured* ω (direct, 1 kHz, has bias) → **model input**.
`groundtruth.txt` = Vicon pose; ω obtained by *differencing* quaternions
(`R1ᵀR2`, body frame; drift-free but finite-difference-noisy at 20 ms) →
**independent scoring**. Scoring `thesis_imu` against the gyro it is fed is
circular; scoring against Vicon is not.

**Units.** `R` is rad/frame. `ω = R/dt` (rad/s). Warm start `R_init = ω_gyro·dt`.
`frame_duration` is both the event-accumulation window and this `dt`.

**Coordinate conventions.** Pixel `[...,0]=x` (cols), `[...,1]=y` (rows).
Forward differences, boundary 0. Quaternions ordered `(qx,qy,qz,qw)` →
`R_wc`. Body-frame ω uses `R1ᵀR2` (right-invariant), **not** `R2R1ᵀ` (world).

**Distortion.** Handled at the *data* level: `undistort_events` (cv2 plumb-bob)
corrects event coordinates before binning, so `V`, `G`, `F` and `C` all live in
the undistorted pinhole frame.
