# InteractingMaps — Architecture & Pipeline Reference

> A line-by-line reference for the InteractingMaps repo (Cook 2011 + Martel 2019
> Thesis Chapter 6). Use this document to answer questions like *"where does
> variable X come from?"*, *"what does this function actually compute?"* and
> *"why is the network behaving this way?"*.

---

## Table of Contentsgranseerstrasse 1 berlin was part of east or west berlin ?

1. [Big-picture overview](#1-big-picture-overview)
2. [The mathematical core (what every module is solving)](#2-the-mathematical-core-what-every-module-is-solving)
3. [The pipeline end-to-end (data → plot)](#3-the-pipeline-end-to-end-data--plot)
4. [Repository map](#4-repository-map)granseerstrasse 1 berlin was part of east or west berlin ?
5. [Module-by-module deep dive](#5-module-by-module-deep-dive)
   - [5.1 `config.py`](#51-configpy)
   - [5.2 `data_loader.py`](#52-data_loaderpy)
   - [5.3 `interacting_maps/camera.py`](#53-interacting_mapscamerapy)
   - [5.4 `interacting_maps/network.py` (Cook 2011)](#54-interacting_mapsnetworkpy-cook-2011)
   - [5.5 `interacting_maps/network_dissertation.py` (Martel 2019)](#55-interacting_mapsnetwork_dissertationpy-martel-2019)
   - [5.6 `simulation/event_sim.py`](#56-simulationevent_simpy)
   - [5.7 `find_segments.py`](#57-find_segmentspy)
   - [5.8 `demo.py`](#58-demopy)
   - [5.9 `demo_single_packet.py`](#59-demo_single_packetpy)
   - [5.10 `validation.py`](#510-validationpy)
6. [Cross-cutting concerns](#6-cross-cutting-concerns)
   - [6.1 The β-scale ambiguity](#61-the-β-scale-ambiguity)
   - [6.2 Inter-frame state management (F_DECAY / G_DECAY)](#62-inter-frame-state-management-f_decay--g_decay)
   - [6.3 Coordinate frames and conventions](#63-coordinate-frames-and-conventions)
   - [6.4 Unit conventions (rad/s vs rad/frame)](#64-unit-conventions-rads-vs-radframe)

---

## 1. Big-picture overview

This repo implements an **energy-based recurrent network** that takes a single
input — `V`, the temporal intensity derivative produced by an event camera —
and *jointly* infers four interpretation maps:

| Map   | Shape                             | Meaning                                                                                       |
| ----- | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `V` | `(H, W)`                        | Temporal derivative of brightness (events accumulated in a time window).**Only input.** |
| `I` | `(H, W)` or `(H+1, W+1)`      | Light intensity (scene appearance).                                                           |
| `G` | `(H, W, 2)`                     | Spatial gradient of `I`. `G[..., 0] = ∂I/∂x`, `G[..., 1] = ∂I/∂y`.                  |
| `F` | `(H, W, 2)`                     | Optical flow (pixel/frame).`F[..., 0] = u`, `F[..., 1] = v`.                              |
| `R` | `(3,)`                          | Camera body-frame angular velocity (rad/frame).                                               |
| `C` | `(H, W, 3)` or `(H, W, 2, 3)` | **Constant** calibration / kinematic matrix.                                            |

These maps are tied together by **three constraints**:

1. **OFCE** (optical flow constraint): `V + F · G = 0`
2. **Spatial**: `G = ∇I`
3. **Kinematics**: `F = C · R` (`C` is a (H, W, 2, 3) matrix mapping ω → flow)

Inference is **iterative relaxation**: each constraint contributes a small
correction toward satisfaction; the corrections circulate until all three
errors are small.

Two distinct networks implement this:

- **`InteractingMaps`** (`network.py`) — Cook 2011 style. **Gauss-Seidel**:
  every quantity is updated in sequence, the next sub-step sees the new value.
- **`InteractingMapsThesis`** (`network_dissertation.py`) — Martel 2019
  thesis. **Jacobi (two-phase)**: all costs first compute messages from the
  *current* state, then every quantity is updated simultaneously.

The three driver scripts use them:

- **`demo.py`** — animated multi-frame visualisation.
- **`demo_single_packet.py`** — one fixed `V` frame, network iterates 100× to
  show map dynamics from cold-start.
- **`validation.py`** — quantitative comparison of network ω vs IMU ω vs
  ground-truth (quaternion-derived) ω over many frames.

---

## 2. The mathematical core (what every module is solving)

### 2.1 OFCE — Optical Flow Constraint Equation

Brightness conservation: if the scene moves with flow `F = (u, v)`, then the
brightness change at a pixel is `∂I/∂t = −∇I · F`. Renaming `V := ∂I/∂t` and
`G := ∇I` gives:

```
V + F · G = 0
```

One scalar equation per pixel with two unknowns per pixel (`F` is 2D). It
becomes solvable only when combined with constraints 2 and 3.

**Cost (squared residual):** `C_OFCE = Σ (V + F·G)²`

**Gradients (nonlinear in F, G):**

- `∂C/∂F = 2 (V + F·G) · G`
- `∂C/∂G = 2 (V + F·G) · F`

Both networks clip these gradients per-pixel (default `max_grad = 5.0` in the
thesis version) to bound the cubic instability that appears when both `|F|`
and `|G|` are large.

### 2.2 Spatial — `G = ∇I`

`G` is *by definition* the spatial gradient of `I`. The forward-difference
discretisation used everywhere in the repo:

```
G_x[v, u] = I[v, u+1] − I[v, u]
G_y[v, u] = I[v+1, u] − I[v, u]
```

(Boundary rows/columns are zero-padded.)

**Cost:** `C_Spat = Σ (G − ∇I)²` (linear in both `G` and `I`).

**Update of I (Eqs. 6.61 of thesis / Eqs. 7–9 of Cook):** the gradient of the
cost w.r.t. `I` is the **discrete negative divergence** of the residual
`Ψ = G − ∇I`. This is why you'll see lines like

```python
grad_I_update += error[:, :, 0] + error[:, :, 1]
grad_I_update[:, 1:] -= error[:, :-1, 0]   # x-shifted residual
grad_I_update[1:, :] -= error[:-1, :, 1]   # y-shifted residual
```

This is just `−div(Ψ)` written out for forward differences on a discrete grid.

### 2.3 Kinematics — `F = C · R`

A purely rotational camera moving at angular velocity `ω = R` produces a flow
field given (in normalised image coords `x' = (u−cx)/fx`, `y' = (v−cy)/fy`) by

```
F_u = fx · [ x'y'·ωx − (x'²+1)·ωy +    y' ·ωz ]
F_v = fy · [ (y'²+1)·ωx − x'y'·ωy −    x' ·ωz ]
```

This is encoded as a **(H, W, 2, 3) matrix** `C_mat` built once by
`build_kinematic_matrix()`. Then `F_target = C_mat @ R` is a single `einsum`.

**Cost:** `C_Kin = Σ ‖F − C·R‖²`

**Closed-form optimal R** (Eq. 6.50, footnote 18 of thesis):

```
R* = (Σ Cᵀ·C)⁻¹ · (Σ Cᵀ·F) = M⁻¹ · v
```

`M = Σ Cᵀ·C` is a constant 3×3 matrix; `M⁻¹` is precomputed at network
construction. The R update is then just a blend toward `R*`.

### 2.4 The β-scale ambiguity (read this!)

For **any β ≠ 0**, the substitution

```
G → β·G    I → β·I    F → F/β    R → R/β
```

satisfies all three constraints simultaneously. There is no information in
`V` alone that fixes β. See [§6.1](#61-the-β-scale-ambiguity) for what this
does to the inference dynamics and how `F_DECAY` / `G_DECAY` break it.

---

## 3. The pipeline end-to-end (data → plot)

```
                ┌────────────────────┐
events.txt ────►│ data_loader        │
calib.txt  ────►│  EventFrameSequence│──► (V_k, t_mid_k)  k = 0…N-1
                └────────────────────┘            │
                                                  ▼
                                       ┌────────────────────┐
                                       │  net.step(V_k,     │
config.py  (deltas, F/G_DECAY) ───────►│   n_iters=N_iter,  │
                                       │   f_decay=F_DECAY, │
                                       │   g_decay=G_DECAY) │
                                       └────────────────────┘
                                                  │
                                                  ▼
                                         net.R, net.F, net.G, net.I

                                                  │
   groundtruth.txt ───► gt_omega_body ────┐       │
   imu.txt         ───► imu_omega    ─────┤       │
                                          ▼       ▼
                                   ┌───────────────────────┐
                                   │  validation: compare  │
                                   │  estimated ω vs       │
                                   │  GT ω vs IMU ω        │
                                   └───────────────────────┘
                                                  │
                                                  ▼
                                          matplotlib plots
```

**Five high-level steps:**

1. **Find a segment of constant ω** — `find_segments.py` slides a window over
   `imu.txt`, detects intervals with low gyro std, and prints a config block.
   The user copies it into `DATASET_CONFIGS` in `config.py`.
2. **Bin events into V frames** — `EventFrameSequence` reads events between
   `t_start` and `t_start + n_frames·Δt`, then yields `(V, t_mid)` per frame.
   Each event contributes `+1` (ON) or `−1` (OFF) at its pixel; the count is
   clipped to `±clip_value` and normalised to `[−1, 1]`.
3. **Build the network** — `InteractingMapsThesis(H, W, fx, fy, cx, cy, **deltas)`
   precomputes `C_mat`, `M`, `M⁻¹`. `initialize_from_rotation(R_init)` sets
   `R = R_init` and `F = C·R_init` (mutually consistent).
4. **Iterate per frame** — `net.step(V, n_iters)`:

   - **Inter-frame**: `F` is pulled toward `C·R_current` by `f_decay`; `G` is
     scaled by `g_decay`.
   - **Then `n_iters` of two-phase message passing**:
     Phase 1 (all costs → gradients) then Phase 2 (all quantities update).
   - **Stability clipping** at every iteration:
     `I ∈ [−10, 10]`, `G ∈ [−5, 5]`, `F ∈ [−10, 10]`, `R ∈ [−1, 1]`.
5. **Read out and plot** — `net.R / Δt` converts back to rad/s. Reference ω
   comes from quaternion differencing of `groundtruth.txt` (drift-free) and
   direct IMU gyro averages from `imu.txt`.

---

## 4. Repository map

```
InteractingMaps/
├── ARCHITECTURE.md                 ← this file
├── README.md
├── requirements.txt
│
├── config.py                       (1) shared knobs: deltas, ITERS_PER_FRAME, F_DECAY, G_DECAY,
│                                       and per-dataset configs (t_start, n_frames, initial_R…)
│
├── data_loader.py                  (2) calibration parser + event bin→V frame + frame-iterator
│
├── interacting_maps/
│   ├── __init__.py
│   ├── camera.py                   (3) compute_calibration + build_kinematic_matrix
│   ├── network.py                  (4) Cook 2011 (Gauss-Seidel sequential)
│   └── network_dissertation.py     (5) Martel 2019 thesis (Jacobi two-phase)
│
├── simulation/
│   ├── __init__.py
│   └── event_sim.py                (6) synthetic image + DVSSimulator for tests
│
├── find_segments.py                (7) discover constant-ω intervals from imu.txt
│
├── demo.py                         (8) multi-frame animation
├── demo_single_packet.py           (9) single-V iteration animation
├── validation.py                  (10) ω comparison vs GT & IMU
│
└── data/<dataset_name>/
    ├── events.txt
    ├── calib.txt
    ├── imu.txt
    └── groundtruth.txt
```

---

## 5. Module-by-module deep dive

### 5.1 `config.py`

Pure-data module imported by every script. Has no logic, only knobs.

**Key constants:**

- `DATASET_CONFIGS` — dict keyed by dataset name. Each entry:

  - `t_start` — start time (s) of the constant-ω segment.
  - `frame_duration` — Δt for one event-packet, default `0.020 s`.
  - `n_frames` — how many packets the validation run consumes.
  - `initial_R` — `(3,)` array, R (rad/frame) for cold-start init. Comes from
    `find_segments.py`.
  - `expected_omega` — `(3,)` array, mean ω (rad/s) of the segment from IMU.
- `THESIS_PARAMS` — relaxation step sizes for `InteractingMapsThesis`:

  - `delta_VFG = 0.20` — OFCE gradient step on F & G. Must be strong because
    it competes with the kinematics cost.
  - `delta_IG = 0.10` — Spatial cost step on G (toward ∇I).
  - `delta_GI = 0.05` — Spatial cost step on I. Gentler (it's a Poisson-like
    step and needs stability).
  - `delta_RF = 0.01` — Kinematics step on F (toward C·R). **Weak** — we want
    OFCE to dominate the local structure of F and only let kinematics provide
    a soft bias.
  - `delta_FR = 0.80` — Kinematics step on R (toward `M⁻¹·v`). Strong because
    the closed-form aggregate is stable.
- `COOK_PARAMS` — same keys but tuned for the Cook 2011 sequential network.
- `ITERS_PER_FRAME = 75` — inner-loop iterations per V frame in the validation
  / demo scripts.
- `F_DECAY = 0.5`, `G_DECAY = 0.7` — inter-frame state-management knobs. See
  [§6.2](#62-inter-frame-state-management-f_decay--g_decay).

**Functions:**

- `get_dataset_paths(dataset_name, base_dir=None)` → dict with absolute paths
  to `events.txt`, `calib.txt`, `imu.txt`, `groundtruth.txt`. Defaults
  `base_dir` to `./data/`.

---

### 5.2 `data_loader.py`

Reads RPG-Event-Camera-Dataset format files (Mueggler et al., 2017) and
converts a time window of events into a V frame.

#### `CameraCalibration`

Parses `calib.txt` (format `fx fy cx cy k1 k2 p1 p2 k3`).

- `self.fx, self.fy` — focal lengths (px).
- `self.cx, self.cy` — principal point (px).
- `self.dist` — 5 distortion coefficients (parsed but **not** applied; the
  network assumes a pinhole model).
- `sensor_size()` — infers `(H, W) = (2·cy, 2·cx)`.
- `crop_calibration(x_start, y_start)` — utility, currently unused.

#### `load_events`, `load_events_fast`

Two loaders for `events.txt` (rows: `t x y polarity`).

- `load_events` — line-by-line scan; safer but slow.
- `load_events_fast` — does one Python scan to count rows to skip until
  `t_start`, then `np.loadtxt(..., skiprows=skip, max_rows=5_000_000)`.
  Returns rows with `t < t_start + duration`. Used by `EventFrameSequence`.

#### `events_to_vframe(events, H, W, x_offset, y_offset, clip_value, normalise)`

The core "binning" function:

```
+1  per ON  event  (polarity == 1)
-1  per OFF event  (polarity == 0)
```

Uses `np.add.at(V, (ys, xs), signed)` for **unbuffered** scatter-add (handles
multiple events at the same pixel correctly). After accumulation:

```python
V = np.clip(V, -clip_value, +clip_value)
if normalise:
    V /= clip_value           # → V is in [-1, +1]
```

#### `EventFrameSequence`

Iterator that yields `(V, t_mid)` per frame. Constructed with:

```python
EventFrameSequence(events_txt, calib_path,
                   frame_duration=0.020,
                   t_start=0.5,
                   n_frames=50,
                   clip_value=3.0)
```

Key fields after construction:

- `self.H, self.W` — full sensor size from calibration (no crop).
- `self.calib` — `CameraCalibration`.
- `self._events` — pre-loaded `(N, 4)` array covering the whole sequence.

`__iter__` walks `k = 0…n_frames−1`, masks events with `t ∈ [t_lo, t_hi)`,
calls `events_to_vframe`, yields `(V, (t_lo + t_hi)/2)`.

> ⚠️ **frame_duration plays two roles** simultaneously: (a) event
> accumulation window, and (b) the "Δt" used to convert `R` (rad/frame) into
> ω (rad/s). They must be the same value.

---

### 5.3 `interacting_maps/camera.py`

#### `compute_calibration(H, W, fx, fy, cx, cy) → C : (H, W, 3)`

For every pixel `(col, row)` computes the **unit direction vector** of the
ray through that pixel:

```
x_n = (col - cx) / fx
y_n = (row - cy) / fy
z_n = 1
C[row, col] = (x_n, y_n, 1) / ‖·‖
```

This `C` is the per-pixel calibration map used in the original Cook 2011
formulation (which uses `cross(R, C)` to compute predicted flow).

#### `build_kinematic_matrix(H, W, fx, fy, cx, cy) → C_mat : (H, W, 2, 3)`

Pre-builds the **pixel-flow-per-unit-ω** matrix (Thesis Eq. 6.38). Each pixel
gets a 2×3 matrix such that

```
F[h, w] = C_mat[h, w] @ R
```

reproduces the perspective-correct rotational flow:

```python
xp = (col - cx) / fx       # normalised x'
yp = (row - cy) / fy       # normalised y'

C_mat[..., 0, 0] = fx · (xp · yp)
C_mat[..., 0, 1] = fx · (−(xp² + 1))
C_mat[..., 0, 2] = fx · yp

C_mat[..., 1, 0] = fy · (yp² + 1)
C_mat[..., 1, 1] = fy · (−xp · yp)
C_mat[..., 1, 2] = fy · (−xp)
```

This is the form **both** networks actually use today (the old `m32` / `m23`
3D-projection helpers are kept commented out at the bottom of the file).

---

### 5.4 `interacting_maps/network.py` (Cook 2011)

`InteractingMaps` — **sequential / Gauss-Seidel** updates. Inside one
iteration each sub-step sees the latest values.

#### State

| Field              | Shape            | Role                                                  |
| ------------------ | ---------------- | ----------------------------------------------------- |
| `self.C`         | `(H, W, 3)`    | unit ray direction per pixel (legacy, still computed) |
| `self._C_mat`    | `(H, W, 2, 3)` | kinematic matrix used in updates                      |
| `self.I`         | `(H+1, W+1)`   | intensity (note: extra row/col for forward diffs)     |
| `self.G`         | `(H, W, 2)`    | gradient                                              |
| `self.F`         | `(H, W, 2)`    | flow                                                  |
| `self.R`         | `(3,)`         | rotation (rad/frame)                                  |
| `self._M_normal` | `(3, 3)`       | `Σ C_matᵀ C_mat` — constant                      |
| `self._M_inv`    | `(3, 3)`       | precomputed inverse                                   |

#### Update rules (all called once per inner iteration in this order)

1. **`update_F_from_VG(V)`** — gradient step on the OFCE cost:

   ```
   e = V + ⟨F, G⟩
   F -= δ_VFG · 2·G·e
   ```
2. **`update_G_from_VF(V)`** — same cost, opposite role:

   ```
   e = V + ⟨F, G⟩
   G -= δ_VFG · 2·F·e
   ```
3. **`update_G_from_I()`** — blend toward ∇I:

   ```
   G ← (1 − δ_IG)·G + δ_IG · ∇I
   ```
4. **`update_I_from_G()`** — discrete negative divergence step:

   - Compute `Ψ = G − ∇I`.
   - `Ψ̂_x[v, u] = Ψ_x[v, u] − Ψ_x[v, u−1]` (boundary zero).
   - `Ψ̂_y[v, u] = Ψ_y[v, u] − Ψ_y[v−1, u]`.
   - `I[v, u] -= δ_GI · (Ψ̂_x + Ψ̂_y)`.
     The minus is because the gradient of `‖G − ∇I‖²` w.r.t. `I` is `+div(Ψ)`,
     but stepping *down* the cost flips the sign.
5. **`update_F_from_RC()`** — blend F toward kinematic prediction:

   ```
   F_pred = C_mat @ R   (einsum)
   F ← (1 − δ_RF)·F + δ_RF·F_pred
   ```
6. **`update_R_from_FC()`** — closed-form least-squares blend:

   ```
   v = Σ C_matᵀ · F      (einsum 'hwji,hwj->i')
   R_new = M⁻¹ · v
   R ← (1 − δ_FR)·R + δ_FR·R_new
   ```

#### Main entry

`step(V, n_iters=20)` — runs all six updates `n_iters` times.

#### Diagnostics

- `residual_VFG(V)` = mean |V + F·G|
- `residual_GI()`   = mean |G − ∇I|

---

### 5.5 `interacting_maps/network_dissertation.py` (Martel 2019)

`InteractingMapsThesis` — the **two-phase Jacobi** message-passing version
based on Algorithm 6.5 of the thesis.

#### Architecture pattern: Quantity + Cost

```python
class Quantity:
    value: ndarray
    gradient_accumulator: ndarray
    def reset_gradient(self):  …
    def add_gradient(self, g): …
    def update(self, lr):       self.value -= lr * self.gradient_accumulator

class Cost:                  # base
    def compute_and_send_gradients(self): …
```

Each iteration:

1. **Phase 1**: all `Quantity.gradient_accumulator`s are zeroed; all `Cost`
   objects compute messages **from the current snapshot** and `add_gradient`
   them.
2. **Phase 2**: every `Quantity.update(1.0)` is called → all quantities move
   simultaneously.

This Jacobi update is fundamentally why `initialize_from_rotation()` matters:
the **first** Phase-1 sees whatever you put in, and if you start with `R≠0`
but `F=0`, the kinematics gradient `error_R = R − M⁻¹·Σ Cᵀ·F` will crush R
to zero before OFCE has time to build any structure in F.

#### The three `Cost` subclasses

**`Cost_OFCE`** — nonlinear OFCE gradient descent:

```python
error  = V + ⟨F, G⟩
grad_F = 2 · error · G    # (H, W, 2)
grad_G = 2 · error · F    # (H, W, 2)
grad_F = clip(grad_F, ±max_grad)  # per-pixel, bounds cubic growth
grad_G = clip(grad_G, ±max_grad)
F.add_gradient(grad_F · δ_VFG)
G.add_gradient(grad_G · δ_VFG)
```

**`Cost_Spatial`** — linear `G = ∇I` blend:

- G receives `error · δ_IG` where `error = G − ∇I`.
- I receives the **discrete negative divergence** of `error`, scaled by
  `δ_GI`. Identical formula to `network.py`'s `update_I_from_G`.

**`Cost_Kinematics`** — linear `F = C·R` blend, closed-form for R:

Precomputed at construction:

```python
M = einsum('hwji,hwjk->ik', C_mat, C_mat)  # (3, 3) — Σ Cᵀ C
M_inv = inv(M)
```

Per iteration:

```python
f_target = einsum('hwij,j->hwi', C_mat, R)        # (H, W, 2)  C·R
error_F  = F - f_target
F.add_gradient(error_F · δ_RF)

v        = einsum('hwji,hwj->i', C_mat, F)        # (3,)  Σ Cᵀ F
R_target = M_inv @ v                              # closed-form optimum
error_R  = R - R_target
R.add_gradient(error_R · δ_FR)
```

Key insight (why R can get stuck): if `F = β · C·R` for *any* β, then
`R_target = β · M⁻¹·M·R = β·R`, so `error_R = R(1 − β)`. When the network
reaches the β-locked state (β=1 on the *current* state), `error_R = 0`.

#### `InteractingMapsThesis` orchestrator

Constructor builds:

- `_C_mat` from `build_kinematic_matrix`.
- Five `Quantity` objects (`q_V, q_I, q_G, q_F, q_R`).
- Three `Cost` objects pulling from a shared `q_dict`.

Convenience properties: `net.I, net.G, net.F, net.R` → the underlying values.

`initialize_from_rotation(R_init)`:

- `q_R.value = R_init`
- `q_F.value = einsum('hwij,j->hwi', _C_mat, R_init)`  ← keeps F kinematically
  consistent with R.
- `q_I.value = small Gaussian noise` (seed `42`) so that `∇I ≠ 0` and G has
  something to bootstrap from.
- G left at 0 — OFCE will rebuild it.

`reset(scale=0.01)` — alternative: pure random init, no R/F consistency.
Used by the Cook network.

`step(V, n_iters, f_decay=0.5, g_decay=0.7)`:

```python
self.q_V.value = V

# --- Inter-frame state management ---
if f_decay > 0.0:
    F_kin = einsum('hwij,j->hwi', _C_mat, R_current)
    F = (1 − f_decay)·F + f_decay·F_kin    # re-anchor

if g_decay < 1.0:
    G *= g_decay                           # decay

# --- Inner loop ---
for _ in range(n_iters):
    for q in [I, G, F, R]: q.reset_gradient()
    for cost in costs:     cost.compute_and_send_gradients()
    for q in [I, G, F, R]: q.update(1.0)

    # Stability clipping
    I  = clip(I, ±10)
    G  = clip(G, ±5)
    F  = clip(F, ±10)
    R  = clip(R, ±1)
```

Setting `f_decay=0, g_decay=1.0` disables both decays (legacy behaviour).
See [§6.2](#62-inter-frame-state-management-f_decay--g_decay).

`residual_VFG(V)`, `residual_GI()` — same diagnostics as Cook.

---

### 5.6 `simulation/event_sim.py`

Used only when no real `events.txt` is found, so the demo scripts still run.

- `make_synthetic_image(H, W, kind='checkerboard'|'random'|'gradient')`.
- `rotation_flow(R, C, f)` — ground-truth F from a known R via the kinematic
  formula (mirrors what `C_mat @ R` does).
- `compute_V(image, flow)` — `V = − dot(flow, ∇image)`. This is the
  brightness-constancy equation used as a *forward* model.
- `DVSSimulator(H, W, f, image_kind, noise_std, rng_seed)` — provides
  `.frame_from_rotation(R)` for one-shot tests in `demo_single_packet.py`.

---

### 5.7 `find_segments.py`

**Not part of the inference pipeline.** A standalone helper to discover
intervals in `imu.txt` where ω is approximately constant:

1. Load IMU (`t, ax, ay, az, gx, gy, gz`).
2. Slide a window of `window_duration` seconds across the gyro signal.
3. Compute per-axis std inside the window.
4. Mark windows where `max(std) < max_std_threshold` and `‖mean(ω)‖ ≥ min_omega`.
5. Merge consecutive matches; print a copy-pasteable `DATASET_CONFIGS` block:

```python
'<dataset>': {
    't_start': ...,
    'frame_duration': 0.020,
    'n_frames': 25,
    'initial_R': np.array([...]) * frame_duration,
    'expected_omega': np.array([...]),
},
```

Run as `python find_segments.py data/<dataset>/imu.txt`.

---

### 5.8 `demo.py`

Multi-frame animated demo. **Top of file** sets `DATASET`, `USE_THESIS_VERSION`.

Flow:

1. `make_real_source()` → `(frames, calib, H, W)` from `EventFrameSequence`.
2. Build net (`InteractingMapsThesis` or `InteractingMaps`), call
   `initialize_from_rotation(initial_R)` (thesis) or `reset()` + `net.R = …`
   (Cook).
3. Matplotlib figure with 6 axes:
   - top row: V (cmap `RdBu`), I (gray), |G| (hot), F (HSV-coded).
   - bottom row: R bar chart + residual semilog-y plot.
4. `FuncAnimation`: every frame consumes the next `(V, _)`, calls
   `net.step(V, n_iters=ITERS_PER_FRAME, f_decay=F_DECAY, g_decay=G_DECAY)`,
   updates all artists, appends to `res_VFG_hist` / `res_GI_hist`.
5. Console prints per frame.

Helper functions:

- `normalise(x)` — min-max to `[0, 1]` for display.
- `flow_to_rgb(F)` — HSV: hue = angle, value = magnitude.
- `flow_norm(F)` — Euclidean norm for `|G|` / `|F|` display.

---

### 5.9 `demo_single_packet.py`

**One V frame, many iterations**. Designed to expose the *internal* network
dynamics (cold-start from `initial_R`, watching maps emerge).

Key knobs (top of file):

- `DATASET`, `MAX_ITERS = 100`, `ANIM_INTERVAL_MS = 120`, `CLIP_VALUE = 10.0`.
- `USE_GT_INIT` — if `True`, computes the exact body-frame ω from
  `groundtruth.txt` at the packet's `t_mid` and uses it as `R_init` (perfect
  conditions). If `False`, falls back to `cfg['initial_R']` (approximate).

Inline helpers:

- `_quat_to_rotmat(q)` — qx,qy,qz,qw → R (`R_wc`).
- `_gt_omega_body(gt_data, t_mid, dt)` — finds the two GT poses bracketing
  `[t_mid − dt/2, t_mid + dt/2]`, computes `dR_body = R1ᵀ R2`, extracts
  axis-angle, returns `axis · angle / actual_dt` (rad/s body frame).

Loading:

- If `events.txt + calib.txt` exist → `EventFrameSequence(n_frames=1)`,
  takes `V = frames[0]`.
- Otherwise → falls back to `DVSSimulator`.

Per animation frame, `update(i)`:

1. `net.step(V, n_iters=1, f_decay=F_DECAY, g_decay=G_DECAY)`.
2. Appends `(i, R, residual_VFG, residual_GI)` to history buffers.
3. Updates the four spatial-map images (I, |G|, F).
4. Redraws R line plot, residual semilog-y plot.
5. Updates suptitle with current iteration / residual / |R|.

Final block prints comparison vs `cfg['expected_omega']` and (if available)
GT ω.

---

### 5.10 `validation.py`

Quantitative per-frame ω validation. Reads:

- `groundtruth.txt`  — `[t, tx, ty, tz, qx, qy, qz, qw]` from a Vicon / OptiTrack
  motion-capture system.
- `imu.txt`         — `[t, ax, ay, az, gx, gy, gz]` from the camera's onboard
  IMU (gyro is **directly** ω in the camera body frame).

#### Helpers

- `load_groundtruth(path)` / `load_imu(path)` — `np.loadtxt` wrappers with
  range-prints.
- `quat_to_rotmat(q)` — same convention as `demo_single_packet.py`.
- `rotmat_to_axisangle(R)` — `cos_a = (trace − 1)/2`, axis from
  skew-symmetric part; returns `(angle ≥ 0, unit_axis)`.
- `gt_omega_body(gt_data, t_mid, dt)`:
  - Finds GT pose indices bracketing `[t_mid − dt/2, t_mid + dt/2]`.
  - `dR_body = R1ᵀ R2` (right-invariant, body frame — **NOT** `R2 R1ᵀ`,
    which gives world-frame ω).
  - Returns `axis · angle / actual_dt` (rad/s).
- `imu_omega(imu_data, t_lo, t_hi)`:
  - Averages gyro columns `[4:7]` of all IMU samples in `[t_lo, t_hi)`.
  - Falls back to nearest sample if the window is empty.

#### Knobs (top of file)

- `DATASET` — picks the entry from `DATASET_CONFIGS`.
- `USE_THESIS_VERSION` — Thesis network vs Cook network.
- `RESEED_R_FROM_GT` — if `True`, **overwrite `net.q_R.value` (or `net.R`)
  with `ω_GT(t_mid) · Δt` at the start of every frame**. This is the
  "perfect oracle" mode: it isolates per-frame inference quality from the
  carry-over dynamics. Setting it to `False` is the standard recurrent run.

#### `run_validation(gt_data, imu_data=None)`

1. Builds `EventFrameSequence`.
2. Builds the network and calls `initialize_from_rotation(initial_R)`
   (thesis) or `reset() + net.R = initial_R` (Cook).
3. Per frame `k`:
   - `omega_ref = gt_omega_body(gt_data, t_mid, FRAME_DURATION)`.
   - If `RESEED_R_FROM_GT`: `net.q_R.value = omega_ref · FRAME_DURATION`.
   - `net.step(V, n_iters=ITERS_PER_FRAME, f_decay=F_DECAY, g_decay=G_DECAY)`.
   - `omega_est = net.R / FRAME_DURATION`.
   - `omega_imu_k = imu_omega(...)`.
   - Print one row of the console table.
4. Returns `(times, omega_est, omega_gt, omega_imu, label)`.

#### `plot_results(...)`

4-panel figure:

- Row 1–3: ω_x, ω_y, ω_z over time. Three curves per panel:
  GT (black solid), IMU (green solid), network estimate (color-coded dashed).
- Row 4: per-frame angular error magnitude vs GT (purple solid) and vs IMU
  (green dashed), with mean lines.

Suptitle includes the validation mode string ("GT-reseeded" or
"R carried over") so different runs can be told apart.

---

## 6. Cross-cutting concerns

### 6.1 The β-scale ambiguity

For any β ≠ 0:

| Substitution                  | Effect on each constraint                              |
| ----------------------------- | ------------------------------------------------------ |
| `G → β·G`, `F → F/β` | `V + (F/β)·(β·G) = V + F·G` unchanged ✓        |
| `I → β·I`                | `∇(β·I) = β·∇I = β·G` ✓ (matches the new G) |
| `R → R/β`                 | `C·(R/β) = (C·R)/β` ✓ (matches the new F)       |

So `V` alone cannot determine the scale. Practically:

- **Without inter-frame decay**, once the network reaches a stable β ≠ 1
  fixed point, `error_R = R − M⁻¹·Σ Cᵀ·F = R − R = 0`, so R **never
  changes** in subsequent frames. This is the "flat-line R" pathology.
- **The OFCE clipping** at `±max_grad` interacts with this: large `|G|·|F|`
  values bound the cubic gradient growth but don't fix the scale.

### 6.2 Inter-frame state management (F_DECAY / G_DECAY)

Implemented inside `InteractingMapsThesis.step()` and configured via
`config.F_DECAY = 0.5` and `config.G_DECAY = 0.7`.

**F_DECAY**: re-anchor F toward the kinematic prediction at the **current** R:

```python
F_kin   = C·R_current
F       = (1 − f_decay)·F_prev + f_decay·F_kin
```

Why this works: it forces F into a state where `OFCE_residual = V + F·G ≠ 0`,
which then produces a non-zero gradient to update G away from the locked-in
β-scaled value.

**G_DECAY**: shrink G each new frame:

```python
G *= g_decay
```

When G is small and F = C·R, OFCE's gradient on G is `2 · V · F`. This pulls
G toward `−V / (C·R)` — which is the **correct-scale** G. Without the decay,
the stale β·G_true from the previous frame keeps satisfying OFCE and OFCE
exerts no corrective pressure on the scale.

Disable: `f_decay=0.0`, `g_decay=1.0`.

### 6.3 Coordinate frames and conventions

- **Pixel convention**: row = `v` (y, vertical), col = `u` (x, horizontal).
  All 2-vector maps use `[..., 0] = x-component, [..., 1] = y-component`.
- **Forward differences**: `∂I/∂x[v, u] = I[v, u+1] − I[v, u]`,
  `∂I/∂y[v, u] = I[v+1, u] − I[v, u]`. Boundary = 0.
- **Quaternions in `groundtruth.txt`**: order `(qx, qy, qz, qw)`. The rotation
  matrix built from this is `R_wc` (camera-to-world, active convention).
- **Angular velocity frames**:
  - `dR/dt = R · [ω_body]×`  ⇒  `R1ᵀ R2 ≈ expm([ω_body Δt])`.
  - `dR/dt = [ω_world]× · R`  ⇒  `R2 R1ᵀ ≈ expm([ω_world Δt])`.
    The network and IMU both measure ω in the **body frame**; therefore GT ω
    must be computed as `R1ᵀ R2`, not `R2 R1ᵀ`.

### 6.4 Unit conventions (rad/s vs rad/frame)

The network's `R` lives in **rad/frame** (it's the rotation that should be
applied during one `frame_duration` to predict `V`).

Conversion appears in three places:

| Conversion                   | Where                                                      | Direction          |
| ---------------------------- | ---------------------------------------------------------- | ------------------ |
| `R_init = ω_GT · Δt`    | `demo_single_packet.py`, `validation.py` (oracle mode) | rad/s → rad/frame |
| `ω_est = R_net / Δt`     | every script                                               | rad/frame → rad/s |
| `expected_omega` in config | rad/s, from IMU                                            | reference only     |

If you ever pass an IMU/GT ω directly into the network without multiplying by
`frame_duration`, R will be off by a factor of ~50 (with `Δt = 20 ms`).

---

## Appendix A — How to debug a flat-R run

1. **Print `|R|`, `|F|`, `|G|`, `|I|` every iteration**. If `|F| ≈ |C·R|`
   exactly across many iterations, you're in the β-locked state.
2. **Check `RESEED_R_FROM_GT = True` in `validation.py`**. If `omega_est`
   tracks GT closely under reseed but flat-lines without it → the inner loop
   is correct, the inter-frame dynamics are at fault → tune `F_DECAY` /
   `G_DECAY`.
3. **If reseed *also* fails** → the inner-loop messages are wrong. Likely
   suspects: sign error in `Cost_Spatial`, wrong `einsum` in
   `Cost_Kinematics` (`hwji,hwj->i` is the correct contraction for
   `Σ Cᵀ·F`), or `max_grad` clip too aggressive (try `max_grad=50`).
4. **Verify `gt_omega_body`** by plotting it against `imu_omega` directly.
   They should agree to within ~0.01 rad/s except for IMU bias drift. If
   `gt_omega_body` is rotated by ~π around an axis, you've used `R2 R1ᵀ` by
   mistake.

## Appendix B — Quick reference: who calls `net.step`?

| Caller                    | `n_iters`               | `f_decay`       | `g_decay`       |
| ------------------------- | ------------------------- | ----------------- | ----------------- |
| `demo.py`               | `ITERS_PER_FRAME` (75)  | `F_DECAY` (0.5) | `G_DECAY` (0.7) |
| `demo_single_packet.py` | `1` (called repeatedly) | `F_DECAY`       | `G_DECAY`       |
| `validation.py`         | `ITERS_PER_FRAME`       | `F_DECAY`       | `G_DECAY`       |

All three pass the same `F_DECAY` / `G_DECAY` so tuning is a one-place change
in `config.py`.
