# Persistence and block-matching network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A third network type, `InteractingMapsPersist`, adding Martel's persistence
term (§6.5) and block-matching flow (§6.8.3) to the thesis network, available as the
models `thesis_persist`, `thesis_bm`, `thesis_persist_bm` in the existing harness.

**Architecture:** The rotation warp is a new stand-alone core module
(`interacting_maps/rotation_warp.py`). It holds the camera rays, exp-map warp,
Jacobian, bilinear sampling, the Gauss–Newton rotation solve and block matching. A
new module `interacting_maps/network_persistence.py` subclasses
`InteractingMapsThesis`, adds the quantity Ǐ, adds two `Cost_*` classes and
overrides `step()`. The harness gets additive model branches only.

**Tech Stack:** Python, NumPy, OpenCV (`cv2.undistortPoints`, `cv2.Rodrigues`),
SciPy (`scipy.ndimage`), as already used in the repo.

**Spec:** `docs/superpowers/specs/2026-11-28-persistence-block-matching-design.md`

## Global Constraints

- `network_dissertation.py`, `network.py`, `angular_velocity.py` are **not edited**. No existing signature changes.
- The existing models behave byte-identically. `PYTHONPATH=. python tests/test_golden_tracking.py` prints the same `[ok]` lines before and after every task.
- Tests are standalone scripts (not pytest), run from the worktree root as `PYTHONPATH=. python tests/<file>.py`. They print `[ok]`/`[FAIL]` and exit non-zero on failure.
- Smoke runs only locally (≤ 3 frames × 10 iterations). Nothing is written under `results/`; test runs go to `experiments/persist_test` and are removed by the test.
- Import only from core modules (`interacting_maps/*`, `config`, `run_config`), never from scripts.
- Units: `R` is rad/frame (the per-frame rotation vector, Martel's R_δ = exp([R]×)). Any ω shown is in deg/s.
- Pixel indexing: `[...,0] = x` (cols), `[...,1] = y` (rows). Flow convention, shared by F, F′ and the warp: `I(p) ≈ Ǐ(p − F(p))` (flow from the previous frame to the current one, px/frame).
- Default `DISTORTION_MODE = 'C_full'`: I lives in distorted pixels. The warp undistorts through `cv2.undistortPoints` and re-distorts with Brown–Conrady `[k1,k2,p1,p2,k3]`, like `camera.build_kinematic_matrix`.
- Frame start, as Martel does it: Ǐ ← I at the start of each frame (from frame 1 on). I is **not** pre-warped. On frame 0 the new costs are inactive.
- Between frames all maps carry over (no reset). The frame-0 R comes from `rc.initial_R`, whatever the harness's R-initialisation flag sets. No new flag.
- Commits: plain message, authored by the local git config. No `Co-authored-by` or "Generated with" lines. Never push.

## Review Focus

1. **Warp leaves the image** (large R, border pixels). Those pixels are excluded from both the residual and the I update. No NaN anywhere. Test in Task 1 (`check_warp_outside_masked`).
2. **Textureless reference** (flat Ǐ, e.g. a frame whose I is still near the init noise level). The Gauss–Newton step must not blow up: it returns a finite R and leaves R unchanged where the system is rank-deficient. Test in Task 2 (`check_gn_flat_reference`).
3. **Frame start, I == Ǐ.** The solve returns R* ≈ 0, as the spec expects. Test in Task 2 (`check_gn_identical_images`).
4. **Block-matching ties in flat regions.** A flat image must give F′ = 0, not an arbitrary shift. Test in Task 3 (`check_bm_flat_zero`).
5. **Sweep with a wrong model/parameter pair.** A persistence delta passed to an old model must be rejected by `RunConfig` (ValueError). Old models' sweep csvs keep exactly their columns. Test in Task 5 (`check_old_model_rejects_persist_delta`, `check_sweep_columns`).

---

### Task 1: Rotation warp core module

**Files:**
- Create: `interacting_maps/rotation_warp.py`
- Test: `tests/test_persistence.py` (create, with a `CHECKS` list and a `main()` that runs every check, prints `[ok] name` / `[FAIL] name: msg`, and returns non-zero on any failure; later tasks append checks)

**Interfaces:**
- Produces:
  - `Camera = namedtuple('Camera', 'fx fy cx cy dist')`, where `dist` is a `None` or `(5,)` array.
  - `pixel_rays(H: int, W: int, cam: Camera) -> np.ndarray`, shape `(H, W, 3)`: unit rays of each native pixel. With `dist`, undistort via `cv2.undistortPoints`.
  - `rotation_matrix(R: np.ndarray) -> np.ndarray`, shape `(3, 3)`: the matrix the warp applies to the rays (exp-map of ±R; the sign is fixed by `check_warp_matches_C`).
  - `compose(delta: np.ndarray, R: np.ndarray) -> np.ndarray`, shape `(3,)`, with `rotation_matrix(compose(d, R)) == rotation_matrix(d) @ rotation_matrix(R)`.
  - `warp_coords(rays, R, cam) -> (u, v)`, each of shape `(H, W)`: T_R(p), the position in Ǐ that pixel p of I came from.
  - `warp_jacobian(rays, R, cam) -> np.ndarray`, shape `(H, W, 2, 3)`: ∂T_{compose(Δ,R)}(p)/∂Δ at Δ = 0. Uses the projection Jacobian and, with `dist`, the Brown–Conrady Jacobian (same formulas as `camera.build_kinematic_matrix`).
  - `bilinear_sample(img, u, v) -> (vals, valid)`: `vals` is `(H, W)`; `valid` is a bool `(H, W)` that is False where `(u, v)` is outside `[0, W-1] × [0, H-1]`; `vals` is 0 there.

- [ ] **Step 1: Write the failing checks** in `tests/test_persistence.py`. Use a pinhole camera `cam0 = Camera(200., 200., 120., 90., None)`, `H, W = 180, 240`, and the poster calibration (`CameraCalibration(get_dataset_paths('poster_rotation')['calib'])`) for the distorted case `camd`.
  - `check_warp_identity`: `warp_coords(rays, zeros(3), cam)` equals the pixel grid within 1e-6 px for `cam0` and within 1e-3 px for `camd`.
  - `check_warp_matches_C`: for `R = [2e-4, -3e-4, 1e-4]`, `warp_coords(rays, R, cam) − grid ≈ −(C·R)` with `C = build_kinematic_matrix(H, W, fx, fy, cx, cy, dist_coeffs=cam.dist)`. Check over interior pixels, abs error < 1e-2 × |C·R| max, for both cameras. This pins the sign convention to F's.
  - `check_compose`: for random small d and R, `rotation_matrix(compose(d, R))` equals `rotation_matrix(d) @ rotation_matrix(R)` within 1e-10.
  - `check_warp_jacobian_fd`: at `R = [0.01, -0.02, 0.015]`, for k = 0..2, `(warp_coords(compose(eps·e_k, R)) − warp_coords(R)) / eps` with `eps = 1e-6` matches `warp_jacobian[..., k]` within 1e-3 relative, for both cameras.
  - `check_warp_outside_masked`: with `R = [0, 0.5, 0]`, `bilinear_sample(img, *warp_coords(...))` gives `valid.sum() < H*W`, `np.isfinite(vals).all()`, and `vals[~valid] == 0`.
- [ ] **Step 2: Run, expect failure:** `PYTHONPATH=. python tests/test_persistence.py` fails with `ModuleNotFoundError: interacting_maps.rotation_warp`.
- [ ] **Step 3: Implement the module** with the signatures above. `rotation_matrix`/`compose` go through `cv2.Rodrigues`. `warp_coords` rotates the rays, projects them with `z` guarded at 1e-9, and applies the forward distortion when `cam.dist` is set.
- [ ] **Step 4: Run, expect all `[ok]`;** also run `PYTHONPATH=. python tests/test_golden_tracking.py` (unchanged `[ok]`).
- [ ] **Step 5: Commit:** `git add interacting_maps/rotation_warp.py tests/test_persistence.py && git commit -m "rotation warp core module for the persistence term"`

### Task 2: Gauss–Newton rotation solve (Martel Eqs. 6.73–6.98)

**Files:**
- Modify: `interacting_maps/rotation_warp.py` (append)
- Test: `tests/test_persistence.py` (append checks)

**Interfaces:**
- Consumes: Task 1's `Camera`, `pixel_rays`, `compose`, `warp_coords`, `warp_jacobian`, `bilinear_sample`.
- Produces: `solve_rotation_gn(I, Icheck, grad_Icheck, rays, cam, R0, n_max=20, tol=1e-8, mask=None) -> np.ndarray`, shape `(3,)`.
  - `grad_Icheck = (gx, gy)` is precomputed by the caller (`np.gradient(Icheck)`, x = axis 1).
  - The residual is ρ = I − Ǐ∘T_R on pixels that are valid (and in `mask`, if given).
  - Each iteration builds `J = ∇Ǐ(T_R)ᵀ · warp_jacobian`, of shape (N, 3), solves `J Δ = −ρ` with `np.linalg.lstsq(rcond=1e-10)` (rank-safe), and updates `R ← compose(Δ, R)`.
  - Stop when `|Δ| < tol` or after `n_max` iterations. If fewer than 3 valid pixels remain, or the result is non-finite, return `R0`.

- [ ] **Step 1: Write the failing checks.**
  - `check_gn_known_rotation`:
    - Ǐ is the first poster APS frame (`load_aps_list(paths['data_dir'])`, `nearest_aps`), grey, scaled to [0, 1], and blurred with `scipy.ndimage.gaussian_filter(sigma=1.0)`.
    - `R_true = [0.002, -0.003, 0.002]`. `I, valid = bilinear_sample(Ǐ, *warp_coords(rays, R_true, camd))`.
    - `solve_rotation_gn(..., R0=zeros(3), mask=valid)` and `R0 = 2*R_true` both return `R_true` within `|·| < 1e-4` rad.
  - `check_gn_identical_images`: with `I = Ǐ` (same poster frame), from `R0 = [0.003, 0.002, -0.002]`, the result is within `1e-4` of 0.
  - `check_gn_flat_reference`: with `Ǐ = I = ones((H, W))`, from `R0 = [0.001, 0, 0]`, the result is finite and equal to `R0` within 1e-12.
- [ ] **Step 2: Run, expect failure** (`ImportError: solve_rotation_gn`).
- [ ] **Step 3: Implement `solve_rotation_gn`** as specified. Bilinear-sample `gx`, `gy` at T_R with the same `bilinear_sample`.
- [ ] **Step 4: Run, expect all `[ok]`.**
- [ ] **Step 5: Commit:** `git commit -am "Gauss-Newton rotation solve for the persistence term"`

### Task 3: Block matching (Martel Eq. 6.138)

**Files:**
- Modify: `interacting_maps/rotation_warp.py` (append)
- Test: `tests/test_persistence.py` (append)

**Interfaces:**
- Produces: `block_match(I, Icheck, half_block=3, radius=4) -> (F_prime, valid)`.
  - `F_prime` has shape `(H, W, 2)`, float: integer displacements `d = (dx, dy)` with `|dx|, |dy| ≤ radius` that minimise the SAD between the `(2·half_block+1)²` block of I at p and the block of Ǐ at `p − d`.
  - `valid` is a bool `(H, W)`: False within `half_block + radius` of the border.
  - Ties go to the smallest `|d|`, so zero wins in flat regions: candidates are visited in order of `|d|` with a strict `<` update.
  - Vectorised: one `scipy.ndimage.uniform_filter` of `|I − shift(Ǐ, d)|` per candidate.

- [ ] **Step 1: Write the failing checks.**
  - `check_bm_shift`: Ǐ is the blurred poster frame. `I = np.roll(Ǐ, shift=(2, -3), axis=(0, 1))` (dy = 2, dx = −3). `F_prime[valid]` equals `(-3, 2)` at ≥ 99 % of valid pixels.
  - `check_bm_flat_zero`: `I = Ǐ = ones`, so `F_prime[valid]` is all 0.
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement `block_match`.**
- [ ] **Step 4: Run, expect all `[ok]`.**
- [ ] **Step 5: Commit:** `git commit -am "block-matching flow for the persistence network"`

### Task 4: The network: `InteractingMapsPersist`

**Files:**
- Create: `interacting_maps/network_persistence.py`
- Test: `tests/test_persistence.py` (append)

**Interfaces:**
- Consumes:
  - `network_dissertation.InteractingMapsThesis`, `Cost`, `Quantity`;
  - all of `rotation_warp` (Tasks 1–3).
- Produces:
  - `Cost_Persistence(q_dict, rays, cam, delta_persist_R, eta_persist_I, gn_iters, gn_tol)`:
    - `.set_reference(Icheck)` computes `np.gradient` once per frame and sets `.active = True`; `.active = False` until then.
    - `compute_and_send_gradients()` returns at once if inactive. Otherwise:
      - R side: `R* = solve_rotation_gn(I, Ǐ, grad, rays, cam, R0=current R)`, then `q['R'].add_gradient(delta_persist_R · (R − R*))`.
      - I side (Eq. 6.100): `q['I'].add_gradient(eta_persist_I · (I − Ǐ∘T_R) · valid)`, with the current network R.
  - `Cost_BlockMatching(q_dict, delta_BM, half_block, radius)`, with `.active` like the above. When active: `F′, valid = block_match(I, Ǐ, ...)`, then `q['F'].add_gradient(delta_BM · (F − F′) · valid[..., None])`. I is not updated.
  - `InteractingMapsPersist(H, W, fx, fy, cx, cy, frame_duration=0.005, *, use_persistence=True, use_block_matching=False, delta_persist_R=0.10, eta_persist_I=0.05, delta_BM=0.05, bm_half_block=3, bm_radius=4, gn_iters=20, gn_tol=1e-8, dist_coeffs=None, include_jacobian=True, poisson='iterative', **thesis_deltas)`:
    - calls the base constructor with the base arguments;
    - adds `self.q_Icheck = Quantity((H, W), "Intensity_previous")` and `q_dict['Icheck']`;
    - builds `Camera(fx, fy, cx, cy, dist_coeffs)` and `pixel_rays` once;
    - appends the enabled costs to `self.costs`.
  - `step(V, n_iters=50, omega_imu=None, events=None)`: from the second call on, first sets `q_Icheck.value = I.copy()` and calls `set_reference` / activates the enabled costs; then `super().step(...)`.
  - Docstrings name the spec deviations 1–5. A comment at the frame-start line says "Martel: Ǐ = I(t−δt), I not pre-warped".

- [ ] **Step 1: Write the failing checks.** Build networks with the poster calibration, `THESIS_PARAMS`, and `initialize_from_rotation(R0)` with `R0 = [0.002, -0.001, 0.001]`. Take V from one poster frame via the same loader `evaluation.experiment_tracking` uses (or a fixed random ±1 sparse V with seed 0 if that is simpler, the same for both networks).
  - `check_frame0_equals_thesis`: for each `(use_persistence, use_block_matching)` in `{(T, F), (F, T), (T, T)}`, after `step(V, n_iters=10)`, `I`, `G`, `F`, `R` are `np.array_equal` to an `InteractingMapsThesis` stepped identically.
  - `check_frame1_reference`: after the first step, store `I1 = net.I.copy()`. Then `step(V, n_iters=1)` gives `net.q_Icheck.value == I1` exactly, the enabled costs are `.active`, and all maps are finite.
  - `check_thesis_untouched`: `InteractingMapsThesis.__init__` and `.step` are the same function objects as before the import of `network_persistence` (no monkey-patching).
- [ ] **Step 2: Run, expect failure** (`ModuleNotFoundError: interacting_maps.network_persistence`).
- [ ] **Step 3: Implement the module.**
- [ ] **Step 4: Run all checks plus the golden test; expect all `[ok]`.**
- [ ] **Step 5: Commit:** `git add interacting_maps/network_persistence.py tests/test_persistence.py && git commit -m "InteractingMapsPersist: persistence and block-matching costs"`

### Task 5: Harness wiring (additive)

**Files:**
- Modify:
  - `config.py` (append after `COOK_PARAMS`)
  - `run_config.py` (`RunConfig.__init__` model branches, `output_dir` base dict, `make_network`)
  - `interacting_maps/style.py` (`MODELS`)
  - `evaluation.py:639-641` (`--model` choices)
  - `report/run_experiments.py` (`_parse_vary`, `cmd_sweep` columns)
  - `CLAUDE.md` (the `--model` list line)
- Test: `tests/test_persistence.py` (append)

**Interfaces:**
- Consumes: Task 4's `InteractingMapsPersist`.
- Produces:
  - `config.PERSIST_MODELS = ('thesis_persist', 'thesis_bm', 'thesis_persist_bm')`.
  - `config.PERSIST_PARAMS = dict(delta_persist_R=0.10, eta_persist_I=0.05, delta_BM=0.05, bm_half_block=3, bm_radius=4, gn_iters=20, gn_tol=1e-8)`.
  - `config.PERSIST_DELTAS = ['delta_persist_R', 'eta_persist_I', 'delta_BM']`.
  - `RunConfig` for a `PERSIST_MODELS` model:
    - `params = {**THESIS_PARAMS, **PERSIST_PARAMS}`, `use_thesis=True`, `use_imu=False`;
    - `self.use_persistence = model in ('thesis_persist', 'thesis_persist_bm')`, `self.use_block_matching = model in ('thesis_bm', 'thesis_persist_bm')`;
    - every other model sets both flags False.
    - `output_dir` compares against `{**THESIS_PARAMS, **PERSIST_PARAMS}` for these models, so defaults add no suffix. Unchanged for the others.
  - `make_network` builds `InteractingMapsPersist(..., use_persistence=rc.use_persistence, use_block_matching=rc.use_block_matching, **rc.params)` for these models, then `initialize_from_rotation(rc.initial_R)`. Other branches unchanged.
  - `style.MODELS`: `'thesis_persist': ('persistence', '#56B4E9')`, `'thesis_bm': ('block-match', '#F0E442')`, `'thesis_persist_bm': ('persist+BM', '#000000')`.
  - `run_experiments`: `_parse_vary` accepts `PERSIST_DELTAS` as float axes. `cmd_sweep` uses `deltas = ALL_DELTAS + (PERSIST_DELTAS if args.model in PERSIST_MODELS else [])` for both the header and the row.

- [ ] **Step 1: Write the failing checks.**
  - `check_runconfig_models`: for each model in `PERSIST_MODELS`, `RunConfig(dataset='poster_rotation', segment='seg_C', model=m, n_frames=3, n_iters=10, out_root='experiments/persist_test')` has the flags above, the `output_dir` has no persistence suffix, and `output_dir` starts with `experiments`.
  - `check_old_model_rejects_persist_delta`: `RunConfig(..., model='thesis', deltas={'delta_BM': 0.1})` raises `ValueError`.
  - `check_old_output_dir_unchanged`: `RunConfig(..., model='thesis').output_dir` and `model='thesis_imu'` equal the strings computed from `git show HEAD~N:run_config.py`. Simplest: hard-code the two expected strings, captured before the edit in Step 2.
  - `check_sweep_columns`:
    - `report/run_experiments._parse_vary('delta_BM=0.1,0.2')` returns `[('delta_BM', [0.1, 0.2])]`.
    - A `cmd_sweep` header for `--model thesis` has exactly the current columns (hard-coded list, captured in Step 2), and for `--model thesis_persist` it adds the three `PERSIST_DELTAS` after `delta_shrinkI`.
    - To make the header testable, extract `_sweep_columns(model) -> list[str]` in `run_experiments.py` and use it in `cmd_sweep`.
  - `check_smoke_runs`: for each new model, `evaluation.experiment_tracking(rc, save_frames=False)` with the config above returns a summary with finite `mean_err_deg_s`. `run.json` and `tracking.csv` exist under `experiments/persist_test`. Remove `experiments/persist_test` at the end with `shutil.rmtree`, as the golden test does. Set `IM_MLFLOW=0` at the top of the test file.
- [ ] **Step 2: Capture the expected old strings** (`output_dir` for `thesis` and `thesis_imu`, the current sweep header) by running the current code once, and paste them into the checks. Run the checks: expect failure.
- [ ] **Step 3: Implement the wiring** listed above. `run_experiments` imports `PERSIST_DELTAS, PERSIST_MODELS` from `config`.
- [ ] **Step 4: Run, expect all `[ok]`:**
  - `PYTHONPATH=. python tests/test_persistence.py`
  - `PYTHONPATH=. python tests/test_golden_tracking.py` (same `[ok]` lines as before Task 1)
  - `PYTHONPATH=. python tests/test_viz.py`
  - CLI smoke: `python evaluation.py --exp 2 --dataset poster_rotation --segment seg_C --model thesis_persist_bm --n_frames 3 --n_iters 10 --no-frames --out-root experiments/persist_test`, then delete `experiments/persist_test`.
- [ ] **Step 5: Commit:** `git commit -am "wire thesis_persist, thesis_bm, thesis_persist_bm into the harness"`

---

After Task 5 (not part of this plan): the cluster runs come next. Those are the three
models on the 5 report sequences, 1.5 s, with frame-0 R from the gyro and with R = 0
(the R-init flag from the walkthrough session's branch), plus `--exp 6`. They come
after merging and the `ADOPT=1` step, since `.py` files changed.
