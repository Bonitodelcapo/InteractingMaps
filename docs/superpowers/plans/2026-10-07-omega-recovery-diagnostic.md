# ω-recovery Diagnostic (rungs 0–1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one standalone script, `diag_omega_ladder.py`, that localizes *why the vision-only Interacting-Maps path fails to recover ω* by running oracle-ladder rungs 0, 1a, and 1b on `ecrot_city` seg_A and resolving a pass/fail decision gate.

**Architecture:** A single self-contained script in the repo's existing standalone-test idiom (run directly, prints `[ok]`/`[FAIL]` lines — *not* pytest; see the Tests section of [CLAUDE.md](../../../CLAUDE.md)). It reuses the real pipeline (`EventFrameSequence`, `InteractingMapsThesis`, the `camera.py` R-solve, `metrics.py`) and **only reads the model and injects ground-truth flow** into `q_F`; it never edits model source. Ground-truth flow for a pure-rotation camera is `F* = C·R_gt`.

**Tech Stack:** Python 3 / NumPy; existing modules `data_loader.py`, `interacting_maps/network_dissertation.py`, `interacting_maps/camera.py`, `eval_io.py`, `metrics.py`, `evaluation.py`, `config.py`.

## Global Constraints

- **Target segment:** `ecrot_city` seg_A — synthetic, pure-rotation, pinhole/zero-distortion. Defined explicitly in the script.
- **Segment window:** `t_start = 0.05`, `frame_duration = 0.02` (dt=20ms), `n_frames = 75`, `sensor_size = (180, 240)`.
- **Distortion:** build the network with `dist_coeffs=None` and the sequence with `undistort=False` → pure pinhole `C` everywhere (ecrot calib has zero distortion, so this is exact and removes `C_full` as a confound).
- **Inner iterations:** `n_iters = config.ITERS_PER_FRAME` (= 75), matching reported thesis runs.
- **Hyper-parameters:** `config.THESIS_PARAMS`, `poisson = config.POISSON_MODE` — never hard-code; import from `config`.
- **Scoring reference:** exact `omega_gt` via `eval_io.omega_gt_at(om_data, t_lo, t_hi)` (avoids dependence on the `SCORE_AGAINST` global). `R_gt = omega_ref · frame_duration` (rad/frame).
- **Metrics:** `err`/`dir`/`β` via `metrics.compute_metrics`; report **mean and median**, and score **direction and scale apart**.
- **Model is read-only:** no edits to any file under `interacting_maps/`, `config.py`, or the harness. The only "write" into the model is setting `net.q_F.value` and replacing `net.q_F.update` with a no-op *on the instance* (monkeypatch), to freeze flow.
- **No provenance/MLflow wiring** (diagnostic, not a reported run).
- **Commits:** authored by the local git config; no attribution/Co-authored-by lines; do not push.

---

## File Structure

- **Create:** `diag_omega_ladder.py` (repo root) — the entire diagnostic. One file, four responsibilities kept as separate functions: `setup()` (segment + loaders + helpers), `rung0_baseline()`, `rung1a_kinematics()`, `rung1b_clamped_flow()`, plus a `main()` dispatcher and a `_decision_gate()` printer.

No other files are created or modified. The script is runnable from the repo root exactly like `test_poisson_solvers.py`.

---

### Task 1: Scaffolding — segment, loaders, and oracle helpers

**Files:**
- Create: `diag_omega_ladder.py`

**Interfaces:**
- Consumes (all existing):
  - `data_loader.EventFrameSequence(events_txt, calib_path, frame_duration, t_start, n_frames, clip_value, sensor_size, undistort)` → iterable of `(V, t_mid)`; attrs `.H`, `.W`, `.calib.{fx,fy,cx,cy}`.
  - `evaluation.get_dataset_paths(name)` → dict with keys `events`, `calib`, `imu`, `groundtruth`, `omega_gt`.
  - `eval_io.load_imu(path)`, `eval_io.get_gyro_for_frame(imu, t_lo, t_hi)`, `eval_io.load_omega_gt(path)`, `eval_io.omega_gt_at(om, t_lo, t_hi)`.
  - `interacting_maps.network_dissertation.InteractingMapsThesis(H, W, fx, fy, cx, cy, frame_duration, dist_coeffs, poisson, **params)`; `.initialize_from_rotation(R_init)`; `.step(V, n_iters, omega_imu=None)`; attrs `.q_F.value`, `.q_G.value`, `.R`, `._C_mat` (shape `(H,W,2,3)`).
  - `config.THESIS_PARAMS`, `config.ITERS_PER_FRAME`, `config.POISSON_MODE`.
- Produces (used by later tasks):
  - `SEG` dict: `{'ds':'ecrot_city','t_start':0.05,'frame_duration':0.02,'n_frames':75,'sensor_size':(180,240)}`.
  - `load_segment()` → `(seq, paths, imu, om, frames)` where `frames = list(seq)` (list of `(V, t_mid)`).
  - `frame_windows(n)` → list of `(t_lo, t_hi)` for `k in range(n)`.
  - `ref_omega(om, t_lo, t_hi)` → `(3,)` exact ω (rad/s).
  - `build_net(seq)` → fresh `InteractingMapsThesis` (pinhole, THESIS_PARAMS), **not yet initialized**.
  - `gt_flow(net, R_gt)` → `(H,W,2)` `= einsum('hwij,j->hwi', net._C_mat, R_gt)`.
  - `score_frames(est_list, ref_list)` → dict with `mean_err`, `median_err`, `mean_dir`, `median_dir`, `mean_beta`, plus per-axis `axis_r` (length-3 Pearson r of est vs ref across frames) and `axis_bias` (length-3 mean(est-ref)).

- [ ] **Step 1: Write the scaffolding with an inline self-check**

```python
"""
diag_omega_ladder.py — ω-recovery oracle ladder, rungs 0 / 1a / 1b.

Run directly (NOT pytest). Prints [ok]/[FAIL] + per-rung tables and a decision
gate. Read-only on the model: it only injects ground-truth flow into q_F.
See docs/superpowers/specs/2026-10-07-omega-recovery-diagnostic-design.md.
"""
import argparse
import numpy as np

import config
from evaluation import get_dataset_paths
from eval_io import load_imu, get_gyro_for_frame, load_omega_gt, omega_gt_at
from data_loader import EventFrameSequence
from interacting_maps.network_dissertation import InteractingMapsThesis
from interacting_maps.camera import build_R_normal_equations, solve_R_lstsq
from metrics import compute_metrics, curl_share

SEG = {'ds': 'ecrot_city', 't_start': 0.05, 'frame_duration': 0.02,
       'n_frames': 75, 'sensor_size': (180, 240)}


def load_segment():
    paths = get_dataset_paths(SEG['ds'])
    seq = EventFrameSequence(
        paths['events'], paths['calib'],
        frame_duration=SEG['frame_duration'], t_start=SEG['t_start'],
        n_frames=SEG['n_frames'], clip_value=10.0,
        sensor_size=SEG['sensor_size'], undistort=False,
    )
    imu = load_imu(paths['imu'])
    om = load_omega_gt(paths.get('omega_gt'))
    frames = list(seq)
    return seq, paths, imu, om, frames


def frame_windows(n):
    t0, dt = SEG['t_start'], SEG['frame_duration']
    return [(t0 + k * dt, t0 + (k + 1) * dt) for k in range(n)]


def ref_omega(om, t_lo, t_hi):
    return omega_gt_at(om, t_lo, t_hi)


def build_net(seq):
    c = seq.calib
    return InteractingMapsThesis(
        H=seq.H, W=seq.W, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy,
        frame_duration=SEG['frame_duration'],
        dist_coeffs=None, poisson=config.POISSON_MODE,
        **config.THESIS_PARAMS,
    )


def gt_flow(net, R_gt):
    return np.einsum('hwij,j->hwi', net._C_mat, R_gt)


def score_frames(est_list, ref_list):
    est = np.asarray(est_list)          # (N, 3) rad/s
    ref = np.asarray(ref_list)          # (N, 3) rad/s
    errs, dirs, betas = [], [], []
    for e, r in zip(est, ref):
        err, d, b = compute_metrics(e, r)
        errs.append(err); dirs.append(d); betas.append(b)
    axis_r = []
    for a in range(3):
        ea, ra = est[:, a], ref[:, a]
        ea = ea - ea.mean(); ra = ra - ra.mean()
        den = np.sqrt((ea * ea).sum() * (ra * ra).sum())
        axis_r.append(float((ea * ra).sum() / den) if den > 0 else float('nan'))
    return {
        'mean_err': float(np.mean(errs)), 'median_err': float(np.median(errs)),
        'mean_dir': float(np.mean(dirs)), 'median_dir': float(np.median(dirs)),
        'mean_beta': float(np.mean(betas)),
        'axis_r': axis_r,
        'axis_bias': (est - ref).mean(axis=0).tolist(),
    }


def _selfcheck():
    seq, paths, imu, om, frames = load_segment()
    assert om is not None, "ecrot_city must ship omega_gt.txt"
    assert len(frames) == SEG['n_frames'], f"got {len(frames)} frames"
    wins = frame_windows(SEG['n_frames'])
    w0 = ref_omega(om, *wins[0])
    assert np.linalg.norm(w0) > 1e-3, "omega_gt should be non-trivial"
    net = build_net(seq)
    assert net._C_mat.shape == (seq.H, seq.W, 2, 3)
    print(f"[ok] setup: {len(frames)} frames, |omega_gt[0]|="
          f"{np.linalg.norm(w0):.3f} rad/s, C={net._C_mat.shape}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', default='all',
                    choices=['setup', '0', '1a', '1b', 'all'])
    args = ap.parse_args()
    if args.rung in ('setup', 'all'):
        _selfcheck()
```

- [ ] **Step 2: Run the self-check and verify it passes**

Run: `python diag_omega_ladder.py --rung setup`
Expected: prints `[ok] setup: 75 frames, |omega_gt[0]|=... rad/s, C=(180, 240, 2, 3)` and exits 0. (If `data/ecrot_city/` is absent, run `./download_gdrive.sh` first.)

- [ ] **Step 3: Commit**

```bash
git add diag_omega_ladder.py
git commit -m "Diagnostic: scaffold omega-recovery ladder (segment, loaders, helpers)"
```

---

### Task 2: Rung 0 — baseline vision-only characterization

**Files:**
- Modify: `diag_omega_ladder.py`

**Interfaces:**
- Consumes: `build_net`, `load_segment`, `frame_windows`, `ref_omega`, `score_frames`, `get_gyro_for_frame`, `curl_share`, `build_R_normal_equations`.
- Produces: `rung0_baseline()` → the `score_frames` dict, plus prints the baseline table (`cond(M)` once, per-frame-mean `curl_share`, axiswise r/bias).

- [ ] **Step 1: Add the rung-0 function with a reproduction self-check**

```python
def rung0_baseline():
    """Vision-only thesis net on the segment; characterize the failure."""
    seq, paths, imu, om, frames = load_segment()
    wins = frame_windows(len(frames))
    net = build_net(seq)
    # Warm start from the gyro at t_start (mirrors reported runs: initial_R None
    # => R_init = omega_gyro(t_start) * dt).
    R_init = get_gyro_for_frame(imu, *wins[0]) * SEG['frame_duration']
    net.initialize_from_rotation(R_init)

    cond_M = float(np.linalg.cond(build_R_normal_equations(net._C_mat)))
    est, ref, curls = [], [], []
    for (V, _t), (t_lo, t_hi) in zip(frames, wins):
        net.step(V, n_iters=config.ITERS_PER_FRAME)      # vision-only (no anchor)
        est.append(net.R / SEG['frame_duration'])
        ref.append(ref_omega(om, t_lo, t_hi))
        curls.append(curl_share(net.G))
    s = score_frames(est, ref)
    s['cond_M'] = cond_M
    s['mean_curl_share'] = float(np.mean(curls))
    print("\n=== RUNG 0  vision-only baseline (ecrot_city seg_A) ===")
    print(f"  err  mean/median : {s['mean_err']:7.2f} / {s['median_err']:7.2f} deg/s")
    print(f"  dir  mean/median : {s['mean_dir']:7.2f} / {s['median_dir']:7.2f} deg")
    print(f"  beta mean        : {s['mean_beta']:7.3f}  (1.0 = perfect scale)")
    print(f"  axis_r  (x,y,z)  : {[round(v,3) for v in s['axis_r']]}")
    print(f"  axis_bias(x,y,z) : {[round(v,4) for v in s['axis_bias']]}")
    print(f"  cond(M)          : {cond_M:7.1f}")
    print(f"  curl_share mean  : {s['mean_curl_share']:7.3f}")
    # Reproduction self-check: the run must complete and produce finite metrics.
    ok = np.isfinite(s['mean_dir']) and np.isfinite(s['mean_err'])
    print(f"[{'ok' if ok else 'FAIL'}] rung0 produced finite metrics")
    return s
```

- [ ] **Step 2: Wire into `main` and run it**

Add to the `__main__` block, after the setup branch:

```python
    if args.rung in ('0', 'all'):
        rung0_baseline()
```

Run: `python diag_omega_ladder.py --rung 0`
Expected: prints the RUNG 0 table and `[ok] rung0 produced finite metrics`. Note the `dir` value — this is the baseline failure (reported `thesis` elsewhere is dir≈75–84°; here on clean pinhole synthetic it may differ, which is itself the finding).

- [ ] **Step 3: Commit**

```bash
git add diag_omega_ladder.py
git commit -m "Diagnostic: rung 0 (vision-only baseline characterization)"
```

---

### Task 3: Rung 1a — kinematic self-consistency

**Files:**
- Modify: `diag_omega_ladder.py`

**Interfaces:**
- Consumes: `build_net`, `load_segment`, `frame_windows`, `ref_omega`, `gt_flow`, `score_frames`, `build_R_normal_equations`, `solve_R_lstsq`.
- Produces: `rung1a_kinematics()` → score dict; asserts mean `dir` ≈ 0.

- [ ] **Step 1: Add the rung-1a function (pure numpy, GT flow → R)**

```python
def rung1a_kinematics():
    """Perfect flow F*=C*R_gt inverted back through the R least-squares.
    Must return R_gt (dir~0); failure means the C-matrix / solve is broken."""
    seq, paths, imu, om, frames = load_segment()
    wins = frame_windows(len(frames))
    net = build_net(seq)                      # only for its C matrix
    C = net._C_mat
    M_inv = build_R_normal_equations(C)
    dt = SEG['frame_duration']
    est, ref = [], []
    for (t_lo, t_hi) in wins:
        w = ref_omega(om, t_lo, t_hi)
        R_gt = w * dt
        F_star = gt_flow(net, R_gt)
        R_hat = solve_R_lstsq(M_inv, C, F_star)
        est.append(R_hat / dt)
        ref.append(w)
    s = score_frames(est, ref)
    print("\n=== RUNG 1a  kinematic self-consistency (F*=C.R_gt -> R) ===")
    print(f"  dir  mean/median : {s['mean_dir']:7.4f} / {s['median_dir']:7.4f} deg")
    print(f"  beta mean        : {s['mean_beta']:7.4f}")
    ok = s['mean_dir'] < 1.0 and abs(s['mean_beta'] - 1.0) < 0.05
    print(f"[{'ok' if ok else 'FAIL'}] rung1a: perfect flow recovers R_gt "
          f"(dir<1 deg, beta~1)")
    if not ok:
        print("  >>> GATE: kinematic inversion is broken -> investigate "
              "camera.py (build_kinematic_matrix / solve_R_lstsq).")
    return s, ok
```

- [ ] **Step 2: Wire into `main` and run it**

Add after the rung-0 branch:

```python
    if args.rung in ('1a', 'all'):
        rung1a_kinematics()
```

Run: `python diag_omega_ladder.py --rung 1a`
Expected: `dir mean` well under 1° and `beta` ≈ 1.0 → `[ok] rung1a...`. If `[FAIL]`, the gate message points at `camera.py` and the ladder stops here (that is a valid, decisive outcome).

- [ ] **Step 3: Commit**

```bash
git add diag_omega_ladder.py
git commit -m "Diagnostic: rung 1a (kinematic self-consistency check)"
```

---

### Task 4: Rung 1b — clamped perfect flow (anchor-free + anchored) + decision gate

**Files:**
- Modify: `diag_omega_ladder.py`

**Interfaces:**
- Consumes: `build_net`, `load_segment`, `frame_windows`, `ref_omega`, `gt_flow`, `score_frames`, `get_gyro_for_frame`.
- Produces: `rung1b_clamped_flow(anchored: bool)` → score dict; `_decision_gate(...)`; final summary.

- [ ] **Step 1: Add the clamped-flow runner, with a freeze-verification self-check**

Freezing mechanism: set `q_F.value = F*` each frame and replace the instance's `q_F.update` with a no-op, so the per-iteration Phase-2 update cannot move `F`. The OFCE and kinematics costs still send their gradients to `G` and `R`; only `F` is held at the oracle value.

```python
def rung1b_clamped_flow(anchored):
    """Clamp F = F* (perfect flow) through the relaxation; let R evolve.
    anchored=False: pure vision (R from kinematics only).
    anchored=True : also anchor R toward omega_gt (do flow and anchor agree?)."""
    seq, paths, imu, om, frames = load_segment()
    wins = frame_windows(len(frames))
    net = build_net(seq)
    dt = SEG['frame_duration']
    R_init = get_gyro_for_frame(imu, *wins[0]) * dt
    net.initialize_from_rotation(R_init)
    net.q_F.update = lambda lr: None         # freeze F (instance-level no-op)

    est, ref = [], []
    max_drift = 0.0
    for (V, _t), (t_lo, t_hi) in zip(frames, wins):
        w = ref_omega(om, t_lo, t_hi)
        F_star = gt_flow(net, w * dt)
        net.q_F.value = F_star.copy()
        net.step(V, n_iters=config.ITERS_PER_FRAME,
                 omega_imu=(w if anchored else None))
        max_drift = max(max_drift, float(np.max(np.abs(net.q_F.value - F_star))))
        est.append(net.R / dt)
        ref.append(w)
    s = score_frames(est, ref)
    tag = 'anchored' if anchored else 'anchor-free'
    print(f"\n=== RUNG 1b  clamped perfect flow [{tag}] ===")
    print(f"  err  mean/median : {s['mean_err']:7.2f} / {s['median_err']:7.2f} deg/s")
    print(f"  dir  mean/median : {s['mean_dir']:7.2f} / {s['median_dir']:7.2f} deg")
    print(f"  beta mean        : {s['mean_beta']:7.3f}")
    frozen_ok = max_drift < 1e-9
    print(f"[{'ok' if frozen_ok else 'FAIL'}] F stayed clamped "
          f"(max drift {max_drift:.2e})")
    return s
```

- [ ] **Step 2: Add the decision gate and wire everything into `main`**

```python
def _decision_gate(s1a_ok, s1b_free):
    print("\n=== DECISION GATE ===")
    if not s1a_ok:
        print("  1a FAILED -> kinematic inversion broken. Fix camera.py; "
              "the rest of the ladder reorients.")
        return
    if s1b_free['mean_dir'] > 10.0:
        print("  1a ok, 1b(anchor-free) dir high -> relaxation cannot exploit "
              "correct flow. Investigate the R-update / inter-cost coupling.")
    else:
        print("  1a ok, 1b(anchor-free) recovers direction -> correct flow => "
              "correct omega. Failure is UPSTREAM in flow estimation "
              "(V+F.G=0 with a bad G). Rung 2 (GT gradient) is justified.")
```

Restructure the `__main__` block so each named rung runs standalone, and `all`
runs the full ladder **once** (capturing returns to feed the gate). Replace the
entire `if __name__ == '__main__':` block from Task 1/2/3 with this final version:

```python
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rung', default='all',
                    choices=['setup', '0', '1a', '1b', 'all'])
    args = ap.parse_args()

    if args.rung == 'setup':
        _selfcheck()
    elif args.rung == '0':
        rung0_baseline()
    elif args.rung == '1a':
        rung1a_kinematics()
    elif args.rung == '1b':
        rung1b_clamped_flow(anchored=False)
        rung1b_clamped_flow(anchored=True)
    elif args.rung == 'all':
        _selfcheck()
        rung0_baseline()
        _, s1a_ok = rung1a_kinematics()
        s1b_free = rung1b_clamped_flow(anchored=False)
        rung1b_clamped_flow(anchored=True)
        _decision_gate(s1a_ok, s1b_free)
```

This replaces the smaller `if args.rung in (...)` snippets added in earlier tasks,
so no rung runs twice.

- [ ] **Step 3: Run the full ladder**

Run: `python diag_omega_ladder.py --rung all`
Expected: setup `[ok]`, RUNG 0 table, RUNG 1a `[ok]` (dir≈0), RUNG 1b anchor-free + anchored tables each with `[ok] F stayed clamped`, then the DECISION GATE line naming the next investigation branch.

- [ ] **Step 4: Commit**

```bash
git add diag_omega_ladder.py
git commit -m "Diagnostic: rung 1b (clamped flow) + decision gate"
```

---

## Self-Review

**1. Spec coverage** (against `2026-10-07-omega-recovery-diagnostic-design.md`):
- Research question (direction vs scale) → `score_frames` reports `dir`/`beta` separately, `axis_r`/`axis_bias` for direction decomposition. ✓
- Target segment `ecrot_city` seg_A, pinhole, dt=20ms, explicit → `SEG` + `dist_coeffs=None`/`undistort=False`. ✓
- Rung 0 (baseline + cond(M) + curl_share + axiswise) → Task 2. ✓
- Rung 1a (F*=C·R_gt → solve, expect dir≈0) → Task 3. ✓
- Rung 1b (clamp q_F, anchor-free **and** anchored) → Task 4. ✓
- Decision-gate table → `_decision_gate`. ✓
- Standalone `[ok]`/`[FAIL]` script, read-only on model → all tasks; only `q_F.value`/`q_F.update` touched. ✓
- GT flow via omega_gt, R_gt=ω·dt → `ref_omega`/`gt_flow`. ✓
- Freeze-mechanism verification (q_F stays = F*) → `max_drift` check in Task 4. ✓

**2. Placeholder scan:** No TBD/TODO. Task 4 Step 2 gives the exact final `__main__` block (real code) that replaces the smaller per-rung snippets from earlier tasks, so nothing runs twice. No "add error handling" vagueness.

**3. Type consistency:** `score_frames` returns the same dict keys consumed by the print blocks and `_decision_gate` (`mean_dir`, `mean_err`, `mean_beta`, `axis_r`, `axis_bias`). `rung1a_kinematics` returns `(s, ok)`; the `all` branch unpacks `_, s1a_ok`. `rung1b_clamped_flow` returns a score dict; `all` captures `s1b_free`. `net._C_mat`, `net.q_F`, `net.R`, `net.G` match `InteractingMapsThesis`. `build_R_normal_equations(C)->M_inv`, `solve_R_lstsq(M_inv, C, F)->R` match `camera.py`. ✓
