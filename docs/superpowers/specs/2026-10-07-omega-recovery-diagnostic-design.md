# ω-recovery diagnostic — rungs 0–1

> Design/spec for a focused diagnostic that localizes **why the vision-only
> Interacting-Maps path fails to recover camera angular velocity ω**. Companion to
> [architecture_review.md](../../../architecture_review.md) (code organization) and
> [architecture.md](../../../architecture.md) (model). This document is about a
> *scientific* diagnostic, not a refactor. Scope is deliberately the first two
> rungs of a larger oracle ladder; the rest is gated on what these find.

## Motivation — what the current numbers say

From `report/experiments/main_dt20ms.csv` (metric defs in [metrics.py](../../../metrics.py),
`floor` = IMU-gyro-vs-Vicon disagreement in [run_experiments.py:288](../../../report/run_experiments.py)):

- **Vision-only (`cook`, `thesis`) produces no angular signal.** `dir` ≈ 75–85°
  across every sequence. 90° is orthogonal — the estimate carries essentially no
  directional information about ω. `err` 43–74 deg/s follows from that.
- **Two *separable* failures, not one.** If the β scale ambiguity were the only
  problem, direction would be ≈ correct and only magnitude wrong. `dir ≈ 85°`
  means a **directional** failure sits *underneath* the β scale failure
  (β = 181/340 for `cook` on synthetic; β ≈ 2.5 for `thesis` on synthetic). Any
  diagnostic must score direction and scale **apart**.
- **The anchor does nearly all the work.** On real pure-rotation data,
  `thesis_imu` and `thesis_cmax` reach the IMU-vs-Vicon floor (err ≈ 6–11,
  dir ≈ 2–12°). `thesis_cmax` reaching it *without an IMU* is the headline
  positive result — but it masks the vision failure rather than fixing it.

**Research question for rungs 0–1:** *Does the pure-rotation kinematic inversion
`F = C·R` recover ω direction at all, and if it is given perfect flow, does the
relaxation hold onto it?* The answer picks which branch of the investigation is
real.

## Goal & success criteria

Produce a **decisive localization signal**: the first stage in the chain
`V → (V+F·G=0) → F → (F=C·R) → R → ω` at which correct ω direction is or is not
present. Success = the diagnostic runs, reproduces the known baseline failure, and
cleanly resolves the decision gate below. This is a *measurement*, not a fix — no
model or harness behavior changes.

## Scope

**In scope**
- Rung 0: characterize the baseline vision-only failure on one segment.
- Rung 1a: kinematic self-consistency (perfect flow → R via least squares).
- Rung 1b: relaxation with clamped perfect flow, run **both** anchor-free and
  anchored.
- One standalone diagnostic script + a short written findings block.

**Out of scope (gated / deferred)**
- Rungs 2–4 (GT gradient, GT intensity warm-start, pre-anchor R scoring) — only
  pursued if rung 1 says the failure is upstream of flow.
- The `thesis_cmax_v2` regression (bicycle 45 vs v1's 7) — tracked separately.
- Any change to the model, the networks, `config.py` defaults, or the harness.
- Multi-segment / real-vs-synthetic sweeps (one cross-check segment at most).

## Target segment

**`ecrot_city` seg_A** (primary). Chosen because it is simultaneously:
- **synthetic** → `data/ecrot_city/omega_gt.txt` gives clean, steady ω ground
  truth (2831 samples), no Vicon differencing noise;
- **pure rotation** (ECRot) → the rotation-only model `F = C·R` is correctly
  specified (no translational-flow confound);
- **pinhole, zero distortion** (`calib.txt`: fx=fy=200, cx=120, cy=90, all Brown–
  Conrady coeffs 0.0) → `C_full` distortion is *not* a confound; `C` is the clean
  pinhole kinematic matrix.

The diagnostic defines its segment explicitly (`t_start`, `frame_duration`,
`n_frames`, `sensor_size = (180, 240)`) rather than depending on a
`DATASET_SEGMENTS` entry, so it is self-contained. **Window: `frame_duration =
0.02` (`dt=20ms`), matching the existing `ecrot_city` runs** (`results/ecrot_city/
.../seg_A_t0.050_dt20ms_...`); `t_start = 0.05`, a first-pass window of ~50–75
frames (≈1–1.5 s). Exact frame count is a minor tuning detail for the plan.

**Optional later cross-check:** `poster_rotation` (local, real, pure-rotation) to
contrast a clean synthetic result against real-data noise — only if rung 1 warrants.

## Metrics (all per-frame, reported as mean + median)

- `err` = ‖ω̂ − ω_gt‖ · 180/π (deg/s) — [metrics.py:15](../../../metrics.py).
- `dir` = angle(ω̂, ω_gt) in degrees — the **direction** failure.
- `β` = ‖ω_gt‖ / ‖ω̂‖ — the **scale** failure (1.0 perfect).
- **Axis-wise direction decomposition:** per-component signed error of ω̂ vs ω_gt
  in the ω_gt frame (is one axis recovered and another dead? the aperture /
  observability fingerprint).
- `cond(M)` where `M = ΣCᵀC` (`build_R_normal_equations`,
  [camera.py](../../../interacting_maps/camera.py)) — conditioning of the R solve,
  free context.
- `curl_share(G)` — [metrics.py:80](../../../metrics.py) — how non-integrable the
  inferred gradient is, free context.

## Ground-truth flow construction

For a **purely rotating** camera the optical flow field is depth-independent and
is exactly the rotational flow `F*(p) = C(p) · R_gt`, where `C` is the pipeline's
own kinematic matrix (`build_kinematic_matrix`) and `R_gt` is the ground-truth
rotation **in rad/frame** for that frame window.

- `R_gt` per frame comes from `omega_gt` via the existing reference helper
  (`get_reference_omega` / `load_omega_gt`, re-exported through `evaluation` /
  `eval_io`): `R_gt = ω_gt · frame_duration`.
- Building `F*` from the *same* `C` the pipeline uses is intentional: rung 1a then
  tests whether `solve_R_lstsq(M⁻¹, C, F*)` returns `R_gt` (it must, up to
  conditioning), which is precisely the self-consistency check.

## Rung 0 — characterize the baseline failure

Run the `thesis` (vision-only) network on the segment via the normal pipeline
(no anchor). For each frame record `err / dir / β`, the axis-wise decomposition,
`cond(M)`, and `curl_share(G)`.

**Purpose:** confirm we reproduce the known failure on this clean pinhole segment
(does vision-only fail here *too*, or does the absence of distortion/real-noise
already recover ω?) and describe *how* the direction is wrong, not just that it is.
Either outcome is informative.

## Rung 1a — kinematic self-consistency (seconds, pure numpy)

For each frame: `F* = C · R_gt` → `R̂ = solve_R_lstsq(M⁻¹, C, F*)` → score
`dir`/`β` of ω̂ = R̂/dt vs ω_gt. **Expect dir ≈ 0.**

- **If 1a fails** → the `C`-matrix / least-squares / `M⁻¹` is wrong; everything
  downstream is built on sand. The investigation reorients onto
  [camera.py](../../../interacting_maps/camera.py).

## Rung 1b — relaxation with clamped perfect flow

Freeze `q_F = F*` throughout the relaxation (inject `F*` and prevent the OFCE / FR
costs from moving `F`), run the network, score `R` per frame. Run **twice**:

1. **Anchor-free** — `update_r=True`, no IMU/CMax anchor. R driven only by `CostFR`
   ([network_dissertation.py:367](../../../interacting_maps/network_dissertation.py))
   from the clamped flow. *The core question:* does vision-only recover R given
   perfect flow?
2. **Anchored** — same, with the anchor on. Reveals whether perfect flow and the
   anchor **agree or fight** (if they disagree, the anchor is compensating for a
   flow→R path that is itself wrong).

- **If 1a passes but 1b (anchor-free) fails** → the relaxation / R-update dynamics
  cannot exploit correct flow; investigate the update scheme and inter-cost
  coupling, not the kinematics.

## Decision gate (the deliverable's point)

| Rung 1a | Rung 1b anchor-free | Conclusion → next step |
|---|---|---|
| fails | — | Kinematic inversion is broken → fix `camera.py`; ladder reorients. |
| passes | fails | Relaxation can't use correct flow → investigate R-update / coupling. |
| passes | passes | Correct flow ⇒ correct ω. Failure is **upstream in flow estimation** (`V+F·G=0` with a bad `G`) → rung 2 justified; proceed to GT-gradient and the translation question. |

The anchored-vs-anchor-free comparison in 1b further tells us whether the anchor is
*adding* information or *masking* a broken flow→R path.

## Deliverable

One standalone script, `diag_omega_ladder.py`, in the repo's existing test idiom
(a script you run directly that prints `[ok]`/`[FAIL]` + a compact per-rung table,
**not** pytest — see the Tests section of [CLAUDE.md](../../../CLAUDE.md)). It:

- only **reads** the pipeline and **injects** GT into `q_F` / the R solve; it does
  not modify any model code;
- defines its segment explicitly and is runnable from the repo root like the other
  `test_*.py` / `diag_*` scripts;
- prints the rung-0/1a/1b tables and writes a short findings block (appended to a
  diagnostics note or printed for the user to paste).

No provenance/MLflow wiring required (diagnostic, not a reported run), though it may
reuse `eval_io` loaders for consistency.

## Risks & notes

- **Rung 1a near-tautology:** building `F*` from the pipeline's own `C` means 1a
  mainly catches `C`/solve bugs and conditioning, not physics. That is the intent;
  its value is as a floor check before trusting 1b.
- **Clamping mechanism:** freezing `q_F` must actually prevent the F-updating costs
  from relaxing it (zero their F-deltas or re-inject `F*` each iteration). Verify
  `q_F` stays equal to `F*` across iterations before trusting 1b. (Implementation
  detail for the plan.)
- **`ecrot_city` may not reproduce the failure.** If vision-only already recovers ω
  on this clean pinhole segment (rung 0 dir small), that is itself a major finding:
  distortion and/or real-data noise, not the core model, breaks direction — and the
  campaign pivots to the real cross-check segment immediately.
