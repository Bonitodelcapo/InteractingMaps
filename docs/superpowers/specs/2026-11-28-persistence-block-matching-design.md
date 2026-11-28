# Persistence and block-matching terms: a third network type

Date: 2026-11-28. Branch `persistence-terms`. Status: reviewed, awaiting approval.

## Intent and why

Vision-only ω does not work in our `thesis` network: it stays at its frame-0 warm
start (sweep2, stale, basin runs, 2026-11-28). Cause: the base model ("3-DoF motion
in a static scene", Martel 2019 §6.4) has no notion of time between frames, so the
gauge F → sF, G → G/s (β ambiguity, incl. sign) is free and nothing restores R.
Martel states this himself (§6.8.1) and shows the base model failing even in a
perfect simulated rotation (Fig. 6.19). His fixes add a term that compares the image
**across frames**, which brings the real time step in:

- **Persistence** (§6.5, Eqs. 6.68–6.100): the current image I must equal the
  previous image Ǐ warped by the inter-frame rotation R_δ.
- **Block-matching flow** (§6.8.3, Eqs. 6.138–6.139, Fig. 6.17): a flow F′ found by
  matching I against Ǐ, tied to F; "all ambiguities are resolved since F′ gets
  properly scaled".

Goal of this work, in the user's words: **a faithful reimplementation first, then
improve where it fails.** The question it answers for the thesis: does Martel's own
fix make vision-only ω work on our data, scored against Vicon / ω_gt?

## Goals

1. A third network type, `InteractingMapsPersist`, with three model strings:
   - `thesis_persist`: base thesis network + persistence cost (Martel §6.5).
   - `thesis_bm`: base thesis network + Ǐ + block-matching cost (Martel Fig. 6.17).
   - `thesis_persist_bm`: both (our combination; Martel shows them separately).
2. Faithful to the thesis equations wherever the thesis gives them; every deviation
   is listed in this document (section "Deviations from Martel") and in the code.
3. Runs through the existing harness unchanged in use: `evaluation.py --exp 2`,
   `report/run_experiments.py --what sweep --model <new>`, run.json, tracking.csv,
   the recon figures.
4. A validation that persistence alone recovers a known rotation on real images
   (see Verification).

## Non-goals

- Martel's world map W and absolute orientation R(t) (Eqs. 6.101–6.111). They serve
  panoramas and long-term drift, not the ω scale. Possible later stage.
- Anchored variants (persistence + IMU/CMax). Not now.
- Any report table/figure change, any change to `report/report_runs.MODELS`.
- Improvements beyond Martel ("then improve where it fails") — a later spec, driven
  by what this one measures.

## Anti-goals (failure even if it works)

- **Any change in behaviour of the existing networks** (`InteractingMaps`,
  `InteractingMapsThesis`) or of the existing models' runs: the golden tracking test
  prints the same `[ok]` lines.
- Changed signatures of existing functions or classes.
- Writing scratch runs under `results/`.

## Constraints

- Code-quality rules of the repo: reuse existing functions, atomic new functions,
  import only from core modules, minimal diffs.
- Smoke tests only locally (~3 frames × 10 iterations); real runs on the cluster.
- ω shown in deg/s everywhere.
- Default distortion mode `C_full`: I lives in distorted pixel coordinates.

## Approaches considered

- **A. New costs inside the existing two-phase loop (picked).** Ǐ stored at the
  frame boundary; `Cost_Persistence` and `Cost_BlockMatching` added to the cost
  list like the existing `Cost_*` classes. Faithful in exactly the terms that carry
  the time scale; fits the codebase's pattern.
- B. A + world map W and absolute R(t): Martel's full persistence model. About twice
  the work; does not add to what fixes the ω scale. Kept as a possible stage 3.
- C. Persistence outside the loop as an anchor (align previous/current image once
  per frame, feed through `Cost_IMU`). Cheapest, but not faithful: an external
  anchor cannot show whether Martel's network fixes itself. Rejected.

## Design (decided)

### Structure

- New class `InteractingMapsPersist(InteractingMapsThesis)` in a **new module**
  `interacting_maps/network_persistence.py`. `network_dissertation.py` is not
  edited. Constructor takes the base class's arguments plus
  `use_persistence`, `use_block_matching`, `delta_persist_R`, `eta_persist_I`,
  `delta_BM` and the block-matching settings; it calls the base constructor, then
  appends the new costs to `self.costs`.
- New quantity `q_Icheck` (Ǐ), shape (H, W), not updated by any cost (Martel: "we
  will not update Ǐ").
- `step()` is overridden: at its start, before calling the base `step()`, set
  Ǐ ← current I (= the previous frame's final I). On the very first frame there is
  no previous image: the new costs are inactive for frame 0 (flag), so frame 0
  behaves exactly like `thesis`.
- **Frame start (user decision: "do as Martel says in the thesis").** I is *not*
  pre-warped: at the first iteration of a frame I equals Ǐ, exactly as in Martel
  (Ǐ = I(t−δt), I then changed only by the events and by the costs, Eq. 6.100 with
  step η). No switch, no alternative start. Known consequence, to be read in the
  cluster results rather than designed around: in the first iterations of each
  frame the Gauss–Newton candidate R* is near 0 (identical images), and it moves
  away only as the events change I. If this is where the model fails, it is input
  for the later "improve where it fails" stage.
- **Kinematics R update (δ_FR) stays on** (my call): Martel's persistence model keeps
  the F → Ω link, so leaving the existing `Cost_Kinematics` unchanged is faithful.
- Mapping to our quantities: our `R` is already the per-frame rotation vector
  Ω·Δt, so Martel's R_δ = exp([R]×) (Rodrigues, Eq. 6.91). His Ω–R relation
  (Eqs. 6.106–6.110) is then an identity and needs no cost.

### Persistence cost (`Cost_Persistence`, quantities I, Ǐ, R)

- Residual per pixel (Eq. 6.72): ρ_i(R) = I(p_i) − Ǐ(T_R(p_i)), with T_R the warp of
  pixel p_i by rotation exp([R]×) through the camera model.
- **R side (faithful, Eqs. 6.73–6.98):** each network iteration, starting from the
  current R, run Gauss–Newton on Σρ_i² in the so(3) parametrisation (Jacobian
  J_ρ = ∇Ǐᵀ · J_proj · J_rot, Eqs. 6.81–6.96) until convergence → candidate R*.
  Then R takes a step toward it: gradient contribution δ_persist_R · (R − R*), the
  same "candidate, then small step" pattern as `Cost_Kinematics`.
- **I side (Eq. 6.100):** I ← I − η (I − Ǐ∘T_R), as gradient contribution
  η · (I − Ǐ∘T_R).
- Pixels whose warped position falls outside the image are excluded from both sides.

### Block-matching cost (`Cost_BlockMatching`, quantities I, Ǐ, F)

- F′ (Eq. 6.138): for each pixel, the integer displacement within a search window
  that minimises the sum of absolute differences between a block of I around the
  pixel and the displaced block of Ǐ.
- Cost (Eq. 6.139 + the F′–F link of Fig. 6.17): F takes a step toward F′,
  gradient contribution δ_BM · (F − F′). Sign/direction convention of F′ must match
  F's (flow from previous to current frame, px/frame).
- I is not updated by this cost (Martel derives only the cost; the F′–F link is
  what Fig. 6.17 adds).

### Harness wiring (additive only)

- `run_config.RunConfig`: three new `model` branches → `use_thesis=True`,
  `use_imu=False`, params = `THESIS_PARAMS` + new `PERSIST_PARAMS` (new dict in
  `config.py`). `output_dir` and `to_dict` unchanged in format; the new deltas
  appear through the existing "extra params" naming.
- `run_config.make_network`: builds `InteractingMapsPersist` for the new models.
- `interacting_maps/style.MODELS`: three new entries (name, colour).
- `evaluation.py --model` choices: the three new strings.
- **Sweeps** (my call): `report/run_experiments.py` `_parse_vary` accepts the new
  deltas (`delta_persist_R`, `eta_persist_I`, `delta_BM`). Their csv columns appear
  **only** for the new models, so existing sweep csvs keep exactly their columns.
  Because this edits a `.py`, the next `run_all.sh` needs the `ADOPT=1` step first.

### Deviations from Martel (named, also as comments in the code)

1. No world map W, no absolute orientation (non-goal).
2. R is our per-frame rotation vector, not a rotation matrix plus Ω.
3. Warp through our camera model, which includes lens distortion in `C_full`
   mode; Martel uses a pinhole K. (Builder: see "Left to the builder".)
4. Persistence and block matching combined in one model (`thesis_persist_bm`) —
   Martel shows them only separately.
5. **Frame-0 R start.** Martel never specifies how R is set for the first frame
   (ch. 6 only says the sign solution picked depends on "initialization"; his
   Gauss–Newton starts near the identity). Our harness warm-starts R from the gyro.
   The new models use the harness's existing R-initialisation flag unchanged (same
   design as for the existing models, user decision) and are run **both ways**:
   gyro warm start and R = 0. R = 0 is the faithful vision-only case; the pair
   separates what persistence recovers alone from what it keeps from the gyro.
   Between frames, all maps (I, G, F, R) carry over as now; no new flag.

## Verification

- **Before and after any edit:** `tests/test_golden_tracking.py` prints the same
  `[ok]` lines (my call: `test_states_replay.py` / `test_network_state.py` exist only
  on the `inspection-viewer` branch, not on `main`).
- New standalone test `tests/test_persistence.py` (repo style, `[ok]`/`[FAIL]`):
  1. **Known rotation, real image:** take a poster APS frame as Ǐ, synthesise I by
     warping it with a known small rotation R_true; the Gauss–Newton solve
     returns R_true within tolerance, from R = 0 and from R = 2·R_true.
  2. Gradient/Jacobian check of the persistence residual vs finite differences.
  3. Block matching on a shifted image returns the shift as F′ (interior pixels).
  4. Frame 0 of each new model equals `thesis` exactly.
  5. A 3-frame × 10-iteration exp-2 smoke run of each new model completes and
     writes under `experiments/`, not `results/`.
- Cluster (not part of this build): the three new models on the 5 report
  sequences, 1.5 s, scored against Vicon / ω_gt, each with frame-0 R from the gyro
  and R = 0 (deviation 5); plus the basin test (`--exp 6`).
  Success for the thesis question: ω moves with the ground truth (path ratio, not
  just error) and β approaches 1 without an anchor.

## Left to the builder

- Gauss–Newton details: convergence tolerance, iteration cap, damping if needed.
- Default values of `delta_persist_R`, `eta_persist_I`, `delta_BM` (start small,
  like the existing deltas; they will be swept on the cluster).
- Block size, search radius (must cover the expected flow, a few px/frame at 20 ms)
  and sub-pixel refinement (none in Martel).
- Interpolation for Ǐ∘T_R (bilinear expected).
- Reuse of the CMax estimator's bearing/distortion/warp helpers
  (`angular_velocity.py`) for T_R and its Jacobian, if they fit, by moving the
  shared part to a core module rather than importing a private method.
