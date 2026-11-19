Interacting Maps · thesis work, TU Berlin

# Interacting Maps Logbook

State as of Tue 27 Oct 2026 · refresh: ask Claude to “refresh my logbook”

Now

## What do I do next?

The science has one solid result: the β ≈ 3 scale error is explained, and two puzzles remain open. The code is split between your branch and Leo's `main`. That merge blocks everything else, and it waits on two decisions only you and Leo can make.

1. **Talk to Leo (30 min).** Agree on (a) the IMU model's name, (b) whether report figures and CSVs live in git, and (c) a short freeze on `main` while you merge.
2. **Merge `main` into your branch.** Take Leo's content (which runs to read, |r| scoring, the read-out column, the image-settings experiment, the anchored-maps fix) and draw it with your `style`/`viz`. About 60 conflicts, most of them output files.
3. **Check whether the paper's anchored numbers are affected.** In the reported runs the anchored models work with flow above the stability bound (poster seg_C: 85 % of pixels at 20 ms). Do their results depend on the `max_grad` clip?

Science map

## What do I know?

The chain of reasoning, from problem to open questions. Each step was established by the one before it.

established, with evidenceopen, with the test that would settle it

1. P

   ### Vision-only models don't recover ω

   Real data, `cook`/`thesis`: direction error **75–85°**, error 43–74 °/s, so essentially no directional signal. Anchored models (IMU, CMax) reach the gyro-vs-Vicon floor. report/experiments/main_dt20ms.csv
2. 0

   ### On clean synthetic data, the scale is wrong, not the direction

   Rung 0, `ecrot_city`, 20 ms: direction **13°**, but **β = 2.83** (ω about 3× too small). The dominant y axis is anti-correlated (r = −0.22).
3. 1

   ### Perfect flow gives perfect ω, so the fault is upstream in the flow

   Rung 1a, the kinematic inverse: exact (dir 0°, β 1). Rung 1b, flow clamped to truth: error 0, β 1.00, with or without anchor. The kinematics and the R update are not the bug.
4. 1c

   ### Even with the true ω, the maps go wrong

   R clamped to ω_gt: I comes out **striped** and the flow error stays about 3 px/frame. So the G update itself misbehaves.
5. B

   ### The explicit G step is unstable above |F| ≈ 2.8 px/frame

   Derived from δ_VFG = 0.12 and δ_IG = 0.10. Above the bound b, G flips sign every iteration; only the `max_grad = 5` clip keeps it bounded, and the Poisson step turns the flips into stripes. The flip curves collapse onto |F|/b across 7 cluster variants.
6. β

   ### That instability is what produces β ≈ 3

   Cluster run, 14 Oct: the free network cannot hold |F| above about **0.8·b**, so β ≈ |F\*| / (0.8·b), within 2–9 % in all 4 capped variants. A numerical property of the update, not of the data. [spec findings](https://github.com/Bonitodelcapo/InteractingMaps/blob/modularity-notebooks/docs/superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md)

Still open

1. ?1

   ### Remove the cap and ω overshoots 1.37×

   With a 5 ms dt or a normalised step, nothing caps F, yet β ≈ **0.73**: the free |F| still sits too high (0.75·b vs a true 0.54·b).

   not designed yet. Find which term pushes |F| above the truth when the step is stable.
2. ?2

   ### A ≈ 14° direction error in every converged variant

   12.5–17.9° whatever the bound, dt or step. It probably ties to the anti-correlated y axis seen in rung 0.

   per-axis breakdown (axis_r) across the 7 OFCE variants.
3. ?3

   ### Is the no-clip blow-up the real-data failure?

   With no `max_grad` clip (E2), G pins at `CLIP_G`, F runs above 2·b, β = 13 and direction 89°. That is close to the ≈ 85° on real data.

   compare E2's signature with a real-data vision-only run.
4. ?4

   ### Do the paper's anchored numbers lean on the clip?

   Anchored models have correctly scaled flow, so on poster seg_C **85 %** of pixels sit above the bound at 20 ms (16 % at 10 ms, 0 % at 5 ms).

   rerun the main table's anchored models at 10 ms or with the normalised step, then compare.

Side thread

### Good ω ≠ good picture

Reconstruction quality and ω error are not linked within a model. The overall ρ = −0.64 comes only from comparing vision-only runs with anchored ones; with ω clamped to truth, I is still striped. Leo's report section "dissociation" makes the same point.

Side thread · paused

### Rung 2: the true gradient

2a works: the event equation fits, and the warp is essential (R² 0.23 vs 0.03). 2b, G clamped to truth, is uninformative because the F step is too slow at the true G scale. Code is in `stash@{0}`.

For your supervisor

### Open questions Q1–Q7

Evidence and the deciding test for each, written 14 Oct (before the β result). [Open the questions page](https://claude.ai/artifact/JwQ59U9HawNSxJFpSvSSx1)

Work map

## Where is the code?

One row per line of work. No commits on any branch since 14 Oct.

Merge Leo's mainmain → modularity-notebooks

blocked

About 60 conflicts: 10 code spots in `make_figures.py` and `run_experiments.py`, plus 27 output files. Your branch is also missing 8 of Leo's changes. Waits on the decisions below.

Your branchmodularity-notebooks

active

Refactor, unified plotting, ω ladder, OFCE study. **2 commits not pushed** (G-flip findings, β figure). 8 commits behind `main`.

Leo's reportorigin/main

active

The report rewritten against fresh runs, |r| scoring, figures at 1.5 s, anchored-maps fix, new image-settings experiment. Last commit 14 Oct.

Paused

Rung 2stash@{0}

paused

The true-gradient rung. 2a works; 2b needs a different design.

Count-based framingworktree const_events

paused

Fixed event count per frame instead of fixed dt. The design Q&A stopped at Q11–Q17; no code.

Stale

Gfinal

decide

About 2.3k lines never merged: an evaluation plan, report analysis and table scripts, a `find_segments` rework.

Cmax, gianluca, claude/…

stale

Superseded or empty. Safe to delete after a quick look.

Waiting on you

## Decisions, and what's parked

- What is the IMU model called?“thesis+IMU” (paper, 21×; Leo) or “IMU-anchor” (your figures)
- Do report figures and CSVs live in git?Leo commits them; your branch ignores them. Decides 27 of the conflicts.
- Gfinal: harvest or drop?Is anything in it still needed?
- Is count-based framing still wanted?Resume the design or close the worktree.
- Push the 2 local commits?The G-flip findings and the β figure.

Parked: not now, on purpose

- Plotting Plan 3 (debug viewers, demo, old plot helpers)
- Name-based `--exp` CLI (architecture review R5)
- Demo's thesis + IMU path uses the wrong `frame_duration`
- `test_ecrot_convert.py` imports a function that no longer exists
- Exp 8 overwrites `run.json`; `iwe_log.csv` duplicates rows on rerun
- `BETA_SEGMENTS` defined twice

Sources: git (all branches, stash, worktrees), the 14 Oct cluster OFCE run, the spec findings, and the InteractingMaps Claude sessions of 7–14 Oct. Numbers are copied from those sources, not recomputed.