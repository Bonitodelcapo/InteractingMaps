# TODO

Open scientific questions on ω recovery. CMax-specific work lives in [cmax_todo.md](cmax_todo.md).

**Priority:** **P1** = could change numbers in the report; do first. **P2** = understanding the model.

Background for all items: the explicit OFCE step on G is unstable above |F| ≈ b = 2.8 px/frame
(δ_VFG = 0.12, δ_IG = 0.10), and the `max_grad = 5` clip is what keeps G bounded there. Measured on
`ecrot_city` city_short with `python diag_omega_ladder.py --mode ofce` (7 variants, cluster, 2026-10-14).
Findings and tables: [the clamped-ω spec](../superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md),
section "Findings: G flipping measured".

## Open

- [ ] **P1 · Tooling · Save the full model state at every frame (per-frame checkpoints).**
  Today a tracking run keeps only `maps_final.npz`, so nothing shows what the model looked like at
  the frame where a curve spikes. After `net.step(...)` in `experiment_tracking` ([evaluation.py](../../evaluation.py)),
  save everything recurrent at the end of each frame's last iteration, not only the plotted maps:
  Cook `I, G, F, R`; thesis / thesis_imu each `Quantity.value` + `gradient_accumulator`;
  CMax V1 also `omega_cmax_prev` (held in the loop); CMax V2 whatever `Cost_CMax` carries across frames.
  Each network gets a `get_state` / `set_state` pair. Acceptance test: replaying frame k from
  checkpoint k−1 reproduces the saved frame k exactly. Then a replay script that re-runs one frame from
  checkpoint k−1 and writes every iteration to disk (the iteration view; reuse `run_experiments --what converge`).
  Applies to every model on every segment; start with a few segments. Size ≈ 1 MB/frame (float32,
  180×240), ≈ 75–100 MB per 75-frame run before compression. Feeds the inspection viewer
  (design in progress: notebook + matplotlib/ipywidgets, 1–5 models side by side, linked to `tracking.csv` curves, GIF/MP4 export).

- [ ] **P1 · Report · Double-check every figure and table in the journal.**
  `make_figures.py` draws `fig_cmax_baseline` and `fig_sensitivity`, and `make_tables.py` writes
  `table_anchor` and `table_grid`, yet none of the four appears in `report/bare_jrnl.tex`
  (scan of 2026-11-25, see [experiments.md](../../experiments.md)). Decide which belong in the
  report, or keep an explicit inclusion list that the scripts and the tex are checked against.

- [ ] **P1 · Report · `fig_converge` iterates frame k from a fresh network, not from the run's state.**
  `cmd_converge` in [report/run_experiments.py](../../report/run_experiments.py) builds a new network
  (zero maps, `initial_R`) and, for CMax, solves with `omega_init = 0`, then iterates frame k. The real
  run reached frame k after frames 0…k−1, with recurrent I, G, F, R. So the figure shows a convergence
  that never happens in the tracking run. Investigate how different the two are: once per-frame
  checkpoints exist, replay frame k from checkpoint k−1 and compare with the current `converge_path`
  snapshots. Then decide whether `fig_converge` moves onto the replay (changes a report figure).

- [ ] **P1 · Do the report's anchored numbers depend on the clip?**
  The anchored models (`thesis_imu`, `thesis_cmax`) have correctly scaled flow, so they run above the
  bound: on poster seg_C, 85 % of pixels exceed b at 20 ms (16 % at 10 ms, 0 % at 5 ms).
  Their reported error may then partly reflect what `max_grad` does, not the model.
  Test: rerun the main-table anchored runs at dt = 10 ms and 5 ms, and with the normalised G step
  (`Cost_OFCE(normalize=True)`), then compare ω error and |r| with Table III. `max_grad` and `normalize`
  are not run parameters yet; they need wiring through `THESIS_PARAMS` before the second part can run.

- [ ] **P2 · Why does cook stay stable above the OFCE bound b?**
  In the scale spike (2026-11-25, `experiments/scale_spike/`, poster seg_C, 30 frames, 20 ms) cook's
  99th-pct |F| sits at ≈ 3.37 px/frame on every frame, above b ≈ 2.8, yet its G and I never break.
  thesis_imu at the same level (≈ 3.35 at frames 6 and 13) breaks one frame later (G spike, striped I
  at frame 14). b was derived for the thesis' simultaneous G step (δ_VFG, δ_IG); cook updates G
  sequentially (Gauss-Seidel: OFCE, then G←∇I, then I←G within one iteration) with its own step sizes.
  Test: derive cook's bound for its sequential update and compare; on the cluster, run both at
  matched |F| and record the fraction of pixels above each model's bound per frame.

- [ ] **P2 · Why does ω overshoot by ≈ 1.37× once the cap is gone?**
  With dt = 5 ms or the normalised step nothing caps F, yet β ≈ 0.73: the free |F| sits at 0.75·b
  against a true 0.54·b. Test: not designed yet; find which term pushes |F| above the truth when the
  step is stable.

- [ ] **P2 · Where does the ≈ 14° direction error come from?**
  12.5–17.9° in every converged variant, whatever b, dt or step. Rung 0 already showed the dominant
  y axis anti-correlated (axis_r = −0.22). Test: per-axis breakdown (axis_r) across the 7 OFCE variants.

- [ ] **P2 · Is the no-clip blow-up the real-data failure?**
  Without the `max_grad` clip (variant E2) G pins at `CLIP_G` on > 99 % of pixels, F runs above 2·b,
  β = 13 and the direction error is 89°, close to the ≈ 85° of vision-only models on real data.
  Test: compare E2's signature (|F|/b occupancy, G at `CLIP_G`, β, axis_r) with a real-data
  vision-only run.

- [ ] **P2 · What causes the period-2 zig-zag in the per-iteration residuals?**
  Seen in the walkthrough notebooks (poster seg_C, frame 0, 75 iterations; figures in
  `experiments/model_walkthrough/<model>_figures/<model>_residuals.png`). Both residuals alternate
  up/down from one iteration to the next. In cook it dies out after ≈ 10 iterations. In
  thesis and thesis_imu, mean |G − ∇I| keeps alternating to iteration 75 (≈ 0.011 ↔ 0.016 on thesis).
  thesis_imu also swings slowly (period ≈ 35 iterations, |V+F·G| 0.2 ↔ 0.5) and never settles.
  Test: record the residual maps per iteration and check whether a spatial
  checkerboard flips sign each iteration, and which update (G←∇I, I←G, OFCE) produces the flip.

- [ ] **P2 · Is the zig-zag caused by the updates interacting, not by one step being too large?**
  Each update on its own looks stable. Cook's I step `I -= δ_GI·div(G − ∇I)` multiplies the
  checkerboard mode, the worst case, by 1 − 8·δ_GI = 0.36 at δ_GI = 0.08, which does not flip sign.
  That bound holds only with G fixed, so the hypothesis is that the coupling between G←∇I, I←G and
  the OFCE steps flips sign (in thesis, G also takes the OFCE and ∇I pulls from the same old state).
  Unverified: no measurement yet. Test: switch off one update at a time and see which removes the
  zig-zag; in thesis, compare the sum of the two G pulls with each pull alone.
