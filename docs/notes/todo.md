# TODO

Open scientific questions on ω recovery. CMax-specific work lives in [cmax_todo.md](cmax_todo.md).

**Priority:** **P1** = could change numbers in the report; do first. **P2** = understanding the model.

Background for all items: the explicit OFCE step on G is unstable above |F| ≈ b = 2.8 px/frame
(δ_VFG = 0.12, δ_IG = 0.10), and the `max_grad = 5` clip is what keeps G bounded there. Measured on
`ecrot_city` city_short with `python diag_omega_ladder.py --mode ofce` (7 variants, cluster, 2026-10-14).
Findings and tables: [the clamped-ω spec](../superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md),
section "Findings: G flipping measured".

## Open

- [ ] **P1 · Do the report's anchored numbers depend on the clip?**
  The anchored models (`thesis_imu`, `thesis_cmax`) have correctly scaled flow, so they run above the
  bound: on poster seg_C, 85 % of pixels exceed b at 20 ms (16 % at 10 ms, 0 % at 5 ms).
  Their reported error may then partly reflect what `max_grad` does, not the model.
  Test: rerun the main-table anchored runs at dt = 10 ms and 5 ms, and with the normalised G step
  (`Cost_OFCE(normalize=True)`), then compare ω error and |r| with Table III. `max_grad` and `normalize`
  are not run parameters yet; they need wiring through `THESIS_PARAMS` before the second part can run.

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
