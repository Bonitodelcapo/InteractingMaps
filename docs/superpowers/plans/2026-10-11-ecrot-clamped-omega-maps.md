# ECRot clamped-ω maps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rung `1c_R` (R clamped to ω_gt) to the ω ladder and a maps run that renders a frames GIF, an iterations GIF, a static grid and a flow-EPE plot comparing rows 1c_R / 1b_free / 0.

**Architecture:** Compute stays in `diag_omega_ladder.py` (`--mode maps`, cluster-ready, writes one `.npz`); plotting reads the npz (`--mode plots`). Reuse core: `cost_kin.update_r` (R freeze), `viz.net_maps/draw_map/draw_series`, `style.frame_times/use_style/save`, `iwe_io.make_gif`. Only new core function: `metrics.flow_epe`.

**Tech Stack:** Python, NumPy, matplotlib, imageio (already in requirements.txt).

Spec: `docs/superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md`.

## Global Constraints

- Import only from core modules (`interacting_maps/*`, `config`); never from scripts.
- Minimal diffs; default code paths of existing rungs unchanged.
- Tests are standalone scripts printing `[ok]`/`[FAIL]` (not pytest).
- `.py` and `.ipynb` updated together (authored walkthrough notebook).
- Commits authored by the user's git config; no attribution lines; never push.
- Heavy runs on the cluster; local runs only with `--limit`.

---

### Task 1: core helpers — `flow_epe`, named `make_gif`

**Files:** Modify `interacting_maps/metrics.py`, `interacting_maps/iwe_io.py:57-71`, `tests/test_viz.py` (`test_iwe_io`).

**Produces:** `metrics.flow_epe(F, F_ref) -> float`; `iwe_io.make_gif(out_dir, fps=15, name='iwe_sequence.gif')`.

- [ ] Add to `test_iwe_io` (after the contrast curve check):
```python
        iwe_io.make_gif(d, fps=5, name='named.gif')
        check('make_gif writes the named file', os.path.exists(os.path.join(d, 'named.gif')))
```
and a metrics test:
```python
def test_metrics():
    print('metrics.py')
    from interacting_maps.metrics import flow_epe
    F = np.zeros((4, 5, 2)); G = np.zeros((4, 5, 2)); G[..., 0], G[..., 1] = 3.0, 4.0
    check('flow_epe: identical is 0', flow_epe(F, F) == 0.0)
    check('flow_epe: (3,4) offset is 5', np.isclose(flow_epe(F, G), 5.0))
```
(call `test_metrics()` in `__main__`).
- [ ] Run `PYTHONPATH=. python tests/test_viz.py` → FAIL (named file missing, flow_epe import error).
- [ ] Implement:
```python
def flow_epe(F, F_ref):
    """Mean endpoint error (px/frame) between two (H, W, 2) flow fields."""
    return float(np.mean(np.linalg.norm(F - F_ref, axis=-1)))
```
and `make_gif(out_dir, fps=15, name='iwe_sequence.gif')` writing `os.path.join(out_dir, name)`.
- [ ] Run again → all ok. Commit `metrics: flow_epe; iwe_io: make_gif output name`.

### Task 2: ladder compute — `clamp_R`, hooks, rung 1c, F_epe column

**Files:** Modify `diag_omega_ladder.py`.

**Produces:** `clamp_R(net, R_star)`, `trace_relaxation(..., on_iter=None)`,
`advance(net, V, k, dt, omega_imu=None, on_iter=None, iter_frame=None)`,
`rung0_baseline(ctx, on_frame=None, on_iter=None, iter_frame=None)`,
`rung1b_clamped_flow(ctx, anchored, on_frame=None, on_iter=None, iter_frame=None)`,
`rung1c_clamped_R(ctx, on_frame=None, on_iter=None, iter_frame=None)`; hooks are
`on_frame(k, net)` and `on_iter(net)`; each scored rung adds `s['F_epe']`.

- [ ] Add `_selfcheck_clamps(ctx)` to `--mode setup` (first available segment): `clamp_R` keeps `net.R` exactly equal to `R_star` after `net.step(V, ITERS_PER_FRAME)`; `trace_relaxation(on_iter=...)` gives maps/R identical to `net.step` and calls `on_iter` `ITERS_PER_FRAME` times; `flow_epe(F, F) == 0`.
- [ ] Run `python diag_omega_ladder.py --mode setup --limit 2` → NameError (FAIL).
- [ ] Implement:
```python
def clamp_R(net, R_star):
    """Hold q_R at R_star: with the kinematic R-update off (the V2 switch) and
    no IMU anchor, no cost sends gradient to R, so the relaxation keeps it."""
    net.cost_kin.update_r = False
    net.q_R.value = R_star.copy()

def advance(net, V, k, dt, omega_imu=None, on_iter=None, iter_frame=None):
    """One frame's relaxation; at frame iter_frame step it one iteration at a
    time through trace_relaxation so on_iter sees every iteration."""
    if on_iter is not None and k == iter_frame:
        trace_relaxation(net, V, config.ITERS_PER_FRAME, dt, omega_imu, on_iter)
    else:
        net.step(V, n_iters=config.ITERS_PER_FRAME, omega_imu=omega_imu)
```
`trace_relaxation` calls `on_iter(net)` after each iteration when given. Rungs 0/1b
iterate with `enumerate`, call `advance`, append `flow_epe(net.F, gt_flow(net, w*dt))`,
call `on_frame(k, net)`. `rung1c_clamped_R` = rung 1b shape with `clamp_R(net, w*dt)`
instead of `clamp_flow`, no anchor. `RESULTS_HEADER += ['F_epe']`; `_row` appends
`round(s['F_epe'], 4)` or `''`; `run_all_segments` adds `_row(ctx, '1c_R', rung1c_clamped_R(ctx))`.
`RUNG_COLORS['1c_R'] = '#8a5cd1'`, `RUNG_LABEL['1c_R'] = 'rung 1c (R clamped)'`.
- [ ] Run setup → all `[ok]`; `--mode run --limit 2` writes CSVs with a `1c_R` row and `F_epe`. Commit.

### Task 3: maps run + plots

**Files:** Modify `diag_omega_ladder.py`.

**Produces:** `MAP_RUNGS = ('1c_R', '1b_free', '0')`, `collect_maps(ctx, K) -> dict`
(keys `V (n,H,W)`, `F_gt (n,H,W,2)`, `I/G/F (3,n,...)`, `I_it/G_it/F_it (3,T,...)`,
`rungs`, `dt`, `t_start`, `K`), `maps_path(seg_name)`, `run_maps(seg_name, K=None)`,
`_map_scales(d)`, `draw_maps_grid(axes, maps_t, scales, rungs)`,
`plot_P4(d, out)`, `plot_P5(d, out)`, `make_maps_gifs(d, out_dir, fps=8)`,
`plot_maps(seg_name)`; `make_all_plots()` guards P1–P3 on CSV existence and calls
`plot_maps` for every `diag_maps_*.npz`. CLI: `--mode maps --segment city_short --frame K`.

- [ ] Implement (code as in the spec: fixed scales from row 0, 3×6 grid via `draw_map`, GIF frames saved with plain `fig.savefig(dpi=72)` so every PNG has the same size, assembled with `make_gif(..., name='gif_frames.gif' | 'gif_iters.gif')`).
- [ ] Smoke: `python diag_omega_ladder.py --mode maps --limit 6 --frame 3` then `--mode plots` → `diag_maps_city_short.npz`, `diag_P4_maps_city_short.png`, `diag_P5_F_epe_city_short.png`, `diag_maps_city_short/{frames,iters}/gif_*.gif`. Inspect the PNGs. Commit.

### Task 4: notebook walkthrough section

**Files:** Modify `diag_omega_ladder.ipynb` (insert after the summary step, before "Across segments").

- [ ] Add markdown "Step 10 — Clamp R to ω_gt: do I and F relax well?" + code cells: `d = dol.collect_maps(ctx, n // 2)` with per-row mean EPE; `plot_P4`/`plot_P5` displayed; `make_maps_gifs` and both GIFs displayed inline. Mention `--mode maps` in the cluster section.
- [ ] Execute the new cells headless (`jupyter nbconvert --execute` or run the cell sources in a script) to confirm they run. Commit.
