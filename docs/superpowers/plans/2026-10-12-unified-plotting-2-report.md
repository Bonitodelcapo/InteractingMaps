# Unified Plotting — Plan 2: report figures

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every figure produced by `report/*` onto the shared conventions in `interacting_maps/style.py` and `interacting_maps/viz.py`, after a few core additions deferred from Plan 1.

**Architecture:** Same as Plan 1: `style.py` holds conventions only; `viz.py` draws. Report scripts keep their own layouts and data loading; they draw every map with `draw_map`, every ω line with `draw_omega`, take names and colours from `MODELS`, and save with `save()` inside `with use_style('paper'):`.

**Tech Stack:** Python 3.12, NumPy, matplotlib 3.10, pandas (analyze_grid only).

**Spec:** `docs/superpowers/specs/2026-10-11-unified-plotting-design.md`. Plan 1: `docs/superpowers/plans/2026-10-11-unified-plotting-1-core-evaluation.md` (complete).

## Global Constraints

- Run everything from the repo root. Tests are standalone scripts: `PYTHONIOENCODING=utf-8 PYTHONPATH=. python tests/<file>.py`; read `[ok ]` / `[FAIL]`.
- **Never write into `results/` and never overwrite files in `report/figures/`.** Visual checks redirect output to `experiments/plotting_check/report/` by overriding the module's figure-directory constant.
- ω is plotted in **deg/s**; time axes are **seconds since segment start at frame midpoints**. Non-time axes are exempt.
- Model names and colours only from `style.MODELS` (Cook, Thesis, IMU-anchor, CMax-anchor, CMax-inloop). A figure comparing models colours by model.
- Scaling rule: one image scaled on its own; panels meant to be compared share one `scale=` (`viz.map_scale`). Intensity maps of *different* runs or methods are scaled per image (I is defined only up to offset and scale).
- User decisions (2026-10-12): `fig_beta_speed`'s two CMax-anchor series share the model colour and differ by marker fill (filled = tuned δ_FR 0.1, hollow = default δ_FR 0.5); error heatmaps use `viridis` with a fixed 0–15 °/s range; `fig_maps` shows G direction-coloured with the square key.
- The `cmax` package was removed by an earlier refactor and its imports are stale (separate task). Verification snippets that run CMax models first alias it: `import sys, interacting_maps.angular_velocity as av; sys.modules['cmax'] = av`. Do not change the imports in this plan.
- Code-quality rules: small single-purpose functions, reuse existing helpers, import only from core modules, smallest diff, no comments that restate code, no compatibility shims.
- Other uncommitted files in the working tree are not yours (architecture.md, config.py, demo.*, eval_pipeline.*, run_config.py; a parallel session may commit diag_* files). Stage only the files a task names, by explicit path. Commits authored by the local git user; no attribution lines; do not push.

## File structure

| File | Change |
|---|---|
| `interacting_maps/style.py` | line styles, neutral colour, key rect, heatmap colormap/range; constrained layout in the base style; `save()` accepts `.png` or `.pdf` |
| `interacting_maps/viz.py` | read line styles and key rect from style; `draw_omega` requires `ref_src` with `ref` |
| `tests/test_viz.py` | checks for the above |
| `report/make_tables.py` | display names from `MODELS` |
| `report/make_figures.py` | all five figures on the conventions; `reference_omega` returns its source |
| `report/run_experiments.py` | `FIGDIR` constant; `converge`, `poisson`, `recon`, `dissoc`, `regfig` figures on the conventions; `DISPLAY` removed |
| `report/analyze_grid.py` | both figures on the conventions; input/output paths as `main()` arguments |

---

### Task 1: Core additions (style, viz, save)

**Files:**
- Modify: `interacting_maps/style.py`, `interacting_maps/viz.py`, `tests/test_viz.py`

**Interfaces:**
- Produces in `style.py`: `NEUTRAL_COLOR = '0.5'`, `EST_STYLE = dict(lw=1.2)`, `ERR_STYLE = dict(lw=1.0)`, `SERIES_STYLE = dict(lw=1.0, marker='.', ms=3)`, `KEY_RECT = (0.02, 0.02, 0.2, 0.2)`, `HEATMAP_CMAP = 'viridis'`, `ERROR_RANGE = (0.0, 15.0)`; `save(fig, path, show=False)` where `path` may end in `.png` or `.pdf` (always writes `<base>.png`, plus `<base>.pdf` under `report/figures/`). Both presets use constrained layout.
- `viz.draw_omega(...)` raises `ValueError('draw_omega: ref_src is required when ref is given')` if `ref` is given without `ref_src`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_viz.py`, add at the end of `test_style()` (inside the function):

```python
    with tempfile.TemporaryDirectory() as d:
        fig, _ = plt.subplots()
        style.save(fig, os.path.join(d, 'y.pdf'))
        check('save accepts a .pdf path (writes the png)',
              os.path.exists(os.path.join(d, 'y.png')) and not os.path.exists(os.path.join(d, 'y.pdf')))
    with style.use_style('screen'):
        fig = plt.figure()
    check('presets use constrained layout',
          type(fig.get_layout_engine()).__name__ == 'ConstrainedLayoutEngine')
    plt.close(fig)
```

In `test_lines()`, add before its final `plt.close(fig)`:

```python
    try:
        viz.draw_omega(ax, t, est, ref=est)
        raised = False
    except ValueError:
        raised = True
    check('draw_omega: ref without ref_src is refused', raised)
```

and after the first `draw_omega` block's colour check (`check('draw_omega: axis colour when no model', ...)`), add:

```python
    check('draw_omega: estimate width from EST_STYLE', lines[-1].get_linewidth() == style.EST_STYLE['lw'])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONIOENCODING=utf-8 PYTHONPATH=. python tests/test_viz.py`
Expected: `AttributeError: module 'interacting_maps.style' has no attribute 'EST_STYLE'` or `[FAIL]` lines for the new checks.

- [ ] **Step 3: Implement in `style.py`**

After `LINE_COLOR = '0.25'` add:

```python
NEUTRAL_COLOR = '0.5'
EST_STYLE = dict(lw=1.2)
ERR_STYLE = dict(lw=1.0)
SERIES_STYLE = dict(lw=1.0, marker='.', ms=3)
KEY_RECT = (0.02, 0.02, 0.2, 0.2)
HEATMAP_CMAP = 'viridis'
ERROR_RANGE = (0.0, 15.0)
```

In `_BASE`, add the entry `'figure.constrained_layout.use': True`.

Replace `save` with:

```python
def save(fig, path, show=False):
    """PNG at 200 dpi (+ PDF for report/figures/), then close or show. path ends in .png or .pdf."""
    import matplotlib.pyplot as plt
    base = os.path.splitext(path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    if _wants_pdf(path):
        fig.savefig(base + '.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)
```

- [ ] **Step 4: Implement in `viz.py`**

Extend the `interacting_maps.style` import with `EST_STYLE, ERR_STYLE, SERIES_STYLE, KEY_RECT`. Then:
- in `draw_map`: `key = ax.inset_axes(KEY_RECT)` (replacing the literal list);
- in `draw_omega`: as its first statement add
  ```python
      if ref is not None and ref_src is None:
          raise ValueError('draw_omega: ref_src is required when ref is given')
  ```
  and change the estimate line to `ax.plot(t, to_deg_s(est), color=color, label=label, **EST_STYLE)`;
- in `draw_error`: `ax.plot(t, err, color=color, label=label, **ERR_STYLE)`;
- in `draw_series`: `ax.plot(t, y, color=color, label=label, **SERIES_STYLE)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONIOENCODING=utf-8 PYTHONPATH=. python tests/test_viz.py`
Expected: `all checks passed`.

- [ ] **Step 6: Re-render the evaluation figures (constrained layout changes them)**

Run (Bash timeout 600000):

```bash
PYTHONIOENCODING=utf-8 IM_MLFLOW=0 PYTHONPATH=. python -c "
import evaluation as E
rc = E.RunConfig(dataset='poster_rotation', model='thesis_imu', segment='seg_C', n_frames=5, out_root='experiments/plotting_check')
E.experiment_tracking(rc, save_frames=True)
print(rc.output_dir)
rc1 = E.RunConfig(dataset='poster_rotation', model='thesis_imu', segment='seg_C', n_iters=20, out_root='experiments/plotting_check')
E.experiment_single_frame_convergence(rc1, max_iters=20)
print(rc1.output_dir)"
```

Expected: no `UserWarning` about layout engines; `tracking_plot.png`, `video_frames/frame_0004.png`, `exp1_convergence.png` written. Report their paths for controller review (the gap under the Exp 1 title should be gone).

- [ ] **Step 7: Commit**

```bash
git add interacting_maps/style.py interacting_maps/viz.py tests/test_viz.py
git commit -m "style: line styles, key rect, heatmap range, constrained layout; save takes .pdf paths; draw_omega needs ref_src"
```

---

### Task 2: `make_tables.py` display names

**Files:**
- Modify: `report/make_tables.py:17-25`

**Interfaces:**
- Consumes: `style.MODELS`

- [ ] **Step 1: Implement**

Replace

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'report', 'experiments')

# display name, csv model key
MODELS = [(r'\texttt{cook}', 'cook'),
          (r'\texttt{thesis}', 'thesis'),
          (r'\texttt{thesis+IMU}', 'thesis_imu'),
          (r'\texttt{CMax-anchor}', 'thesis_cmax'),
          (r'\texttt{CMax-inloop}', 'thesis_cmax_v2')]
```

with

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'report', 'experiments')
sys.path.insert(0, ROOT)
from interacting_maps.style import MODELS as MODEL_STYLE

# display name, csv model key
MODELS = [(rf'\texttt{{{MODEL_STYLE[k][0]}}}', k)
          for k in ('cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2')]
```

- [ ] **Step 2: Verify**

Run: `PYTHONIOENCODING=utf-8 python report/make_tables.py 2>/dev/null | grep -c 'IMU-anchor'`
Expected: a count ≥ 1 (one row per sequence with data), and `grep -c 'thesis+IMU'` on the same output gives 0.

- [ ] **Step 3: Commit**

```bash
git add report/make_tables.py
git commit -m "make_tables: model display names from style.MODELS (thesis+IMU -> IMU-anchor)"
```

---

### Task 3: `make_figures.py` on the conventions

**Files:**
- Modify: `report/make_figures.py`

**Interfaces:**
- Consumes: `style.MODELS, PAPER_WIDTH, PAPER_WIDE, NEUTRAL_COLOR, LABEL_BETA, frame_times, to_deg_s, use_style, save`; `viz.draw_map, draw_omega, map_scale`.
- Produces: `reference_omega(E, ds, t0, n=N_TRACK) -> (ndarray (n,3) rad/s, ref_src str)` (was: array only).

- [ ] **Step 1: Imports and removal of local style**

Delete `import matplotlib.colors as mcolors`. After the line `import experiment_tracking  # merged run.json reader ...` add:

```python
from interacting_maps.style import (MODELS, PAPER_WIDTH, PAPER_WIDE, NEUTRAL_COLOR, LABEL_BETA,
                                    frame_times, to_deg_s, use_style, save)
from interacting_maps.viz import draw_map, draw_omega, map_scale
```

Delete the `plt.rcParams.update({...})` block and the whole `_save` function.

In the module docstring, replace `Outputs PDFs (used by the LaTeX) and PNGs (for quick inspection) into` / `report/figures/.` with `Outputs PDFs (used by the LaTeX) and PNGs (for quick inspection) into report/figures/ via style.save.`

- [ ] **Step 2: `reference_omega` returns its source**

Replace the function body with:

```python
def reference_omega(E, ds, t0, n=N_TRACK):
    p = E.get_dataset_paths(ds)
    gt = E.load_groundtruth(p['groundtruth'])
    imu = E.load_imu(p['imu'])
    om = E.load_omega_gt(p.get('omega_gt'))
    refs = [E.get_reference_omega(gt, imu, t0 + k*DT, t0 + (k+1)*DT, om) for k in range(n)]
    return np.array([w for w, _ in refs]), refs[0][1]
```

- [ ] **Step 3: Replace `fig_beta_speed`**

```python
def fig_beta_speed(E):
    # (model, delta_FR, delta_anchor, filled marker)
    series = {
        'default': ('thesis_cmax',    0.5, 0.3,  False),
        'tuned':   ('thesis_cmax',    0.1, 0.5,  True),
        'inloop':  ('thesis_cmax_v2', 0.5, None, True),
    }
    data = {k: ([], []) for k in series}
    for ds, sid, t0 in BETA_SEGMENTS:
        w = to_deg_s(np.linalg.norm(reference_omega(E, ds, t0)[0], axis=1).mean())
        for key, (model, d_fr, d_a, _) in series.items():
            d = find_run(ds, sid, t0, model, d_fr, d_a)
            if d is None:
                print(f'    (missing {ds}/{sid}/{model} dFR={d_fr})')
                continue
            data[key][0].append(w)
            data[key][1].append(experiment_tracking.read_summary(d)['mean_beta'])

    with use_style('paper'):
        fig, ax = plt.subplots(figsize=(PAPER_WIDTH, 2.5))
        for key, (model, d_fr, _, filled) in series.items():
            x, y = np.array(data[key][0]), np.array(data[key][1])
            if len(x) < 2:
                continue
            name, col = MODELS[model]
            r = np.corrcoef(x, y)[0, 1]
            ax.scatter(x, y, s=26, zorder=3, color=col if filled else 'none', edgecolors=col,
                       linewidths=0.8, label=rf'{name}, $\delta_{{FR}}={d_fr}$  ($r={r:+.2f}$)')
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=col, lw=1.0, alpha=0.55,
                    ls='-' if filled else '--', zorder=2)
        ax.axhline(1.0, color=NEUTRAL_COLOR, lw=0.8, ls=':', zorder=1)
        ax.set_xlabel(r'rotation speed $\|\omega\|$ (°/s)')
        ax.set_ylabel(LABEL_BETA)
        ax.legend(loc='upper left')
        save(fig, os.path.join(FIG, 'fig_beta_speed.png'))
    print('  wrote fig_beta_speed.pdf / .png')
```

- [ ] **Step 4: Replace `fig_tracking`**

```python
def fig_tracking(E):
    ds, sid, t0 = 'poster_rotation', 'seg_C', 8.816
    ref, ref_src = reference_omega(E, ds, t0)
    t = frame_times(DT, N_TRACK)
    runs = [('thesis_cmax', find_run(ds, sid, t0, 'thesis_cmax', 0.1, 0.5)),
            ('thesis', find_run(ds, sid, t0, 'thesis', 0.1))]

    with use_style('paper'):
        fig, axes = plt.subplots(3, 1, figsize=(PAPER_WIDTH, 3.6), sharex=True)
        first = True
        for model, d in runs:
            if d is None:
                print(f'    (missing run for {MODELS[model][0]})')
                continue
            rows = list(csv.DictReader(open(os.path.join(d, 'tracking.csv'))))
            est = np.array([[float(r['est_wx']), float(r['est_wy']), float(r['est_wz'])]
                            for r in rows])[:N_TRACK]
            for i in range(3):
                draw_omega(axes[i], t[:len(est)], est[:, i], axis=i, model=model,
                           ref=ref[:len(est), i] if first else None, ref_src=ref_src)
            first = False
        for ax in axes:
            ax.label_outer()
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
        save(fig, os.path.join(FIG, 'fig_tracking.png'))
    print('  wrote fig_tracking.pdf / .png')
```

- [ ] **Step 5: Replace `_draw_maps` and `fig_distortion`'s drawing**

```python
def _draw_maps(maps, labels, name, height_per_row=1.72):
    with use_style('paper'):
        fig, axes = plt.subplots(len(labels), 4, figsize=(PAPER_WIDE, height_per_row * len(labels)))
        axes = np.atleast_2d(axes)
        for r, (arrs, lab) in enumerate(zip(maps, labels)):
            for c, (kind, arr) in enumerate(zip('VIGF', arrs)):
                draw_map(axes[r, c], kind, arr, title=None if r == 0 else '')
            axes[r, 0].set_ylabel(lab)
        save(fig, os.path.join(FIG, f'{name}.png'))
    print(f'  wrote {name}.pdf / .png')
```

In `fig_distortion`, replace everything from `    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.15))` to the end of the function with:

```python
    panels = [(V_raw, r'raw events (lens modelled in $\mathcal{C}$)'),
              (V_und, 'undistorted events'),
              (V_und[10:80, 140:230], 'detail (undistorted)')]
    scale = map_scale('V', V_und)
    with use_style('paper'):
        fig, ax = plt.subplots(1, 3, figsize=(PAPER_WIDE, 2.15))
        for a, (img, ttl) in zip(ax, panels):
            draw_map(a, 'V', img, title=ttl, scale=scale)
        save(fig, os.path.join(FIG, 'fig_distortion.png'))
    print('  wrote fig_distortion.pdf / .png')
```

- [ ] **Step 6: Render every figure for review (outputs redirected, cmax aliased)**

Run (Bash timeout 600000):

```bash
mkdir -p experiments/plotting_check/report && cp report/figures/.maps_cache.npz experiments/plotting_check/report/
PYTHONIOENCODING=utf-8 IM_MLFLOW=0 PYTHONPATH=. python -c "
import sys, importlib.util, numpy as np
import interacting_maps.angular_velocity as av; sys.modules['cmax'] = av
sys.argv = ['x']
import evaluation as E
spec = importlib.util.spec_from_file_location('MF', 'report/make_figures.py')
MF = importlib.util.module_from_spec(spec); spec.loader.exec_module(MF)
import os; MF.FIG = os.path.abspath('experiments/plotting_check/report')
runs = {}
for m, kw in (('thesis_cmax', dict(delta_FR=0.1, delta_IMU=0.5)), ('thesis', dict(delta_FR=0.1))):
    rc = E.RunConfig(dataset='poster_rotation', model=m, segment='seg_C', n_frames=25, out_root='experiments/plotting_check', **kw)
    E.experiment_tracking(rc, save_frames=False); runs[m] = rc.output_dir
MF.N_TRACK = 25
MF.find_run = lambda ds, sid, t0, model, *a, **k: runs.get(model, runs['thesis'])
MF.BETA_SEGMENTS = [('poster_rotation', 'seg_A', 14.166), ('poster_rotation', 'seg_C', 8.816), ('poster_rotation', 'seg_D', 1.866)]
rng = np.random.default_rng(0)
MF.experiment_tracking.read_summary = lambda d: {'mean_beta': float(rng.uniform(0.8, 1.4))}
for f in (MF.fig_tracking, MF.fig_beta_speed, MF.fig_maps, MF.fig_appendix, MF.fig_distortion):
    f(E)
print(MF.FIG)"
```

Expected: five PNGs in `experiments/plotting_check/report/` (no PDFs, since it is not `report/figures/`), nothing written under `report/figures/` or `results/` (`git status --short report/figures` shows nothing). `fig_beta_speed` uses fake β values — check only styling. Report the PNG paths.

- [ ] **Step 7: Commit**

```bash
git add report/make_figures.py
git commit -m "make_figures: all report figures via style/viz (deg/s, model colours, draw_map, paper preset)"
```

---

### Task 4: `run_experiments.py` figures on the conventions

**Files:**
- Modify: `report/run_experiments.py` (module header, `cmd_converge`, `cmd_poisson`, `cmd_recon`, `cmd_dissoc`, `cmd_regfig`)

**Interfaces:**
- Consumes: `style.MODELS` (imported as `MODEL_STYLE`, since the module already defines `MODELS`), `PAPER_WIDE, use_style, save`; `viz.draw_map, map_scale, net_maps`.
- Produces: module constant `FIGDIR = os.path.join(ROOT, 'report', 'figures')`, the only place report figures are written by this module.

- [ ] **Step 1: Header**

After `import experiment_tracking  # merged run.json reader ...` add:

```python
from interacting_maps.style import MODELS as MODEL_STYLE, PAPER_WIDE, use_style, save
from interacting_maps.viz import draw_map, map_scale, net_maps
```

Delete the `#: names as the report writes them, for figure titles` comment and the `DISPLAY = {...}` dict. After `OUTDIR = ...` / `os.makedirs(OUTDIR, exist_ok=True)` add:

```python
FIGDIR = os.path.join(ROOT, 'report', 'figures')
```

- [ ] **Step 2: `cmd_converge`**

Replace

```python
                    I = net.I if net.I.shape == (H, W) else net.I[:H, :W]
                    snaps[it] = I.copy()
```

with

```python
                    snaps[it] = net_maps(net, H, W)['I'].copy()
```

Replace everything from `        fig, axes = plt.subplots(len(MODELS), len(checkpoints),` through `        plt.close(fig)` (keep the following `print`) with:

```python
        last = checkpoints[-1]
        with use_style('paper'):
            fig, axes = plt.subplots(len(MODELS), len(checkpoints),
                                     figsize=(PAPER_WIDE, 1.0 * len(MODELS) + 0.4))
            for r, model in enumerate(MODELS):
                scale = map_scale('I', rows[model][last])
                for c, it in enumerate(checkpoints):
                    draw_map(axes[r, c], 'I', rows[model][it], scale=scale,
                             title=f'iter {it}' if r == 0 else '')
                axes[r, 0].set_ylabel(MODEL_STYLE[model][0])
            fig.suptitle(f'{lab} {sid}: intensity map across message-passing '
                         f'iterations (frame {args.frame})')
            save(fig, os.path.join(FIGDIR, f'fig_converge_{lab}.png'))
```

- [ ] **Step 3: `cmd_poisson`**

Replace everything from `        print(f"\n{lab}: {'low_freq':>10} {'contrast':>10}")` through `        plt.close(fig)` (keep the following `print(f"  -> {out}")`) with:

```python
        print(f"\n{lab}: {'low_freq':>10} {'contrast':>10}")
        out = os.path.join(root, f'poisson_{lab}{args.tag}.png')
        with use_style('paper'):
            fig, ax = plt.subplots(1, len(panels), figsize=(PAPER_WIDE, 1.9))
            for a, (ttl, img, _) in zip(ax, panels):
                if img is None:
                    a.set_axis_off()
                    continue
                st = E._intensity_stats(img)
                print(f"  {ttl:<22} {st['low_freq_share']:10.3f} {st['contrast']:10.3f}")
                draw_map(a, 'aps' if ttl == 'APS reference' else 'I', img, title=ttl)
            fig.suptitle(f'{lab}: the same G, five ways of turning it into I')
            save(fig, out)
```

- [ ] **Step 4: `cmd_recon`**

Replace from `    fig, axes = plt.subplots(len(seqs), 4, figsize=(7.1, 1.85 * len(seqs)))` through `    plt.close(fig)` (keep the final `print`) with:

```python
    with use_style('paper'):
        fig, axes = plt.subplots(len(seqs), 4, figsize=(PAPER_WIDE, 1.85 * len(seqs)))
        axes = np.atleast_2d(axes)
        for row, (ds, sid, t0, lab) in enumerate(seqs):
            got = {}
            for dt in (0.02, 0.01):
                nf = int(round(dur / dt))
                rc = _cfg(E, ds, sid, t0, args.model, dt, nf, out_root=root,
                          poisson='iterative', deltas=dict(base))
                npz = os.path.join(rc.output_dir, 'maps_final.npz')
                if not os.path.exists(npz):
                    print(f"  running {lab} dt={dt*1000:.0f}ms ...", flush=True)
                    _run(E, rc, not args.no_frames, args.frame_stride)
                got[dt] = np.load(npz)
            aps = _aps(E, ds, t0 + dur)
            ref = np.log(aps.astype(np.float64) + 1.0) if aps is not None else None
            panels = [
                ('APS', aps, 'aps'),
                (r'iterative, $\Delta t = 20$ ms', got[0.02]['I'], 'I'),
                (r'iterative, $\Delta t = 10$ ms', got[0.01]['I'], 'I'),
                (r'exact read-out, $\Delta t = 10$ ms',
                 solve_poisson_exact(got[0.01]['G'], 'fft'), 'I'),
            ]
            for col, (ttl, img, kind) in enumerate(panels):
                a = axes[row, col]
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=ttl if row == 0 else '')
                if col == 0:
                    a.set_ylabel(lab)
                if col > 0 and ref is not None:
                    a.set_xlabel(f'$r={_corr(_crop(img), _crop(ref)):.2f}$')
            print(f"  {lab}: done", flush=True)
        save(fig, os.path.join(FIGDIR, 'fig_recon.png'))
```

- [ ] **Step 5: `cmd_dissoc`**

Replace from `    cols = len(models) + 1` through `    plt.close(fig)` (keep the final `print`) with:

```python
    cols = len(models) + 1
    with use_style('paper'):
        fig, axes = plt.subplots(len(seqs), cols, figsize=(PAPER_WIDE, 1.65 * len(seqs)))
        axes = np.atleast_2d(axes)
        for r_i, (ds, sid, t0, lab) in enumerate(seqs):
            aps = _aps(E, ds, t0 + nf * args.dt)
            panels = [('APS', aps, None, 'aps')]
            for m in models:
                # Exact path from RunConfig, not a glob: results/ holds dozens of
                # runs per model that differ only in parameters the pattern would
                # not distinguish, and the oldest of them predate maps_final.npz.
                rc = _cfg(E, ds, sid, t0, m, args.dt, nf)
                npz = os.path.join(rc.output_dir, 'maps_final.npz')
                img = None
                if os.path.exists(npz):
                    img = solve_poisson_exact(np.load(npz)['G'], 'fft')
                else:
                    print(f'    missing: {npz}', flush=True)
                panels.append((MODEL_STYLE[m][0], img, stats.get((ds, m)), 'I'))
            for c_i, (title, img, st, kind) in enumerate(panels):
                a = axes[r_i, c_i]
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=title if r_i == 0 else '')
                if c_i == 0:
                    a.set_ylabel(lab)
                if st is not None:
                    a.set_xlabel(f"$r={float(st['recon']):.2f}$,  "
                                 f"$\\omega$ err $={float(st['err']):.1f}$ °/s")
            print(f'  {lab}: done', flush=True)
        save(fig, os.path.join(FIGDIR, 'fig_dissociation.png'))
```

- [ ] **Step 6: `cmd_regfig`**

Replace from `    fig, axes = plt.subplots(len(seqs), len(cols),` through `    plt.close(fig)` (keep the final `print`) with:

```python
    with use_style('paper'):
        fig, axes = plt.subplots(len(seqs), len(cols), figsize=(PAPER_WIDE, 1.5 * len(seqs)))
        axes = np.atleast_2d(axes)
        for row, (ds, sid, t0, lab) in enumerate(seqs):
            ref = None
            aps = _aps(E, ds, t0 + nf * args.dt)
            if aps is not None:
                ref = np.log(aps.astype(np.float64) + 1.0)
            for col, (title, suffix) in enumerate(cols):
                a = axes[row, col]
                kind, img = 'aps', aps
                if col > 0:
                    kind, img = 'I', None
                    pat = (f'{root}/{ds}/{args.model}/{sid}_t{t0:.3f}_'
                           f'dt{int(args.dt*1000)}ms_n{nf}_*C_full{suffix}')
                    hits = [d for d in sorted(glob.glob(pat))
                            if os.path.basename(d).endswith('C_full' + suffix)]
                    if hits:
                        img = solve_poisson_exact(
                            np.load(os.path.join(hits[0], 'maps_final.npz'))['G'], 'fft')
                if img is None:
                    a.set_axis_off()
                    continue
                draw_map(a, kind, img, title=title if row == 0 else '')
                if col == 0:
                    a.set_ylabel(lab)
                elif ref is not None and ref.shape == img.shape:
                    a.set_xlabel(f'$r={_corr(_crop(img), _crop(ref)):.2f}$')
            print(f'  {lab}: done', flush=True)
        save(fig, os.path.join(FIGDIR, 'fig_regularisers.png'))
```

- [ ] **Step 7: Confirm nothing else uses the removed names**

Run: Grep `report/run_experiments.py` for `DISPLAY`, `fontsize=`, `tight_layout`, `savefig`, `cmap=`.
Expected: no matches.

- [ ] **Step 8: Render the figures for review (outputs redirected, cmax aliased)**

Run (Bash timeout 600000):

```bash
PYTHONIOENCODING=utf-8 IM_MLFLOW=0 PYTHONPATH=. python -c "
import sys, os, importlib.util
from types import SimpleNamespace as N
import interacting_maps.angular_velocity as av; sys.modules['cmax'] = av
sys.argv = ['x']
import evaluation as E
spec = importlib.util.spec_from_file_location('RE', 'report/run_experiments.py')
RE = importlib.util.module_from_spec(spec); spec.loader.exec_module(RE)
RE.FIGDIR = os.path.abspath('experiments/plotting_check/report')
common = dict(sequences='poster', dt=0.02, frame=0, no_frames=True, frame_stride=1, tag='', base='')
RE.cmd_converge(E, N(checkpoints='1,3,10', **common))
RE.cmd_poisson(E, N(model='thesis_imu', duration=0.2, modes='iterative,fft', name='plotting_check_poisson', **common))
RE.cmd_recon(E, N(model='thesis_imu', duration=0.2, name='plotting_check_recon', **common))
RE.cmd_dissoc(E, N(models='thesis,thesis_imu', duration=3.0, **common))
RE.cmd_regfig(E, N(model='thesis_imu', duration=0.2, name='plotting_check_regfig', **common))
print(RE.FIGDIR)"
```

Expected: `fig_converge_poster.png`, `fig_recon.png`, `fig_dissociation.png`, `fig_regularisers.png` in `experiments/plotting_check/report/`, and `poisson_poster.png` in `experiments/plotting_check_poisson/`. `fig_dissociation` and `fig_regularisers` show only the APS column locally (their runs live in `results/` and `experiments/ablation/`, absent here) — that is expected. Nothing written under `report/figures/` or `results/` (`git status --short report/figures` shows nothing). Report the PNG paths.

- [ ] **Step 9: Commit**

```bash
git add report/run_experiments.py
git commit -m "run_experiments: report figures via style/viz (draw_map, MODELS names, shared converge scale, FIGDIR)"
```

---

### Task 5: `analyze_grid.py` on the conventions

**Files:**
- Modify: `report/analyze_grid.py`

**Interfaces:**
- Consumes: `style.MODELS, HEATMAP_CMAP, ERROR_RANGE, LABEL_ERROR, NEUTRAL_COLOR, SCREEN_WIDTH, use_style, save`.
- Produces: `main(grid_csv='results/parameter_grid.csv', out_dir='results/analysis')`.

- [ ] **Step 1: Imports and signature**

Replace the import block (lines 8–11) with:

```python
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interacting_maps.style import (MODELS, HEATMAP_CMAP, ERROR_RANGE, LABEL_ERROR,
                                    NEUTRAL_COLOR, SCREEN_WIDTH, use_style, save)
```

Replace `def main():` / `    df = pd.read_csv('results/parameter_grid.csv')` with:

```python
def main(grid_csv='results/parameter_grid.csv', out_dir='results/analysis'):
    df = pd.read_csv(grid_csv)
```

Replace `    os.makedirs('results/analysis', exist_ok=True)` with `    os.makedirs(out_dir, exist_ok=True)`.

- [ ] **Step 2: Replace section 3 (heatmap)**

Replace from `    fig, axes = plt.subplots(1, 3, figsize=(15, 5))` through the heatmap's `    plt.close()` with:

```python
    lo, hi = ERROR_RANGE
    with use_style('screen'):
        fig, axes = plt.subplots(1, 3, figsize=(SCREEN_WIDTH, 4.5))
        fig.suptitle(rf"{MODELS['thesis_imu'][0]}: error by $\delta_{{FR}}$ × $\delta_{{IMU}}$ "
                     '(mean over segments)')
        for ax, n_fr in zip(axes, [25, 50, 150]):
            sub = imu_df[imu_df['n_frames'] == n_fr]
            pivot = sub.pivot_table(values='mean_err_deg_s', index='delta_FR',
                                    columns='delta_IMU', aggfunc='mean')
            im = ax.imshow(pivot.values, cmap=HEATMAP_CMAP, aspect='auto', vmin=lo, vmax=hi)
            ax.set_xticks(range(len(pivot.columns)), [f'{v:.2f}' for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [f'{v:.2f}' for v in pivot.index])
            ax.set_xlabel(r'$\delta_{IMU}$')
            ax.set_ylabel(r'$\delta_{FR}$')
            ax.set_title(f'{n_fr} frames')
            ax.grid(False)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                            color='w' if v < (lo + hi) / 2 else 'k')
        fig.colorbar(im, ax=axes, label=LABEL_ERROR)
        save(fig, os.path.join(out_dir, 'heatmap_FR_vs_IMU.png'))
```

- [ ] **Step 3: Replace section 4 (error vs |ω|)**

Replace from `    fig, ax = plt.subplots(figsize=(8, 6))` through that section's `    plt.close()` with:

```python
    with use_style('screen'):
        fig, ax = plt.subplots(figsize=(8, 6))
        for model, marker in [('cook', 'o'), ('thesis', 's')]:
            mdf = df[df['model'] == model]
            best_per_seg = mdf.groupby(['dataset', 'segment_id']).apply(
                lambda g: g.loc[g['mean_err_deg_s'].idxmin()])
            name, col = MODELS[model]
            ax.scatter(best_per_seg['omega_gt_deg_s'], best_per_seg['mean_err_deg_s'],
                       marker=marker, s=60, color=col, label=name)
        x_range = np.linspace(0, ax.get_xlim()[1], 50)
        ax.plot(x_range, x_range, ls='--', color=NEUTRAL_COLOR, label=r'error $=\|\omega\|$')
        ax.set_xlabel(r'$\|\omega_{ref}\|$ (°/s)')
        ax.set_ylabel('best achievable ' + LABEL_ERROR)
        ax.set_title('Error scales with rotation speed (no anchor)')
        ax.legend()
        save(fig, os.path.join(out_dir, 'error_vs_omega.png'))
```

- [ ] **Step 4: Names in the printed table and remaining paths**

In section 5, replace the header line `f"{'Cook':>8} {'Thesis':>8} {'Th+IMU':>8}")` with
`f"{MODELS['cook'][0]:>8} {MODELS['thesis'][0]:>8} {MODELS['thesis_imu'][0]:>11}")` and widen the matching value format for thesis_imu from `:>7.1f}%` to `:>10.1f}%`.

In section 7, replace both occurrences of `'results/analysis/parameter_grid_enhanced.csv'` with `os.path.join(out_dir, 'parameter_grid_enhanced.csv')`, and `"Plots saved in results/analysis/"` with `f"Plots saved in {out_dir}/"`.

- [ ] **Step 5: Render with a synthetic grid (no real grid exists locally)**

Run:

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. python -c "
import importlib.util, itertools, os, numpy as np, pandas as pd
rng = np.random.default_rng(0); rows = []
for (ds, seg), m, nf, dfr in itertools.product(
        [('poster_rotation', s) for s in 'ABCDE'] + [('boxes_rotation', s) for s in 'ABC'],
        ['cook', 'thesis', 'thesis_imu'], [25, 50, 150], [0.1, 0.2, 0.5]):
    for dimu in ([0.1, 0.2, 0.5] if m == 'thesis_imu' else [0.0]):
        rows.append(dict(dataset=ds, segment_id='seg_' + seg, model=m, n_frames=nf, n_iters=75,
                         delta_FR=dfr, delta_IMU=dimu, mean_err_deg_s=rng.uniform(1, 20),
                         mean_dir_err_deg=rng.uniform(1, 30)))
d = 'experiments/plotting_check/grid'; os.makedirs(d, exist_ok=True)
pd.DataFrame(rows).to_csv(f'{d}/parameter_grid.csv', index=False)
spec = importlib.util.spec_from_file_location('AG', 'report/analyze_grid.py')
AG = importlib.util.module_from_spec(spec); spec.loader.exec_module(AG)
AG.main(f'{d}/parameter_grid.csv', d)" > /dev/null
ls experiments/plotting_check/grid
```

Expected: `heatmap_FR_vs_IMU.png`, `error_vs_omega.png`, `parameter_grid_enhanced.csv` listed. Report the two PNG paths.

- [ ] **Step 6: Commit**

```bash
git add report/analyze_grid.py
git commit -m "analyze_grid: viridis heatmap with fixed range, model colours, mathtext, legend includes y=x"
```
