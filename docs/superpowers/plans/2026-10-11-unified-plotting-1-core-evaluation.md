# Unified Plotting — Plan 1: core modules + evaluation.py

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the shared plotting conventions (`interacting_maps/style.py`) and drawing functions (`interacting_maps/viz.py`), then move `iwe_io.py` and every figure in `evaluation.py` onto them.

**Architecture:** `style.py` holds every convention (model names/colours, ω-axis colours, reference/IMU line styles, units, time axis, label text, size presets, `save`) and draws nothing. `viz.py` holds map renderers (array → RGB array, no figures), one map drawer `draw_map`, a crop helper `net_maps`, and line-plot drawers `draw_omega` / `draw_error` / `draw_series`. Callers compose their own layouts from these.

**Tech Stack:** Python 3.12, NumPy, matplotlib 3.10 (only plotting dependency).

**Spec:** `docs/superpowers/specs/2026-10-11-unified-plotting-design.md`

**This is plan 1 of 3.** Plan 2 migrates `report/*`; plan 3 migrates `debug_visual.py`, `debug_pipeline.py`, `diag_omega_ladder.py`, `demo.py(+ipynb)`, `eval_pipeline.py(+ipynb)`, `find_segments.py`, and finally deletes `normalise` / `flow_to_rgb` / `grad_to_rgb` from `viz.py`. Those three old functions stay in `viz.py` during plan 1 because `debug_visual.py` and `demo.ipynb` still import them.

## Global Constraints

- **Precondition:** the working-tree refactor (modules moved into `interacting_maps/`, e.g. untracked `interacting_maps/viz.py`, `interacting_maps/iwe_io.py`, modified `evaluation.py`) must be committed by the user before Task 1. Every commit in this plan must contain only the files its task names.
- Run everything from the repo root. Tests are standalone scripts, not pytest: `PYTHONPATH=. python tests/<file>.py`, read the `[ok ]` / `[FAIL]` lines; exit code 1 on failure.
- Never write into `results/`. Visual checks use `RunConfig(..., out_root='experiments/plotting_check')` and `IM_MLFLOW=0`.
- ω is plotted in **deg/s**; data stays in rad/s. Time axes are **seconds since segment start at frame midpoints**, `(k + 0.5)·dt`. Non-time axes (iterations, parameter values) are exempt.
- Model names: Cook, Thesis, IMU-anchor, CMax-anchor, CMax-inloop. A figure colours lines by ω-axis when it shows one model, by model when it compares models, never both.
- Code-quality rules: small single-purpose functions, reuse existing helpers, import only from core modules (never from scripts), smallest diff, no comments that restate code, no compatibility shims.
- Commits are authored by the local git user; no attribution lines. Do not push.

## Spec amendments made in Task 1

These refine the approved spec and are written into it in Task 1:

1. V colormap is `bwr`, not `RdBu_r`: `bwr` is exactly white at zero, matching the white base of G/F. `RdBu_r`'s midpoint is `#f7f7f7`.
2. `use_style(preset)` is a context manager (`with use_style('screen'):`) so library code such as `evaluation.py` does not change global rcParams for its caller.
3. `save(fig, path, show=False)`: `show=True` calls `plt.show()` instead of closing, so notebooks display and save.
4. Percentile scales fall back to the max when the 99th percentile is ~0 (maps sparser than 1 % non-zero, common for IWE).
5. `iwe_io.plot_contrast_curve(out_dir, t_start)` takes `t_start` so its x axis is seconds since segment start.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `interacting_maps/style.py` | create | conventions only: names, colours, styles, units, labels, presets, `save` |
| `interacting_maps/viz.py` | modify | add renderers, `draw_map`, `net_maps`, line drawers; keep `normalise_robust`; keep old helpers until plan 3 |
| `interacting_maps/iwe_io.py` | modify | use `render_IWE`; contrast curve over seconds; drop the import-time `matplotlib.use('Agg')` |
| `evaluation.py` | modify | `_plot_tracking`, `_save_3col_frame`, Exp 1, Exp 3, Exp 6 figures; contrast-curve call |
| `tests/test_viz.py` | create | numeric checks of every convention |
| `docs/superpowers/specs/2026-10-11-unified-plotting-design.md` | modify | the five amendments above |

---

### Task 1: `style.py` — conventions

**Files:**
- Create: `interacting_maps/style.py`
- Create: `tests/test_viz.py`
- Modify: `docs/superpowers/specs/2026-10-11-unified-plotting-design.md`

**Interfaces:**
- Produces (used by every later task):
  - `MODELS: dict[str, tuple[str, str]]` — key → (display name, hex colour)
  - `OMEGA_AXIS_COLORS: tuple[str, str, str]`
  - `REF_STYLE: dict`, `IMU_STYLE: dict`, `LINE_COLOR: str`
  - `LABEL_TIME, LABEL_OMEGA, LABEL_ERROR, LABEL_DIR_ERROR, LABEL_BETA: str`
  - `MAP_TITLES: dict[str, str]` for kinds `'V','I','aps','G','F','IWE'`
  - `PAPER_WIDTH = 3.4`, `PAPER_WIDE = 7.05`, `SCREEN_WIDTH = 12.0` (inches)
  - `omega_label(axis: int) -> str`
  - `ref_label(ref_src: str) -> str` for `'omega_gt' | 'groundtruth' | 'imu'`
  - `to_deg_s(omega_rad_s) -> ndarray | float`
  - `frame_times(dt: float, n: int) -> ndarray` (seconds, frame midpoints)
  - `use_style(preset: str)` → context manager, preset `'paper' | 'screen'`
  - `save(fig, path: str, show: bool = False) -> None` (`path` ends in `.png`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_viz.py`:

```python
"""Checks that the plotting conventions in style.py and viz.py hold numerically.

    PYTHONPATH=. python tests/test_viz.py
"""
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from interacting_maps import style

FAILURES = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))
    if not ok:
        FAILURES.append(name)


def test_style():
    print('style.py')
    check('five models with name and colour',
          set(style.MODELS) == {'cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2'})
    check('thesis_imu is IMU-anchor', style.MODELS['thesis_imu'][0] == 'IMU-anchor')
    check('model colours are distinct', len({c for _, c in style.MODELS.values()}) == 5)
    check('frame_times are midpoints',
          np.allclose(style.frame_times(0.02, 3), [0.01, 0.03, 0.05]))
    check('to_deg_s converts pi rad/s', np.isclose(style.to_deg_s(np.pi), 180.0))
    check('ref_label omega_gt', 'omega' in style.ref_label('omega_gt'))
    check('ref_label groundtruth', 'Vicon' in style.ref_label('groundtruth'))
    check('ref_label imu', 'IMU' in style.ref_label('imu'))
    check('omega_label y', style.omega_label(1) == r'$\omega_y$ (°/s)')
    with style.use_style('paper'):
        inside = matplotlib.rcParams['font.size']
    check('use_style is scoped', inside == 8 and matplotlib.rcParams['font.size'] != 8)
    check('pdf only under report/figures',
          style._wants_pdf(os.path.join('report', 'figures', 'x.png'))
          and not style._wants_pdf(os.path.join('experiments', 'x.png')))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'sub', 'x.png')
        fig, _ = plt.subplots()
        style.save(fig, path)
        check('save writes png and closes', os.path.exists(path)
              and not os.path.exists(path[:-4] + '.pdf')
              and not plt.fignum_exists(fig.number))


if __name__ == '__main__':
    test_style()
    print('all checks passed' if not FAILURES else f"FAILED: {', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: `ImportError: cannot import name 'style' from 'interacting_maps'`

- [ ] **Step 3: Write `interacting_maps/style.py`**

```python
"""
Plot conventions shared by every figure: model names and colours, ω-axis
colours, reference/IMU line styles, units, time axis, label text, size presets
and saving. Nothing here draws; viz.py does. Change a convention here and every
figure follows.
"""

import os
import numpy as np
import matplotlib

MODELS = {
    'cook':           ('Cook',        '#E69F00'),
    'thesis':         ('Thesis',      '#D55E00'),
    'thesis_imu':     ('IMU-anchor',  '#009E73'),
    'thesis_cmax':    ('CMax-anchor', '#0072B2'),
    'thesis_cmax_v2': ('CMax-inloop', '#CC79A7'),
}
OMEGA_AXIS_COLORS = ('#4e79a7', '#f28e2b', '#e15759')
REF_STYLE = dict(color='k', lw=1.2, ls='-')
IMU_STYLE = dict(color='0.55', lw=0.9, ls=':', label='IMU gyro (input)')
LINE_COLOR = '0.25'

LABEL_TIME = 'time (s)'
LABEL_OMEGA = r'$\omega$ (°/s)'
LABEL_ERROR = 'error (°/s)'
LABEL_DIR_ERROR = 'direction error (°)'
LABEL_BETA = r'scale $\beta$'
MAP_TITLES = {'V': r'events $V$', 'I': r'intensity $I$', 'aps': 'APS (log)',
              'G': r'gradient $G$', 'F': r'flow $F$', 'IWE': 'IWE'}
_REF_NAMES = {'omega_gt': r'$\omega_{gt}$', 'groundtruth': 'Vicon', 'imu': 'IMU'}

PAPER_WIDTH, PAPER_WIDE, SCREEN_WIDTH = 3.4, 7.05, 12.0
_BASE = {'axes.grid': True, 'grid.alpha': 0.3, 'axes.spines.top': False,
         'axes.spines.right': False, 'legend.frameon': False,
         'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'}
_PRESETS = {
    'paper':  {'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
               'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7},
    'screen': {'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
               'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9},
}
_REPORT_FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'report', 'figures')


def omega_label(axis):
    return rf'$\omega_{"xyz"[axis]}$ (°/s)'


def ref_label(ref_src):
    return f'reference ({_REF_NAMES[ref_src]})'


def to_deg_s(omega_rad_s):
    return np.degrees(omega_rad_s)


def frame_times(dt, n):
    """Seconds since segment start at each frame's midpoint."""
    return (np.arange(n) + 0.5) * dt


def use_style(preset):
    return matplotlib.rc_context({**_BASE, **_PRESETS[preset]})


def _wants_pdf(path):
    folder = os.path.normcase(os.path.dirname(os.path.abspath(path)))
    return folder == os.path.normcase(_REPORT_FIGURES)


def save(fig, path, show=False):
    """PNG at 200 dpi (+ PDF for report/figures/), then close or show."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    if _wants_pdf(path):
        fig.savefig(os.path.splitext(path)[0] + '.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: every line `[ok ]`, last line `all checks passed`.

- [ ] **Step 5: Amend the spec**

In `docs/superpowers/specs/2026-10-11-unified-plotting-design.md`:
- Decisions table, row "V": replace `` `RdBu_r` (red +, blue −, white 0) `` with `` `bwr` (red +, blue −, exactly white at 0) ``.
- Renderer table, row `render_V`: replace `` `RdBu_r` `` with `` `bwr` ``.
- In the `style.py` section, replace the `use_style` bullet with: ``- `use_style('paper' | 'screen')` returns a context manager (`with use_style('screen'):`) applying rcParams only inside the block. Both presets: …`` (keep the rest of the bullet).
- Replace the `save(fig, path)` bullet with: ``- `save(fig, path, show=False)`: PNG at 200 dpi, `bbox_inches='tight'`; additionally a PDF when `path` is in `report/figures/`; then closes the figure, or calls `plt.show()` when `show=True` (notebooks).``
- After the "Degenerate input" paragraph add: `Percentile scales fall back to the maximum when the 99th percentile is ~0 (maps with under 1 % non-zero pixels).`
- In Migration step 1 append: `` `plot_contrast_curve(out_dir, t_start)` gains `t_start` for its seconds axis. ``

- [ ] **Step 6: Commit**

```bash
git add interacting_maps/style.py tests/test_viz.py docs/superpowers/specs/2026-10-11-unified-plotting-design.md
git commit -m "Add style.py: one place for plot names, colours, units, presets and saving"
```

---

### Task 2: `viz.py` — map renderers, `net_maps`, `draw_map`

**Files:**
- Modify: `interacting_maps/viz.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Consumes: `style.MAP_TITLES`
- Produces:
  - `render_V(V, scale=None) -> ndarray (H,W,3)` — `scale` is a positive float
  - `render_I(I, scale=None) -> ndarray (H,W,3)` — `scale` is `(lo, hi)`
  - `render_vec(v, scale=None) -> ndarray (H,W,3)` — `v` is (H,W,2); `scale` positive float
  - `render_IWE(iwe, scale=None) -> ndarray (H,W,3)`
  - `vec_key(n=65) -> ndarray (n,n,3)`
  - `net_maps(net, H, W) -> {'I': (H,W), 'G': (H,W,2), 'F': (H,W,2)}`
  - `draw_map(ax, kind, arr, title=None, scale=None) -> None`, `kind` in `'V','I','aps','G','F','IWE'`; `title=None` → default from `MAP_TITLES`, `title=''` → no title

- [ ] **Step 1: Write the failing tests**

In `tests/test_viz.py`, add after `from interacting_maps import style`:

```python
from types import SimpleNamespace
from interacting_maps import viz
```

Add this function above `if __name__ == '__main__':`

```python
def test_maps():
    print('viz.py maps')
    white = np.ones(3)
    check('render_vec: zero vector is white',
          np.allclose(viz.render_vec(np.zeros((4, 4, 2)))[0, 0], white))
    plus_x = viz.render_vec(np.array([[[1.0, 0.0]]]), scale=1.0)[0, 0]
    key = viz.vec_key(65)
    check('render_vec: +x matches right-edge centre of key', np.allclose(plus_x, key[32, -1]))
    check('vec_key: centre is white', np.allclose(key[32, 32], white))
    check('render_vec: weak vector is pale (white base)',
          viz.render_vec(np.array([[[0.1, 0.0]]]), scale=1.0)[0, 0].min() > 0.85)
    V = np.zeros((10, 10)); V[0, 0], V[0, 1] = 1.0, -1.0
    rv = viz.render_V(V)
    check('render_V: zero is white', rv[5, 5].min() > 0.99)
    check('render_V: positive is red', rv[0, 0, 0] > 0.9 and rv[0, 0, 2] < 0.1)
    check('render_V: all-zero is white', np.allclose(viz.render_V(np.zeros((3, 3))), 1.0))
    check('render_I: constant is mid-grey',
          np.allclose(viz.render_I(np.full((5, 5), 3.0)), 0.5))
    check('render_I: explicit scale', np.allclose(viz.render_I(np.array([[2.0]]), scale=(0.0, 4.0)), 0.5))
    check('render_IWE: all-zero is black', np.allclose(viz.render_IWE(np.zeros((3, 3))), 0.0))
    sparse = np.zeros((100, 100)); sparse[0, 0] = 5.0
    check('render_IWE: sparse map falls back to max', viz.render_IWE(sparse)[0, 0].min() > 0.99)
    H, W = 6, 8
    cook = SimpleNamespace(I=np.zeros((H + 1, W + 1)), G=np.zeros((H, W, 2)), F=np.zeros((H, W, 2)))
    thesis = SimpleNamespace(I=np.zeros((H, W)), G=np.zeros((H, W, 2)), F=np.zeros((H, W, 2)))
    check('net_maps crops Cook and thesis alike',
          all(viz.net_maps(n, H, W)['I'].shape == (H, W) for n in (cook, thesis)))
    fig, axes = plt.subplots(1, 6)
    arrs = {'V': V, 'I': np.random.rand(10, 10), 'aps': np.random.rand(10, 10, 3),
            'G': np.random.rand(10, 10, 2) - 0.5, 'F': np.random.rand(10, 10, 2) - 0.5,
            'IWE': np.random.rand(10, 10)}
    for ax, (kind, a) in zip(axes, arrs.items()):
        viz.draw_map(ax, kind, a)
    check('draw_map: default titles', [a.get_title() for a in axes] == [style.MAP_TITLES[k] for k in arrs])
    check('draw_map: key inset only for G and F',
          [len(a.child_axes) for a in axes] == [0, 0, 0, 1, 1, 0])
    plt.close(fig)
```

Change the main block to:

```python
if __name__ == '__main__':
    test_style()
    test_maps()
    print('all checks passed' if not FAILURES else f"FAILED: {', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: `AttributeError: module 'interacting_maps.viz' has no attribute 'render_vec'`

- [ ] **Step 3: Implement**

In `interacting_maps/viz.py`, replace the module docstring (lines 1–9) with:

```python
"""
Drawing helpers. Every convention (names, colours, units, labels) comes from
style.py; nothing here defines its own.

Maps: renderers turn an array into an RGB (H, W, 3) array in [0, 1] and never
touch figures; draw_map puts one on an axis. Line plots: draw_omega,
draw_error and draw_series fill one axis each, time in seconds.
"""
```

Add after `import numpy as np`:

```python
from interacting_maps.style import MAP_TITLES
```

Append to the end of `interacting_maps/viz.py`:

```python
def _robust_max(a, pct=99.0):
    hi = float(np.percentile(a, pct))
    return hi if hi > 1e-12 else float(a.max())


def _gray(g):
    return np.repeat(g[..., None], 3, axis=-1)


def render_V(V, scale=None):
    """bwr: red positive, blue negative, white zero; symmetric ±scale (default p99 |V|)."""
    from matplotlib import colormaps
    s = _robust_max(np.abs(V)) if scale is None else scale
    if s <= 1e-12:
        return np.ones(V.shape + (3,))
    return colormaps['bwr'](0.5 + 0.5 * np.clip(V / s, -1.0, 1.0))[..., :3]


def render_I(I, scale=None):
    """Grey; 1-99 percentile by default, or an explicit (lo, hi)."""
    if scale is None:
        return _gray(normalise_robust(I))
    lo, hi = scale
    if hi - lo < 1e-9:
        return _gray(np.full(I.shape, 0.5))
    return _gray(np.clip((I - lo) / (hi - lo), 0.0, 1.0))


def render_vec(v, scale=None):
    """Hue = direction (image coords, y down), saturation = |v|/scale, white base."""
    from matplotlib.colors import hsv_to_rgb
    mag = np.hypot(v[..., 0], v[..., 1])
    s = _robust_max(mag) if scale is None else scale
    if s <= 1e-12:
        return np.ones(v.shape[:2] + (3,))
    hue = (np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi)) % 1.0
    sat = np.clip(mag / s, 0.0, 1.0)
    return hsv_to_rgb(np.stack([hue, sat, np.ones_like(hue)], axis=-1))


def render_IWE(iwe, scale=None):
    """Grey, white = many events; |IWE| / scale (default p99)."""
    a = np.abs(iwe)
    s = _robust_max(a) if scale is None else scale
    if s <= 1e-12:
        return np.zeros(iwe.shape + (3,))
    return _gray(np.clip(a / s, 0.0, 1.0))


def vec_key(n=65):
    """Square direction key, centre white; coloured by render_vec itself."""
    u = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(u, u)
    return render_vec(np.stack([X, Y], axis=-1), scale=1.0)


def _render_aps(aps, scale=None):
    a = np.asarray(aps, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    return render_I(np.log1p(a), scale)


_RENDERERS = {'V': render_V, 'I': render_I, 'aps': _render_aps,
              'G': render_vec, 'F': render_vec, 'IWE': render_IWE}


def net_maps(net, H, W):
    """I, G, F cropped to the H x W pixel grid (Cook's I is (H+1) x (W+1))."""
    return {'I': net.I[:H, :W], 'G': net.G[:H, :W], 'F': net.F[:H, :W]}


def draw_map(ax, kind, arr, title=None, scale=None):
    ax.imshow(_RENDERERS[kind](arr, scale=scale), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(MAP_TITLES[kind] if title is None else title)
    if kind in ('G', 'F'):
        key = ax.inset_axes([0.02, 0.02, 0.2, 0.2])
        key.imshow(vec_key(), interpolation='nearest')
        key.set_axis_off()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: every line `[ok ]`, `all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add interacting_maps/viz.py tests/test_viz.py
git commit -m "viz: map renderers, square direction key, net_maps crop, draw_map"
```

---

### Task 3: `viz.py` — line-plot drawers

**Files:**
- Modify: `interacting_maps/viz.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Consumes: `style.MODELS, OMEGA_AXIS_COLORS, REF_STYLE, IMU_STYLE, LINE_COLOR, LABEL_TIME, LABEL_ERROR, omega_label, ref_label, to_deg_s`
- Produces:
  - `draw_omega(ax, t, est, ref=None, imu=None, axis=0, model=None, ref_src=None, label=None) -> None` — `est`, `ref`, `imu` are 1-D arrays of one ω component in **rad/s**; plotted in deg/s. `ref_src` is required when `ref` is given. `imu` is skipped when `ref_src == 'imu'`.
  - `draw_error(ax, t, err, model=None, label=None) -> None` — `err` already in deg/s
  - `draw_series(ax, t, y, ylabel, model=None, label=None) -> None`
  - All three set xlabel `LABEL_TIME`; in stacked figures the caller calls `ax.label_outer()`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_viz.py`, add above the main block:

```python
def test_lines():
    print('viz.py line plots')
    t = style.frame_times(0.02, 4)
    est = np.full(4, np.pi)
    fig, ax = plt.subplots()
    viz.draw_omega(ax, t, est, ref=est, imu=est, axis=2, ref_src='groundtruth')
    lines = ax.get_lines()
    check('draw_omega: ref, imu, estimate', len(lines) == 3)
    check('draw_omega: plotted in deg/s', np.allclose(lines[-1].get_ydata(), 180.0))
    check('draw_omega: x is seconds at midpoints', np.allclose(lines[-1].get_xdata(), t))
    check('draw_omega: axis colour when no model',
          lines[-1].get_color() == style.OMEGA_AXIS_COLORS[2])
    check('draw_omega: labels', ax.get_ylabel() == style.omega_label(2)
          and ax.get_xlabel() == style.LABEL_TIME)
    plt.close(fig)
    fig, ax = plt.subplots()
    viz.draw_omega(ax, t, est, ref=est, imu=est, axis=0, model='thesis_cmax', ref_src='imu')
    lines = ax.get_lines()
    check('draw_omega: IMU skipped when it is the reference', len(lines) == 2)
    check('draw_omega: model colour and name',
          lines[-1].get_color() == style.MODELS['thesis_cmax'][1]
          and lines[-1].get_label() == 'CMax-anchor')
    plt.close(fig)
    fig, ax = plt.subplots()
    viz.draw_error(ax, t, np.ones(4))
    viz.draw_series(ax, t, np.ones(4), 'contrast')
    check('draw_error/draw_series: one line each', len(ax.get_lines()) == 2)
    plt.close(fig)
```

Change the main block to call `test_style()`, `test_maps()`, `test_lines()`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: `AttributeError: module 'interacting_maps.viz' has no attribute 'draw_omega'`

- [ ] **Step 3: Implement**

In `interacting_maps/viz.py`, replace the line `from interacting_maps.style import MAP_TITLES` with:

```python
from interacting_maps.style import (MAP_TITLES, MODELS, OMEGA_AXIS_COLORS, REF_STYLE,
                                    IMU_STYLE, LINE_COLOR, LABEL_TIME, LABEL_ERROR,
                                    omega_label, ref_label, to_deg_s)
```

Append to the end of the file:

```python
def _line(model, label):
    if model is None:
        return LINE_COLOR, label
    name, color = MODELS[model]
    return color, label or name


def draw_omega(ax, t, est, ref=None, imu=None, axis=0, model=None, ref_src=None, label=None):
    """One ω component (inputs rad/s, drawn in deg/s) against time in seconds."""
    if ref is not None:
        ax.plot(t, to_deg_s(ref), label=ref_label(ref_src), **REF_STYLE)
    if imu is not None and ref_src != 'imu':
        ax.plot(t, to_deg_s(imu), **IMU_STYLE)
    if model is None:
        color, label = OMEGA_AXIS_COLORS[axis], label or 'estimate'
    else:
        color, label = _line(model, label)
    ax.plot(t, to_deg_s(est), color=color, lw=1.2, label=label)
    ax.set_ylabel(omega_label(axis))
    ax.set_xlabel(LABEL_TIME)


def draw_error(ax, t, err, model=None, label=None):
    color, label = _line(model, label)
    ax.plot(t, err, color=color, lw=1.0, label=label)
    ax.set_ylabel(LABEL_ERROR)
    ax.set_xlabel(LABEL_TIME)


def draw_series(ax, t, y, ylabel, model=None, label=None):
    color, label = _line(model, label)
    ax.plot(t, y, color=color, lw=1.0, marker='.', ms=3, label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(LABEL_TIME)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: every line `[ok ]`, `all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add interacting_maps/viz.py tests/test_viz.py
git commit -m "viz: draw_omega, draw_error, draw_series with seconds and deg/s"
```

---

### Task 4: `iwe_io.py` on the shared conventions

**Files:**
- Modify: `interacting_maps/iwe_io.py`
- Modify: `evaluation.py:385` (the `plot_contrast_curve` call)
- Modify: `tests/test_viz.py`

**Interfaces:**
- Consumes: `viz.render_IWE`, `viz.draw_series`, `style.use_style`, `style.save`, `style.SCREEN_WIDTH`
- Produces: `plot_contrast_curve(out_dir: str, t_start: float) -> None` (was `plot_contrast_curve(out_dir)`); `save_iwe` and `make_gif` signatures unchanged.

- [ ] **Step 1: Write the failing test**

In `tests/test_viz.py`, add above the main block:

```python
def test_iwe_io():
    print('iwe_io.py')
    from interacting_maps import iwe_io
    with tempfile.TemporaryDirectory() as d:
        iwe = np.zeros((6, 8)); iwe[2, 3] = 4.0
        for k in range(3):
            iwe_io.save_iwe(iwe, np.zeros(3), 1.0 + k, d, k, t_mid=10.0 + (k + 0.5) * 0.02)
        img = plt.imread(os.path.join(d, 'frame_0000.png'))
        check('save_iwe: event pixel white, background black',
              img[2, 3, :3].min() > 0.99 and img[0, 0, :3].max() < 0.01)
        iwe_io.plot_contrast_curve(d, t_start=10.0)
        check('contrast curve written', os.path.exists(os.path.join(d, 'contrast_curve.png')))
```

Add `test_iwe_io()` to the main block.

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: `TypeError: plot_contrast_curve() got an unexpected keyword argument 't_start'`

- [ ] **Step 3: Implement**

In `interacting_maps/iwe_io.py`:

Replace lines 8–9 of the docstring (`Brightness: per-frame normalisation …` / `sequence brightness is a future option.`) with:

```
Rendering follows viz.render_IWE (grey, |IWE| at the 99th percentile per frame).
```

Replace the import block (lines 12–18) with:

```python
import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

from interacting_maps.style import SCREEN_WIDTH, use_style, save
from interacting_maps.viz import render_IWE, draw_series
```

Delete `_to_uint8` (lines 21–27) and in `save_iwe` replace

```python
    plt.imsave(os.path.join(out_dir, f'frame_{frame_idx:04d}.png'),
               _to_uint8(iwe), cmap='gray')
```

with

```python
    plt.imsave(os.path.join(out_dir, f'frame_{frame_idx:04d}.png'), render_IWE(iwe))
```

Replace the whole `plot_contrast_curve` function with:

```python
def plot_contrast_curve(out_dir, t_start):
    """IWE contrast against seconds since segment start -> contrast_curve.png."""
    log_path = os.path.join(out_dir, 'iwe_log.csv')
    if not os.path.exists(log_path):
        return
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    t = np.array([float(r['t_mid']) for r in rows]) - t_start
    contrast = np.array([float(r['contrast']) for r in rows])
    with use_style('screen'):
        fig, ax = plt.subplots(figsize=(SCREEN_WIDTH, 4))
        draw_series(ax, t, contrast, 'IWE contrast (variance)')
        ax.set_title('final-IWE contrast per frame (higher = sharper focus)')
        save(fig, os.path.join(out_dir, 'contrast_curve.png'))
```

In `evaluation.py`, replace

```python
        plot_contrast_curve(iwe_dir)
```

with

```python
        plot_contrast_curve(iwe_dir, rc.t_start)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. python tests/test_viz.py`
Expected: every line `[ok ]`, `all checks passed`.

- [ ] **Step 5: Commit**

```bash
git add interacting_maps/iwe_io.py evaluation.py tests/test_viz.py
git commit -m "iwe_io: render_IWE frames, contrast curve over seconds, no forced backend"
```

---

### Task 5: `evaluation.py` — tracking plot

**Files:**
- Modify: `evaluation.py:82` (imports), `evaluation.py:458-497` (`_plot_tracking`)

**Interfaces:**
- Consumes: `viz.draw_omega`, `viz.draw_error`; `style.MODELS, SCREEN_WIDTH, frame_times, ref_label, use_style, save`
- Produces: `_plot_tracking(rows, rc)` unchanged signature; writes `{rc.output_dir}/tracking_plot.png`.

- [ ] **Step 1: Replace the viz import**

In `evaluation.py`, replace line 82

```python
from interacting_maps.viz import normalise, normalise_robust, flow_to_rgb, grad_to_rgb
```

with

```python
from interacting_maps.viz import draw_map, draw_omega, draw_error, net_maps
from interacting_maps.style import (MODELS, OMEGA_AXIS_COLORS, REF_STYLE, LABEL_OMEGA,
                                    LABEL_ERROR, MAP_TITLES, SCREEN_WIDTH, frame_times,
                                    to_deg_s, ref_label, use_style, save)
```

- [ ] **Step 2: Replace `_plot_tracking`**

Replace the whole function (from `def _plot_tracking(rows, rc: RunConfig):` through `plt.close()`) with:

```python
def _plot_tracking(rows, rc: RunConfig):
    """ω components and error against seconds since segment start."""
    t = frame_times(rc.frame_duration, len(rows))
    col = lambda p: np.array([[r[f'{p}_wx'], r[f'{p}_wy'], r[f'{p}_wz']] for r in rows])
    est, ref, imu = col('est'), col('gt'), col('imu')
    ref_src = rows[0]['ref_source']

    with use_style('screen'):
        fig, axes = plt.subplots(4, 1, figsize=(SCREEN_WIDTH, 10), sharex=True)
        fig.suptitle(f'ω tracking — {MODELS[rc.model][0]}   [{rc.dataset} {rc.segment_id}, '
                     f't_start {rc.t_start:.3f} s]\n'
                     f'dt {rc.frame_duration*1000:.0f} ms, {rc.n_iters} iters, '
                     f'{rc.duration_s:.1f} s, scored vs {ref_label(ref_src)}')
        for i in range(3):
            draw_omega(axes[i], t, est[:, i], ref=ref[:, i], imu=imu[:, i], axis=i,
                       ref_src=ref_src)
            axes[i].legend(loc='upper right')
        err = np.array([r['err_deg_s'] for r in rows])
        draw_error(axes[3], t, err)
        axes[3].set_title(f'error: mean {err.mean():.1f} °/s, median {np.median(err):.1f} °/s')
        for ax in axes:
            ax.label_outer()
        save(fig, os.path.join(rc.output_dir, 'tracking_plot.png'))
```

- [ ] **Step 3: Run the golden regression (numbers must not change)**

Run: `PYTHONPATH=. python tests/test_golden_tracking.py`
Expected: `OK: all N models match` (plot code runs for every model; metrics unchanged).

- [ ] **Step 4: Render one tracking plot for visual review**

Run:

```bash
IM_MLFLOW=0 PYTHONPATH=. python -c "
import evaluation as E
rc = E.RunConfig(dataset='poster_rotation', model='thesis_imu', segment='seg_C', n_frames=25, out_root='experiments/plotting_check')
E.experiment_tracking(rc, save_frames=False)
print(rc.output_dir)"
```

Expected: prints the run directory. Open `<that dir>/tracking_plot.png`: x axis "time (s)" from ~0.01 to 0.49 s, y axes in °/s, black reference labelled "reference (Vicon)", grey dotted IMU line, blue/orange/red estimate per axis. Show it to the user before continuing.

- [ ] **Step 5: Commit**

```bash
git add evaluation.py
git commit -m "evaluation: tracking plot in deg/s over seconds, correct reference label"
```

---

### Task 6: `evaluation.py` — per-frame video frame

**Files:**
- Modify: `evaluation.py:716-815` (`_save_3col_frame`)

**Interfaces:**
- Consumes: `viz.draw_map`, `viz.net_maps`; `style.MODELS, SCREEN_WIDTH, to_deg_s, use_style, save`; `metrics._corr_to_aps` (already imported); `interacting_maps.network_dissertation.solve_poisson_exact`
- Produces: `_save_3col_frame(out_dir, k, V, net, H, W, gt_images, rc)` unchanged signature; writes `{out_dir}/frame_{k:04d}.png`.

- [ ] **Step 1: Replace `_save_3col_frame`**

Replace the whole function (from `def _save_3col_frame(` through `plt.close(fig)`) with:

```python
def _save_3col_frame(out_dir, k, V, net, H, W, gt_images, rc: RunConfig):
    """
    One inspection frame per time step: V | in-loop I | I read out from G by
    the exact solve (what recon_r scores) | G | F | nearest APS frame.

    The in-loop I keeps a display range smoothed across frames, so a map that
    loses contrast shows washed out instead of being re-stretched; its std and
    correlation with APS are in the title.
    """
    from interacting_maps.network_dissertation import solve_poisson_exact
    maps = net_maps(net, H, W)
    I, G = maps['I'], maps['G']
    t_rel = (k + 0.5) * rc.frame_duration
    t_mid = rc.t_start + t_rel

    lo, hi = np.percentile(I, [1.0, 99.0])
    prev = getattr(rc, '_I_disp_range', None)
    if prev is None or k == 0:
        rng = (lo, hi)
    else:
        a = 0.05                      # slow: ~20 frames to follow a real change
        rng = (prev[0] + a * (lo - prev[0]), prev[1] + a * (hi - prev[1]))
    rc._I_disp_range = rng
    I_ro = solve_poisson_exact(G, 'fft')
    fmt_r = lambda r: f'   r = {r:+.2f}' if np.isfinite(r) else ''
    r_in = _corr_to_aps(np.asarray(I, dtype=np.float64), gt_images, t_mid)
    r_ro = _corr_to_aps(I_ro, gt_images, t_mid)
    w = to_deg_s(np.linalg.norm(net.R) / rc.frame_duration)

    with use_style('screen'):
        fig, axes = plt.subplots(1, 6, figsize=(SCREEN_WIDTH, 2.6))
        draw_map(axes[0], 'V', V)
        draw_map(axes[1], 'I', I, scale=rng, title=f'I in-loop   std {I.std():.3f}{fmt_r(r_in)}')
        draw_map(axes[2], 'I', I_ro, title=f'I from G (exact){fmt_r(r_ro)}')
        draw_map(axes[3], 'G', G)
        draw_map(axes[4], 'F', maps['F'], title=f'flow $F$   |ω| {w:.0f} °/s')
        if gt_images:
            t_aps, path = min(gt_images, key=lambda x: abs(x[0] - t_mid))
            draw_map(axes[5], 'aps', plt.imread(path),
                     title=f'APS (log)   Δt {1000 * abs(t_aps - t_mid):.0f} ms')
        else:
            draw_map(axes[5], 'aps', np.zeros((H, W)), title='APS (n/a)')
        anchor = f'   δ_anchor {rc.params["delta_IMU"]:.2f}' if rc.use_imu else ''
        fig.suptitle(f'{MODELS[rc.model][0]}   frame {k:04d}   t {t_rel:.3f} s '
                     f'(segment start {rc.t_start:.3f} s)   δ_FR {rc.params["delta_FR"]:.2f}'
                     f'{anchor}   dt {1000 * rc.frame_duration:.0f} ms   {rc.distortion_mode}')
        save(fig, os.path.join(out_dir, f'frame_{k:04d}.png'))
```

- [ ] **Step 2: Fix the stale "3-col" docstrings**

In `evaluation.py`:
- line 21: replace `4. Qualitative video (tracking with 3-column frames: Events | I | GT APS)` with `4. Qualitative video (tracking with per-frame panels: V | I | I read-out | G | F | APS)`
- in `experiment_tracking`'s docstring, replace `- video_frames/frame_XXXX.png (3-col: Events | I | GT)` with `- video_frames/frame_XXXX.png (V | I | I read-out | G | F | APS)`

- [ ] **Step 3: Run the golden regression**

Run: `PYTHONPATH=. python tests/test_golden_tracking.py`
Expected: `OK: all N models match`.

- [ ] **Step 4: Render video frames for visual review**

Run:

```bash
IM_MLFLOW=0 PYTHONPATH=. python -c "
import evaluation as E
for m in ('thesis_imu', 'cook'):
    rc = E.RunConfig(dataset='poster_rotation', model=m, segment='seg_C', n_frames=5, out_root='experiments/plotting_check')
    E.experiment_tracking(rc, save_frames=True)
    print(rc.output_dir)"
```

Expected: two run directories. Open `video_frames/frame_0004.png` in each: V white background with red/blue events; G and F on white with the square key bottom-left; APS in log grey; Cook frame panels all the same size (crop works). Show both to the user.

- [ ] **Step 5: Commit**

```bash
git add evaluation.py
git commit -m "evaluation: video frame from draw_map (shared encodings, square key, log APS)"
```

---

### Task 7: `evaluation.py` — Exp 1, Exp 3, Exp 6 figures

**Files:**
- Modify: `evaluation.py:145-152` and `:162-169` (Exp 1 snapshots), `:171-208` (Exp 1 figure), `:683-704` (Exp 3 figure), `:955-970` (Exp 6 figure)

**Interfaces:**
- Consumes: `viz.draw_map`, `viz.net_maps`; `style.MODELS, OMEGA_AXIS_COLORS, REF_STYLE, LABEL_OMEGA, LABEL_ERROR, MAP_TITLES, SCREEN_WIDTH, to_deg_s, ref_label, use_style, save`
- Produces: same output files `exp1_convergence.png`, `exp3_parameters.png`, `exp6_basin.png`.

- [ ] **Step 1: Crop Exp 1 snapshots in both branches**

In `experiment_single_frame_convergence`, replace the thesis-branch snapshot

```python
                snapshots[it] = {
                    'I': net.I.copy(), 'G': net.G.copy(),
                    'F': net.F.copy(), 'R': net.R.copy(),
                }
```

and the Cook-branch snapshot

```python
                snapshots[it] = {
                    'I': net.I[:H, :W].copy(), 'G': net.G.copy(),
                    'F': net.F.copy(), 'R': net.R.copy(),
                }
```

each with

```python
                snapshots[it] = {**{n: m.copy() for n, m in net_maps(net, H, W).items()},
                                 'R': net.R.copy()}
```

- [ ] **Step 2: Replace the Exp 1 figure**

Replace from `    # --- Save plot ---` through `    print(f"  Saved: {rc.output_dir}/exp1_convergence.png")` with:

```python
    # --- Save plot ---
    rc.save_params()
    n_snaps = len(snapshot_iters)
    path = os.path.join(rc.output_dir, 'exp1_convergence.png')
    with use_style('screen'):
        fig, axes = plt.subplots(5, n_snaps, figsize=(SCREEN_WIDTH, 11))
        fig.suptitle(f'Exp 1: map convergence — {MODELS[rc.model][0]}   '
                     f'[{rc.dataset} {rc.segment_id}, t_start {rc.t_start:.3f} s, frame {frame_idx}]')
        for col, it in enumerate(snapshot_iters):
            snap = snapshots[it]
            draw_map(axes[0, col], 'V', V, title=f'iter {it}')
            for row, kind in enumerate(('I', 'G', 'F'), start=1):
                draw_map(axes[row, col], kind, snap[kind], title='')
            err = to_deg_s(np.linalg.norm(snap['R'] - gt_R) / rc.frame_duration)
            ax = axes[4, col]
            ax.bar(range(3), to_deg_s(snap['R'] / rc.frame_duration), color=OMEGA_AXIS_COLORS)
            ax.bar(range(3), to_deg_s(gt_omega), color='none', edgecolor=REF_STYLE['color'],
                   linewidth=1.5, label=ref_label('imu'))
            ax.set_xticks(range(3), [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'])
            ax.set_title(f'error {err:.1f} °/s')
        for row, kind in enumerate(('V', 'I', 'G', 'F')):
            axes[row, 0].set_ylabel(MAP_TITLES[kind])
        axes[4, 0].set_ylabel(LABEL_OMEGA)
        axes[4, 0].legend()
        save(fig, path)
    print(f"  Saved: {path}")
```

- [ ] **Step 3: Replace the Exp 3 figure**

Replace from `    # Plot` (after the frame-duration sweep loop) through `    plt.close()` with:

```python
    # Plot
    color = MODELS[rc.model][1]
    with use_style('screen'):
        fig, axes = plt.subplots(1, 2, figsize=(SCREEN_WIDTH, 4.5))
        fig.suptitle(f'Exp 3: parameter influence — {MODELS[rc.model][0]}   '
                     f'[{rc.dataset} {rc.segment_id}, t_start {rc.t_start:.3f} s, '
                     f'reference = IMU]')
        axes[0].plot([r['iters'] for r in iter_results],
                     [r['err'] for r in iter_results], 'o-', color=color)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('iterations per frame')
        axes[0].set_title(f'(a) iterations (dt {rc.frame_duration*1000:.0f} ms)')
        axes[1].plot([r['dt_ms'] for r in dt_results],
                     [r['err'] for r in dt_results], 's-', color=color)
        axes[1].set_xlabel('frame duration (ms)')
        axes[1].set_title(f'(b) frame duration ({rc.n_iters} iters)')
        for ax in axes:
            ax.set_ylabel(LABEL_ERROR)
        save(fig, os.path.join(rc.output_dir, 'exp3_parameters.png'))
```

- [ ] **Step 4: Replace the Exp 6 figure**

Replace from `    # Plot` (after the perturbation loop) through `    plt.close()` with:

```python
    # Plot: distances as angular-velocity error in deg/s; filled = converged
    color = MODELS[rc.model][1]
    to_w = lambda d: to_deg_s(d / rc.frame_duration)
    with use_style('screen'):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results:
            x, y = to_w(r['init_dist']), to_w(r['final_dist'])
            ax.scatter(x, y, s=80, zorder=5, color=color if r['converged'] else 'none',
                       edgecolors=color, linewidths=1.5)
            ax.annotate(r['label'], (x, y), fontsize=8)
        max_d = to_w(max(r['init_dist'] for r in results)) * 1.1
        ax.plot([0, max_d], [0, max_d], ls='--', color='0.5', label='no improvement')
        ax.set_xlabel(r'initial $\|\omega - \omega_{ref}\|$ (°/s)')
        ax.set_ylabel(r'final $\|\omega - \omega_{ref}\|$ (°/s)')
        ax.set_title(f'Exp 6: basin of attraction — {MODELS[rc.model][0]}   '
                     f'[{rc.dataset} {rc.segment_id}, t_start {rc.t_start:.3f} s, '
                     f'reference = IMU; filled = converged]')
        ax.legend()
        save(fig, os.path.join(rc.output_dir, 'exp6_basin.png'))
```

- [ ] **Step 5: Render all three for visual review**

Run:

```bash
IM_MLFLOW=0 PYTHONPATH=. python -c "
import evaluation as E
rc = E.RunConfig(dataset='poster_rotation', model='thesis_imu', segment='seg_C', n_iters=20, out_root='experiments/plotting_check')
E.experiment_single_frame_convergence(rc, max_iters=20)
E.experiment_parameter_influence(rc)
E.experiment_basin_of_attraction(rc)
print(rc.output_dir)"
```

Expected: `exp1_convergence.png`, `exp3_parameters.png`, `exp6_basin.png` in the printed directory; all errors in °/s; Exp 1 rows V/I/G/F with keys on G and F. Show them to the user.

- [ ] **Step 6: Run all checks**

Run: `PYTHONPATH=. python tests/test_viz.py && PYTHONPATH=. python tests/test_golden_tracking.py`
Expected: `all checks passed` and `OK: all N models match`.

- [ ] **Step 7: Commit**

```bash
git add evaluation.py
git commit -m "evaluation: Exp 1/3/6 figures on shared conventions (deg/s, model colour, cropped maps)"
```
