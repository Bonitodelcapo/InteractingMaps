"""Inspection viewer: 1-5 runs of one segment, linked to their curves.

In a notebook (inspect_viewer.ipynb, guide: docs/inspection_viewer.md):

    %matplotlib widget
    from inspect_viewer import Inspector
    Inspector(root='experiments/inspect').show()      # pick runs in the Runs tab
    Inspector(['<run dir>', '<run dir>']).show()       # or open them directly

Tabs: Runs (scan a folder, dataset -> segment -> models, run info), View (play,
step, frame/iteration sliders, jump to the worst frame or the next spike, curve
and map choice, scale mode), Export (GIF/MP4 over a frame range or over the
iterations of the current frame, PNG of the current view). Click a curve to
jump to that frame; click a map pixel to plot its I, |G|, |F| over the frames
(and over the iterations of a replayed frame). Replayed frames also show the
residuals per iteration. Everything is read from files; nothing is computed
beyond these display quantities.
"""
import os

import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

import experiment_tracking
from interacting_maps.style import MODELS, use_style
from interacting_maps.inspect_io import load_run, check_comparable, scan_runs, MAX_RUNS
from interacting_maps.inspect_fig import (MAP_KINDS, fixed_scales, draw_inspector,
                                          export_animation, frame_from_click)
from interacting_maps.inspect_nav import (worst_frame, next_spike, pixel_series,
                                          pixel_iterations, iteration_residuals)
from interacting_maps import states_io

DEFAULT_CURVES = ('err_deg_s', 'beta')
SCALE_MODES = ('fixed', 'shared', 'per frame')
#: titles of the side plots (math text: the report font has no ∇ glyph)
AUX_TITLES = {'|V+F·G|': r'mean $|V+F\cdot G|$', '|G−∇I|': r'mean $|G-\nabla I|$',
              '|R|': r'$|\omega|$ (°/s)', '|G|': '$|G|$', '|F|': '$|F|$'}


def _set_options(dropdown, options):
    """Replace a dropdown's options and select the first one."""
    dropdown.options = tuple(options)
    dropdown.value = dropdown.options[0] if dropdown.options else None


class Inspector:
    def __init__(self, run_dirs=None, data_root=None, root='experiments/inspect'):
        self.runs, self.catalog, self.probe = [], [], None
        self._scales, self._hold = {}, False
        self._build_runs_tab(root, data_root)
        self._build_view_tab()
        self._build_export_tab()
        self.fig = plt.figure(figsize=(12, 9))
        self.aux = plt.figure(figsize=(12, 3))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.scan()
        if run_dirs:
            self.open(run_dirs)

    # ================================================================ Runs tab
    def _build_runs_tab(self, root, data_root):
        self.root = widgets.Text(root, description='runs in')
        self.data = widgets.Text(data_root or '', placeholder='default: ./data',
                                 description='data in')
        self.scan_btn = widgets.Button(description='scan')
        self.ds = widgets.Dropdown(description='dataset')
        self.seg = widgets.Dropdown(description='segment', layout={'width': '480px'})
        self.models = widgets.SelectMultiple(description='models', rows=5)
        self.load_btn = widgets.Button(description='load', button_style='primary')
        self.info = widgets.HTML()
        self.scan_btn.on_click(lambda _b: self.scan())
        self.load_btn.on_click(lambda _b: self.load())
        self.ds.observe(lambda _c: self._fill_segments(), names='value')
        self.seg.observe(lambda _c: self._fill_models(), names='value')

    def scan(self):
        self.catalog = scan_runs(self.root.value) if os.path.isdir(self.root.value) else []
        _set_options(self.ds, sorted({e['dataset'] for e in self.catalog}))
        self._fill_segments()

    def _fill_segments(self):
        _set_options(self.seg, sorted({e['segment'] for e in self.catalog
                                       if e['dataset'] == self.ds.value}))
        self._fill_models()

    def _picked(self):
        return [e for e in self.catalog
                if e['dataset'] == self.ds.value and e['segment'] == self.seg.value]

    def _fill_models(self):
        self.models.options = tuple(sorted({e['model'] for e in self._picked()}))

    def load(self):
        dirs = [e['dir'] for e in self._picked() if e['model'] in self.models.value]
        if dirs:
            self.open(dirs[:MAX_RUNS])

    def open(self, run_dirs):
        data_root = self.data.value or None
        runs = [load_run(d, data_root) for d in run_dirs]
        check_comparable(runs)
        self._hold = True
        self.runs, self._scales, self.probe = runs, {}, None
        n = runs[0].n_frames
        names = sorted(set().union(*(r.curves() for r in runs)) - {'frame', 'time'})
        self.curves.options = names + ['omega_fullrate']
        self.curves.value = tuple(c for c in DEFAULT_CURVES if c in names)
        self.spike_curve.options = names
        self.spike_curve.value = self.curves.value[0] if self.curves.value else names[0]
        for w in (self.frame, self.play):
            w.max, w.value = n - 1, 0
        self.iteration.max = max(r.n_iters for r in runs)
        self.frames_range.max = n - 1
        self.frames_range.value = (0, n - 1)
        self.info.value = self._info_html()
        self._hold = False
        self.redraw()

    def _info_html(self):
        rows = []
        for r in self.runs:
            rec = experiment_tracking.read_run(r.dir)
            s, p = rec['summary'], rec['provenance']
            rep = [k for k in range(r.n_frames)
                   if os.path.exists(states_io.replay_path(r.dir, k))]
            commit = p.get('git_commit_short', '?') + (' (dirty)' if p.get('git_dirty') else '')
            err = s.get('mean_err_deg_s')
            beta = s.get('mean_beta')
            deltas = ', '.join(f"{k.replace('delta_', '')}={v:g}"
                               for k, v in sorted(r.config.get('params', {}).items()) if v)
            rows.append(f"<tr><td><b>{MODELS.get(r.label, (r.label,))[0]}</b> ({r.label})</td>"
                        f"<td>{r.n_iters} it</td><td>{deltas}</td><td>{commit}</td>"
                        f"<td>{'' if err is None else f'{err:.1f} °/s'}</td>"
                        f"<td>{'' if beta is None else f'β {beta:.2f}'}</td>"
                        f"<td>replays: {rep or 'none'}</td></tr>")
        return "<table>" + ''.join(rows) + "</table>"

    # ================================================================ View tab
    def _build_view_tab(self):
        self.play = widgets.Play(interval=400, description='play')
        self.speed = widgets.IntSlider(400, 100, 2000, step=100, description='ms/frame')
        self.frame = widgets.IntSlider(0, 0, 0, description='frame')
        widgets.jslink((self.play, 'value'), (self.frame, 'value'))
        widgets.jslink((self.speed, 'value'), (self.play, 'interval'))
        self.prev_btn = widgets.Button(description='◀', layout={'width': '40px'})
        self.next_btn = widgets.Button(description='▶', layout={'width': '40px'})
        self.iteration = widgets.IntSlider(0, 0, 0, description='iteration')
        self.curves = widgets.SelectMultiple(description='curves', rows=6)
        self.panels = widgets.SelectMultiple(options=MAP_KINDS, value=MAP_KINDS,
                                             description='maps', rows=3)
        self.scale_mode = widgets.ToggleButtons(options=SCALE_MODES, description='scale')
        self.spike_curve = widgets.Dropdown(description='spikes in')
        self.threshold = widgets.FloatText(0.0, description='above')
        self.worst_btn = widgets.Button(description='worst frame')
        self.prev_spike_btn = widgets.Button(description='◀ spike')
        self.next_spike_btn = widgets.Button(description='spike ▶')
        self.clear_probe_btn = widgets.Button(description='clear pixel probe')
        self.prev_btn.on_click(lambda _b: self.step(-1))
        self.next_btn.on_click(lambda _b: self.step(+1))
        self.worst_btn.on_click(lambda _b: self.goto_worst())
        self.prev_spike_btn.on_click(lambda _b: self.goto_spike(-1))
        self.next_spike_btn.on_click(lambda _b: self.goto_spike(+1))
        self.clear_probe_btn.on_click(lambda _b: self.set_probe(None))
        for w in (self.frame, self.iteration, self.curves, self.panels, self.scale_mode):
            w.observe(lambda _c: self.redraw(), names='value')

    def step(self, n):
        self.frame.value = int(np.clip(self.frame.value + n, 0, self.frame.max))

    def _spike_signal(self):
        ys = [r.curves().get(self.spike_curve.value) for r in self.runs]
        ys = [y for y in ys if y is not None]
        with np.errstate(all='ignore'):
            return np.nanmax(np.vstack(ys), axis=0)

    def goto_worst(self):
        self.frame.value = worst_frame(self._spike_signal())

    def goto_spike(self, direction):
        k = next_spike(self._spike_signal(), self.frame.value, self.threshold.value, direction)
        if k is not None:
            self.frame.value = k

    def set_probe(self, xy):
        self.probe = xy
        self.redraw()

    # ================================================================ Export tab
    def _build_export_tab(self):
        self.frames_range = widgets.IntRangeSlider(value=(0, 0), min=0, max=0,
                                                   description='frames')
        self.fps = widgets.IntSlider(5, 1, 30, description='fps')
        self.fmt = widgets.Dropdown(options=('gif', 'mp4', 'png'), description='format')
        self.out_dir = widgets.Text('experiments/inspect/exports', description='to folder')
        self.exp_frames_btn = widgets.Button(description='export frames')
        self.exp_iters_btn = widgets.Button(description='export iterations')
        self.exp_png_btn = widgets.Button(description='save current view')
        self.msg = widgets.Output()
        self.exp_frames_btn.on_click(lambda _b: self._say(self.export_frames))
        self.exp_iters_btn.on_click(lambda _b: self._say(self.export_iterations))
        self.exp_png_btn.on_click(lambda _b: self._say(self.export_current))

    def _say(self, fn):
        with self.msg:
            try:
                print(f'-> {fn()}')
            except Exception as e:                      # shown in the panel, not lost
                print(f'export failed: {e!r}')

    def _name(self, tail, ext):
        c = self.runs[0].config
        models = '+'.join(r.label for r in self.runs)
        os.makedirs(self.out_dir.value, exist_ok=True)
        return os.path.join(self.out_dir.value,
                            f"{c['dataset']}_{c['segment_id']}_{models}_{tail}.{ext}")

    def export_frames(self):
        a, b = self.frames_range.value
        ext = self.fmt.value if self.fmt.value != 'png' else 'gif'
        return export_animation(self.runs, self._name(f'f{a:03d}-{b:03d}', ext),
                                frames=range(a, b + 1), fps=self.fps.value, **self.draw_kw())

    def export_iterations(self):
        k = self.frame.value
        ext = self.fmt.value if self.fmt.value != 'png' else 'gif'
        return export_animation(self.runs, self._name(f'f{k:03d}_iterations', ext),
                                it_frame=k, fps=self.fps.value, **self.draw_kw())

    def export_current(self):
        it = self._it()
        path = self._name(f'f{self.frame.value:03d}' + (f'_it{it}' if it else ''), 'png')
        self.fig.savefig(path, dpi=120)
        return path

    def export(self, path=None, over_iterations=False):
        """Back-compatible one-call export (default name when path is None)."""
        if path is None:
            return self.export_iterations() if over_iterations else self.export_frames()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if over_iterations:
            return export_animation(self.runs, path, it_frame=self.frame.value, **self.draw_kw())
        return export_animation(self.runs, path, **self.draw_kw())

    # ================================================================ drawing
    def _kinds(self):
        return [p for p in self.panels.value if p in MAP_KINDS]

    def scales(self):
        if self.scale_mode.value == 'per frame':
            return [{} for _ in self.runs]
        key = (tuple(self._kinds()), self.scale_mode.value)
        if key not in self._scales:
            self._scales[key] = fixed_scales(self.runs, list(key[0]) + ['V'],
                                             share=key[1] == 'shared')
        return self._scales[key]

    def _has_replay(self, k):
        return any(os.path.exists(states_io.replay_path(r.dir, k)) for r in self.runs)

    def _it(self):
        return None if self.iteration.disabled else (self.iteration.value or None)

    def draw_kw(self):
        return dict(panels=tuple(self._kinds()), curves=tuple(self.curves.value),
                    scales=self.scales())

    def redraw(self):
        if self._hold or not self.runs:
            return
        k = self.frame.value
        self.iteration.disabled = not self._has_replay(k)
        draw_inspector(self.runs, k, it=self._it(), fig=self.fig, probe=self.probe,
                       **self.draw_kw())
        self.fig.canvas.draw_idle()
        self.draw_aux()

    def draw_aux(self):
        """Pixel probe over frames (and iterations), residuals over iterations."""
        k, it = self.frame.value, self._it()
        rows = []
        if self.probe is not None:
            rows.append(('probe', [pixel_series(r, *self.probe) for r in self.runs]))
        res = [iteration_residuals(r, k) for r in self.runs]
        if any(x is not None for x in res):
            rows.append(('residuals', res))
            if self.probe is not None:
                rows.append(('probe_it', [pixel_iterations(r, k, *self.probe)
                                          for r in self.runs]))
        self.aux.clear()
        with use_style('screen'):
            if not rows:
                self.aux.text(0.5, 0.5, 'click a map pixel to probe it; replayed frames '
                              'also show the residuals per iteration', ha='center')
            else:
                self.aux.set_size_inches(12, 2.4 * len(rows))
                gs = self.aux.add_gridspec(len(rows), 3)
                for i, (kind, data) in enumerate(rows):
                    self._aux_row(gs, i, kind, data, k, it)
                self.aux.tight_layout()
        self.aux.canvas.draw_idle()

    def _aux_row(self, gs, i, kind, data, k, it):
        keys = list(next(d for d in data if d is not None))
        for j, key in enumerate(keys[:3]):
            ax = self.aux.add_subplot(gs[i, j])
            ax._inspect_aux = 'residuals' if kind == 'residuals' else 'probe'
            for r, d in zip(self.runs, data):
                if d is None:
                    continue
                x = r.times if kind == 'probe' else np.arange(1, len(d[key]) + 1)
                color = MODELS.get(r.label, (r.label, None))[1]
                ax.plot(x, d[key], color=color, lw=1, marker='.', ms=3, label=r.label)
            if kind == 'probe':
                ax.axvline(self.runs[0].times[k], color='k', lw=0.8, ls='--')
                ax.set_xlabel('time (s)')
            else:
                if it:
                    ax.axvline(it, color='k', lw=0.8, ls='--')
                ax.set_xlabel(f'iteration (frame {k})')
            where = f' at pixel {self.probe}' if kind != 'residuals' else ''
            ax.set_title(AUX_TITLES.get(key, key) + where)
        if i == 0:
            ax.legend(fontsize='small')

    def on_click(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        if getattr(ax, '_inspect_curve', None):
            self.frame.value = frame_from_click(event.xdata, self.runs[0].times)
        elif getattr(ax, '_inspect_role', None) == 'map' and event.ydata is not None:
            r = self.runs[0]
            x = int(np.clip(round(event.xdata), 0, r.W - 1))
            y = int(np.clip(round(event.ydata), 0, r.H - 1))
            self.set_probe((x, y))

    # ================================================================ layout
    def show(self):
        from IPython.display import display
        runs_tab = widgets.VBox([widgets.HBox([self.root, self.scan_btn]), self.data,
                                 self.ds, self.seg, widgets.HBox([self.models, self.load_btn]),
                                 self.info])
        view_tab = widgets.VBox([
            widgets.HBox([self.play, self.prev_btn, self.frame, self.next_btn, self.speed]),
            widgets.HBox([self.iteration, self.scale_mode]),
            widgets.HBox([self.spike_curve, self.threshold, self.prev_spike_btn,
                          self.worst_btn, self.next_spike_btn]),
            widgets.HBox([self.curves, self.panels, self.clear_probe_btn])])
        export_tab = widgets.VBox([self.frames_range, widgets.HBox([self.fps, self.fmt]),
                                   self.out_dir,
                                   widgets.HBox([self.exp_frames_btn, self.exp_iters_btn,
                                                 self.exp_png_btn]), self.msg])
        tabs = widgets.Tab(children=[runs_tab, view_tab, export_tab])
        for i, t in enumerate(('Runs', 'View', 'Export')):
            tabs.set_title(i, t)
        tabs.selected_index = 1 if self.runs else 0
        display(tabs)
        display(self.fig.canvas)
        display(self.aux.canvas)
