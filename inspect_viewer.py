"""Inspection viewer: 1-5 runs of one segment, linked to their curves.

In a notebook (inspect_viewer.ipynb):

    %matplotlib widget
    from inspect_viewer import Inspector
    Inspector(['experiments/inspect/poster_rotation/cook/<run>',
               'experiments/inspect/poster_rotation/thesis_imu/<run>']).show()

Frame slider, iteration slider (on when a compared run has a replay of that
frame; 0 = the frame's saved state), curve and map dropdowns, a shared-scale
switch, and click on any curve to jump to that frame. Export writes the same
figure over frames, or over the iterations of the current frame, as GIF/MP4.
The runs come from `python report/run_experiments.py --what inspect` (states)
and `python replay_frame.py` (iterations); nothing is computed here.
"""
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt

from interacting_maps import states_io
from interacting_maps.inspect_io import load_run, check_comparable
from interacting_maps.inspect_fig import (MAP_KINDS, fixed_scales, draw_inspector,
                                          export_animation, frame_from_click)

DEFAULT_CURVES = ('err_deg_s', 'beta')


class Inspector:
    def __init__(self, run_dirs, data_root=None):
        self.runs = [load_run(d, data_root) for d in run_dirs]
        check_comparable(self.runs)
        r0 = self.runs[0]
        names = sorted(set().union(*(r.curves() for r in self.runs)) - {'frame', 'time'})
        self.frame = widgets.IntSlider(0, 0, r0.n_frames - 1, description='frame')
        self.iteration = widgets.IntSlider(0, 0, max(r.n_iters for r in self.runs),
                                           description='iteration')
        self.curves = widgets.SelectMultiple(
            options=names + ['omega_fullrate'], description='curves',
            value=tuple(c for c in DEFAULT_CURVES if c in names))
        self.panels = widgets.SelectMultiple(options=MAP_KINDS, value=MAP_KINDS,
                                             description='maps')
        self.share = widgets.Checkbox(False, description='share scale across models')
        self.path = widgets.Text('experiments/inspect/inspect.gif', description='export to')
        self.btn_frames = widgets.Button(description='export frames')
        self.btn_iters = widgets.Button(description='export iterations')
        self.msg = widgets.Output()
        self._scales = {}
        self.fig = plt.figure(figsize=(12, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        for w in (self.frame, self.iteration, self.curves, self.panels, self.share):
            w.observe(lambda _change: self.redraw(), names='value')
        self.btn_frames.on_click(lambda _b: self._export_msg(over_iterations=False))
        self.btn_iters.on_click(lambda _b: self._export_msg(over_iterations=True))
        self.redraw()

    # ------------------------------------------------------------ drawing
    def _kinds(self):
        return [p for p in self.panels.value if p in MAP_KINDS]

    def scales(self):
        key = (tuple(self._kinds()), self.share.value)
        if key not in self._scales:
            self._scales[key] = fixed_scales(self.runs, list(key[0]) + ['V'], share=key[1])
        return self._scales[key]

    def _has_replay(self, k):
        return any(os.path.exists(states_io.replay_path(r.dir, k)) for r in self.runs)

    def draw_kw(self):
        return dict(panels=tuple(self._kinds()), curves=tuple(self.curves.value),
                    share_scale=self.share.value, scales=self.scales())

    def redraw(self):
        k = self.frame.value
        self.iteration.disabled = not self._has_replay(k)
        it = None if self.iteration.disabled else (self.iteration.value or None)
        draw_inspector(self.runs, k, it=it, fig=self.fig, **self.draw_kw())
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or not getattr(ax, '_inspect_curve', None):
            return
        self.frame.value = frame_from_click(event.xdata, self.runs[0].times)

    # ------------------------------------------------------------ export
    def export(self, path=None, over_iterations=False):
        path = path or self.path.value
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if over_iterations:
            return export_animation(self.runs, path, it_frame=self.frame.value, **self.draw_kw())
        return export_animation(self.runs, path, **self.draw_kw())

    def _export_msg(self, over_iterations):
        with self.msg:
            print(f'-> {self.export(over_iterations=over_iterations)}')

    def show(self):
        from IPython.display import display
        controls = widgets.VBox([
            widgets.HBox([self.frame, self.iteration]),
            widgets.HBox([self.curves, self.panels, self.share]),
            widgets.HBox([self.path, self.btn_frames, self.btn_iters]), self.msg])
        display(controls)
        display(self.fig.canvas)
