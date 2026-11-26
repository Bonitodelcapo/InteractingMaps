"""Pixel probe (I, |G|, |F| over frames, and over the iterations of a replayed
frame) and residuals per iteration, for the runs in "curves for"."""
import numpy as np
import pyqtgraph as pg

PROBE = ('I', '|G|', '|F|')
RESIDUALS = ('|V+F·G|', '|G−∇I|', '|R|')
TITLES = {'|R|': '|ω| (°/s)', '|V+F·G|': 'mean |V+F·G|', '|G−∇I|': 'mean |G−∇I|'}


class SidePlots:
    def __init__(self, app):
        self.app = app
        self.layout = pg.GraphicsLayoutWidget()
        self.widget = self.layout
        self.plots, self.markers = {}, {}

    def rebuild(self):
        self.layout.clear()
        self.plots, self.markers = {}, {}
        for row, (prefix, keys, xlabel) in enumerate((('probe', PROBE, 'time (s)'),
                                                      ('probe_it', PROBE, 'iteration'),
                                                      ('res', RESIDUALS, 'iteration'))):
            for col, key in enumerate(keys):
                p = self.layout.addPlot(row=row, col=col)
                p.setTitle(TITLES.get(key, key) + (' at pixel' if prefix != 'res' else ''), size='8pt')
                p.setLabel('bottom', xlabel)
                m = pg.InfiniteLine(angle=90, pen=pg.mkPen('k', style=pg.QtCore.Qt.DashLine))
                p.addItem(m, ignoreBounds=True)
                self.plots[f'{prefix}:{key}'] = p
                self.markers[f'{prefix}:{key}'] = m

    def _fill(self, prefix, keys, series, x_of, marker_x):
        for key in keys:
            p = self.plots[f'{prefix}:{key}']
            for item in p.listDataItems():
                p.removeItem(item)
            for c, s in series:
                if s is not None:
                    p.plot(x_of(c, s[key]), np.asarray(s[key], float), pen=pg.mkPen(c.color, width=1.5))
            self.markers[f'{prefix}:{key}'].setValue(marker_x)
            p.setVisible(True)

    def _hide(self, prefix, keys):
        for key in keys:
            self.plots[f'{prefix}:{key}'].setVisible(False)

    def update(self):
        app = self.app
        if not self.plots:
            return
        runs = [app.caches[i] for i in sorted(app.curves_for) if i < len(app.caches)]
        k, it = app.frame, app.current_iteration() or 0
        iters = lambda c, y: np.arange(1, len(y) + 1)
        if app.probe is None:
            self._hide('probe', PROBE)
            self._hide('probe_it', PROBE)
        else:
            x, y = app.probe
            self._fill('probe', PROBE, [(c, c.pixel_series(x, y)) for c in runs],
                       lambda c, s: c.times, runs[0].times[k] if runs else 0)
            if app.has_replay(k):
                self._fill('probe_it', PROBE, [(c, c.pixel_iterations(k, x, y)) for c in runs], iters, it)
            else:
                self._hide('probe_it', PROBE)
        if app.has_replay(k):
            self._fill('res', RESIDUALS, [(c, c.iteration_residuals(k)) for c in runs], iters, it)
        else:
            self._hide('res', RESIDUALS)
