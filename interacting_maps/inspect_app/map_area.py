"""Map rows (one per run in "maps for") and the raw row (events V, APS).

Images are created when the rows change and afterwards only receive new RGB
data. Colours come from viz.render_* at the scale mode's scale; rendered RGB is
kept per (run, kind, frame, iteration, mode), so a frame seen once is instant.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from interacting_maps.viz import render_V, render_I, render_vec, _render_aps
from interacting_maps.inspect_fig import _pool

_RENDER = {'I': render_I, 'G': render_vec, 'F': render_vec, 'V': render_V}


class _Image(pg.ImageItem):
    def __init__(self, on_click):
        super().__init__()
        self._on_click = on_click

    def mouseClickEvent(self, ev):
        p = ev.pos()
        self._on_click(p.x(), p.y())


def _rgb(x):
    return (np.asarray(x) * 255).astype(np.uint8)


class MapArea:
    def __init__(self, app):
        self.app = app
        self.layout = pg.GraphicsLayoutWidget()
        self.widget = QtWidgets.QScrollArea()
        self.widget.setWidgetResizable(True)
        self.widget.setWidget(self.layout)
        self.images, self.raw_items, self.views, self.markers = {}, {}, [], []
        self.run_rows = []
        self.reset()

    def reset(self):
        self._rgb, self._scales = {}, {}
        self._generation = getattr(self, '_generation', 0) + 1     # stops a running prefill

    def prefill(self, blocking=False):
        """Render every frame of the shown runs at the current scale mode ahead of
        time (background thread), so even the first pass through the frames is fast."""
        import threading
        app, gen = self.app, self._generation
        jobs = [(i, kind, k) for k in range(app.caches[0].n_frames) if app.caches
                for i in sorted(app.maps_for) for kind in app.map_kinds]

        def work():
            for i, kind, k in jobs:
                if gen != self._generation or i >= len(app.caches):
                    return
                self._image(i, kind, k, None)
        if blocking:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ scales
    def scales(self) -> list:
        """One {kind: scale} per loaded run for the current mode ({} = per frame)."""
        app, mode = self.app, self.app.scale_mode
        if mode == 'per frame':
            return [{} for _ in app.caches]
        if mode not in self._scales:
            kinds = ('I', 'G', 'F', 'V')
            per_run = [{k: _pool(k, c.frame_scales.get(k, [])) for k in kinds} for c in app.caches]
            if mode == 'shared':
                pooled = {k: _pool(k, [s for c in app.caches for s in c.frame_scales.get(k, [])])
                          for k in kinds}
                per_run = [dict(pooled) for _ in app.caches]
            self._scales[mode] = per_run
        return self._scales[mode]

    # ------------------------------------------------------------ build
    def _view(self, row, col, title):
        vb = self.layout.addViewBox(row=row, col=col, lockAspect=True, enableMenu=False)
        vb.invertY(True)
        if self.views:
            vb.setXLink(self.views[0])
            vb.setYLink(self.views[0])
        self.views.append(vb)
        self.layout.addLabel(title, row=row + 1, col=col, size='8pt')
        mk = pg.ScatterPlotItem(size=12, symbol='+', pen=pg.mkPen('k', width=2), brush=None)
        vb.addItem(mk)
        self.markers.append(mk)
        return vb

    def rebuild(self):
        app = self.app
        self.layout.clear()
        self.images, self.raw_items, self.views, self.markers = {}, {}, [], []
        self.run_rows = sorted(i for i in app.maps_for if i < len(app.caches))
        row = 0
        for i in self.run_rows:
            c = app.caches[i]
            self.layout.addLabel(c.display, row=row, col=0, angle=-90, color=c.color)
            for j, kind in enumerate(app.map_kinds):
                img = _Image(self.on_image_click)
                self._view(row, j + 1, kind).addItem(img)
                self.images[(i, kind)] = img
            row += 2
        if app.caches:
            self.layout.addLabel('raw', row=row, col=0, angle=-90)
            for j, kind in enumerate(('V', 'aps')):
                img = _Image(self.on_image_click)
                self._view(row, j + 1, 'events V' if kind == 'V' else 'APS').addItem(img)
                self.raw_items[kind] = img
            row += 2
        self.layout.setMinimumHeight(max(300, 110 * row))

    # ------------------------------------------------------------ update
    def _image(self, i, kind, k, it):
        key = (i, kind, k, it, self.app.scale_mode)
        if key not in self._rgb:
            c = self.app.caches[i]
            sc = self.scales()[i].get(kind)
            if kind == 'V':
                data = None if c.V is None else c.V[k]
                self._rgb[key] = None if data is None else _rgb(render_V(data, scale=sc))
            elif kind == 'aps':
                data = c.aps_at(k)
                self._rgb[key] = None if data is None else _rgb(_render_aps(data))
            else:
                self._rgb[key] = _rgb(_RENDER[kind](c.map_at(kind, k, it), scale=sc))
        return self._rgb[key]

    def update(self):
        app = self.app
        k, it = app.frame, app.current_iteration()
        for (i, kind), img in self.images.items():
            img.setImage(self._image(i, kind, k, it if app.caches[i].has_replay(k) else None),
                         autoLevels=False, levels=(0, 255))
        for kind, img in self.raw_items.items():
            rgb = self._image(0, kind, k, None)
            if rgb is None:
                img.clear()
            else:
                img.setImage(rgb, autoLevels=False, levels=(0, 255))
        pts = [] if app.probe is None else [{'pos': app.probe}]
        for mk in self.markers:
            mk.setData(pts)

    def on_image_click(self, x, y):
        c = self.app.caches[0]
        self.app.set_probe(int(np.clip(round(x), 0, c.W - 1)), int(np.clip(round(y), 0, c.H - 1)))
