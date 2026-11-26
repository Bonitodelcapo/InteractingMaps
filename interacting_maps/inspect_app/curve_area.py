"""Curve plots: one per chosen curve, one line per loaded run, a shared time
axis and a draggable frame marker (drag it, or click a plot, to change frame).
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets


class CurveArea:
    def __init__(self, app):
        self.app = app
        self.layout = pg.GraphicsLayoutWidget()
        self.widget = self.layout
        self.plots, self.lines, self.markers = {}, {}, {}
        self._moving = False

    def rebuild(self):
        app = self.app
        self.layout.clear()
        self.plots, self.lines, self.markers = {}, {}, {}
        first = None
        for row, name in enumerate(app.curve_names):
            p = self.layout.addPlot(row=row, col=0)
            p.setLabel('left', name)
            p.showGrid(x=True, y=True, alpha=0.2)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if name == 'omega_fullrate':
                fr = app.caches[0].fullrate if app.caches else {}
                if fr:
                    self.lines[(name, 'ref')] = p.plot(fr['t_gt'], fr['gt'], pen=pg.mkPen('k'))
                    self.lines[(name, 'imu')] = p.plot(fr['t_imu'], fr['imu'],
                                                       pen=pg.mkPen((140, 140, 140), style=pg.QtCore.Qt.DotLine))
            else:
                for i, c in enumerate(app.caches):
                    y = c.curves.get(name)
                    if y is not None:
                        self.lines[(name, i)] = p.plot(c.times, np.asarray(y, float),
                                                       pen=pg.mkPen(c.color, width=1.5), name=c.display)
            m = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('k', style=pg.QtCore.Qt.DashLine))
            m.sigPositionChanged.connect(self._dragged)
            m.sigPositionChangeFinished.connect(self._dragged)
            p.addItem(m)
            p.scene().sigMouseClicked.connect(lambda ev, p=p: self._clicked(ev, p))
            self.markers[name] = m
            self.plots[name] = p
        p_last = self.plots[app.curve_names[-1]] if app.curve_names else None
        if p_last is not None:
            p_last.setLabel('bottom', 'time (s)')
        self.apply_visibility()

    def apply_visibility(self):
        for (name, i), line in self.lines.items():
            if isinstance(i, int):
                line.setVisible(i in self.app.curves_for)

    def _nearest(self, x):
        return int(np.argmin(np.abs(self.app.caches[0].times - x)))

    def _dragged(self, m):
        if not self._moving and self.app.caches:
            self.app.set_frame(self._nearest(m.value()))

    def _clicked(self, ev, p):
        if not self.app.caches or not p.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.app.set_frame(self._nearest(p.vb.mapSceneToView(ev.scenePos()).x()))

    def update(self):
        t = self.app.caches[0].times[self.app.frame]
        self._moving = True
        for m in self.markers.values():
            m.setValue(t)
        self._moving = False
