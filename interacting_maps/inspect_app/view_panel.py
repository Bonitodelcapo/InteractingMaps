"""View tab: play / step / frame and iteration sliders, which runs show maps and
curves, which curves and map kinds, scale mode, spike navigation, probe."""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from interacting_maps.inspect_nav import worst_frame, next_spike
from interacting_maps.inspect_registry import curve_names
from . import SCALE_MODES, MAP_KINDS


def _checklist(items, checked, colors=None):
    w = QtWidgets.QListWidget()
    for i, text in enumerate(items):
        it = QtWidgets.QListWidgetItem(text)
        it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
        it.setCheckState(QtCore.Qt.Checked if i in checked else QtCore.Qt.Unchecked)
        if colors:
            it.setForeground(QtGui.QColor(colors[i]))
        w.addItem(it)
    return w


class ViewPanel:
    def __init__(self, app):
        self.app = app
        w = self.widget = QtWidgets.QWidget()
        self.form = f = QtWidgets.QFormLayout(w)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.play = QtWidgets.QPushButton('▷ play')
        self.play.setCheckable(True)
        self.play.toggled.connect(self._play)
        self.ms = QtWidgets.QSpinBox()
        self.ms.setRange(20, 2000)
        self.ms.setValue(80)
        self.ms.setSuffix(' ms/frame')
        prev, nxt = QtWidgets.QPushButton('◀'), QtWidgets.QPushButton('▶')
        row = QtWidgets.QHBoxLayout()
        for x in (self.play, prev, nxt, self.ms):
            row.addWidget(x)
        f.addRow(row)
        self.frame = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iteration = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_label, self.it_label = QtWidgets.QLabel(), QtWidgets.QLabel()
        f.addRow(self.frame_label, self.frame)
        f.addRow(self.it_label, self.iteration)
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(SCALE_MODES)
        f.addRow('scale', self.scale)
        self.kinds = [QtWidgets.QCheckBox(k) for k in MAP_KINDS]
        row = QtWidgets.QHBoxLayout()
        for cb in self.kinds:
            cb.setChecked(True)
            row.addWidget(cb)
        f.addRow('maps', row)
        self.maps_for = self.curves_for = self.curves = None
        self.spike_curve = QtWidgets.QComboBox()
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(-1e9, 1e9)
        self.threshold.setDecimals(3)
        f.addRow('spikes in', self.spike_curve)
        f.addRow('above', self.threshold)
        row = QtWidgets.QHBoxLayout()
        b_prev, b_worst, b_next = (QtWidgets.QPushButton(t) for t in ('◀ spike', 'worst frame', 'spike ▶'))
        for b in (b_prev, b_worst, b_next):
            row.addWidget(b)
        f.addRow(row)
        clear = QtWidgets.QPushButton('clear pixel probe')
        f.addRow(clear)
        self._lists_row = f.rowCount()

        prev.clicked.connect(lambda: app.set_frame(app.frame - 1))
        nxt.clicked.connect(lambda: app.set_frame(app.frame + 1))
        self.frame.valueChanged.connect(lambda v: self._quiet or app.set_frame(v))
        self.iteration.valueChanged.connect(lambda v: self._quiet or app.set_iteration(v))
        self.scale.currentTextChanged.connect(lambda m: self._quiet or app.set_scale_mode(m))
        for cb in self.kinds:
            cb.toggled.connect(lambda _x: self._quiet or app.set_map_kinds(
                [c.text() for c in self.kinds if c.isChecked()]))
        b_prev.clicked.connect(lambda: self.goto_spike(-1))
        b_next.clicked.connect(lambda: self.goto_spike(+1))
        b_worst.clicked.connect(self.goto_worst)
        clear.clicked.connect(app.clear_probe)
        self._quiet = False

    # ------------------------------------------------------------ runs loaded
    def sync_runs(self):
        app, f = self.app, self.form
        while f.rowCount() > self._lists_row:
            f.removeRow(self._lists_row)
        labels = [c.display for c in app.caches]
        colors = [c.color for c in app.caches]
        self.maps_for = _checklist(labels, app.maps_for, colors)
        self.curves_for = _checklist(labels, app.curves_for, colors)
        names = curve_names(app.caches[0]) if app.caches else []
        self.curves = _checklist(names, {i for i, n in enumerate(names) if n in app.curve_names})
        f.addRow('maps for', self.maps_for)
        f.addRow('curves for', self.curves_for)
        f.addRow('curves', self.curves)
        self.maps_for.itemChanged.connect(lambda _i: app.set_maps_for(self._checked(self.maps_for)))
        self.curves_for.itemChanged.connect(lambda _i: app.set_curves_for(self._checked(self.curves_for)))
        self.curves.itemChanged.connect(lambda _i: app.set_curve_names(
            [self.curves.item(i).text() for i in sorted(self._checked(self.curves))]))
        self._quiet = True
        self.spike_curve.clear()
        self.spike_curve.addItems([n for n in names if n in app.caches[0].curves])
        if 'err_deg_s' in names:
            self.spike_curve.setCurrentText('err_deg_s')
        n = app.caches[0].n_frames if app.caches else 1
        self.frame.setRange(0, n - 1)
        self.iteration.setRange(0, max((c.n_iters for c in app.caches), default=0))
        self._quiet = False

    @staticmethod
    def _checked(lst) -> set:
        return {i for i in range(lst.count()) if lst.item(i).checkState() == QtCore.Qt.Checked}

    def sync(self):
        app = self.app
        self._quiet = True
        self.frame.setValue(app.frame)
        self.iteration.setEnabled(app.has_replay())
        self.iteration.setValue(app.iteration)
        self.scale.setCurrentText(app.scale_mode)
        t = app.caches[0].times[app.frame]
        self.frame_label.setText(f'frame {app.frame}  ({t:.3f} s)')
        self.it_label.setText('iteration ' + (str(app.iteration) if app.current_iteration() else '—'))
        self._quiet = False

    # ------------------------------------------------------------ play, spikes
    def _play(self, on):
        self.play.setText('❚❚ pause' if on else '▷ play')
        self.timer.start(self.ms.value()) if on else self.timer.stop()

    def tick(self):
        app = self.app
        if app.caches:
            app.set_frame((app.frame + 1) % app.caches[0].n_frames)

    def _signal(self):
        app, name = self.app, self.spike_curve.currentText()
        ys = [np.asarray(app.caches[i].curves[name], float) for i in sorted(app.curves_for)
              if name in app.caches[i].curves]
        if not ys:
            return None
        with np.errstate(all='ignore'):
            return np.nanmax(np.vstack(ys), axis=0)

    def goto_worst(self):
        y = self._signal()
        if y is not None:
            self.app.set_frame(worst_frame(y))

    def goto_spike(self, direction):
        y = self._signal()
        if y is not None:
            k = next_spike(y, self.app.frame, self.threshold.value(), direction)
            if k is not None:
                self.app.set_frame(k)
