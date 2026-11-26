"""Export tab: GIF/MP4 over a frame range or over the iterations of the current
frame, PNG of the current frame (report style, via inspect_fig), and a
screenshot of the window exactly as shown."""
import os
import re

from PySide6 import QtWidgets

from interacting_maps.inspect_fig import draw_inspector, export_animation

MAX_NAME = 150


class ExportPanel:
    def __init__(self, app):
        self.app = app
        w = self.widget = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.frame_from, self.frame_to, self.fps = (QtWidgets.QSpinBox() for _ in range(3))
        self.fps.setRange(1, 60)
        self.fps.setValue(8)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.frame_from)
        row.addWidget(QtWidgets.QLabel('to'))
        row.addWidget(self.frame_to)
        f.addRow('frames', row)
        f.addRow('fps', self.fps)
        self.fmt = QtWidgets.QComboBox()
        self.fmt.addItems(['gif', 'mp4', 'png'])
        f.addRow('format', self.fmt)
        self.folder = QtWidgets.QLineEdit('experiments/inspect/exports')
        f.addRow('to folder', self.folder)
        self.msg = QtWidgets.QLabel('')
        self.msg.setWordWrap(True)
        for text, fn in (('export frames', self.export_frames),
                         ('export iterations', self.export_iterations),
                         ('save current frame (PNG)', self.save_current),
                         ('screenshot of the window', self.screenshot)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _c=False, fn=fn: self._say(fn))
            f.addRow(b)
        f.addRow(self.msg)

    def sync_runs(self):
        n = self.app.caches[0].n_frames if self.app.caches else 1
        for sb in (self.frame_from, self.frame_to):
            sb.setRange(0, n - 1)
        self.frame_from.setValue(0)
        self.frame_to.setValue(n - 1)

    def _say(self, fn):
        try:
            self.msg.setText(f'-> {fn()}')
        except Exception as e:                           # shown, not lost
            self.msg.setText(f'export failed: {type(e).__name__}: {e}')

    # ------------------------------------------------------------ helpers
    def _runs(self):
        app = self.app
        idx = sorted(app.maps_for) or sorted(app.curves_for) or [0]
        return idx, [app.caches[i].run for i in idx]

    def _draw_kw(self, idx):
        app = self.app
        return dict(panels=app.map_kinds,
                    curves=tuple(n for n in app.curve_names if n != 'omega_fullrate') or ('err_deg_s',),
                    scales=[app.map_area.scales()[i] for i in idx])

    def _name(self, tail, ext) -> str:
        app = self.app
        c = app.caches[0].config
        labels = '+'.join(app.caches[i].display for i in self._runs()[0])
        labels = re.sub(r'[^\w.=+-]+', '_', labels)
        head = f"{c['dataset']}_{c['segment_id']}_"
        room = MAX_NAME - len(head) - len(tail) - len(ext) - 2
        name = f"{head}{labels[:max(room, 0)]}_{tail}.{ext}"
        os.makedirs(self.folder.text(), exist_ok=True)
        return os.path.join(self.folder.text(), name)

    def _ext(self):
        return self.fmt.currentText() if self.fmt.currentText() != 'png' else 'gif'

    # ------------------------------------------------------------ exports
    def export_frames(self) -> str:
        a, b = self.frame_from.value(), self.frame_to.value()
        idx, runs = self._runs()
        return export_animation(runs, self._name(f'f{a:03d}-{b:03d}', self._ext()),
                                frames=range(a, b + 1), fps=self.fps.value(), **self._draw_kw(idx))

    def export_iterations(self) -> str:
        k = self.app.frame
        idx, runs = self._runs()
        return export_animation(runs, self._name(f'f{k:03d}_iterations', self._ext()),
                                it_frame=k, fps=self.fps.value(), **self._draw_kw(idx))

    def save_current(self) -> str:
        app = self.app
        idx, runs = self._runs()
        it = app.current_iteration()
        path = self._name(f'f{app.frame:03d}' + (f'_it{it}' if it else ''), 'png')
        draw_inspector(runs, app.frame, it=it, **self._draw_kw(idx)).savefig(path, dpi=120)
        return path

    def screenshot(self) -> str:
        path = self._name(f'f{self.app.frame:03d}_screenshot', 'png')
        self.app.grab().save(path)
        return path
