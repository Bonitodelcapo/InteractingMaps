"""Qt inspection app: up to 10 runs of one segment, linked to their curves.

    python inspect_app.py [--root experiments/inspect] [--data DIR] [run dirs ...]

InspectApp holds the state (loaded RunCaches, frame, iteration, probe, which
runs show maps / curves, scale mode) and the panels; every panel only reads that
state and swaps data in place, so stepping is fast. Spec:
docs/superpowers/specs/2026-11-26-inspect-app-qt-design.md
"""
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

pg.setConfigOptions(imageAxisOrder='row-major', antialias=False)

MAX_RUNS = 10
MAPS_SHOWN = 5                       # runs with map rows right after loading
SCALE_MODES = ('fixed', 'shared', 'per frame')
DEFAULT_CURVES = ('err_deg_s', 'beta')
MAP_KINDS = ('I', 'G', 'F')


def qt_app():
    """The one QApplication (created on first use)."""
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class InspectApp(QtWidgets.QMainWindow):
    def __init__(self, root='experiments/inspect', data_root=None):
        super().__init__()
        from .runs_panel import RunsPanel
        from .view_panel import ViewPanel
        from .export_panel import ExportPanel
        from .map_area import MapArea
        from .curve_area import CurveArea
        from .side_plots import SidePlots
        self.setWindowTitle('Inspection')
        self.caches, self.frame, self.iteration, self.probe = [], 0, 0, None
        self.maps_for, self.curves_for = set(), set()
        self.scale_mode, self.map_kinds, self.curve_names = 'fixed', MAP_KINDS, DEFAULT_CURVES

        self.map_area, self.curve_area, self.side_plots = MapArea(self), CurveArea(self), SidePlots(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        for w in (self.map_area.widget, self.curve_area.widget, self.side_plots.widget):
            split.addWidget(w)
        split.setSizes([500, 250, 200])
        self.setCentralWidget(split)

        self.runs_panel = RunsPanel(self, root, data_root)
        self.view_panel, self.export_panel = ViewPanel(self), ExportPanel(self)
        self.tabs = QtWidgets.QTabWidget()
        for w, title in ((self.runs_panel.widget, 'Runs'), (self.view_panel.widget, 'View'),
                         (self.export_panel.widget, 'Export')):
            self.tabs.addTab(w, title)
        dock = QtWidgets.QDockWidget('Controls')
        dock.setWidget(self.tabs)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.resize(1500, 950)

    # ------------------------------------------------------------ runs
    def set_runs(self, caches):
        from interacting_maps.inspect_labels import run_labels, run_colors
        cfgs = [{**c.config, '_dir': c.dir} for c in caches]
        for c, lab, col in zip(caches, run_labels(cfgs), run_colors(cfgs)):
            c.display = c.run.display = lab
            c.color = c.run.color = col
        self.caches = list(caches)
        n = len(self.caches)
        self.frame, self.iteration, self.probe = 0, 0, None
        self.maps_for = set(range(min(MAPS_SHOWN, n)))
        self.curves_for = set(range(n))
        names = set(self.caches[0].curves) if n else set()
        self.curve_names = tuple(c for c in DEFAULT_CURVES if c in names)
        self.map_area.reset()
        for part in (self.map_area, self.curve_area, self.side_plots):
            part.rebuild()
        self.view_panel.sync_runs()
        self.export_panel.sync_runs()
        self.runs_panel.fill_info()
        self.refresh()
        self.map_area.prefill()

    # ------------------------------------------------------------ state
    def has_replay(self, k=None):
        k = self.frame if k is None else k
        return any(c.has_replay(k) for c in self.caches)

    def current_iteration(self):
        return self.iteration if (self.iteration and self.has_replay()) else None

    def set_frame(self, k):
        if not self.caches:
            return
        self.frame = int(max(0, min(int(k), self.caches[0].n_frames - 1)))
        self.refresh()

    def set_iteration(self, it):
        self.iteration = int(it)
        self.refresh()

    def set_probe(self, x, y):
        self.probe = (int(x), int(y))
        self.refresh()

    def clear_probe(self):
        self.probe = None
        self.refresh()

    def set_maps_for(self, idx):
        self.maps_for = set(idx)
        self.map_area.rebuild()
        self.refresh()
        self.map_area.prefill()

    def set_curves_for(self, idx):
        self.curves_for = set(idx)
        self.curve_area.apply_visibility()
        self.refresh()

    def set_scale_mode(self, mode):
        assert mode in SCALE_MODES, mode
        self.scale_mode = mode
        self.refresh()
        self.map_area.prefill()

    def set_map_kinds(self, kinds):
        self.map_kinds = tuple(k for k in MAP_KINDS if k in kinds)
        self.map_area.rebuild()
        self.refresh()

    def set_curve_names(self, names):
        self.curve_names = tuple(names)
        self.curve_area.rebuild()
        self.refresh()

    def refresh(self):
        if not self.caches:
            return
        self.map_area.update()
        self.curve_area.update()
        self.side_plots.update()
        self.view_panel.sync()
