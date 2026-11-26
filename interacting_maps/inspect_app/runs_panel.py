"""Runs tab: scan a folder, pick dataset -> segment -> run folders (<= MAX_RUNS,
any models, filterable by model), load them in the background, run info table."""
import os
import threading

from PySide6 import QtCore, QtWidgets

import experiment_tracking
from interacting_maps.inspect_io import scan_runs, check_comparable
from interacting_maps.inspect_cache import RunCache
from . import MAX_RUNS


class _Signals(QtCore.QObject):
    progress = QtCore.Signal(float)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class RunsPanel:
    def __init__(self, app, root, data_root):
        self.app, self.catalog, self.last_error = app, [], None
        w = self.widget = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.root = QtWidgets.QLineEdit(root)
        scan = QtWidgets.QPushButton('scan')
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.root)
        row.addWidget(scan)
        f.addRow('runs in', row)
        self.data = QtWidgets.QLineEdit(data_root or '')
        self.data.setPlaceholderText('default: ./data')
        f.addRow('data in', self.data)
        self.ds, self.seg, self.model_filter = (QtWidgets.QComboBox() for _ in range(3))
        f.addRow('dataset', self.ds)
        f.addRow('segment', self.seg)
        f.addRow('model', self.model_filter)
        self.run_list = QtWidgets.QListWidget()
        f.addRow('runs', self.run_list)
        load = QtWidgets.QPushButton('load')
        self.progress = QtWidgets.QProgressBar()
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        f.addRow(load, self.progress)
        f.addRow(self.status)
        self.info = QtWidgets.QTableWidget(0, 8)
        self.info.setHorizontalHeaderLabels(['run', 'model', 'it', 'deltas', 'commit',
                                             'err °/s', 'β', 'replays'])
        f.addRow(self.info)
        scan.clicked.connect(self.scan)
        load.clicked.connect(lambda: self.load())
        self.ds.currentTextChanged.connect(self._fill_segments)
        self.seg.currentTextChanged.connect(self._fill_runs)
        self.model_filter.currentTextChanged.connect(self._filter)
        self.run_list.itemChanged.connect(self._limit)
        self.signals = _Signals()
        self.signals.progress.connect(lambda f: self.progress.setValue(int(100 * f)))
        self.signals.done.connect(self._loaded)
        self.signals.failed.connect(self._failed)

    # ------------------------------------------------------------ picking
    def scan(self):
        root = self.root.text()
        self.catalog = scan_runs(root) if os.path.isdir(root) else []
        self.ds.blockSignals(True)
        self.ds.clear()
        self.ds.addItems(sorted({e['dataset'] for e in self.catalog}))
        self.ds.blockSignals(False)
        self._fill_segments()
        self.status.setText(f'{len(self.catalog)} runs with saved states in {root}')

    def _fill_segments(self):
        self.seg.blockSignals(True)
        self.seg.clear()
        self.seg.addItems(sorted({e['segment'] for e in self.catalog
                                  if e['dataset'] == self.ds.currentText()}))
        self.seg.blockSignals(False)
        self._fill_runs()

    def _entries(self):
        return [e for e in self.catalog if e['dataset'] == self.ds.currentText()
                and e['segment'] == self.seg.currentText()]

    def _fill_runs(self):
        entries = self._entries()
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for e in entries:
            it = QtWidgets.QListWidgetItem(f"{os.path.basename(e['dir'])}   [{e['model']}]")
            it.setData(QtCore.Qt.UserRole, e['dir'])
            it.setData(QtCore.Qt.UserRole + 1, e['model'])
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)
            self.run_list.addItem(it)
        self.run_list.blockSignals(False)
        self.model_filter.blockSignals(True)
        self.model_filter.clear()
        self.model_filter.addItems(['all'] + sorted({e['model'] for e in entries}))
        self.model_filter.blockSignals(False)

    def _items(self):
        return [self.run_list.item(i) for i in range(self.run_list.count())]

    def _filter(self, model):
        for it in self._items():
            it.setHidden(model not in ('all', '') and it.data(QtCore.Qt.UserRole + 1) != model)

    def visible_runs(self) -> int:
        return sum(not it.isHidden() for it in self._items())

    def checked_dirs(self) -> list:
        return [it.data(QtCore.Qt.UserRole) for it in self._items()
                if it.checkState() == QtCore.Qt.Checked]

    def _limit(self, item):
        if item.checkState() == QtCore.Qt.Checked and len(self.checked_dirs()) > MAX_RUNS:
            self.run_list.blockSignals(True)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.run_list.blockSignals(False)
            self.status.setText(f'at most {MAX_RUNS} runs')

    def check_all(self):
        for it in self._items():
            if not it.isHidden() and len(self.checked_dirs()) < MAX_RUNS:
                it.setCheckState(QtCore.Qt.Checked)

    # ------------------------------------------------------------ loading
    def load(self, blocking=False):
        dirs = self.checked_dirs()
        if dirs:
            self.load_dirs(dirs, blocking)

    def load_dirs(self, dirs, blocking=False):
        data_root = self.data.text() or None
        self.last_error = None
        sig = self.signals

        def work():
            try:
                caches = []
                for i, d in enumerate(dirs):
                    caches.append(RunCache.open(
                        d, data_root, progress=lambda f, i=i: sig.progress.emit((i + f) / len(dirs))))
                check_comparable([c.run for c in caches], max_runs=MAX_RUNS)
                return caches, None
            except Exception as e:                       # report, keep the previous runs
                return None, f'{type(e).__name__}: {e}'

        if blocking:
            caches, err = work()
            self._failed(err) if err else self._loaded(caches)
            return
        self.status.setText(f'loading {len(dirs)} runs ...')

        def run():
            caches, err = work()
            sig.failed.emit(err) if err else sig.done.emit(caches)
        threading.Thread(target=run, daemon=True).start()

    def _loaded(self, caches):
        self.progress.setValue(100)
        self.status.setText(f'{len(caches)} runs loaded')
        self.app.set_runs(caches)
        self.app.tabs.setCurrentIndex(1)

    def _failed(self, err):
        self.last_error = err
        self.status.setText(f'load failed: {err}')

    # ------------------------------------------------------------ info
    def fill_info(self):
        caches = self.app.caches
        self.info.setRowCount(len(caches))
        for r, c in enumerate(caches):
            rec = experiment_tracking.read_run(c.dir)
            s, p = rec['summary'], rec['provenance']
            deltas = ', '.join(f"{k.replace('delta_', '')}={v:g}"
                               for k, v in sorted(c.config.get('params', {}).items()) if v)
            commit = p.get('git_commit_short', '?') + (' (dirty)' if p.get('git_dirty') else '')
            err, beta = s.get('mean_err_deg_s'), s.get('mean_beta')
            vals = [c.display, c.config['model'], str(c.n_iters), deltas, commit,
                    '' if err is None else f'{err:.1f}', '' if beta is None else f'{beta:.2f}',
                    ', '.join(map(str, sorted(c.replays))) or 'none']
            for col, v in enumerate(vals):
                self.info.setItem(r, col, QtWidgets.QTableWidgetItem(v))
        self.info.resizeColumnsToContents()
