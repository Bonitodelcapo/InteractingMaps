"""The comparison app (ipywidgets): 1-20 run slots, each with cascading dropdowns
and its own output checkboxes, and the comparison grid, one band per output type.

    from interacting_maps.compare_runs import CompareApp
    CompareApp().show()
"""
import io
import os

import ipywidgets as wd
import matplotlib.pyplot as plt
from IPython.display import display

from . import bands, catalog, outputs

BANDS = ['curves', 'table', 'figures', 'maps', 'frames']
CELL_WIDTH = '24%'                       # figure / frame cells: four per row


def _fig_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return wd.Image(value=buf.getvalue(), format='png')


def _file_widget(path):
    ext = os.path.splitext(path)[1].lstrip('.').lower()
    with open(path, 'rb') as f:
        data = f.read()
    if ext == 'mp4':
        return wd.Video(value=data, format='mp4', autoplay=False, loop=False,
                        layout=wd.Layout(width='100%'))
    return wd.Image(value=data, format=ext, layout=wd.Layout(width='100%'))


def _head(text, color='black'):
    return wd.HTML(f'<b style="color:{color}">{text}</b>')


class RunSlot:
    """One run of the comparison: cascading dropdowns -> one folder, plus its outputs."""

    def __init__(self, app, slot):
        self.app, self.slot = app, slot
        self.color = bands.SLOT_COLORS[slot - 1]
        self._busy = False
        self.drop = {k: wd.Dropdown(description=k, options=[('any', 'any')], value='any',
                                    layout=wd.Layout(width='250px'))
                     for k in catalog.choice_keys(app.cat)}
        self.run = wd.Dropdown(description='run', layout=wd.Layout(width='760px'))
        self.check = {o.key: wd.Checkbox(description=o.label, indent=False,
                                         layout=wd.Layout(width='240px'))
                      for o in outputs.OUTPUTS}
        for w in self.drop.values():
            w.observe(self.refresh, names='value')
        self.run.observe(self._run_changed, names='value')
        per_band = [wd.HBox([wd.Label(b, layout=wd.Layout(width='60px')),
                             wd.Box([self.check[o.key] for o in outputs.OUTPUTS if o.band == b],
                                    layout=wd.Layout(flex_flow='row wrap'))])
                    for b in BANDS]
        body = wd.VBox([wd.Box(list(self.drop.values()), layout=wd.Layout(flex_flow='row wrap')),
                        self.run, *per_band])
        self.box = wd.Accordion(children=[body], selected_index=0)
        self.refresh()

    # -------------------------------------------------------------- state
    @property
    def folder(self):
        return self.run.value or None

    @property
    def ticked(self):
        return {k for k, cb in self.check.items() if cb.value and not cb.disabled}

    def choices(self):
        return {k: w.value for k, w in self.drop.items()}

    def copy_from(self, other):
        """Take another slot's choices, run and ticks."""
        self._busy = True
        try:
            for k, v in other.choices().items():
                if k in self.drop:
                    self.drop[k].options = [('any', 'any')] + ([(str(v), v)] if v != 'any' else [])
                    self.drop[k].value = v
        finally:
            self._busy = False
        self.refresh()
        if other.folder in dict(self.run.options).values():
            self.run.value = other.folder
        for k, cb in other.check.items():
            self.check[k].value = cb.value

    # -------------------------------------------------------------- cascade
    def refresh(self, change=None):
        """Refill every dropdown from the runs the other choices leave, then the run list."""
        if self._busy:
            return
        self._busy = True
        try:
            cat = self.app.cat
            opts = catalog.options(cat, self.choices())
            for k, w in self.drop.items():
                keep = w.value
                w.options = [('any', 'any')] + [(str(v), v) for v in opts.get(k, [])]
                w.value = keep if keep in [v for _, v in w.options] else 'any'
            hits = catalog.matches(cat, self.choices())
            keep = self.run.value
            self.run.options = [(f'{os.path.relpath(f, r)}   err={e:.1f}°/s', f)
                                for f, r, e in zip(hits.folder, hits.root, hits.err_deg_s)]
            folders = list(hits.folder)
            self.run.value = keep if keep in folders else (folders[0] if folders else None)
        finally:
            self._busy = False
        self._run_changed()

    def _run_changed(self, change=None):
        have = outputs.available(self.folder, self.app.data_root) if self.folder else set()
        for k, cb in self.check.items():
            cb.disabled = k not in have
        n = len(self.run.options)
        name = os.path.basename(self.folder) if self.folder else 'no run'
        self.box.set_title(0, f'#{self.slot}  {name}   ({n} match{"es" if n != 1 else ""})')


class CompareApp:
    """Up to max_runs saved runs side by side, output by output. Viewer only."""

    def __init__(self, roots=('results', 'experiments'), max_runs=20, data_root='data'):
        self.max_runs, self.data_root = max_runs, data_root
        self.roots = wd.Text(value=', '.join(roots), description='roots')
        self.cat = catalog.scan(self._roots())
        self.rescan_btn = wd.Button(description='rescan')
        self.add_btn = wd.Button(description='＋ add run')
        self.remove_btn = wd.Button(description='− remove run')
        self.compare_btn = wd.Button(description='compare', button_style='primary')
        self.only_diff = wd.Checkbox(description='table: only differing columns', value=True)
        self.frame = wd.IntSlider(description='frame', min=0, max=0, continuous_update=False)
        self.info = wd.HTML()
        self.slots_box, self.view, self.frames_box = wd.VBox(), wd.VBox(), wd.VBox()
        self.slots = []
        self.rescan_btn.on_click(lambda _: self.rescan())
        self.add_btn.on_click(lambda _: self.add())
        self.remove_btn.on_click(lambda _: self.remove())
        self.compare_btn.on_click(lambda _: self.render())
        self.only_diff.observe(lambda _: self.render(), names='value')
        self.frame.observe(lambda _: self._render_frames(), names='value')
        self.add()
        self.box = wd.VBox([wd.HBox([self.roots, self.rescan_btn]), self.info,
                            wd.HBox([self.add_btn, self.remove_btn]), self.slots_box,
                            wd.HBox([self.compare_btn, self.only_diff]), self.view])

    def _roots(self):
        return [r.strip() for r in self.roots.value.split(',') if r.strip()]

    def show(self):
        display(self.box)

    # -------------------------------------------------------------- slots
    def _buttons(self):
        self.add_btn.disabled = len(self.slots) >= self.max_runs
        self.remove_btn.disabled = len(self.slots) <= 1
        self.slots_box.children = [s.box for s in self.slots]
        self.info.value = f'{len(self.cat)} saved runs under {", ".join(self._roots())}'

    def add(self):
        if len(self.slots) >= self.max_runs:
            return
        s = RunSlot(self, len(self.slots) + 1)
        if self.slots:
            s.copy_from(self.slots[-1])
        self.slots.append(s)
        self._buttons()

    def remove(self):
        if len(self.slots) > 1:
            self.slots.pop()
            self._buttons()

    def rescan(self):
        outputs.clear_cache()
        self.cat = catalog.scan(self._roots())
        for s in self.slots:
            s.refresh()
        self._buttons()

    # -------------------------------------------------------------- comparison
    def runs(self):
        """The slots that have a run folder, as bands.Run."""
        out = []
        for s in self.slots:
            if s.folder:
                row = self.cat[self.cat.folder == s.folder].iloc[0]
                out.append(bands.Run(s.slot, s.color, row, s.ticked))
        return out

    def _cells(self, runs, key, lab, frame=None):
        cells = []
        for slot, path in bands.image_cells(runs, key, frame):
            r = next(r for r in runs if r.slot == slot)
            body = _file_widget(path) if path else wd.HTML(
                '—' if frame is None else f'no frame {frame}')
            cells.append(wd.VBox([_head(lab[slot], r.color), body],
                                 layout=wd.Layout(width=CELL_WIDTH)))
        return wd.Box(cells, layout=wd.Layout(flex_flow='row wrap'))

    def _band(self, runs, band, lab):
        out = []
        for o in [o for o in outputs.OUTPUTS if o.band == band]:
            sel = [r for r in runs if o.key in r.ticked]
            if not sel:
                continue
            out.append(_head(o.label))
            if band == 'curves':
                fig, missing = bands.curve_figure(sel, o.key, lab)
                out.append(_fig_image(fig))
                if missing:
                    out.append(wd.HTML('not in: ' + ', '.join(f'#{s}' for s in missing)))
            elif band == 'table':
                out.append(wd.HTML(bands.table_html(sel, self.only_diff.value, lab)))
            elif band == 'maps':
                out.append(_fig_image(bands.map_figure(sel, o.key, lab)))
            else:
                out.append(self._cells(sel, o.key, lab))
        return out

    def _frame_count(self, runs):
        n = [len(outputs.frame_paths(r.row.folder, k)) for r in runs for k in outputs.SEQUENCES
             if k in r.ticked]
        return max(n, default=0)

    def _render_frames(self):
        runs, k = self._runs, self.frame.value
        out = []
        for key in outputs.SEQUENCES:
            sel = [r for r in runs if key in r.ticked]
            if not sel:
                continue
            out.append(_head(outputs.BY_KEY[key].label + f' — frame {k}'))
            out.append(_fig_image(bands.state_figure(sel, k, self._lab)) if key == 'states'
                       else self._cells(sel, key, self._lab, frame=k))
        for key in outputs.MOVIES:
            sel = [r for r in runs if key in r.ticked]
            if sel:
                out += [_head(key), self._cells(sel, key, self._lab)]
        self.frames_box.children = out

    def render(self):
        """Redraw the comparison grid from the slots' runs and ticks."""
        runs = self._runs = self.runs()
        if not runs:
            self.view.children = [wd.HTML('<b>no run chosen</b>')]
            return
        self._lab = bands.labels(runs)
        kids = []
        warn = bands.segment_warning(runs)
        if warn:
            kids.append(wd.HTML(f'<b style="color:#b00">⚠ {warn}</b>'))
        kids.append(wd.HTML(' &nbsp; '.join(f'<b style="color:{r.color}">{self._lab[r.slot]}</b>'
                                            for r in runs)))
        for band in BANDS[:-1]:
            kids += self._band(runs, band, self._lab)
        n = self._frame_count(runs)
        self.frame.max = max(n - 1, 0)
        self._render_frames()
        if self.frames_box.children:
            kids += [self.frame, self.frames_box]
        self.view.children = kids
