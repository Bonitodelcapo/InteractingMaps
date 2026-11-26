"""
test_experiments_walkthrough.py -- executes experiments_walkthrough.ipynb headless.

Standalone script (repo idiom, not pytest). From the repo root:
    PYTHONPATH=. python tests/test_experiments_walkthrough.py          # all but live
    PYTHONPATH=. python tests/test_experiments_walkthrough.py live     # tracking runs: cluster
    PYTHONPATH=. python tests/test_experiments_walkthrough.py setup A2 # some

Each test runs the notebook's 'core' cells (setup, widgets, helpers) plus the
cells of the sections it needs, with the widget values it sets, and reads the
printed output. Prints [ok] / [FAIL] per test.
"""

import os
import sys

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.environ.get('WALK_NB') or os.path.join(ROOT, 'experiments_walkthrough.ipynb')   # WALK_NB: a scratch build
FAILS = []


def check(name, ok, detail=''):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f'  ({detail})' if detail and not ok else ''))
    if not ok:
        FAILS.append(name)


def execute(mode='Redraw', overrides=None, sections=(), extra=(), pre=(), allow_errors=True):
    """Run the core cells + the cells tagged with `sections`, widgets set from
    `overrides`, then the `extra` code cells. `pre` cells go right after the
    A2 definitions (to patch them). Returns the executed notebook."""
    nb = nbformat.read(NB, as_version=4)
    keep = []
    for c in nb.cells:
        tags = set(c.metadata.get('tags', []))
        if 'core' in tags or tags & set(sections):
            keep.append(c)
            if 'widgets' in tags:
                sets = {'mode': mode, **(overrides or {})}
                keep.append(nbformat.v4.new_code_cell(
                    '\n'.join(f"W[{k!r}].value = {v!r}" for k, v in sets.items())))
            if 'A2-defs' in tags:
                keep += [nbformat.v4.new_code_cell(src) for src in pre]
    keep += [nbformat.v4.new_code_cell(src) for src in extra]
    nb.cells = keep
    NotebookClient(nb, timeout=3600, kernel_name='python3', allow_errors=allow_errors,
                   resources={'metadata': {'path': ROOT}}).execute()
    return nb


def text(nb):
    out = []
    for c in nb.cells:
        for o in c.get('outputs', []):
            if o.output_type == 'stream':
                out.append(o.text)
            elif o.output_type == 'error':
                out.append(f'ERROR {o.ename}: {o.evalue}\n')
    return ''.join(out)


def errors(nb):
    return [f'{o.ename}: {o.evalue}' for c in nb.cells
            for o in c.get('outputs', []) if o.output_type == 'error']


# ------------------------------------------------------------------- Task 1
def test_setup():
    from config import THESIS_PARAMS
    nb = execute(extra=[
        "p = params(); print('P', p['seg'][3], p['n_iters'], p['dt'], p['delta_FR'])",
        "W['delta_FR'].value = 0.3; print('LIVE', params()['delta_FR'])",
        "import make_figures, io\n"
        "orig = make_figures.FIG\n"
        "try:\n"
        "    with redirect(make_figures, FIG='x'):\n"
        "        raise RuntimeError('boom')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "print('RESTORED', make_figures.FIG == orig)",
        "print('NEED', need(['nope.csv'], ['main_poster_cook']))",
        "print('CK1', check('x', 1.005, 1.0)); print('CK2', check('x', 1.1, 1.0))",
    ])
    t = text(nb)
    check('setup_runs: no cell errors', not errors(nb), errors(nb))
    check('setup_runs: report defaults',
          f"P poster 75 0.02 {THESIS_PARAMS['delta_FR']}" in t, t[-500:])
    check('params_reads_widgets_live', 'LIVE 0.3' in t)
    check('redirect_restores_on_error', 'RESTORED True' in t)
    check('need_reports_missing', 'NEED False' in t and 'missing: nope.csv' in t
          and 'main_poster_cook' in t)
    check('check_loose', 'CK1 True' in t and 'CK2 False' in t and '[DIFF] x' in t)


# ------------------------------------------------------------------- Task 2
PROTECTED = ['results', os.path.join('report', 'experiments'), os.path.join('report', 'figures')]


def snapshot():
    """Newest modification time under each protected folder."""
    out = {}
    for top in PROTECTED:
        newest = 0.0
        for dirpath, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, f)))
        out[top] = newest
    return out


#: columns that a stale run on disk (older recon_r scoring) differs in; any other
#: DIFF means the unrolled code and the report code disagree
STALE_OK = ('recon', 'recon_inloop')


def unexplained(diffs):
    """DIFF lines not explained by a stale run (only STALE_OK columns differ)."""
    import re
    bad = []
    for l in diffs:
        cols = re.findall(r'(\w+) [-\d.e+naninf]+ vs', l.split(' in ', 1)[-1]) if ' in ' in l else []
        if not cols or any(c not in STALE_OK for c in cols):
            bad.append(l)
    return bad


def test_A2_redraw():
    nb = execute('Redraw', sections=['A2'])
    t = text(nb)
    have = all(os.path.exists(os.path.join(ROOT, 'report', 'experiments',
                                           f'main_base_poster_{m}.csv'))
               for m in ('cook', 'thesis'))
    check('A2_redraw: no cell errors', not errors(nb), errors(nb))
    check('A2_redraw: omega in deg/s only', 'rad/s' not in t)
    if have:
        diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
        check('A2_redraw: checks pass', '[ok]' in t and not unexplained(diffs), diffs)
        check('A2_redraw: track_length checked',
              any(l.startswith(('[ok] text_numbers.track_length', '[DIFF] text_numbers.track_length'))
                  for l in t.splitlines()))
        for l in diffs:
            print(f'   stale run on disk (flag to user): {l}')
    else:
        check('A2_redraw: reports missing', 'missing:' in t)


def test_A2_live_short():
    before = snapshot()
    for _ in range(2):          # twice: a repeated live run overwrites its own folder
        nb = execute('Live', {'model': 'thesis', 'segment': 'poster', 'n_frames': 3},
                     sections=['A2'])
        t = text(nb)
        check('A2_live: no cell errors', not errors(nb), errors(nb))
        check('A2_live: config check ok', '[ok] config' in t, t[-800:])
    check('A2_live: writes walkthrough dir',
          os.path.isdir(os.path.join(ROOT, 'experiments', 'walkthrough', 'A2', 'poster_rotation')))
    check('live_writes_only_walkthrough', snapshot() == before)


def test_live_missing_dataset_skips():
    nb = execute('Live', {'segment': 'boxes', 'n_frames': 3}, sections=['A2'])
    t = text(nb)
    check('live_missing_dataset: no cell errors', not errors(nb), errors(nb))
    check('live_missing_dataset: reported',
          f"missing: {os.path.join('data', 'boxes_rotation')}" in t, t[-500:])


def test_redraw_note():
    nb = execute('Redraw', {'delta_FR': 0.3, 'n_frames': 10}, sections=['A2'])
    t = text(nb)
    check('redraw_says_rerun_needed',
          "No saved run has the panel's delta_FR 0.3, n_frames 10 (saved: 150): "
          "rerun it in Live mode." in t, t[:600])
    nb = execute('Redraw', {'n_frames': 150}, sections=['A2'])
    check('redraw_quiet_when_saved', 'No saved run' not in text(nb))


# ------------------------------------------------------------------- Task 3
def test_A3_redraw():
    before = snapshot()
    nb = execute('Redraw', sections=['A2-defs', 'A3'])
    t = text(nb)
    check('A3_redraw: no cell errors', not errors(nb), errors(nb))
    check('A3_redraw: omega in deg/s only', 'rad/s' not in t)
    for what in ('table_main', 'fig_tracking', 'text_numbers.unanchored', 'text_numbers.reference',
                 'text_numbers.intensity', 'fig_dissociation'):
        hit = [l for l in t.splitlines() if what in l and l.startswith(('[ok]', '[DIFF]'))]
        check(f'A3_redraw: {what} checked or missing', hit or 'missing:' in t)
    diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
    check('A3_redraw: no unexplained DIFF', not unexplained(diffs), diffs)
    for l in diffs:
        print(f'   stale run on disk (flag to user): {l}')
    check('A3_redraw: report folders untouched', snapshot() == before)


def test_A3_live_short():
    nb = execute('Live', {'model': 'thesis_cmax', 'segment': 'poster', 'n_frames': 3},
                 sections=['A2-defs', 'A3'])
    t = text(nb)
    check('A3_live: no cell errors', not errors(nb), errors(nb))
    check('A3_live: config check ok', '[ok] config' in t, t[-800:])


# ------------------------------------------------------ final review fixes
def test_redraw_ignores_dt():
    nb = execute('Redraw', {'dt': 0.01}, sections=['A2'])
    diffs = [l for l in text(nb).splitlines() if l.startswith('[DIFF]')]
    check('fix1 redraw_ignores_dt: no unexplained DIFF', not unexplained(diffs), diffs)


def test_check_empty_is_skip():
    nb = execute(extra=["print('E', check('e', [], []))"])
    t = text(nb)
    check('fix2 check_empty_is_skip', '[skip] e' in t and '[ok] e' not in t, t[-300:])


def test_fig_tracking_runs_missing():
    pre = ["_rd = run_dir\n"
           "def run_dir(seg, m, nf, **kw):\n"
           "    return None if m in ('thesis', 'thesis_cmax') else _rd(seg, m, nf, **kw)\n"
           "make_figures.find_run = lambda *a, **k: None"]
    nb = execute('Redraw', sections=['A2-defs', 'A3'], pre=pre)
    check('fix3 fig_tracking_runs_missing: no cell errors', not errors(nb), errors(nb))


def test_section_raises():
    from nbclient.exceptions import CellExecutionError
    try:
        execute(extra=["%%section X\nraise ValueError('inside')"], allow_errors=False)
        raised = False
    except CellExecutionError:
        raised = True
    check('fix4 section_magic_fails_the_cell', raised)


def test_ready_after_failed_live():
    pre = ["def _boom(*a, **k):\n    raise RuntimeError('run failed')\nE.experiment_tracking = _boom"]
    nb = execute('Live', {'n_frames': 2}, sections=['A2-defs', 'A2'], pre=pre,
                 extra=["print('READY', READY.get('A2'))"])
    check('fix5 ready_false_after_failed_live', 'READY False' in text(nb), text(nb)[-400:])


def test_live_check_not_circular():
    pre = ["_mc = main_cfg\n"
           "def main_cfg(p, nf, out_root):\n"
           "    return _mc({**p, 'deltas': dict(THESIS_PARAMS)}, nf, out_root)"]
    nb = execute('Live', {'model': 'thesis', 'n_frames': 2, 'delta_FR': 0.3},
                 sections=['A2-defs', 'A2'], pre=pre)
    check('fix6 live_check_not_circular', '[DIFF] config' in text(nb), text(nb)[-400:])


# ------------------------------------------------------ stage-2 review fixes
def test_sections_stand_alone():
    """A16 and A17 run without A2-defs: they must import ADOPTED themselves."""
    for secs in (['A16'], ['image-defs', 'A17']):
        nb = execute('Redraw', sections=secs, extra=["print('HAS', 'ADOPTED' in globals())"])
        check(f'review1 {secs[-1]} defines ADOPTED', 'HAS True' in text(nb), text(nb)[-300:])


def test_cook_keeps_changed_delta():
    """cmd_main passes --vary deltas to cook; main_cfg must too (A4 Live with cook)."""
    nb = execute('Live', {'model': 'cook', 'delta_curl': 0.6}, sections=['A2-defs'],
                 extra=["print('CURL', main_cfg(params(), 2, None).params.get('delta_curl'))"])
    check('review2 cook keeps delta_curl', 'CURL 0.6' in text(nb), text(nb)[-300:])


def test_full_redraw():
    """The whole notebook top to bottom in Redraw (cluster: reads every saved run).
    Sections were tested one at a time; this catches one clobbering another."""
    import make_figures
    ids = ['A0', 'A2-defs', 'sweep-defs', 'image-defs'] + [f'A{i}' for i in range(1, 20)]
    nb = execute('Redraw', sections=ids, extra=[
        "import make_figures as _mf; print('FIG', repr(_mf.FIG)); print('SAVE', _mf.save.__module__)"])
    t = text(nb)
    check('full_redraw: no cell errors', not errors(nb), errors(nb))
    check('full_redraw: omega in deg/s only', 'rad/s' not in t)
    check('full_redraw: make_figures restored', f"FIG {make_figures.FIG!r}" in t
          and 'SAVE interacting_maps.style' in t, t[-300:])
    diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
    check('full_redraw: no unexplained DIFF', not unexplained(diffs), diffs)


def test_live_missing_dataset_skips_more():
    """Live on a segment whose dataset is absent: A5 and A14 skip, no error (no run)."""
    for ID, secs in (('A5', ['A2-defs', 'A5']), ('A14', ['image-defs', 'A14'])):
        nb = execute('Live', {'segment': 'boxes', 'n_frames': 3}, sections=secs)
        check(f'{ID} live_missing_dataset: no cell errors', not errors(nb), errors(nb))
        check(f'{ID} live_missing_dataset: reported', 'missing:' in text(nb), text(nb)[-300:])


TESTS = {'setup': [test_setup],
         'fixes': [test_redraw_ignores_dt, test_check_empty_is_skip, test_fig_tracking_runs_missing,
                   test_section_raises, test_sections_stand_alone, test_cook_keeps_changed_delta],
         'A3': [test_A3_redraw],
         'A2': [test_A2_redraw, test_redraw_note],
         # tracking runs: on the cluster only, never by default
         'full': [test_full_redraw],          # cluster: whole notebook, Redraw
         'live': [test_live_missing_dataset_skips_more, test_A2_live_short, test_A3_live_short, test_live_missing_dataset_skips,
                  test_ready_after_failed_live, test_live_check_not_circular]}


def section_tests():
    """Stage-2 sections: one test module per group of sections (tests/walk_*.py).
    Each module's tests run against this file imported as a module, so their
    [FAIL]s collect in that module's FAILS. Returns (groups, that FAILS list)."""
    import tests.test_experiments_walkthrough as me
    from tests import walk_A1_A4 as a, walk_A5_A7 as b, walk_A8_A13 as c, walk_A14_A19 as d
    groups = {'A1': [a.test_A1_redraw], 'A4': [a.test_A4_redraw],
              'A5': [b.test_A5_redraw], 'A6': [b.test_A6_redraw, b.test_A6_live_not_beta_segment],
              'A7': [b.test_A7_redraw],
              'sweep': [c.test_sweep_points_extent, c.test_sweep_missing_column_is_diff],
              **{f'A{i}': [getattr(c, f'test_A{i}_redraw')] for i in range(8, 14)},
              **{f'A{i}': [getattr(d, f'test_A{i}_redraw')] for i in range(14, 20)}}
    groups['live'] = [a.test_A1_live, a.test_A4_live, b.test_A5_live, b.test_A6_live,
                      b.test_A7_live, *c.LIVE, *d.TESTS['live']]
    return groups, me.FAILS


if __name__ == '__main__':
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    more, more_fails = section_tests()
    TESTS['live'] += more.pop('live')
    TESTS.update(more)
    # no argument: every group except 'live' and 'full' (cluster only). Locally,
    # run one group at a time as a smoke test, e.g. `... test_experiments_walkthrough.py A5`
    for key in (sys.argv[1:] or [k for k in TESTS if k not in ('live', 'full')]):
        for fn in TESTS[key]:
            fn()
    fails = FAILS + more_fails
    print(f"\n{'ALL OK' if not fails else f'{len(fails)} FAILED: {fails}'}")
    sys.exit(1 if fails else 0)
