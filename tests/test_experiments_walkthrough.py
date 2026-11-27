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
    check('redraw_says_report_runs', "Redraw: the report's saved runs" in t, t[:600])
    check('redraw_no_rerun_line', 'rerun it in Live mode' not in t, t[:600])


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


# ------------------------------------------------------- run registry (lookup + Live guard)
def _readable_runs():
    import glob, json
    n = 0
    for f in glob.glob('results/**/run.json', recursive=True) + \
             glob.glob('experiments/**/run.json', recursive=True):
        try:
            n += 'config' in json.load(open(f))
        except (OSError, ValueError):
            pass
    return n


def test_registry_scan():
    nb = execute('Redraw', extra=[
        "print('NROWS', len(REG)); print('COLS', sorted(REG.columns));"
        "print('NANS', int(REG[['model', 'dataset']].isna().sum().sum()))"])
    t = text(nb)
    check('registry_scan: no errors', not errors(nb), errors(nb))
    check('registry_scan: one row per run.json', f'NROWS {_readable_runs()}' in t, t[-400:])
    for col in ('folder', 'root', 'model', 'segment', 'segment_id', 'dt', 'n_frames',
                'n_iters', 'poisson', 'delta_FR', 'err_deg_s', 'dir_err_deg', 'beta'):
        check(f'registry_scan: column {col}', f"'{col}'" in t)
    check('registry_scan: model/dataset filled', 'NANS 0' in t, t[-200:])


def test_registry_cascade():
    nb = execute('Redraw', extra=[
        "a = available(REG, {'model': 'thesis_cmax'})\n"
        "b = available(REG, {'model': 'thesis_cmax', 'segment': 'poster_rotation seg_C'})\n"
        "print('SUBSET', set(b['delta_FR']) <= set(a['delta_FR']))\n"
        "print('HAS01', 0.1 in b['delta_FR'])\n"
        "print('OWNKEPT', len(b['segment']) == len(a['segment']) > 1)"])
    t = text(nb)
    check('registry_cascade: no errors', not errors(nb), errors(nb))
    for k in ('SUBSET True', 'HAS01 True', 'OWNKEPT True'):
        check(f'registry_cascade: {k}', k in t, t[-300:])


def test_registry_no_match():
    nb = execute('Redraw', extra=[
        "print('EMPTY', matching(REG, {'model': 'cook', 'delta_FR': 123.0}).empty)\n"
        "LOOK['model'].value = 'cook'\n"
        "print(show_matches({'model': 'cook', 'delta_FR': 123.0}))"])
    t = text(nb)
    check('registry_no_match: no errors', not errors(nb), errors(nb))
    check('registry_no_match: empty', 'EMPTY True' in t, t[-300:])
    check('registry_no_match: said', 'no saved run matches' in t, t[-300:])


def test_registry_empty_roots():
    nb = execute('Redraw', extra=[
        "e = scan_runs(('nonexistent_root',)); print('EMPTYREG', e.empty, 'delta_FR' in e.columns)"])
    check('registry_empty_roots', 'EMPTYREG True True' in text(nb) and not errors(nb),
          text(nb)[-300:] + str(errors(nb)))


def test_registry_old_config():
    import json, shutil
    base = os.path.join('experiments', 'walkthrough', '_reg_test')
    d = os.path.join(base, 'poster_rotation', 'cook', 'seg_C_old')
    os.makedirs(d, exist_ok=True)
    json.dump({'config': {'dataset': 'poster_rotation', 'model': 'cook', 't_start': 8.816,
                          'frame_duration': 0.02, 'n_frames': 5, 'n_iters': 75,
                          'distortion_mode': 'C_full', 'params': {'delta_FR': 0.1}},
               'summary': {}}, open(os.path.join(d, 'run.json'), 'w'))
    try:
        nb = execute('Redraw', extra=[
            "r = REG[REG.folder.str.contains('_reg_test')].iloc[0]\n"
            "print('OLD', r.poisson, r.delta_curl, r.segment_id, r.root.replace(os.sep, '/'))"])
    finally:
        shutil.rmtree(base)
    t = text(nb)
    check('registry_old_config', 'OLD iterative 0.0 seg_C experiments/walkthrough/_reg_test' in t,
          t[-300:] + str(errors(nb)))


def test_model_prints_dict():
    nb = execute('Redraw', extra=["LOOK['model'].value = 'thesis_cmax'; print(INFO.value)"])
    t = text(nb)
    check('model_prints_dict', "'delta_FR': [" in t and "'n_frames': [" in t and not errors(nb),
          t[-400:] + str(errors(nb)))


GUARD_RC = ("import types\n"
            "r = REG[REG.root == 'results'].iloc[0]\n"
            "rc = types.SimpleNamespace(dataset=r.dataset, model=r.model, segment_id=r.segment_id,\n"
            "    n_frames=int(r.n_frames), frame_duration=r['dt'], t_start=r.t_start, n_iters=int(r.n_iters),\n"
            "    poisson=r.poisson, distortion_mode=r.distortion_mode, out_root='experiments/walkthrough/_none',\n"
            "    params={k: float(r[k]) for k in DELTAS if k in experiment_tracking.read_config(r.folder).get('params', {})})\n"
            "TRACK0 = lambda *a, **k: print('RAN')\n")


def test_guard_stops_saved():
    nb = execute('Live', extra=[GUARD_RC + "E.experiment_tracking(rc, save_frames=False)"])
    t = text(nb)
    check('guard_stops_saved: said', 'already saved:' in t, t[-400:])
    check('guard_stops_saved: not run', 'RAN' not in t, t[-400:])
    check('guard_stops_saved: no traceback', 'Traceback' not in t and not errors(nb),
          t[-400:] + str(errors(nb)))


def test_guard_runs_new():
    nb = execute('Live', extra=[GUARD_RC + "rc.n_frames = 7\nE.experiment_tracking(rc, save_frames=False)"])
    t = text(nb)
    check('guard_runs_new', 'RAN' in t and 'already saved' not in t, t[-400:] + str(errors(nb)))


def test_guard_unstored_delta():
    nb = execute('Live', extra=[
        GUARD_RC.replace("REG.root == 'results'", "(REG.root == 'results') & (REG.delta_curl == 0)") +
        "rc.params = {**rc.params, 'delta_curl': 0.0}; print('ZERO', bool(saved_runs(rc)))\n"
        "rc.params = {**rc.params, 'delta_curl': 0.6}; print('SIX', bool(saved_runs(rc)))"])
    t = text(nb)
    check('guard_unstored_delta', 'ZERO True' in t and 'SIX False' in t, t[-300:] + str(errors(nb)))


def test_redirect_bypasses_guard():
    nb = execute('Live', extra=[GUARD_RC +
        "with redirect(run_experiments, _run=lambda *a: print('REC')):\n"
        "    run_experiments._run(E, rc, False, 1)"])
    t = text(nb)
    check('redirect_bypasses_guard', 'REC' in t and 'already saved' not in t, t[-300:])


def test_guard_incomplete_and_new_root():
    """A run.json without a summary (crashed run) is not 'saved'; a complete run in
    a root the registry has not scanned yet (rc.out_root) is."""
    import json, shutil
    base = os.path.join('experiments', 'walkthrough', '_reg_test')
    cfg = {'dataset': 'poster_rotation', 'model': 'cook', 'segment_id': 'seg_C', 't_start': 8.816,
           'frame_duration': 0.02, 'n_frames': 4, 'n_iters': 75, 'distortion_mode': 'C_full',
           'poisson': 'iterative', 'params': {'delta_FR': 0.1}}
    d = os.path.join(base, 'poster_rotation', 'cook', 'seg_C_x')
    rc = ("import types\nrc = types.SimpleNamespace(dataset='poster_rotation', model='cook', segment_id='seg_C',"
          " n_frames=4, frame_duration=0.02, t_start=8.816, n_iters=75, poisson='iterative',"
          " distortion_mode='C_full', params={'delta_FR': 0.1}, out_root=%r)\n" % base)
    try:
        os.makedirs(d, exist_ok=True)
        json.dump({'config': cfg, 'summary': None}, open(os.path.join(d, 'run.json'), 'w'))
        nb = execute('Live', extra=[rc + "print('CRASHED', bool(saved_runs(rc)))"])
        json.dump({'config': cfg, 'summary': {'mean_err_deg_s': 1.0}}, open(os.path.join(d, 'run.json'), 'w'))
        nb2 = execute('Live', extra=["REG = REG[~REG.folder.str.contains('_reg_test')]\n" + rc +
                                     "print('NEWROOT', bool(saved_runs(rc)))"])
    finally:
        shutil.rmtree(base)
    check('guard_crashed_run_not_saved', 'CRASHED False' in text(nb), text(nb)[-300:] + str(errors(nb)))
    check('guard_new_root_found', 'NEWROOT True' in text(nb2), text(nb2)[-300:] + str(errors(nb2)))


def test_redraw_shows_segment():
    nb = execute('Redraw', extra=["print('SEGSHOWN', W['segment'] in TOP.children)"])
    check('redraw_shows_segment', 'SEGSHOWN True' in text(nb), text(nb)[-300:] + str(errors(nb)))


TESTS = {'setup': [test_setup],
         'registry': [test_registry_scan, test_registry_cascade, test_registry_no_match,
                      test_registry_empty_roots, test_registry_old_config, test_model_prints_dict,
                      test_guard_stops_saved, test_guard_runs_new, test_guard_unstored_delta,
                      test_redirect_bypasses_guard, test_guard_incomplete_and_new_root,
                      test_redraw_shows_segment],
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
