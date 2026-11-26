"""Smoke tests of sections A8-A13 and the sweep-defs cell (scratch notebook WALK_NB).

    set WALK_NB=<SP>\\nb_c.ipynb
    PYTHONPATH=. python -c "import sys; sys.path.insert(0, r'<SP>'); import tests_c as T; T.test_A8_redraw(); print('FAILS', T.FAILS)"

Redraw tests read saved runs only. *_live tests and test_sweep_full_announces_size
run the network: on the cluster only.
"""
import os
import shutil

from tests.test_experiments_walkthrough import (execute, text, errors, check, unexplained, snapshot,
                                                FAILS, ROOT)

SWEEP = ['A2-defs', 'sweep-defs']


def _redraw(ID, names):
    """Redraw one section: no errors, deg/s only, every check named or 'missing:',
    no unexplained DIFF, report folders and make_figures.FIG untouched."""
    fig0 = os.path.join(ROOT, 'report', 'figures')       # make_figures.FIG
    before = snapshot()
    nb = execute('Redraw', sections=SWEEP + [ID],
                 extra=["import make_figures; print('FIG', make_figures.FIG)"])
    t = text(nb)
    check(f'{ID}_redraw: no cell errors', not errors(nb), errors(nb))
    check(f'{ID}_redraw: omega in deg/s only', 'rad/s' not in t)
    for what in names:
        hit = [l for l in t.splitlines() if what in l and l.startswith(('[ok]', '[DIFF]'))]
        check(f'{ID}_redraw: {what} checked or missing', hit or 'missing:' in t)
    diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
    check(f'{ID}_redraw: no unexplained DIFF', not unexplained(diffs), diffs)
    for l in diffs:
        print(f'   DIFF (flag to user): {l}')
    check(f'{ID}_redraw: report folders untouched', snapshot() == before)
    check(f'{ID}_redraw: make_figures.FIG restored', f'FIG {fig0}' in t, t[-300:])
    return t


# ------------------------------------------------------------- sweep-defs
def test_sweep_points_extent():
    # the jobs' counts, from run_all.sh: sens_cmax 6+7+4+4 one at a time, grid 5x5
    full = {'sens_cmax': 21, 'grid': 25}
    code = ("print('NFULL', len(run_experiments._points(job_axes('grid'), 'grid', {})))\n""for ext in ('one point', 'one axis', 'full'):\n"
            "    W['sweep'].value = ext\n"
            "    for j in ('sens_cmax', 'grid'):\n"
            "        print('N', ext, j, len(sweep_points(j, params())))\n"
            "W['sweep'].value = 'one axis'; W['delta_FR'].value = 0.3\n"
            "print('N axisFR sens_cmax', len(sweep_points('sens_cmax', params())))\n"
            "print('G', sweep_points('grid', params())[0])")
    nb = execute('Live', sections=SWEEP, extra=[code])
    t = text(nb)
    check('sweep_points: no cell errors', not errors(nb), errors(nb))
    for ext, j, n in (('one point', 'sens_cmax', 1), ('one point', 'grid', 1),
                      ('one axis', 'sens_cmax', 6), ('one axis', 'grid', 5),
                      ('full', 'sens_cmax', full['sens_cmax']), ('full', 'grid', full['grid'])):
        check(f'sweep_points: {ext} {j} = {n}', f'N {ext} {j} {n}\n' in t, t[-600:])
    check('sweep_points: one axis follows the changed delta_FR', 'N axisFR sens_cmax 7\n' in t)
    check('sweep_points: grid holds the other axis at the adopted value',
          "G {'delta_FR': 0.02, 'delta_IMU': 0.5}" in t, t[-300:])


def test_sweep_missing_column_is_diff():
    d = os.path.join(ROOT, 'experiments', 'walkthrough', '_coltest')
    os.makedirs(d, exist_ok=True)
    try:
        with open(os.path.join(d, 'old.csv'), 'w') as f:      # an older column set: no blowup
            f.write('sequence,segment,model,err\nposter_rotation,seg_C,thesis_cmax,7.0\n')
        nb = execute('Redraw', sections=SWEEP, extra=[
            "r = sweep_rows('walkthrough/_coltest')\n"
            "print('GATE', have_cols(r, ['err', 'blowup'], 'T'))"])
        t = text(nb)
        check('missing_column: no cell errors', not errors(nb), errors(nb))
        check('missing_column: is a DIFF naming it',
              any(l.startswith('[DIFF] old.csv') and 'blowup' in l for l in t.splitlines()), t[-500:])
        check('missing_column: gates the section', 'GATE False' in t)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------- redraw
def test_A8_redraw():
    _redraw('A8', ['table_sweep', 'fig_sensitivity', 'text_numbers.sensitivity_both',
                   'text_numbers.sweep_ends', 'text_numbers.anchor_ablation',
                   'text_numbers.anchors_across_weights', 'text_numbers.kappa_rows'])


def test_A9_redraw():
    _redraw('A9', ['table_grid', 'table_params', 'text_numbers.grid'])


def test_A10_redraw():
    _redraw('A10', ['text_numbers.synthetic_beta_original'])


def test_A11_redraw():
    _redraw('A11', ['table_regularisers', 'fig_regularisers', 'text_numbers.regularisers'])


def test_A12_redraw():
    _redraw('A12', ['image_search', 'table_image', 'fig_image_settings', 'text_numbers.image_window'])


def test_A13_redraw():
    _redraw('A13', ['compute_maps', 'fig_maps', 'fig_appendix', 'text_numbers.polarity'])


# ------------------------------------------------------------- live: cluster only
def _live(ID, overrides=None):
    before = snapshot()
    nb = execute('Live', {'segment': 'poster', 'n_frames': 2, 'sweep': 'one point', **(overrides or {})},
                 sections=SWEEP + [ID])
    t = text(nb)
    check(f'{ID}_live: no cell errors', not errors(nb), errors(nb))
    check(f'{ID}_live: config check ok', '[ok] config' in t, t[-800:])
    diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
    check(f'{ID}_live: rows equal cmd_sweep', not diffs, diffs)
    check(f'{ID}_live: writes only walkthrough', snapshot() == before)
    check(f'{ID}_live: walkthrough dir', os.path.isdir(os.path.join(ROOT, 'experiments', 'walkthrough', ID)))


def test_A8_live():
    _live('A8', {'model': 'thesis_cmax', 'sweep': 'one axis', 'delta_GI': 0.1, 'n_frames': 2})


def test_A9_live():
    _live('A9', {'model': 'thesis_cmax'})


def test_A10_live():
    _live('A10', {'model': 'thesis_cmax'})


def test_A11_live():
    _live('A11', {'model': 'thesis_cmax'})


def test_A12_live():
    _live('A12', {'model': 'thesis_cmax'})


def test_A13_live():
    _live('A13', {'model': 'thesis_cmax'})


def test_sweep_full_announces_size():
    """'full' prints the number of runs and the total frames before the first run."""
    nb = execute('Live', {'segment': 'poster', 'n_frames': 2, 'sweep': 'full'},
                 sections=SWEEP + ['A9'])
    t = text(nb)
    check('full_announces: no cell errors', not errors(nb), errors(nb))
    i = t.find('Live: job grid')
    check('full_announces: runs and frames printed first',
          i >= 0 and '25 runs, 50 frames in total' in t[i:i + 200]
          and t.find('delta_FR=') > i, t[:800])


REDRAW = [test_sweep_points_extent, test_sweep_missing_column_is_diff, test_A8_redraw,
          test_A9_redraw, test_A10_redraw, test_A11_redraw, test_A12_redraw, test_A13_redraw]
LIVE = [test_A8_live, test_A9_live, test_A10_live, test_A11_live, test_A12_live, test_A13_live,
        test_sweep_full_announces_size]
