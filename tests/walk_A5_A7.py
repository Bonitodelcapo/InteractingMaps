"""Smoke tests of sections A5, A6, A7 (scratch notebook in WALK_NB).

    set WALK_NB=SP\\nb_b.ipynb
    PYTHONPATH=. python -c "import sys; sys.path.insert(0, r'SP'); import tests_b as T; T.test_A5_redraw(); print('FAILS', T.FAILS)"

*_redraw run locally (no network); *_live run tracking / CMax: cluster only.
"""
import os

from tests.test_experiments_walkthrough import (execute, text, errors, check, unexplained,
                                                snapshot, FAILS, ROOT)


def _checked(ID, t, what):
    """The check named `what` printed [ok]/[DIFF], or the section said what is missing."""
    hit = [l for l in t.splitlines() if what in l and l.startswith(('[ok]', '[DIFF]'))]
    check(f'{ID}: {what} checked or missing', bool(hit) or 'missing:' in t)


def _redraw(ID, names):
    before = snapshot()
    nb = execute('Redraw', sections=['A2-defs', ID])
    t = text(nb)
    check(f'{ID}_redraw: no cell errors', not errors(nb), errors(nb))
    check(f'{ID}_redraw: omega in deg/s only', 'rad/s' not in t)
    check(f'{ID}_redraw: section ran', f'(skipped: {ID}' not in t or 'missing:' in t, t[-600:])
    for what in names:
        _checked(ID, t, what)
    diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
    check(f'{ID}_redraw: no unexplained DIFF', not unexplained(diffs), diffs)
    for l in diffs:
        print(f'   DIFF (flag to user): {l}')
    check(f'{ID}_redraw: report folders untouched', snapshot() == before)
    return t


def _live(ID, overrides, names):
    before = snapshot()
    nb = execute('Live', overrides, sections=['A2-defs', ID])
    t = text(nb)
    check(f'{ID}_live: no cell errors', not errors(nb), errors(nb))
    check(f'{ID}_live: omega in deg/s only', 'rad/s' not in t)
    for what in names:
        _checked(ID, t, what)
    diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
    check(f'{ID}_live: no DIFF', not diffs, diffs)
    check(f'{ID}_live: writes walkthrough dir',
          os.path.isdir(os.path.join(ROOT, 'experiments', 'walkthrough')))
    check(f'{ID}_live: writes only walkthrough', snapshot() == before)
    return t


# ------------------------------------------------------------------- A5
def test_A5_redraw():
    _redraw('A5', ['baseline row cmax', 'baseline row gyro', 'table_baseline',
                   'fig_cmax_baseline lines', 'fig_cmax_baseline IWE images',
                   'text_numbers.maps_add'])


def test_A5_live():                     # cluster: runs CMax twice on 3 frames
    _live('A5', {'segment': 'poster', 'n_frames': 3},
          ['cmd_baseline row cmax', 'cmd_baseline row gyro', 'cmd_baseline series cmax'])


# ------------------------------------------------------------------- A6
def test_A6_redraw():
    _redraw('A6', ['beta per run vs read_summary', 'fig_beta_speed points',
                   'fig_beta_speed lines', 'text_numbers.beta_speed',
                   'text_numbers.synthetic_beta_original'])


POSTER_C = ('poster_rotation', 'seg_C', 8.816)
ADOPTED_CFG = ('thesis_cmax', 0.1, 0.5)          # BETA_CONFIGS[1]


def test_A6_live():                     # cluster: one tracking run of 3 frames
    t = _live('A6', {'configs': [ADOPTED_CFG], 'segments': [POSTER_C], 'n_frames': 3},
              ['cmd_beta config', 'beta per run vs read_summary'])
    check('A6_live: one run', '1 of 1 runs found' in t, t[:800])


def test_A6_form_subset():              # Redraw, one config on one segment
    nb = execute('Redraw', {'configs': [ADOPTED_CFG], 'segments': [POSTER_C]},
                 sections=['A2-defs', 'A6'])
    t = text(nb)
    check('A6_form_subset: no cell errors', not errors(nb), errors(nb))
    check('A6_form_subset: one pair only', '1 of 1 runs found' in t or 'no saved run for poster' in t,
          t[:800])
    check('A6_form_subset: report-only steps skip', '[skip] not the report configuration' in t
          or '(skipped: A6' in t, t[-600:])


# ------------------------------------------------------------------- A7
def test_A7_redraw():
    _redraw('A7', ['gyro_floor', 'table_anchor'])


def test_A7_live():                     # cluster: one tracking run of 3 frames
    t = _live('A7', {'models': ['thesis'], 'segments': ['seg_A'], 'n_frames': 3},
              ['summary vs tracking.csv', 'gyro_floor'])
    check('A7_live: announces size', '1 run(s) of 3 frames' in t, t[:800])


def test_A7_form_one_segment():         # Redraw, one poster segment
    nb = execute('Redraw', {'segments': ['seg_D']}, sections=['A2-defs', 'A7'])
    t = text(nb)
    check('A7_form_one_segment: no cell errors', not errors(nb), errors(nb))
    rows = [l for l in t.splitlines() if l.startswith(('seg_A ', 'seg_C ', 'seg_D '))]
    check('A7_form_one_segment: only seg_D runs', rows and all(l.startswith('seg_D') for l in rows)
          or 'no saved run for poster seg_D' in t, t[:800])
