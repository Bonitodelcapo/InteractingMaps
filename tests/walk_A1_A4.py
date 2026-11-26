"""Smoke tests of sections A1 and A4 (scratch). Run from the repo root with
WALK_NB pointing at the scratch notebook nb_a.ipynb.

    test_A1_redraw, test_A4_redraw  -- run locally (no network)
    test_A1_live, test_A4_live      -- cluster only (A4 runs the network)
"""
import os

from tests.test_experiments_walkthrough import (execute, text, errors, check, unexplained,
                                                snapshot, FAILS, ROOT)


def _checked(t, what):
    """the check named `what` printed [ok]/[DIFF], or a `missing:` line was printed"""
    return any(what in l and l.startswith(('[ok]', '[DIFF]')) for l in t.splitlines()) \
        or 'missing:' in t


def _common(name, nb, t, before, names):
    check(f'{name}: no cell errors', not errors(nb), errors(nb))
    check(f'{name}: omega in deg/s only', 'rad/s' not in t)
    for what in names:
        check(f'{name}: {what} checked or missing', _checked(t, what))
    diffs = [l for l in t.splitlines() if l.startswith('[DIFF]')]
    check(f'{name}: no unexplained DIFF', not unexplained(diffs), diffs)
    for l in diffs:
        print(f'   stale run on disk (flag to user): {l}')
    check(f'{name}: report folders untouched', snapshot() == before)


# ------------------------------------------------------------------- A1
A1_CHECKS = ('cmd_window rows', 'window_choice.csv', 'text_numbers.segment_speeds',
             'text_numbers.intensity', 'text_numbers.track_length')


def test_A1_redraw():
    before = snapshot()
    nb = execute('Redraw', sections=['A1'])
    t = text(nb)
    _common('A1_redraw', nb, t, before, A1_CHECKS)
    check('A1_redraw: [ok] cmd_window rows', '[ok] cmd_window rows' in t, t[-600:])


def test_A1_live():
    before = snapshot()
    nb = execute('Live', {'dt': 0.01}, sections=['A1'])
    t = text(nb)
    _common('A1_live', nb, t, before, ('cmd_window rows',))
    check('A1_live: panel dt used', '10 ms' in t, t[:800])


# ------------------------------------------------------------------- A4
A4_CHECKS = ('main_curl row', 'table_curl', 'text_numbers.inloop_identical')


def test_A4_redraw():
    before = snapshot()
    nb = execute('Redraw', sections=['A2-defs', 'A4'])
    t = text(nb)
    _common('A4_redraw', nb, t, before, A4_CHECKS)


def test_A4_live():
    before = snapshot()
    nb = execute('Live', {'model': 'thesis_cmax', 'segment': 'poster', 'n_frames': 3},
                 sections=['A2-defs', 'A4'])
    t = text(nb)
    check('A4_live: no cell errors', not errors(nb), errors(nb))
    check('A4_live: sets curl 0.6', 'delta_curl 0.6' in t, t[:800])
    check('A4_live: config check ok', '[ok] config' in t, t[-800:])
    check('A4_live: writes walkthrough dir',
          os.path.isdir(os.path.join(ROOT, 'experiments', 'walkthrough', 'A4', 'poster_rotation')))
    check('A4_live: report folders untouched', snapshot() == before)
