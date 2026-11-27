"""Tests of sections A14-A19 (plan Task 8). Redraw: run locally (smoke).
Live: write only, they run on the cluster.

    set WALK_NB=<SP>\\nb_d.ipynb
    PYTHONPATH=. python -c "import sys; sys.path.insert(0, r'<SP>'); import tests_d as T; T.test_A14_redraw()"
"""
import os

from tests.test_experiments_walkthrough import (execute, text, errors, check, unexplained,
                                                snapshot, FAILS, ROOT)

#: the sections each test runs (the shared image-defs cell lives in A14's file)
SECTIONS = {'A14': ['image-defs', 'A14'], 'A15': ['image-defs', 'A15'],
            'A16': ['A16'], 'A17': ['image-defs', 'A17'],
            'A18': ['image-defs', 'A18'], 'A19': ['A19']}
#: the same-code checks of each section, by the name they print
CHECKS = {'A14': ['fig_converge'],
          'A15': ['fig_recon pixels', 'fig_recon |r|'],
          'A16': ['readout.csv rows', 'table_recon', 'text_numbers.intensity 10 vs 20 ms'],
          'A17': ['text_numbers.intensity_update', 'cmd_poisson stats', 'cmd_poisson pixels'],
          'A18': ['distortion_numbers vs csv', 'distortion_numbers', 'fig_distortion pixels'],
          'A19': ['table_params tex', 'table_params vs report tex', 'panel defaults = table_params']}
LIVE_CHECKS = {'A14': ['cmd_converge snapshots', 'fig_converge'],
               'A15': ['config', 'fig_recon pixels'],
               'A16': ['config', 'cmd_readout rows'],
               'A17': ['config', 'cmd_poisson stats'],
               'A18': ['distortion_numbers'],
               'A19': ['table_params tex', 'panel defaults = table_params']}


def _redraw(ID):
    before = snapshot()
    nb = execute('Redraw', sections=SECTIONS[ID])
    t = text(nb)
    check(f'{ID}_redraw: no cell errors', not errors(nb), errors(nb))
    check(f'{ID}_redraw: omega in deg/s only', 'rad/s' not in t)
    lines = t.splitlines()
    for what in CHECKS[ID]:
        hit = [l for l in lines if l.startswith((f'[ok] {what}', f'[DIFF] {what}'))]
        check(f'{ID}_redraw: {what} checked or missing', hit or 'missing:' in t, t[-600:])
    diffs = [l for l in lines if l.startswith('[DIFF]')]
    check(f'{ID}_redraw: no unexplained DIFF', not unexplained(diffs), diffs)
    for l in diffs:
        print(f'   DIFF (flag to user): {l}')
    check(f'{ID}_redraw: report folders untouched', snapshot() == before)
    return t


def test_A14_redraw():
    t = _redraw('A14')
    have = all(os.path.exists(os.path.join(ROOT, 'experiments', 'converge', f))
               for f in ('dynamic_frame45.npz', 'bicycle_frame40.npz'))
    if not have:
        check('A14_redraw: skips cleanly', 'missing:' in t and '(skipped: A14' in t, t[-400:])


def test_A15_redraw():
    _redraw('A15')


def test_A16_redraw():
    _redraw('A16')


def test_A17_redraw():
    _redraw('A17')


def test_A18_redraw():
    t = _redraw('A18')
    if os.path.exists(os.path.join(ROOT, 'data', 'poster_rotation', 'events.txt')):
        check('A18_redraw: ran (raw events present)', '[ok] distortion_numbers' in t, t[-600:])


def test_A19_redraw():
    t = _redraw('A19')
    check('A19_redraw: defaults equal the table', '[ok] panel defaults = table_params' in t)


# ------------------------------------------------- live: on the cluster only
def _live(ID, overrides):
    before = snapshot()
    nb = execute('Live', overrides, sections=SECTIONS[ID])
    t = text(nb)
    check(f'{ID}_live: no cell errors', not errors(nb), errors(nb))
    check(f'{ID}_live: omega in deg/s only', 'rad/s' not in t)
    for what in LIVE_CHECKS[ID]:
        check(f'{ID}_live: [ok] {what}', f'[ok] {what}' in t, t[-800:])
    check(f'{ID}_live: writes only the walkthrough dir', snapshot() == before)
    check(f'{ID}_live: walkthrough dir',
          os.path.isdir(os.path.join(ROOT, 'experiments', 'walkthrough', ID)))


def test_A14_live():
    old = os.path.join(ROOT, 'experiments', 'walkthrough', 'converge', 'poster_frame74.npz')
    if os.path.exists(old):             # a saved case is shown, not recomputed: start fresh
        os.remove(old)
    _live('A14', {'cases': [('poster', 74)], 'n_iters': 75})


def test_A15_live():
    _live('A15', {'model': 'thesis_cmax', 'segments': ['poster'], 'duration': 0.06})


def test_A16_live():
    _live('A16', {'model': 'thesis_cmax', 'segment': 'poster', 'durations': [0.06]})


def test_A17_live():
    _live('A17', {'model': 'thesis_cmax', 'segments': ['poster'], 'n_frames': 3})


def test_A18_live():
    _live('A18', {'segment': 'poster', 'n_frames': 3})


def test_A19_live():
    _live('A19', {})


TESTS = {'redraw': [test_A14_redraw, test_A15_redraw, test_A16_redraw, test_A17_redraw,
                    test_A18_redraw, test_A19_redraw],
         'live': [test_A14_live, test_A15_live, test_A16_live, test_A17_live,
                  test_A18_live, test_A19_live]}
