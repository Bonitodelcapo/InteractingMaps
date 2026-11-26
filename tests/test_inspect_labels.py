"""Run labels and colours (Task 3).

Standalone script:  PYTHONPATH=. python tests/test_inspect_labels.py
"""
import sys

from interacting_maps.inspect_labels import run_labels, run_colors
from interacting_maps.style import MODELS

FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


def cfg(model, d_fr=0.1, n_iters=75, folder='run'):
    return {'model': model, 'n_iters': n_iters, 'poisson': 'iterative',
            'distortion_mode': 'C_full', 'params': {'delta_FR': d_fr, 'delta_IG': 0.1},
            '_dir': f'experiments/inspect/x/{model}/{folder}'}


two = [cfg('cook'), cfg('thesis_imu')]
check('different models -> display names', run_labels(two) == ['cook', 'thesis+IMU'])
check('different models -> model colours',
      run_colors(two) == [MODELS['cook'][1], MODELS['thesis_imu'][1]])

sweep = [cfg('thesis_imu', d_fr=round(0.1 * (i + 1), 1), folder=f'r{i}') for i in range(10)]
labels = run_labels(sweep)
check('sweep of one model -> 10 distinct labels', len(set(labels)) == 10
      and all(l.startswith('thesis+IMU dFR=') for l in labels))
check('sweep label format', labels[0] == 'thesis+IMU dFR=0.1')
check('sweep -> 10 distinct colours', len(set(run_colors(sweep))) == 10)

both = [cfg('thesis_imu', d_fr=0.1, n_iters=75), cfg('thesis_imu', d_fr=0.5, n_iters=50)]
lb = run_labels(both)
check('two differing fields both shown', 'dFR=0.5' in lb[1] and 'it=50' in lb[1])

same = [cfg('thesis', folder='a'), cfg('thesis', folder='b')]
check('identical configs -> folder names', run_labels(same) == ['a', 'b'])

sys.exit(1 if FAILED else 0)
