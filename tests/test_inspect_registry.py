"""Quantity registry of the inspection app (Task 1).

Standalone script:  PYTHONPATH=. python tests/test_inspect_registry.py
"""
import sys
from types import SimpleNamespace

import numpy as np

from interacting_maps.inspect_registry import Quantity, register, unregister, of_kind, curve_names

FAILED = []


def check(name, ok):
    print(f'[{"ok" if ok else "FAIL"}] {name}', flush=True)
    if not ok:
        FAILED.append(name)


check('built-in maps', {'I', 'G', 'F', 'V', 'aps'} <= {q.name for q in of_kind('map')})
check('built-in iteration curves',
      [q.name for q in of_kind('iter_curve')] == ['|V+F·G|', '|G−∇I|', '|R|'])
check('omega_fullrate is a fullrate quantity', [q.name for q in of_kind('fullrate')] == ['omega_fullrate'])

register(Quantity('dummy', 'map', 'derived', lambda c, k, it=None: np.zeros((2, 2)), 'I'))
check('registered quantity appears', 'dummy' in {q.name for q in of_kind('map')})
unregister('dummy')
check('unregistered quantity gone', 'dummy' not in {q.name for q in of_kind('map')})

fake = SimpleNamespace(curves={'frame': np.arange(3), 'time': np.arange(3),
                               'err_deg_s': np.ones(3), 'n_events': np.ones(3)})
names = curve_names(fake)
check('curve names: csv columns, events, fullrate; no frame/time',
      {'err_deg_s', 'n_events', 'omega_fullrate'} <= set(names)
      and not {'frame', 'time'} & set(names))

sys.exit(1 if FAILED else 0)
