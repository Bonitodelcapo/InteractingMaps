"""What the inspection app can show: one registry entry per quantity.

A Quantity says what it is (kind), where it comes from (source), how to get it
from a RunCache (compute) and how to draw it (render: a viz renderer / scale
kind). The app builds its panels and lists from the registry, so a new output
or a derived quantity is one `register(...)` call, not new UI code.

    kind     'map'        compute(cache, k, it=None) -> (H, W) or (H, W, 2) array
             'curve'      compute(cache)             -> (n_frames,) array
             'iter_curve' compute(cache, k)          -> (n_iters,) array or None
             'fullrate'   compute(cache)             -> dict of time/value arrays
    source   'state' | 'replay' | 'csv' | 'raw' | 'derived'
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Quantity:
    name: str
    kind: str
    source: str
    compute: Callable
    render: Optional[str] = None
    unit: str = ''


REGISTRY: dict = {}
KINDS = ('map', 'curve', 'iter_curve', 'fullrate')


def register(q: Quantity) -> Quantity:
    if q.kind not in KINDS:
        raise ValueError(f'unknown kind {q.kind!r}; one of {KINDS}')
    REGISTRY[q.name] = q
    return q


def unregister(name) -> None:
    REGISTRY.pop(name, None)


def of_kind(kind) -> list:
    return [q for q in REGISTRY.values() if q.kind == kind]


def curve_names(cache) -> list:
    """Curves one run offers: its numeric tracking.csv columns (and events per
    frame), then the registered curves and full-rate quantities."""
    names = [n for n in cache.curves if n not in ('frame', 'time')]
    names += [q.name for q in of_kind('curve') if q.name not in names]
    return names + [q.name for q in of_kind('fullrate')]


# ------------------------------------------------------------------ built-ins
def _state_map(name):
    return lambda c, k, it=None: c.map_at(name, k, it)


for _n in ('I', 'G', 'F'):
    register(Quantity(_n, 'map', 'state', _state_map(_n), render=_n))
register(Quantity('V', 'map', 'raw', lambda c, k, it=None: None if c.V is None else c.V[k],
                  render='V', unit='events'))
register(Quantity('aps', 'map', 'raw', lambda c, k, it=None: c.aps_at(k), render='aps'))
register(Quantity('n_events', 'curve', 'raw', lambda c: c.curves.get('n_events'),
                  unit='events/frame'))
register(Quantity('omega_fullrate', 'fullrate', 'raw', lambda c: c.fullrate, unit='°/s'))


def _residual(name):
    return lambda c, k: (c.iteration_residuals(k) or {}).get(name)


for _n, _u in (('|V+F·G|', ''), ('|G−∇I|', ''), ('|R|', '°/s')):
    register(Quantity(_n, 'iter_curve', 'derived', _residual(_n), unit=_u))
