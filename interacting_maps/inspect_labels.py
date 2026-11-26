"""Names and colours for up to 10 loaded runs, which may share a model.

A label is the model's display name plus every config field that differs
between the loaded runs (e.g. 'thesis+IMU dFR=0.1'); runs that still cannot be
told apart are named by their run folder. Colours are the report's model
colours when every run is a different model, a 10-colour palette otherwise.
"""
import os

from interacting_maps.style import MODELS

#: matplotlib's tab10, distinct and colour-blind friendly enough for 10 lines
PALETTE = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
_SHORT = {'delta_IMU': 'dimu', 'n_iters': 'it', 'poisson': '', 'distortion_mode': ''}


def _fields(cfg) -> dict:
    out = {f'params.{k}': v for k, v in cfg.get('params', {}).items()}
    for k in ('n_iters', 'poisson', 'distortion_mode'):
        out[k] = cfg.get(k)
    return out


def _fmt(key, value) -> str:
    name = key.split('.', 1)[-1]
    short = _SHORT.get(name, 'd' + name[len('delta_'):] if name.startswith('delta_') else name)
    val = f'{value:g}' if isinstance(value, (int, float)) else str(value)
    return f'{short}={val}' if short else val


def run_labels(configs) -> list:
    fields = [_fields(c) for c in configs]
    keys = sorted({k for f in fields for k in f})
    differ = [k for k in keys if len({repr(f.get(k)) for f in fields}) > 1]
    labels = []
    for c, f in zip(configs, fields):
        parts = [MODELS.get(c['model'], (c['model'],))[0]]
        parts += [_fmt(k, f.get(k)) for k in differ]
        labels.append(' '.join(parts))
    folders = [os.path.basename(os.path.normpath(c.get('_dir', ''))) for c in configs]
    dup = [labels.count(lab) > 1 for lab in labels]
    return [folders[i] if dup[i] else lab for i, lab in enumerate(labels)]


def run_colors(configs) -> list:
    models = [c['model'] for c in configs]
    if len(set(models)) == len(models) and all(m in MODELS for m in models):
        return [MODELS[m][1] for m in models]
    return [PALETTE[i % len(PALETTE)] for i in range(len(configs))]
