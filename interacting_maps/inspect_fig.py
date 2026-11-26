"""The inspection figure: 1-5 runs of one segment at frame k (or one iteration
of frame k), the raw data of that window, and the chosen curves with a marker
at frame k. One drawing path for the live viewer and the GIF/MP4 export.

Map colour scales are fixed for the whole sequence, per model (95th percentile
of the per-frame viz.map_scale values; for I the 5th pct of the lows and the
95th of the highs), so a breakdown shows up instead of being re-stretched, and
frame k looks the same however it is reached. share=True pools all runs.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from interacting_maps.style import MODELS, SCREEN_WIDTH, REF_STYLE, IMU_STYLE, use_style
from interacting_maps.viz import draw_map, map_scale, draw_series

MAP_KINDS = ('I', 'G', 'F')
RAW_TITLES = {'V': 'events V', 'aps': 'APS'}


# ------------------------------------------------------------------ scales
def _finite_scale(kind, arr):
    a = np.asarray(arr, dtype=np.float64)
    ok = np.isfinite(a).all(-1) if kind in ('G', 'F') else np.isfinite(a)
    if not ok.any():
        return None
    return map_scale(kind, a[ok])


def _pool(kind, scales):
    s = [x for x in scales if x is not None and np.all(np.isfinite(x))]
    if not s:
        return None
    if kind == 'I':
        return (float(np.percentile([x[0] for x in s], 5)),
                float(np.percentile([x[1] for x in s], 95)))
    return float(np.percentile(s, 95))


def _per_frame(run, kind):
    if kind == 'V':
        if not run.has_data():
            return []
        return [_finite_scale('V', run.raw(k)['V']) for k in range(run.n_frames)]
    return [_finite_scale(kind, run.state(k)[kind]) for k in range(run.n_frames)]


def fixed_scales(runs, kinds, share=False) -> list:
    """One {kind: scale} per run; identical dicts when share."""
    per = [{kind: _per_frame(r, kind) for kind in kinds} for r in runs]
    if share:
        pooled = {kind: _pool(kind, [s for p in per for s in p[kind]]) for kind in kinds}
        return [dict(pooled) for _ in runs]
    return [{kind: _pool(kind, p[kind]) for kind in kinds} for p in per]


# ------------------------------------------------------------------ figure
def _maps_at(run, k, it):
    """(maps, note): the checkpoint, or iteration `it` (1-based) of a replay."""
    if it:
        rep = run.iterations(k)
        if rep is not None:
            i = min(it, len(rep['R'])) - 1
            return {kind: rep[kind][i] for kind in MAP_KINDS}, f'it {i + 1}'
        return run.state(k), 'no replay'
    return run.state(k), ''


def _draw_raw(axes, run, k, scale_V):
    raw = run.raw(k)
    for ax, kind in zip(axes, ('V', 'aps')):
        ax._inspect_role = 'raw'
        data = raw[kind] if kind == 'V' else (raw['aps'][1] if raw['aps'] else None)
        if data is None:
            ax.set_axis_off()
            ax.set_title('n/a')
            continue
        title = RAW_TITLES[kind]
        if kind == 'aps':
            title += f'  (t {raw["aps"][0]:.3f} s)'
        draw_map(ax, kind, data, title=title, scale=scale_V if kind == 'V' else None)


def _draw_fullrate(ax, run):
    fr = run.omega_fullrate()
    if not fr:
        ax.set_title('n/a')
        return
    ax.plot(fr['t_gt'], fr['gt'], label='reference', **REF_STYLE)
    ax.plot(fr['t_imu'], fr['imu'], **IMU_STYLE)
    ax.set_ylabel('|ω| (°/s)')
    ax.legend(loc='upper right', fontsize='small')


def _draw_curve(ax, runs, name, k):
    ax._inspect_role, ax._inspect_curve = 'curve', name
    if name == 'omega_fullrate':
        _draw_fullrate(ax, runs[0])
    else:
        for r in runs:
            y = r.curves().get(name)
            if y is not None:
                draw_series(ax, r.times, y, name, model=r.label)
    ax.axvline(runs[0].times[k], color='k', lw=0.8, ls='--')
    ax.set_xlim(0, runs[0].n_frames * runs[0].dt)


def draw_inspector(runs, k, it=None, panels=MAP_KINDS, curves=('err_deg_s', 'beta'),
                   share_scale=False, scales=None, fig=None) -> Figure:
    """Rows: one per run (map panels), one raw row (V, APS), one per curve.
    `it` (1-based) shows that iteration of frame k's replay; `scales` from
    fixed_scales (computed here when None); `fig` is cleared and reused."""
    kinds = [p for p in panels if p in MAP_KINDS]
    if scales is None:
        scales = fixed_scales(runs, kinds + ['V'], share=share_scale)
    ncol = max(len(kinds), 2)
    nrow = len(runs) + 1 + len(curves)
    heights = [2.0] * (len(runs) + 1) + [1.4] * len(curves)
    if fig is None:
        fig = Figure(figsize=(SCREEN_WIDTH, sum(heights) + 0.6))
    fig.clear()
    with use_style('screen'):
        gs = fig.add_gridspec(nrow, ncol, height_ratios=heights)
        for i, (r, sc) in enumerate(zip(runs, scales)):
            maps, note = _maps_at(r, k, it)
            for j, kind in enumerate(kinds):
                ax = fig.add_subplot(gs[i, j])
                ax._inspect_role = 'map'
                draw_map(ax, kind, maps[kind], title=f'{kind} {note}'.strip(),
                         scale=sc.get(kind))
                if j == 0:
                    ax.set_ylabel(MODELS.get(r.label, (r.label,))[0])
        _draw_raw([fig.add_subplot(gs[len(runs), j]) for j in range(2)],
                  runs[0], k, scales[0].get('V'))
        for c, name in enumerate(curves):
            _draw_curve(fig.add_subplot(gs[len(runs) + 1 + c, :]), runs, name, k)
        c0 = runs[0].config
        fig.suptitle(f'{c0["dataset"]} {c0["segment_id"]}   frame {k}   '
                     f't {runs[0].times[k]:.3f} s' + (f'   iteration {it}' if it else ''))
        fig.tight_layout()
    return fig


# ------------------------------------------------------------------ export, click
def _to_rgb(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def export_animation(runs, path, frames=None, it_frame=None, fps=5, **draw_kw) -> str:
    """GIF (or MP4 by extension) over frames, or over the iterations of it_frame."""
    import imageio.v2 as imageio
    kinds = [p for p in draw_kw.get('panels', MAP_KINDS) if p in MAP_KINDS]
    draw_kw.setdefault('scales', fixed_scales(runs, kinds + ['V'],
                                              share=draw_kw.get('share_scale', False)))
    if it_frame is not None:
        n_it = max(r.n_iters for r in runs)
        images = [_to_rgb(draw_inspector(runs, it_frame, it=i, **draw_kw))
                  for i in range(1, n_it + 1)]
    else:
        frames = range(runs[0].n_frames) if frames is None else frames
        images = [_to_rgb(draw_inspector(runs, k, **draw_kw)) for k in frames]
    if str(path).lower().endswith('.mp4'):
        imageio.mimsave(path, images, fps=fps)
    else:
        imageio.mimsave(path, images, duration=1.0 / fps, loop=0)
    return path


def frame_from_click(x, times) -> int:
    return int(np.argmin(np.abs(np.asarray(times) - x)))
