"""make_tables.py -- write every results table of the report from the runs.

    python report/make_tables.py                # all tables
    python report/make_tables.py --only main    # one of: main baseline curl anchor
                                                #   sweep params grid recon
                                                #   regularisers image

Each table is written as a complete tabular environment to
report/experiments/table_<name>.tex; bare_jrnl.tex pulls it in with \\input, so
no number in the report is typed by hand. Generated rather than typed so a
table cannot drift from the runs behind it, which it had done several times.

Sources (all written by report/run_all.sh):
    main, curl      report/experiments/dissoc_1p5s_*.csv, main_curl_*.csv (1.5 s)
    baseline        report/experiments/baseline_dt20ms.csv (+ dissoc for the network)
    anchor          results/poster_rotation/*/*/run.json (3 s), matched on the
                    config stored in run.json, not on the folder name
    sweep           sens_*.csv (1.5 s, around the adopted configuration), from
                    experiments/sens/ or, failing that, report/experiments/
    grid            experiments/grid/*.csv (1.5 s)
    params          config.py
    recon           experiments/readout_fft_vs_dct/readout.csv (20 ms) and
                    experiments/readout_dt10ms/readout.csv (10 ms)
    regularisers    experiments/ablation/*.csv (1.5 s)
    image           the Table III and curl runs in results/ and the 10 ms runs
                    in experiments/best_image/ (run_experiments.image_search)
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'report', 'experiments')
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from config import THESIS_PARAMS, COOK_PARAMS, ITERS_PER_FRAME
import experiment_tracking  # run.json reader, also for older params.json runs

# display name in the report, model key
MODELS = [(r'\texttt{cook}', 'cook'), (r'\texttt{thesis}', 'thesis'),
          (r'\texttt{thesis+IMU}', 'thesis_imu'),
          (r'\texttt{CMax-anchor}', 'thesis_cmax'),
          (r'\texttt{CMax-inloop}', 'thesis_cmax_v2')]
# dataset key, display label, label in baseline npz, extra lines under the label
SEQS = [('boxes_rotation', r'\texttt{boxes}', 'boxes', []),
        ('poster_rotation', r'\texttt{poster}', 'poster', []),
        ('dynamic_rotation', r'\texttt{dynamic}', 'dynamic', ['(person)']),
        ('ecrot_bicycle', r'\texttt{bicycle}', 'bicycle', ['(synthetic)']),
        ('ecrot_street', r'\texttt{street}', 'street', ['(synthetic)'])]
SYNTHETIC = {'ecrot_bicycle', 'ecrot_street'}
# poster segments of tab:anchor, slowest last (3 s each)
ANCHOR_SEGMENTS = ['seg_A', 'seg_C', 'seg_D']
# configuration the report adopts (tab:params), marked in tab:grid
ADOPTED = (THESIS_PARAMS['delta_FR'], THESIS_PARAMS['delta_IMU'])


# ------------------------------------------------------------------ helpers
def load(patterns, root=OUTDIR):
    """All rows from every csv matching the patterns, keyed (sequence, model)."""
    rows = {}
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            for r in csv.DictReader(open(path)):
                rows[(r['sequence'], r['model'])] = r
    return rows


def load_list(pattern):
    """Every row of every csv matching the pattern, unkeyed."""
    return [r for path in sorted(glob.glob(pattern)) for r in csv.DictReader(open(path))]


def fmt(v, nd=1):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '---'
    return '---' if np.isnan(v) else f'{v:.{nd}f}'


def mark_best(cells, key, mark=r'\textbf'):
    """cells: {name: displayed string}. Wrap the best value(s) in `mark`.

    `key` maps a displayed value to a score, lower is better. Ties on the
    displayed value are all marked, so the mark never depends on digits the
    table does not show. '---' cells are never marked.
    """
    num = {k: float(v) for k, v in cells.items() if v != '---'}
    if not num:
        return dict(cells)
    best = min(key(v) for v in num.values())
    return {k: (rf'{mark}{{{v}}}' if k in num and key(num[k]) == best else v)
            for k, v in cells.items()}


def floor_label(ds):
    return r'\emph{gyro vs ref.}'


def speed(lab):
    """Mean |omega| of the reference over the 1.5 s segment, rad/s (baseline npz)."""
    path = os.path.join(OUTDIR, f'baseline_dt20ms_{lab}.npz')
    if not os.path.exists(path):
        return None
    return float(np.linalg.norm(np.load(path)['ref'], axis=1).mean())


def tabular(spec, head, body):
    return '\n'.join([rf'\begin{{tabular}}{{{spec}}}', r'\toprule', *head,
                      *body, r'\bottomrule', r'\end{tabular}', ''])


def write(name, text):
    path = os.path.join(OUTDIR, f'table_{name}.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'% generated by report/make_tables.py -- do not edit\n{text}')
    print(f'  wrote {os.path.relpath(path, ROOT)}')


#: models whose cost has an anchor term; for the others delta_IMU is unused
ANCHORED = ('thesis_imu', 'thesis_cmax')


def expected_params(model):
    p = dict(COOK_PARAMS if model == 'cook' else THESIS_PARAMS)
    if model not in ANCHORED:
        p.pop('delta_IMU', None)
    return p




# ------------------------------------------------------------------- tables
def table_main():
    """tab:main: every model on every sequence, 1.5 s, adopted parameters."""
    rows = load(['dissoc_1p5s_*.csv'])
    body = []
    for ds, disp, lab, extra in SEQS:
        sp = speed(lab)
        head = [disp] + ([f'${sp:.2f}$\\,rad/s'] if sp else []) + extra
        body += [r'\midrule',
                 r'\multirow{6}{*}{\shortstack[l]{' + r'\\'.join(head) + '}}']
        floor = None
        # displayed cells per model; bold marks the best of the five models on
        # this sequence: lowest err and dir, beta closest to 1, highest |r|
        cols = {c: {} for c in ('err', 'dir', 'beta', 'recon')}
        for mdisp, key in MODELS:
            r = rows.get((ds, key))
            if r is None:
                continue
            floor = (r['floor'], r['floor_dir'])
            cols['err'][key], cols['dir'][key] = fmt(r['err']), fmt(r['dir'])
            cols['beta'][key] = fmt(r['beta'], 0 if float(r['beta']) >= 100 else 2)
            cols['recon'][key] = fmt(r['recon'], 2)
        cols = {'err': mark_best(cols['err'], lambda v: v),
                'dir': mark_best(cols['dir'], lambda v: v),
                'beta': mark_best(cols['beta'], lambda v: abs(v - 1.0)),
                'recon': mark_best(cols['recon'], lambda v: -v)}
        for mdisp, key in MODELS:
            if key not in cols['err']:
                body.append(f' & {mdisp:<22}& --- & --- & --- & --- \\\\')
                continue
            body.append(f' & {mdisp:<22}& {cols["err"][key]} & {cols["dir"][key]} & '
                        f'{cols["beta"][key]} & {cols["recon"][key]} \\\\')
        if floor:
            nd = 2 if float(floor[0]) < 1 else 1      # sub-degree floors keep two digits
            body.append(f' & {floor_label(ds):<22}& \\emph{{{fmt(floor[0], nd)}}} & '
                        f'\\emph{{{fmt(floor[1])}}} & --- & --- \\\\')
    head = [r'Sequence & Model & err ($^\circ$/s) & dir ($^\circ$) & $\beta$ & $|r|$ \\']
    return tabular('@{}llrrrr@{}', head, body)


def table_baseline():
    """tab:baseline: CMax and gyro alone vs the CMax-anchored network, 1.5 s."""
    base = {}
    for r in csv.DictReader(open(os.path.join(OUTDIR, 'baseline_dt20ms.csv'))):
        base[(r['sequence'], r['source'])] = float(r['err'])
    net = load(['dissoc_1p5s_*.csv'])
    body = []
    for i, (ds, disp, _lab, _extra) in enumerate(SEQS):
        if i and ds in SYNTHETIC and SEQS[i - 1][0] not in SYNTHETIC:
            body.append(r'\midrule')
        cm, gy = base.get((ds, 'cmax')), base.get((ds, 'gyro'))
        nw = net.get((ds, 'thesis_cmax'))
        nw = float(nw['err']) if nw else None
        best = min(v for v in (cm, nw) if v is not None) if (cm or nw) else None
        cell = lambda v: (rf'\textbf{{{fmt(v, 2)}}}' if v is not None and v == best
                          else fmt(v, 2))
        body.append(f'{disp:<18}& {cell(cm)} & {fmt(gy, 2)} & {cell(nw)} \\\\')
    head = [r'sequence & CMax alone & gyro alone & network \\',
            r' & [$^\circ$/s] & [$^\circ$/s] & [$^\circ$/s] \\', r'\midrule']
    return tabular('@{}lccc@{}', head, body)


def table_curl():
    """tab:curl: every model, 1.5 s, without and with delta_curl = 0.6."""
    none, curl = load(['dissoc_1p5s_*.csv']), load(['main_curl_*_*.csv'])
    body = []
    for ds, disp, _lab, _extra in SEQS:
        body.append(r'\midrule')
        g = lambda r, k, nd: fmt(r[k], nd) if r else '---'
        # bold: on each sequence, the lowest error and the highest |r| of the ten runs
        err = mark_best({(key, w): g(src.get((ds, key)), 'err', 1) for _, key in MODELS
                         for w, src in (('none', none), ('curl', curl))}, lambda v: v)
        rec = mark_best({(key, w): g(src.get((ds, key)), 'recon', 2) for _, key in MODELS
                         for w, src in (('none', none), ('curl', curl))}, lambda v: -v)
        for j, (mdisp, key) in enumerate(MODELS):
            body.append(f'{disp if j == 0 else "":<22}& {mdisp:<22}& '
                        f'{err[key, "none"]} & {err[key, "curl"]} & '
                        f'{rec[key, "none"]} & {rec[key, "curl"]} \\\\')
    head = [r' & & \multicolumn{2}{c}{$\wvec$ err [$^\circ$/s]} & \multicolumn{2}{c}{$|r|$} \\',
            r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
            r'Sequence & Model & none & curl & none & curl \\']
    return tabular('@{}llrrrr@{}', head, body)


def gyro_floor(run_dir):
    """Mean error and direction error of the gyro against the reference, plus |omega|."""
    rows = list(csv.DictReader(open(os.path.join(run_dir, 'tracking.csv'))))
    g = np.array([[float(r[f'imu_w{a}']) for a in 'xyz'] for r in rows])
    t = np.array([[float(r[f'gt_w{a}']) for a in 'xyz'] for r in rows])
    cos = (g * t).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(t, axis=1))
    return (np.degrees(np.linalg.norm(g - t, axis=1)).mean(),
            np.degrees(np.arccos(np.clip(cos, -1, 1))).mean(),
            np.linalg.norm(t, axis=1).mean())


def table_anchor():
    """tab:anchor: poster, three segments of 3 s, every model plus the gyro floor."""
    body = []
    for sid in ANCHOR_SEGMENTS:
        runs = {k: experiment_tracking.find_run('results', 'poster_rotation', k, sid, 150,
                                                expected_params(k)) for _, k in MODELS}
        ref = runs.get('thesis_imu')
        fl = gyro_floor(ref) if ref else None
        body += [r'\midrule',
                 rf'\multirow{{6}}{{*}}{{${fl[2]:.2f}$}}' if fl else r'\multirow{6}{*}{---}']
        for mdisp, key in MODELS:
            d = runs[key]
            if d is None:
                print(f'    missing: poster {sid} {key}', file=sys.stderr)
                body.append(f' & {mdisp:<22}& --- & --- & --- \\\\')
                continue
            s = experiment_tracking.read_summary(d)
            body.append(f' & {mdisp:<22}& {fmt(s["mean_err_deg_s"])} & '
                        f'{fmt(s["mean_dir_err_deg"])} & '
                        f'{fmt(s["mean_beta"], 1 if s["mean_beta"] >= 100 else 2)} \\\\')
        if fl:
            body.append(rf' & {floor_label("poster_rotation")}  & \emph{{{fmt(fl[0])}}} & '
                        rf'\emph{{{fmt(fl[1])}}} & --- \\')
    head = [r'$\|\wvec\|$ & Model & err ($^\circ$/s) & dir ($^\circ$) & $\beta$ \\']
    return tabular('@{}llrrr@{}', head, body)


def _sens_rows():
    """Rows of the sensitivity sweep (sens_*.csv), keyed by model and parameters."""
    paths = (glob.glob(os.path.join('experiments', 'sens', 'sens_*.csv'))
             or glob.glob(os.path.join(OUTDIR, 'sens_*.csv')))
    cell = {}
    for path in sorted(paths):
        for r in csv.DictReader(open(path)):
            key = (r['model'],) + tuple(round(float(r[k]), 6) for k in SENS_KEYS)
            cell.setdefault(key, {})[r['sequence']] = r
    return cell


#: parameters that identify a sweep point, in the order of the cell key
SENS_KEYS = ('delta_VFG', 'delta_IG', 'delta_GI', 'delta_RF', 'delta_FR', 'delta_IMU',
             'n_iters', 'dt_ms')


def table_sweep():
    """tab:sweep: one parameter at a time around the adopted configuration, 1.5 s."""
    cell = _sens_rows()
    base = dict({k: THESIS_PARAMS[k] for k in SENS_KEYS[:6]},
                n_iters=ITERS_PER_FRAME, dt_ms=20.0)

    # The adopted value of each block is bold; the lowest error of each block
    # on each sequence is underlined (ties on the displayed value share it),
    # leaving out the ablation row, which is not a setting one would choose.
    def block(title, model, name, values, fmt_label=lambda v: f'{v:g}'):
        out = [r'\midrule', rf'\multicolumn{{12}}{{@{{}}l}}{{{title}}}\\']
        rows = []
        for v in values:
            p = dict(base, **{name: v})
            rows.append((v, cell.get((model,) + tuple(round(float(p[k]), 6) for k in SENS_KEYS), {})))
        err = {}
        for ds, *_ in SEQS:
            e = mark_best({v: fmt(rs[ds]['err']) if ds in rs else '---'
                           for v, rs in rows if not (name == 'delta_IMU' and v == 0)},
                          lambda x: x, mark=r'\underline')
            err[ds] = e
        for v, rs in rows:
            label = fmt_label(v)
            lab = rf'$\mathbf{{{label}}}$' if abs(v - base[name]) < 1e-9 else f'${label}$'
            vals = ' & '.join((f'{err[ds].get(v, fmt(rs[ds]["err"]))} & {fmt(rs[ds]["dir"])}')
                              if ds in rs else '--- & ---' for ds, *_ in SEQS)
            out.append(f'& {lab:<20}& {vals} \\\\')
        return out

    ms = lambda v: rf'{v:g}\,\mathrm{{ms}}'
    body = (block(r'\emph{anchor weight} $\delta_{\text{anchor}}$, contrast anchor', 'thesis_cmax',
                  'delta_IMU', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                  lambda v: r'0\ \text{(abl.)}' if v == 0 else f'{v:g}')
            + block(r'\emph{anchor weight} $\delta_{\text{anchor}}$, gyroscope', 'thesis_imu',
                    'delta_IMU', [0.1, 0.3, 0.5, 0.7, 0.9])
            + block(r'\emph{kinematic weight} $\delta_{F\!R}$', 'thesis_cmax',
                    'delta_FR', [0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
            + block(r'\emph{flow constraint} $\delta_{V\!F\!G}$', 'thesis_cmax',
                    'delta_VFG', [0.03, 0.06, 0.12, 0.24, 0.48])
            + block(r'\emph{flow from} $\wvec$, $\delta_{R\!F}$', 'thesis_cmax',
                    'delta_RF', [0.01, 0.02, 0.05, 0.1, 0.2])
            + block(r'\emph{gradient from intensity} $\delta_{I\!G}$', 'thesis_cmax',
                    'delta_IG', [0.03, 0.1, 0.2, 0.4])
            + block(r'\emph{intensity from gradient} $\delta_{G\!I}$', 'thesis_cmax',
                    'delta_GI', [0.02, 0.05, 0.1, 0.2])
            + block(r'\emph{iterations per frame}', 'thesis_cmax', 'n_iters', [25, 50, 75, 150])
            + block(r'\emph{accumulation window} $\Delta t$', 'thesis_cmax', 'dt_ms',
                    [5.0, 10.0, 20.0, 40.0], ms))
    head = ['& & ' + '\n& '.join(rf'\multicolumn{{2}}{{c}}{{{d}}}' for _, d, *_ in SEQS) + r' \\',
            ''.join(rf'\cmidrule(lr){{{3 + 2*i}-{4 + 2*i}}}' for i in range(len(SEQS))),
            '& & ' + ' & '.join(['err & dir'] * len(SEQS)) + r' \\']
    return tabular('@{}ll' + 'rr' * len(SEQS) + '@{}', head, body)


def table_params():
    """tab:params: the step sizes in config.py, so the table cannot disagree with the code."""
    rows = [(r'$\delta_{\Vmap\!\Fmap\!\Gmap}$', r'flow constraint~\eqref{eq:ofce}', 'delta_VFG'),
            (r'$\delta_{I\!G}$', r'$\Gmap$ from $\Imap$', 'delta_IG'),
            (r'$\delta_{G\!I}$', r'$\Imap$ from $\Gmap$', 'delta_GI'),
            (r'$\delta_{R\!F}$', r'$\Fmap$ from $\wvec$', 'delta_RF'),
            (r'$\delta_{F\!R}$', r'$\wvec$ from $\Fmap$', 'delta_FR'),
            (r'$\delta_{\text{anchor}}$', r'external $\wvec$ anchor~\eqref{eq:anchorcost}', 'delta_IMU')]
    p = lambda d, k: f'{d[k]:.2f}' if k in d else '---'
    body = [f'{s} & {rel} & {p(COOK_PARAMS, k)} & {p(THESIS_PARAMS, k)} \\\\' for s, rel, k in rows]
    body += [r'\midrule',
             rf'\multicolumn{{2}}{{@{{}}l}}{{iterations per frame}} & \multicolumn{{2}}{{c}}{{{ITERS_PER_FRAME}}} \\',
             r'\multicolumn{2}{@{}l}{window $\Delta t$} & \multicolumn{2}{c}{20\,ms} \\']
    return tabular('@{}llcc@{}', [r'Symbol & Relation & Cook & Martel \\', r'\midrule'], body)


def table_grid():
    """tab:grid: the (delta_FR, anchor) grid reduced to its optimum per sequence.

    The error depends on the two only through kappa = delta_FR / delta_anchor
    (equal kappa, equal error), so each sequence is summarised by the best cell
    and the adopted one.
    """
    rows = load_list(os.path.join('experiments', 'grid', '*.csv'))
    by = {}
    for r in rows:
        by.setdefault(r['sequence'], {})[(float(r['delta_FR']), float(r['delta_IMU']))] = float(r['err'])
    body = []
    for ds, disp, *_ in SEQS:
        g = by.get(ds)
        if not g:
            continue
        (fr, an), e = min(g.items(), key=lambda kv: kv[1])
        ad = g.get(ADOPTED)
        body.append(f'{disp:<18}& {fmt(ad, 2)} & {fmt(ADOPTED[0] / ADOPTED[1], 2)} & '
                    f'{fmt(e, 2)} & {fr:g} & {an:g} & {fmt(fr / an, 2)} \\\\')
    head = [r' & \multicolumn{2}{c}{adopted} & \multicolumn{4}{c}{grid optimum} \\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-7}',
            r'sequence & err & $\kappa$ & err & $\delta_{F\!R}$ & $\delta_{\text{anchor}}$ & $\kappa$ \\',
            r'\midrule']
    return tabular('@{}lrrrrrr@{}', head, body)


def table_recon():
    """tab:recon: mean |r| of the three read-outs at 20 and 10 ms, and fft-over-dct wins."""
    body = []
    for dt, name in ((20, 'readout_fft_vs_dct'), (10, 'readout_dt10ms')):
        path = os.path.join('experiments', name, 'readout.csv')
        if not os.path.exists(path):
            body.append(f'${dt}$\\,ms & --- & --- & --- & --- \\\\')
            continue
        r, runs = {}, {}
        for row in csv.DictReader(open(path)):
            v = abs(float(row['r_crop']))
            r.setdefault(row['variant'], []).append(v)
            runs.setdefault((row['sequence'], row['duration_s']), {})[row['variant']] = v
        mean = {k: float(np.mean(v)) for k, v in r.items()}
        best = max(mean.values())
        cell = lambda k: (rf'$\mathbf{{{mean[k]:.2f}}}$' if dt == 10 and mean[k] == best
                          else f'${mean[k]:.2f}$')
        wins = sum(v['fft'] > v['dct'] for v in runs.values())
        body.append(f'${dt}$\\,ms & {cell("iterative")} & {cell("fft")} & {cell("dct")} & '
                    f'${wins}/{len(runs)}$ \\\\')
    head = [r'$\Delta t$ & iterative~\eqref{eq:cook_poisson}',
            r'           & exact, FFT & exact, DCT & FFT $>$ DCT \\', r'\midrule']
    return tabular('@{}lcccc@{}', head, body)


#: rows of tab:regularisers: label, the term the ablation run switched on
REGULARISERS = [('none', {}),
                (r'$\delta_{\text{shrink}I}=1$ (Eq.~6.61)', {'delta_shrinkI': 1.0}),
                (r'$\delta_{\text{map}}=0.01$~\cite{guo2024emba}', {'delta_map': 0.01}),
                (r'$\delta_{\text{map}}=0.03$~\cite{guo2024emba}', {'delta_map': 0.03}),
                (r'$\delta_{\text{curl}}=0.3$ (ours)', {'delta_curl': 0.3}),
                (r'$\delta_{\text{curl}}=0.6$ (ours)', {'delta_curl': 0.6})]
#: street duplicates bicycle's trajectory, so the means leave it out
ABLATION_SEQS = ['boxes_rotation', 'poster_rotation', 'dynamic_rotation', 'ecrot_bicycle']


def table_regularisers():
    """tab:regularisers: each regulariser alone, CMax-anchor, 1.5 s, mean of four sequences."""
    rows = load_list(os.path.join('experiments', 'ablation', '*.csv'))
    terms = ('delta_shrinkI', 'delta_map', 'delta_curl')
    stats = []
    for label, on in REGULARISERS:
        want = {k: on.get(k, 0.0) for k in terms}
        hit = {r['sequence']: r for r in rows
               if r['model'] == 'thesis_cmax' and int(float(r['n_iters'])) == ITERS_PER_FRAME
               and all(abs(float(r[k]) - v) < 1e-9 for k, v in want.items())}
        got = [hit[ds] for ds in ABLATION_SEQS if ds in hit]
        mean = (lambda k: float(np.mean([abs(float(r[k])) for r in got]))
                if len(got) == len(ABLATION_SEQS) else None)
        stats.append((label, [mean('err'), mean('recon_mean'), mean('recon_inloop'),
                              mean('blowup')]))
    # bold: the best value of each column (lowest error and blow-up, highest |r|)
    pick = [min, max, max, min]
    best = [f([s[1][c] for s in stats if s[1][c] is not None], default=None)
            for c, f in enumerate(pick)]
    body = []
    for label, vals in stats:
        cells = []
        for c, (v, nd) in enumerate(zip(vals, (2, 3, 3, 3))):
            t = fmt(v, nd)
            cells.append(rf'\textbf{{{t}}}' if v is not None and v == best[c] else t)
        body.append(f'{label:<38} & ' + ' & '.join(cells) + r' \\')
    head = [r' & $\wvec$ err & $|r|$ read-out & $|r|$ in-loop & blow-up \\',
            r' & [$^\circ$/s] & & & \\', r'\midrule']
    return tabular('@{}lcccc@{}', head, body)


def table_image():
    """tab:image: the image-quality search of report_runs.image_search, 1.5 s."""
    from report_runs import IMG_SEARCH, image_search
    score, best, _ = image_search(1.5)
    body = []
    for model, title in (('thesis', r'none (\texttt{thesis})'),
                         ('thesis_cmax', r'\texttt{CMax-anchor}')):
        if body:
            body.append(r'\midrule')
        for i, cfg in enumerate(c for c in IMG_SEARCH if c[0] == model):
            cells = []
            for kind in ('real', 'synthetic'):
                if (kind, cfg) not in score:
                    cells.append('--- & ---')
                    continue
                r, err = score[kind, cfg]
                rr = rf'\textbf{{{r:.2f}}}' if cfg == best[kind] else f'{r:.2f}'
                cells.append(f'{rr} & {err:.1f}')
            body.append(f'{title if i == 0 else "":<22} & {cfg[1] * 1000:.0f} & {cfg[2]:g} & '
                        + ' & '.join(cells) + r' \\')
    head = [r' & & & \multicolumn{2}{c}{real} & \multicolumn{2}{c}{synthetic} \\',
            r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
            r'Anchor & $\Delta t$ & $\delta_{\text{curl}}$ & $|r|$ & err & $|r|$ & err \\',
            r'\midrule']
    return tabular('@{}lccrrrr@{}', head, body)


TABLES = {'main': table_main, 'baseline': table_baseline, 'curl': table_curl,
          'anchor': table_anchor, 'sweep': table_sweep, 'params': table_params,
          'grid': table_grid, 'recon': table_recon, 'regularisers': table_regularisers,
          'image': table_image}


def main():
    ap = argparse.ArgumentParser(description='write the report tables')
    ap.add_argument('--only', choices=sorted(TABLES), help='one table only')
    args = ap.parse_args()
    for name in ([args.only] if args.only else TABLES):
        write(name, TABLES[name]())


if __name__ == '__main__':
    main()
