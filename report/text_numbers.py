"""
text_numbers.py -- every number the report's text quotes that no table holds.

The tables come from make_tables.py. The text also quotes numbers worked out
from the same runs (one frame of a track, a ratio of two runs, a correlation
over segments, ...). This script recomputes each of them from the runs and
writes them, grouped by the section that quotes them, to
report/experiments/text_numbers.txt. It runs nothing: it only reads what
run_all.sh produced.

    python report/text_numbers.py                       # the runs in this checkout
    python report/text_numbers.py --runs results_19.11 --csv results_19.11/report_experiments
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'report'))
os.chdir(ROOT)

import experiment_tracking
from config import THESIS_PARAMS
from report_runs import SEGMENTS, BETA_SEGMENTS, BETA_CONFIGS, ADOPTED, SYNTHETIC

RUNS, CSV = '.', os.path.join('report', 'experiments')
REAL = [s for s in SEGMENTS if s[3] not in SYNTHETIC]
SYN = [s for s in SEGMENTS if s[3] in SYNTHETIC]
#: one-at-a-time axes of the sens sweep and their adopted values
BASE = {'dt_ms': 20.0, 'n_iters': 75.0,
        **{k: THESIS_PARAMS[k] for k in ('delta_VFG', 'delta_IG', 'delta_GI', 'delta_RF',
                                         'delta_FR', 'delta_IMU')}}
OUT = []


def say(*parts):
    line = ' '.join(str(p) for p in parts)
    print(line)
    OUT.append(line)


def head(title):
    say('')
    say(f'== {title}')


# ------------------------------------------------------------------ loading
def run(seg, model, nf=75, d_fr=None, d_anchor='adopted', curl=0.0, root='results'):
    """tracking.csv of one run as arrays, or None if the run is missing."""
    ds, sid, t0 = seg[:3]
    params = {'delta_FR': ADOPTED['delta_FR'] if d_fr is None else d_fr,
              'delta_shrinkI': 0.0, 'delta_map': 0.0, 'delta_curl': curl}
    if model in ('thesis_imu', 'thesis_cmax') and d_anchor is not None:
        params['delta_IMU'] = ADOPTED['delta_IMU'] if d_anchor == 'adopted' else d_anchor
    d = experiment_tracking.find_run(os.path.join(RUNS, root), ds, model, sid, nf, params,
                                     dt=0.02, t_start=t0)
    if d is None:
        say(f'   (missing run: {ds}/{sid} {model} n{nf} {params})')
        return None
    rows = list(csv.DictReader(open(os.path.join(d, 'tracking.csv'))))
    col = lambda *k: np.array([[float(r[c]) for c in k] for r in rows])
    return {'est': col('est_wx', 'est_wy', 'est_wz'), 'gt': col('gt_wx', 'gt_wy', 'gt_wz'),
            'imu': col('imu_wx', 'imu_wy', 'imu_wz'), 'err': col('err_deg_s')[:, 0],
            'dir': col('dir_err_deg')[:, 0], 'beta': col('beta')[:, 0],
            'r': np.abs(col('recon_r')[:, 0]), 'I_std': col('I_std')[:, 0]}


def rows_of(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        out += list(csv.DictReader(open(p)))
    return out


def sens_rows():
    """sens_*.csv of the one-at-a-time sweep: experiments/sens/, else the csv folder."""
    return (rows_of(os.path.join(RUNS, 'experiments', 'sens', 'sens_*.csv'))
            or rows_of(os.path.join(CSV, 'sens_*.csv')))


def axis(rows, key, seq, model='thesis_cmax'):
    """{value: row} of the rows of one sequence that vary `key` alone."""
    out = {}
    for r in rows:
        if r['sequence'] != seq or r['model'] != model:
            continue
        if all(abs(float(r[k]) - v) < 1e-9 for k, v in BASE.items() if k != key):
            out[float(r[key])] = r
    return dict(sorted(out.items()))


def f(r, k='err'):
    return float(r[k])


def deg(a, b):
    """Angle between two vectors, degrees."""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def mag(v):
    return np.linalg.norm(v, axis=-1)


DEG = np.degrees(1.0)


# ------------------------------------------------------------------- evaluation
def reference():
    """Evaluation, Angular velocity: the gyroscope against the reference, per frame."""
    from config import get_dataset_paths
    from interacting_maps import eval_io as io
    head('Evaluation: the gyroscope against a one-frame and a +-50 ms reference (75 frames)')
    segs = REAL + [s + ('',) for s in BETA_SEGMENTS if s not in [r[:3] for r in REAL]] + SYN
    for seg in segs:
        ds, sid, t0 = seg[:3]
        p = get_dataset_paths(ds)
        imu, gt, om = (io.load_imu(p['imu']), io.load_groundtruth(p['groundtruth']),
                       io.load_omega_gt(p.get('omega_gt')))
        one, wide = [], []
        for k in range(75):
            lo, hi = t0 + k * 0.02, t0 + (k + 1) * 0.02
            g = io.get_gyro_for_frame(imu, lo, hi)
            if om is not None:                     # exact omega over the frame itself
                m = (om[:, 0] >= lo) & (om[:, 0] < hi)
                ref1 = om[m, 1:4].mean(0) if m.any() else io.omega_gt_at(om, lo, hi)
            else:
                ref1 = io.gt_omega_body(gt, lo, hi)
            one.append(np.linalg.norm(g - ref1) * DEG)
            wide.append(np.linalg.norm(g - io.get_reference_omega(gt, imu, lo, hi, om)[0]) * DEG)
        say(f'{ds[:14]:14s} {sid}: gyro vs one-frame reference {np.mean(one):5.2f} deg/s, '
            f'vs +-50 ms {np.mean(wide):5.2f}')


def segment_speeds():
    """Evaluation, Segments: is each reported segment the one of median speed?"""
    from config import DATASET_SEGMENTS, get_dataset_paths
    from interacting_maps import eval_io as io
    head('Evaluation: gyroscope speed of every candidate segment over 150 frames (* = reported)')
    for ds in dict.fromkeys(s[0] for s in REAL):
        imu = io.load_imu(get_dataset_paths(ds)['imu'])
        chosen = [s[1] for s in REAL if s[0] == ds]
        speeds = {}
        for c in DATASET_SEGMENTS[ds]:
            w = np.array([io.get_gyro_for_frame(imu, c['t_start'] + k * 0.02,
                                                c['t_start'] + (k + 1) * 0.02)
                          for k in range(150)])
            m = mag(w) * DEG
            speeds[c['id']] = m.mean()
            say(f"{ds[:14]:14s} {c['id']}{'*' if c['id'] in chosen else ' '} "
                f"mean |w| {m.mean():6.1f} deg/s, std/mean {m.std() / m.mean():.2f}, "
                f"range {m.min():5.1f}-{m.max():5.1f}")
        order = sorted(speeds, key=speeds.get)
        n = len(order)
        med = order[(n - 1) // 2: n // 2 + 1]
        say(f"   median-speed segment(s): {', '.join(med)}; reported: {', '.join(chosen)}")


# ------------------------------------------------------- Comparison of the models
def unanchored():
    head('Comparison: unanchored models (cook, thesis), 1.5 s runs of Table main')
    for seg in REAL + SYN:
        for m in ('cook', 'thesis'):
            t = run(seg, m)
            if t is None:
                continue
            e, g = t['est'], t['gt']
            m0 = np.linalg.norm(e[0])
            say(f'{seg[3]:8s} {m:7s} |w| within {100*np.max(np.abs(mag(e)/m0 - 1)):5.1f}% '
                f'of frame 0, direction within {max(deg(x, e[0]) for x in e):5.1f} deg; '
                f'frame 0 |w|/gyro = {m0/np.linalg.norm(t["imu"][0]):.3f}; '
                f'last/first |w| = {mag(e)[-1]/m0:.3f}')
        g = run(seg, 'thesis')
        if g is not None:
            say(f'{seg[3]:8s} reference: axis turns up to {max(deg(x, g["gt"][0]) for x in g["gt"]):5.1f} '
                f'deg, speed {mag(g["gt"]).min():.2f}-{mag(g["gt"]).max():.2f} rad/s')


def reversal():
    head('Comparison: the frame of largest CMax-anchor error on each real segment')
    for seg in REAL:
        c, i = run(seg, 'thesis_cmax'), run(seg, 'thesis_imu')
        if c is None or i is None:
            continue
        k = int(np.argmax(c['err']))
        keep = np.arange(len(c['err'])) != k
        ax = int(np.argmax(np.abs(c['gt']).mean(0)))          # dominant rotation axis
        near = c['gt'][max(k - 3, 0):k + 4, ax]
        flip = bool((np.diff(np.sign(near)) != 0).any())
        say(f'{seg[3]:8s} frame {k}: |w_ref| = {mag(c["gt"])[k]:.3f} rad/s '
            f'({"dominant axis reverses within 3 frames" if flip else "no reversal within 3 frames"}; '
            f'{int((mag(c["gt"]) <= 0.2).sum())} frames of the track at <= 0.2 rad/s); '
            f'err CMax-anchor {c["err"][k]:6.1f}, thesis+IMU {i["err"][k]:4.1f} deg/s; '
            f'mean without it: CMax-anchor {c["err"][keep].mean():5.2f}, '
            f'thesis+IMU {i["err"][keep].mean():5.2f}, gap {c["err"][keep].mean()-i["err"][keep].mean():5.2f} '
            f'(all frames: gap {c["err"].mean()-i["err"].mean():5.2f})')


def anchors_across_weights():
    head('Comparison: the two anchors at every anchor weight (sens sweep, 1.5 s)')
    rows = sens_rows()
    worst = 0.0
    for seg in SEGMENTS:
        c, i = axis(rows, 'delta_IMU', seg[0]), axis(rows, 'delta_IMU', seg[0], 'thesis_imu')
        diffs = {w: f(c[w]) - f(i[w]) for w in c if w in i and w > 0}
        if seg in REAL:
            worst = max([worst] + [abs(v) for v in diffs.values()])
        say(f'{seg[3]:8s} CMax minus gyro: ' + ', '.join(f'{w:g}: {v:+.1f}' for w, v in diffs.items())
            + (f';  no anchor: {f(c[0.0]):.1f}' if 0.0 in c else ''))
    say(f'largest |CMax - gyro| on the real sequences: {worst:.1f} deg/s')


def poster_segments():
    head('Comparison: the poster segments, 3 s (fastest-segment gap, error relative to speed)')
    for seg in BETA_SEGMENTS:
        c = run(seg, 'thesis_cmax', nf=150, d_fr=0.1, d_anchor=0.5)   # the BETA_CONFIGS series
        i = run(seg, 'thesis_imu', nf=150) if seg[0] == 'poster_rotation' else None
        if c is None:
            continue
        w = mag(c['gt']).mean()
        floor = mag(c['imu'] - c['gt']).mean() * DEG
        line = (f'{seg[0][:6]} {seg[1]}: |w| {w:.2f} rad/s; CMax-anchor {c["err"].mean():5.2f} deg/s '
                f'= {100*c["err"].mean()/(w*DEG):4.1f}% of |w|; gyro floor {floor:4.2f} '
                f'= {100*floor/(w*DEG):4.1f}%')
        if i is not None:
            line += f'; thesis+IMU {i["err"].mean():5.2f}, gap {c["err"].mean()-i["err"].mean():4.2f}'
        say(line)


def beta_speed():
    head('What the maps add: beta against speed, seven segments, 3 s (fig_beta_speed)')
    data = {}
    for cfg in BETA_CONFIGS:
        model, d_fr, d_a = cfg
        pts = []
        for seg in BETA_SEGMENTS:
            t = run(seg, model, nf=150, d_fr=d_fr, d_anchor=d_a)
            if t is not None:
                pts.append((mag(t['gt']).mean(), t['beta'].mean(), t['err'].mean(),
                            t['dir'].mean(), t['est']))
        data[cfg] = pts
        if len(pts) < 3:
            continue
        w, b = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        r, p = stats.pearsonr(w, b)
        say(f'{model} dFR {d_fr} anchor {d_a}: r = {r:+.2f} (p = {p:.3g}, n = {len(w)}), '
            f'beta {b.min():.2f}-{b.max():.2f}, beta at slowest {b[np.argmin(w)]:.2f}, '
            f'at fastest {b[np.argmax(w)]:.2f}; speeds {w.min():.2f}-{w.max():.2f} rad/s')
    orig, adopt, inl = (data.get(c, []) for c in BETA_CONFIGS)
    if len(orig) >= 2:
        fit = np.polyfit([p[0] for p in orig], [p[1] for p in orig], 1)
        syn = [run(s, 'thesis_cmax') for s in SYN]
        ws = np.mean([mag(t['gt']).mean() for t in syn if t is not None])
        say(f'original fit predicts beta {np.polyval(fit, ws):.2f} at the synthetic {ws:.2f} rad/s')
    if len(inl) == len(orig) == len(adopt):
        for (o, a, v, seg) in zip(orig, adopt, inl, BETA_SEGMENTS):
            say(f'{seg[0][:6]} {seg[1]}: dir CMax-anchor {o[3]:.1f} / {a[3]:.1f}, CMax-inloop {v[3]:.1f}; '
                f'err CMax-anchor {o[2]:.1f} / {a[2]:.1f}, CMax-inloop {v[2]:.1f} (original / adopted)')


def inloop_identical():
    head('Comparison: CMax-inloop omega in runs that differ only in the maps')
    for seg in SEGMENTS:
        a, b = run(seg, 'thesis_cmax_v2'), run(seg, 'thesis_cmax_v2', curl=0.6)
        if a is not None and b is not None:
            say(f'{seg[3]:8s} curl 0 vs 0.6 (1.5 s): max |dw| = {np.abs(a["est"]-b["est"]).max():.2e} rad/s')
    for seg in [b for b in BETA_SEGMENTS if b in [s[:3] for s in SEGMENTS]]:   # 3 s main_* runs exist
        a = run(seg, 'thesis_cmax_v2', nf=150, d_fr=0.5, d_anchor=None)
        b = run(seg, 'thesis_cmax_v2', nf=150)
        if a is not None and b is not None:
            say(f'{seg[0][:6]} {seg[1]}: dFR 0.5 vs {ADOPTED["delta_FR"]} (3 s): '
                f'max |dw| = {np.abs(a["est"]-b["est"]).max():.2e} rad/s')


def synthetic_beta_original():
    head('What the maps add: synthetic beta at the original step sizes (sweep base point, 3 s)')
    for s in SYN:
        for r in rows_of(os.path.join(RUNS, 'experiments', 'sweep', f'sweep_{s[3]}_cmax_delta_FR=0.5.csv')):
            say(f'{s[3]:8s} dFR {r["delta_FR"]} anchor {r["delta_IMU"]}: beta {f(r, "beta"):.3f}, '
                f'err {f(r):.2f}')


# ------------------------------------------------------------- what the maps add
def maps_add():
    head('What the maps add: network against its own anchor (baseline, 1.5 s)')
    base = {(r['sequence'], r['source']): r for r in rows_of(os.path.join(CSV, 'baseline_dt20ms.csv'))}
    net = {r['sequence']: r for r in rows_of(os.path.join(CSV, 'dissoc_1p5s_*_thesis_cmax.csv'))}
    for seg in SEGMENTS:
        b, n = base.get((seg[0], 'cmax')), net.get(seg[0])
        if b and n:
            say(f'{seg[3]:8s} CMax alone {f(b):.2f} (median {f(b, "median"):.2f}), network {f(n):.2f} '
                f'(median {f(n, "median"):.2f}): mean {100*(1-f(n)/f(b)):+.0f}%, '
                f'median {100*(1-f(n, "median")/f(b, "median")):+.0f}%')
    rows = sens_rows()
    say('delta_VFG (CMax-anchor):')
    for seg in SEGMENTS:
        a, b = axis(rows, 'delta_VFG', seg[0]), base.get((seg[0], 'cmax'))
        if not a or not b:
            continue
        lo, hi = a[min(a)], a[max(a)]
        dirs = [f(r, 'dir') for r in a.values()]
        say(f'  {seg[3]:8s} at {min(a):g}: err {f(lo):.2f} vs CMax alone {f(b):.2f} '
            f'(diff {f(lo)-f(b):+.2f}); dir range {max(dirs)-min(dirs):.2f} deg; '
            f'beta {f(lo, "beta"):.2f} -> {f(hi, "beta"):.2f}; at {max(a):g} '
            f'{100*(1-f(hi)/f(b)):+.0f}% vs CMax alone')


def kappa_rows():
    head('What the maps add: kappa in the one-at-a-time sweep')
    rows = sens_rows()
    for seg in SEGMENTS:
        an, fr = axis(rows, 'delta_IMU', seg[0]), axis(rows, 'delta_FR', seg[0])
        if 0.1 in an and 0.5 in fr:
            b = [f(r, 'beta') for r in fr.values()]
            mono = all(x < y for x, y in zip(b, b[1:]))
            say(f'{seg[3]:8s} anchor 0.1 vs dFR 0.5 (kappa 1): {f(an[0.1]):.2f} vs {f(fr[0.5]):.2f}; '
                f'beta over dFR {min(fr):g}-{max(fr):g}: {b[0]:.2f} -> {b[-1]:.2f}, '
                f'{"rises at every step" if mono else "NOT monotonic"}')


# ------------------------------------------------------------------- sensitivity
def grid():
    head('Sensitivity: the 5x5 grid of delta_FR and delta_anchor (CMax-anchor, 1.5 s)')
    rows = rows_of(os.path.join(RUNS, 'experiments', 'grid', '*.csv'))
    if not rows:
        say('   (no grid csv)')
        return
    tab = {}
    for r in rows:
        an = r.get('delta_IMU', r.get('delta_anchor'))     # older grid csv: delta_anchor
        tab[(r['sequence'], round(f(r, 'delta_FR'), 4), round(float(an), 4))] = f(r)
    seqs = [s for s in SEGMENTS if any(k[0] == s[0] for k in tab)]
    spread = 0.0
    for s in seqs:
        by_k = {}
        for (q, fr, an), e in tab.items():
            if q == s[0]:
                by_k.setdefault(round(fr / an, 3), []).append(e)
        own = max([0.0] + [max(v) - min(v) for v in by_k.values() if len(v) > 1])
        spread = max(spread, own)
        ks = sorted(by_k)
        mean = {k: np.mean(by_k[k]) for k in ks}
        best = min(ks, key=mean.get)
        errs = [mean[k] for k in ks]
        mono = all(x <= y for x, y in zip(errs, errs[1:]))
        ad = tab.get((s[0], ADOPTED['delta_FR'], ADOPTED['delta_IMU']))
        say(f'{s[3]:8s} best kappa {best:g} ({mean[best]:.2f}); smallest kappa {ks[0]:g} '
            f'({mean[ks[0]]:.2f}){" error rises with kappa throughout" if mono else ""}; '
            f'adopted {ad:.2f}; kappa 0.6 (0.3/0.5) {tab.get((s[0], 0.3, 0.5), np.nan):.2f}; '
            f'start (0.5/0.3) {tab.get((s[0], 0.5, 0.3), np.nan):.2f}; '
            f'equal-kappa spread {own:.2f}')
    say(f'largest spread between settings of equal kappa: {spread:.2f} deg/s')
    for name, group in (('real', REAL), ('synthetic', SYN)):
        m = lambda fr, an: np.mean([tab[(s[0], fr, an)] for s in group if (s[0], fr, an) in tab])
        say(f'{name} mean: adopted {m(ADOPTED["delta_FR"], ADOPTED["delta_IMU"]):.2f}, '
            f'start {m(0.5, 0.3):.2f}, kappa 0.6 {m(0.3, 0.5):.2f}')


def sweep_ends():
    head('Sensitivity: which end of each range is best (one-at-a-time sweep, CMax-anchor)')
    rows = sens_rows()
    for key in ('delta_FR', 'delta_IMU', 'delta_VFG', 'delta_RF', 'delta_IG', 'delta_GI',
                'n_iters', 'dt_ms'):
        cells = []
        for s in SEGMENTS:
            a = {k: v for k, v in axis(rows, key, s[0]).items() if not (key == 'delta_IMU' and k == 0)}
            if a:
                e = {k: f(v) for k, v in a.items()}
                cells.append(f'{s[3]} best {min(e, key=e.get):g} (range {max(e.values())-min(e.values()):.2f})')
        say(f'{key:10s} ' + '; '.join(cells))


def anchor_ablation():
    head('Sensitivity: anchor weight (sens sweep, 1.5 s)')
    rows = sens_rows()
    for model in ('thesis_cmax', 'thesis_imu'):
        for s in SEGMENTS:
            a = axis(rows, 'delta_IMU', s[0], model)
            say(f'{model:11s} {s[3]:8s} ' + ', '.join(f'{k:g}: {f(v):.1f}' for k, v in a.items()))
    for model in ('thesis_cmax', 'thesis_imu'):
        syn = [axis(rows, 'delta_IMU', s[0], model) for s in SYN]
        ws = sorted(set.intersection(*[set(a) for a in syn])) if syn else []
        if ws:
            say(f'{model} synthetic mean: ' + ', '.join(f'{w:g}: {np.mean([f(a[w]) for a in syn]):.1f}' for w in ws))


def little_effect_and_window():
    head('Sensitivity: delta_GI and the reconstruction; the accumulation window')
    rows = sens_rows()
    gi = [axis(rows, 'delta_GI', s[0]) for s in SEGMENTS]
    if all(gi):
        lo, hi = min(gi[0]), max(gi[0])
        say(f'delta_GI {lo:g} -> {hi:g}: |r| ' + ', '.join(
            f'{s[3]} {f(a[lo], "recon_mean"):.2f}->{f(a[hi], "recon_mean"):.2f}' for s, a in zip(SEGMENTS, gi))
            + f'; mean {np.mean([f(a[lo], "recon_mean") for a in gi]):.2f} -> '
              f'{np.mean([f(a[hi], "recon_mean") for a in gi]):.2f}; curl share on the real ones '
              f'{100*np.mean([f(a[lo], "curl") for a, s in zip(gi, SEGMENTS) if s in REAL]):.0f}% -> '
              f'{100*np.mean([f(a[hi], "curl") for a, s in zip(gi, SEGMENTS) if s in REAL]):.0f}%')
    its = {s: axis(rows, 'n_iters', s[0]) for s in SEGMENTS}
    if all(its.values()):
        say('iterations per frame (err deg/s | read-out |r| | in-loop |r|):')
        for s in SEGMENTS:
            say(f'  {s[3]:8s} ' + ' | '.join(f'{k:g}: {f(r):5.2f} {f(r, "recon_mean"):.2f} {f(r, "recon_inloop"):.2f}'
                                            for k, r in its[s].items()))
        say('  largest change over the iteration counts: err '
            f'{max(max(f(r) for r in a.values()) - min(f(r) for r in a.values()) for a in its.values()):.2f} deg/s, read-out |r| '
            f'{max(max(f(r, "recon_mean") for r in a.values()) - min(f(r, "recon_mean") for r in a.values()) for a in its.values()):.3f}; '
            'in-loop |r| rises with the count on every sequence: '
            + str(all(np.all(np.diff([f(r, "recon_inloop") for r in a.values()]) > 0) for a in its.values())))
    dts = {s: axis(rows, 'dt_ms', s[0]) for s in SEGMENTS}
    for dt in sorted(set().union(*[set(a) for a in dts.values()])):
        for name, group in (('real', REAL), ('synthetic', SYN)):
            g = [dts[s][dt] for s in group if dt in dts[s]]
            if g:
                say(f'dt {dt:g} ms {name:9s}: err {np.mean([f(r) for r in g]):6.1f}, '
                    f'beta {np.mean([f(r, "beta") for r in g]):9.3g}, |r| {np.mean([f(r, "recon_mean") for r in g]):.2f}, '
                    f'curl share ' + ', '.join(f'{s[3]} {100*f(dts[s][dt], "curl"):.0f}%' for s in group if dt in dts[s]))



def sensitivity_both():
    head('Sensitivity section and Fig. fig_sensitivity: angular velocity and image per swept value '
         '(CMax-anchor unless named, 1.5 s)')
    say('value: err real/synthetic (deg/s) | read-out |r| real/synthetic | in-loop |r| real/synthetic '
        '| fraction of frames in an excursion real/synthetic')

    def line(label, rows):
        cells = []
        for k, fmt in (('err', '5.1f'), ('recon_mean', '.2f'), ('recon_inloop', '.2f'), ('blowup', '.3f')):
            m = [np.mean([f(r, k) for r in rows if r['sequence'] in [s[0] for s in g]]) for g in (REAL, SYN)]
            cells.append(f'{m[0]:{fmt}} {m[1]:{fmt}}')
        say(f'  {label:<12s} ' + ' | '.join(cells))

    grid = [r for r in rows_of(os.path.join(RUNS, 'experiments', 'grid', '*.csv')) if r['model'] == 'thesis_cmax']
    by_k = {}
    for r in grid:
        an = r.get('delta_IMU', r.get('delta_anchor'))
        by_k.setdefault(round(f(r, 'delta_FR') / float(an), 4), []).append(r)
    if by_k:
        say('grid, by kappa = delta_FR / delta_anchor:')
        for k in sorted(by_k):
            line(f'kappa {k:g}', by_k[k])
        spread = 0.0
        for rs in by_k.values():
            for s in SEGMENTS:
                v = [f(r, 'recon_mean') for r in rs if r['sequence'] == s[0]]
                if len(v) > 1:
                    spread = max(spread, max(v) - min(v))
        say(f'  largest |r| spread between settings of equal kappa: {spread:.3f}')
    rows = sens_rows()
    for model, keys in (('thesis_cmax', ('delta_IMU', 'delta_FR', 'delta_VFG', 'delta_RF', 'delta_IG',
                                         'delta_GI', 'n_iters', 'dt_ms')),
                        ('thesis_imu', ('delta_IMU',))):
        for key in keys:
            by_v = {}
            for s in SEGMENTS:
                for v, r in axis(rows, key, s[0], model).items():
                    by_v.setdefault(v, []).append(r)
            if by_v:
                say(f'{model} {key}:')
                for v in sorted(by_v):
                    line(f'{v:g}', by_v[v])
    lo, hi = 0.01, BASE['delta_RF']
    cells = []
    for s in REAL:
        a = axis(rows, 'delta_RF', s[0])
        if lo in a and hi in a:
            cells.append(f'{s[3]} err {f(a[lo]):.1f} vs {f(a[hi]):.1f}, |r| {f(a[lo], "recon_mean"):.2f} vs '
                         f'{f(a[hi], "recon_mean"):.2f}')
    if cells:
        say(f'delta_RF {lo:g} vs {hi:g} per real sequence: ' + '; '.join(cells))

# ------------------------------------------------------------ intensity sections
def intensity():
    head('Intensity: poster track of the anchored and the unanchored network (1.5 s)')
    poster = [s for s in SEGMENTS if s[3] == 'poster'][0]
    c, t = run(poster, 'thesis_cmax'), run(poster, 'thesis')
    if c is not None and t is not None:
        k = int(np.argmax(c['r']))
        say(f'CMax-anchor I std {c["I_std"].min():.2f}-{c["I_std"].max():.2f}; '
            f'thesis (no anchor) {t["I_std"].min():.2f}-{t["I_std"].max():.2f}')
        b = c['I_std'] > 5 * np.median(c['I_std'])
        say(f'CMax-anchor omega error on the {b.sum()} frames with I std above 5x its median '
            f'{c["err"][b].mean():.1f} deg/s, on the other {(~b).sum()} frames {c["err"][~b].mean():.1f} deg/s')
        say(f'CMax-anchor I std: median {np.median(c["I_std"]):.2f}, above 5x the median at frames '
            f'{np.where(b)[0].min()}-{np.where(b)[0].max()}, peak {c["I_std"].max():.2f} at frame {int(np.argmax(c["I_std"]))}')
        say(f'CMax-anchor |r|: highest at frame {k} ({c["r"][k]:.2f}), track mean {c["r"].mean():.2f}')
    say('frames with I std above 5x its median, per model (Table main runs, 1.5 s):')
    for s in SEGMENTS:
        n = {m: run(s, m) for m in ('cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2')}
        say(f'  {s[3]:8s} ' + ', '.join(f'{m} {int((r["I_std"] > 5 * np.median(r["I_std"])).sum())}/{len(r["I_std"])}'
                                       for m, r in n.items() if r is not None))
    allr = [run(poster, m) for m in ('cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2')]
    if all(a is not None for a in allr):
        m = np.mean([a['r'] for a in allr], axis=0)
        say(f'mean |r| over the five models: highest at frame {int(np.argmax(m))}')
    say('curl share of G, CMax-anchor at the adopted configuration (sens base point):')
    rows = sens_rows()
    for s in SEGMENTS:
        a = axis(rows, 'delta_FR', s[0])
        if ADOPTED['delta_FR'] in a:
            say(f'  {s[3]:8s} {100*f(a[ADOPTED["delta_FR"]], "curl"):.0f}%')
    win = rows_of(os.path.join(CSV, 'window_choice.csv'))
    if win:
        for name, labs in (('real', [s[3] for s in REAL]), ('synthetic', [s[3] for s in SYN])):
            w = [f(r, 'omega_mean') for r in win if r['sequence'] in labs and f(r, 'duration_s') == 1.0]
            say(f'image travel per 20 ms window, centre pixel ({name}): {199.0*np.mean(w)*0.02:.1f} px')
    head('Intensity: 10 against 20 ms on the same read-outs (readout jobs)')
    r20 = rows_of(os.path.join(RUNS, 'experiments', 'readout_fft_vs_dct', 'readout.csv'))
    r10 = rows_of(os.path.join(RUNS, 'experiments', 'readout_dt10ms', 'readout.csv'))
    key = lambda r: (r['sequence'], r['duration_s'], r['variant'])
    a, b = {key(r): abs(f(r, 'r_crop')) for r in r20}, {key(r): abs(f(r, 'r_crop')) for r in r10}
    for v in ('iterative', 'fft', 'dct'):
        ks = [k for k in a if k[2] == v and k in b]
        if ks:
            say(f'{v:9s}: 10 ms better on {sum(b[k] > a[k] for k in ks)}/{len(ks)} runs; '
                f'mean {np.mean([a[k] for k in ks]):.2f} -> {np.mean([b[k] for k in ks]):.2f}')
    for name, tab in (('20 ms', a), ('10 ms', b)):
        pairs = {(k[0], k[1]) for k in tab}
        for x, y in (('fft', 'iterative'), ('dct', 'iterative'), ('fft', 'dct')):
            n = [p for p in pairs if (p + (x,)) in tab and (p + (y,)) in tab]
            if n:
                say(f'{name}: {x} beats {y} on {sum(tab[p + (x,)] > tab[p + (y,)] for p in n)}/{len(n)}')


def intensity_update():
    """Intensity update: the exact solve inside the loop (the poisson job)."""
    head('Intensity update: iterative, fft and dct inside the loop (poisson job)')
    aps_ref = None
    for s in SEGMENTS:
        ds, sid, t0, lab = s
        runs = {}
        for d in sorted(glob.glob(os.path.join(RUNS, 'experiments', 'poisson_solvers',
                                               ds, 'thesis_cmax', f'{sid}_*'))):
            try:
                runs[experiment_tracking.read_config(d).get('poisson', 'iterative')] = d
            except (OSError, ValueError):
                continue
        for mode in ('iterative', 'fft', 'dct'):
            if mode not in runs:
                continue
            c = experiment_tracking.read_config(runs[mode])
            sm = experiment_tracking.read_summary(runs[mode])
            say(f'{lab:8s} {mode:9s} ({c["n_frames"] * c["frame_duration"]:.1f} s): '
                f'err {sm["mean_err_deg_s"]:.2f}, low-freq share of the final I '
                f'{sm["low_freq_share"]:.3f}, track-mean |r| in-loop {sm["recon_r_mean_inloop"]:.2f} '
                f'read-out {sm["recon_r_mean"]:.2f}, median I std {sm["I_std_median"]:.2f}, '
                f'blow-up {sm["blowup_frac"]:.2f}')
        if runs and lab == 'poster':
            try:
                import evaluation as E
                from report_runs import aps_frame
                c = experiment_tracking.read_config(runs['iterative'])
                aps = aps_frame(ds, t0 + c['n_frames'] * c['frame_duration'])
                say(f'{lab:8s} APS frame at the end of the track: low-freq share '
                    f'{E._intensity_stats(np.log(aps + 1.0))["low_freq_share"]:.3f}')
            except Exception as e:      # no APS images in this checkout
                say(f'   (APS frame not available: {e})')


def regularisers():
    """Regularisation of the Maps: per-sequence rows of the ablation (CMax-anchor, 1.5 s)."""
    head('Regularisation: each term alone, per sequence (ablation job)')
    rows = rows_of(os.path.join(RUNS, 'experiments', 'ablation', 'sweep_*.csv'))
    terms = (('none', {}), ('shrinkI=1', {'delta_shrinkI': 1.0}),
             ('map=0.01', {'delta_map': 0.01}), ('map=0.03', {'delta_map': 0.03}),
             ('curl=0.3', {'delta_curl': 0.3}), ('curl=0.6', {'delta_curl': 0.6}))
    reg = ('delta_shrinkI', 'delta_map', 'delta_curl')
    for s in SEGMENTS:
        mine = [r for r in rows if r['sequence'] == s[0]]
        for name, on in terms:
            hit = [r for r in mine if all(abs(f(r, k) - on.get(k, 0.0)) < 1e-9 for k in reg)]
            if hit:
                r = hit[0]
                say(f'{s[3]:8s} {name:10s} err {f(r):6.2f}  |r| read-out {f(r, "recon_mean"):.3f}  '
                    f'in-loop {f(r, "recon_inloop"):.3f}  blow-up {f(r, "blowup"):.2f}  '
                    f'curl share {f(r, "curl"):.2f}')
    say('curl share of the final G, CMax-anchor, Table curl runs (none -> curl 0.6):')
    for s in SEGMENTS:
        a = run_dir(s, 'thesis_cmax', curl=0.0)
        b = run_dir(s, 'thesis_cmax', curl=0.6)
        if a and b:
            say(f'  {s[3]:8s} {experiment_tracking.read_summary(a)["curl_share"]:.2f} -> '
                f'{experiment_tracking.read_summary(b)["curl_share"]:.2f}')


def run_dir(seg, model, curl=0.0, nf=75):
    """Folder of one Table III / Table curl run, or None."""
    ds, sid, t0 = seg[:3]
    params = {'delta_FR': ADOPTED['delta_FR'], 'delta_curl': curl, 'delta_map': 0.0,
              'delta_shrinkI': 0.0}
    if model in ('thesis_imu', 'thesis_cmax'):
        params['delta_IMU'] = ADOPTED['delta_IMU']
    return experiment_tracking.find_run(os.path.join(RUNS, 'results'), ds, model, sid, nf,
                                        params, dt=0.02, t_start=t0)


def polarity():
    """Appendix: the inverted frames of the unanchored models (Figs. appendix cook, thesis)."""
    head('Appendix: sign of the read-out at the frames shown, unanchored models')
    from interacting_maps.network_dissertation import solve_poisson_exact
    from interacting_maps.metrics import crop_interior
    from report_runs import aps_frame
    for s in REAL:
        ds, sid, t0, lab = s
        root, nf = ((os.path.join('experiments', 'maps_poster'), 44) if lab == 'poster'
                    else ('results', 75))
        for m in ('cook', 'thesis'):
            d = experiment_tracking.find_run(os.path.join(RUNS, root), ds, m, sid, nf,
                                             {'delta_FR': ADOPTED['delta_FR'], 'delta_curl': 0.0,
                                              'delta_map': 0.0, 'delta_shrinkI': 0.0},
                                             dt=0.02, t_start=t0)
            if d is None or not os.path.exists(os.path.join(d, 'maps_final.npz')):
                say(f'   (missing maps: {ds} {m} n{nf})')
                continue
            try:
                aps = np.log(aps_frame(ds, t0 + (nf - 0.5) * 0.02).astype(np.float64) + 1.0)
            except Exception as e:      # no APS images in this checkout
                say(f'   (APS frame not available: {e})')
                return
            img = solve_poisson_exact(np.load(os.path.join(d, 'maps_final.npz'))['G'], 'fft')
            a, b = crop_interior(img).ravel(), crop_interior(aps).ravel()
            a, b = a - a.mean(), b - b.mean()
            r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
            last = list(csv.DictReader(open(os.path.join(d, 'tracking.csv'))))[-1]
            e = np.array([float(last[f'est_w{k}']) for k in 'xyz'])
            g = np.array([float(last[f'gt_w{k}']) for k in 'xyz'])
            cos = float(e @ g / (np.linalg.norm(e) * np.linalg.norm(g)))
            say(f'{lab:8s} {m:7s} frame {int(last["frame"])}: signed r of the read-out {r:+.2f}, '
                f'cosine between estimated and true omega {cos:+.2f}')


def track_length():
    """Sensitivity: track length. The 3 s runs of every model cut after N frames; the network is causal,
    so their first 75 frames are the 1.5 s runs of Table main (checked below)."""
    head('Track length: mean error [deg/s] and read-out |r| over the first N frames of the 3 s runs')
    Ns = (25, 50, 75, 100, 150)
    same = []
    for model in ('cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2'):
        say(model)
        for s in SEGMENTS:
            long_, short = run(s, model, nf=150), run(s, model)
            if long_ is None:
                continue
            if short is not None:
                same.append(np.array_equal(long_['err'][:75], short['err']) and np.array_equal(long_['r'][:75], short['r']))
            say(f'  {s[3]:8s} ' + '  '.join(f'{n * 0.02:.1f}s {long_["err"][:n].mean():6.1f} {long_["r"][:n].mean():.2f}'
                                            for n in Ns))
    say(f'first 75 frames of the 3 s runs identical to the 1.5 s runs: {sum(same)} of {len(same)}')
    T = {(m, s[3]): run(s, m, nf=150) for m in ('cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2')
         for s in SEGMENTS}
    if any(v is None for v in T.values()):
        return
    E = lambda m, lab, n: T[m, lab]['err'][:n].mean()
    Rr = lambda m, lab, n: T[m, lab]['r'][:n].mean()
    real = [s[3] for s in SEGMENTS if s in REAL]
    labs = [s[3] for s in SEGMENTS]
    say('anchored (thesis_imu, thesis_cmax), largest change of the error from 1.5 to 3 s: '
        f'{max(abs(E(m, l, 150) - E(m, l, 75)) for m in ("thesis_imu", "thesis_cmax") for l in labs):.1f} deg/s')
    order = lambda l, n: sorted(('thesis_imu', 'thesis_cmax', 'thesis_cmax_v2'), key=lambda m: E(m, l, n))
    say('order of the three anchored models the same at 0.5, 1.5 and 3 s on every sequence: '
        + str(all(order(s[3], n) == order(s[3], 75) for s in SEGMENTS for n in (25, 150))))
    for n in (25, 75, 150):
        v = [E(m, l, n) for m in ('cook', 'thesis') for l in real]
        say(f'unanchored on the real sequences over {n * 0.02:g} s: {min(v):.1f}-{max(v):.1f} deg/s')
    for n in (25, 75):
        v = [T[m, l]['dir'][:n].mean() for m in ('cook', 'thesis') for l in real]
        say(f'unanchored direction error on the real sequences over {n * 0.02:g} s: {min(v):.0f}-{max(v):.0f} deg')
    say('unanchored error over 1.5 s as a share of the 3 s value, real sequences: at least '
        f'{min(E(m, l, 75) / E(m, l, 150) for m in ("cook", "thesis") for l in real):.0%}')
    anch = ('thesis_imu', 'thesis_cmax', 'thesis_cmax_v2')
    say('anchored read-out lowest over 0.5 s on every real sequence: '
        + str(all(Rr(m, l, 25) < min(Rr(m, l, 75), Rr(m, l, 150)) for m in anch for l in real))
        + f'; largest change from 1.5 to 3 s {max(abs(Rr(m, l, 150) - Rr(m, l, 75)) for m in anch for l in labs):.2f}')
    say('best read-out over 3 s per sequence: ' + ', '.join(
        f'{s[3]} {max(T, key=lambda k: Rr(*k, 150) if k[1] == s[3] else -1)[0]} {max(Rr(m, s[3], 150) for m in ("cook", "thesis", *anch)):.2f}'
        for s in SEGMENTS))
    wc = os.path.join(CSV, 'window_choice.csv')
    if os.path.exists(wc):
        rows = list(csv.DictReader(open(wc)))
        say('sign changes of the dominant axis of the reference (run_experiments window): ' + '; '.join(
            f'{lab} ' + ', '.join(f'{r["duration_s"]} s {r["sign_flips"]}' for r in rows if r['sequence'] == lab
                                  and float(r['duration_s']) in (0.5, 1.5, 3.0)) for lab in real))


def image_window():
    """Image-Optimised Configuration: halving the window, per sequence, with and without the anchor."""
    head('Image configuration: track-mean |r| of the read-out at 20 -> 10 ms, per sequence (best_image job)')
    from report_runs import image_search, track_stats
    _, _, run_dir = image_search(1.5)
    for ds, sid, t0, lab in SEGMENTS:
        parts = []
        for model in ('thesis', 'thesis_cmax'):
            for curl in (0.0, 0.6):
                r20 = track_stats(run_dir(ds, sid, t0, model, 0.02, curl))[0]
                r10 = track_stats(run_dir(ds, sid, t0, model, 0.01, curl))[0]
                parts.append(f'{model} curl {curl:g}: {r20:.2f} -> {r10:.2f}')
        say(f'  {lab:8s} ' + ' | '.join(parts))


SECTIONS = [reference, segment_speeds, unanchored, reversal, anchors_across_weights, poster_segments, inloop_identical,
            maps_add, kappa_rows, beta_speed, synthetic_beta_original, grid, sweep_ends,
            sensitivity_both, anchor_ablation, little_effect_and_window, intensity, intensity_update,
            regularisers, track_length, image_window, polarity]


def main():
    global RUNS, CSV
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='.', help='folder holding results/ and experiments/')
    ap.add_argument('--csv', default=os.path.join('report', 'experiments'),
                    help='folder holding the csv files of run_experiments.py')
    a = ap.parse_args()
    RUNS, CSV = a.runs, a.csv
    for fn in SECTIONS:
        try:
            fn()
        except Exception as e:                        # one missing input must not hide the rest
            say(f'   ({fn.__name__} failed: {type(e).__name__}: {e})')
    path = os.path.join(CSV, 'text_numbers.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(OUT) + '\n')
    print(f'\n-> {path}')


if __name__ == '__main__':
    main()
