# ω-ladder: multi-segment evaluation + plots — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `diag_omega_ladder.py` to run the existing rungs (0, 1a, 1b) across **four synthetic segments** (city & street, short & long windows), write tidy result files, and render three dataviz-grounded comparison plots — keeping the single `.py` + `.ipynb` structure.

**Architecture:** Same single-file diagnostic, parametrized by a `SEGMENTS` list instead of one `SEG`. The compute path (the rung runs) is **cluster-ready** — the user runs it on the cluster; it writes `diag_ladder_results.csv` (aggregates, one row per segment×rung) and `diag_ladder_frames.csv` (rung-0 per-frame ω for the time-series plot). A separate `--plots` mode runs locally, reads those CSVs, and saves three PNG figures. The `.ipynb` is regenerated from the `.py` `# %%` cells as before.

**Tech Stack:** Python 3 / NumPy / matplotlib 3.10; existing pipeline modules; the nb-sync converter in the scratchpad.

## Global Constraints

- **Keep the single-file structure.** One `diag_omega_ladder.py` + one `diag_omega_ladder.ipynb`; no new module files. Regenerate the notebook from the `.py` with the existing scratchpad converter after each task, and commit both.
- **Rungs unchanged.** Only rungs 0, 1a, 1b. Rung 2/3 are deferred (filed as a separate task) — do not add them here.
- **Segments (4), all dt=20ms:**
  - `city_short` — ds `ecrot_city`, t_start 0.05, 75 frames
  - `city_long` — ds `ecrot_city`, t_start 0.05, 250 frames
  - `street_short` — ds `street_sinthetic`, t_start 1.001, 75 frames
  - `street_long` — ds `street_sinthetic`, t_start 1.001, 250 frames
  - `sensor_size = (180, 240)` for all; `dist_coeffs=None`, `undistort=False` (pinhole).
- **Coverage cap:** before running a segment, read its `omega_gt` extent and cap `n_frames` so the window stays within available GT; print a `[warn]` when capped. (Local `ecrot_city` covers only ~1.5 s, so `city_long` caps to ~75 frames locally; the cluster city sequence is expected to be longer.)
- **Missing data = skip, don't crash:** if a dataset dir / `events.txt` is absent (e.g. street not downloaded locally), print `[skip]` and continue. Lets the city path verify locally while the full 4-segment run happens on the cluster.
- **Compute on the cluster:** the full run is heavy; local verification uses `--limit N` (tiny frame count). Never run the full 4-segment job locally.
- **Palette (pre-validated dataviz default, fixed order):** rung categorical colors `0=#2a78d6`, `1a=#eb6834`, `1b_free=#1baf7a`, `1b_anchored=#eda100`. P2 diverging (axis_r ∈ [−1,1], 0 = neutral): orange `#eb6834` → neutral `#f0efec` → blue `#2a78d6`, centered at 0. Node is unavailable locally so `scripts/validate_palette.js` is not run here; these are the dataviz reference palette's own slots, used in fixed order — re-run the validator on the cluster if a custom palette is ever substituted.
- **Commits:** authored by the local git config; no attribution lines; do not push.

---

## File Structure

- **Modify:** `diag_omega_ladder.py` — parametrize by `SEGMENTS`, add aggregate run + CSV output, add `--plots` mode with three figures.
- **Regenerate:** `diag_omega_ladder.ipynb` — via the scratchpad `nb_sync.py` (no manual edits).
- **Produced at runtime (not committed):** `diag_ladder_results.csv`, `diag_ladder_frames.csv`, `diag_P1_dir_beta.png`, `diag_P2_axis_heatmap.png`, `diag_P3_omega_timeseries.png`.

---

### Task 1: Parametrize the ladder by a SEGMENTS list

**Files:**
- Modify: `diag_omega_ladder.py`

**Interfaces:**
- Produces (used by Task 2/3):
  - `SEGMENTS` — list of dicts `{name, ds, t_start, dt, n_frames, sensor_size}`.
  - `cap_to_coverage(seg, om)` → int capped frame count.
  - `load_segment(seg)` → `(seq, paths, imu, om, frames, n)` or `None` if data missing.
  - `build_net(seg, seq)`, `frame_windows(seg, n)`, `ref_omega(om, t_lo, t_hi)`, `gt_flow(net, R_gt)`, `score_frames(est, ref)` (unchanged behavior, now seg-aware where needed).
  - `rung0_baseline(seg)` → `(s, est, ref)`; `rung1a_kinematics(seg)` → `(s, ok)`; `rung1b_clamped_flow(seg, anchored)` → `s`.

- [ ] **Step 1: Replace the `SEG` constant + helpers cell with the seg-parametric version**

Replace the first two `# %%` cells' segment/helper code (the `SEG = {...}` block and the helper functions `load_segment`/`frame_windows`/`ref_omega`/`build_net`/`gt_flow`/`score_frames`/`_selfcheck`) with:

```python
import os

SEGMENTS = [
    {'name': 'city_short',   'ds': 'ecrot_city',      't_start': 0.05,  'dt': 0.02, 'n_frames': 75,  'sensor_size': (180, 240)},
    {'name': 'city_long',    'ds': 'ecrot_city',      't_start': 0.05,  'dt': 0.02, 'n_frames': 250, 'sensor_size': (180, 240)},
    {'name': 'street_short', 'ds': 'street_sinthetic', 't_start': 1.001, 'dt': 0.02, 'n_frames': 75,  'sensor_size': (180, 240)},
    {'name': 'street_long',  'ds': 'street_sinthetic', 't_start': 1.001, 'dt': 0.02, 'n_frames': 250, 'sensor_size': (180, 240)},
]


def cap_to_coverage(seg, om):
    """Cap n_frames so the window stays within the available omega_gt extent."""
    n = seg['n_frames']
    if om is None:
        return n
    t_max = float(om[:, 0].max())
    max_f = int((t_max - seg['t_start']) / seg['dt'])
    if max_f < n:
        print(f"[warn] {seg['name']}: omega_gt covers to {t_max:.2f}s; "
              f"capping {n} -> {max_f} frames")
        n = max(max_f, 0)
    return n


def load_segment(seg):
    """Load one segment, or None if its data is not present on this machine."""
    paths = get_dataset_paths(seg['ds'])
    if not os.path.exists(paths['events']):
        print(f"[skip] {seg['name']}: no data at {paths['events']}")
        return None
    om = load_omega_gt(paths.get('omega_gt'))
    n = cap_to_coverage(seg, om)
    if n < 2:
        print(f"[skip] {seg['name']}: too few frames after coverage cap")
        return None
    seq = EventFrameSequence(
        paths['events'], paths['calib'],
        frame_duration=seg['dt'], t_start=seg['t_start'],
        n_frames=n, clip_value=10.0,
        sensor_size=seg['sensor_size'], undistort=False,
    )
    imu = load_imu(paths['imu'])
    frames = list(seq)
    return seq, paths, imu, om, frames, len(frames)


def frame_windows(seg, n):
    t0, dt = seg['t_start'], seg['dt']
    return [(t0 + k * dt, t0 + (k + 1) * dt) for k in range(n)]


def ref_omega(om, t_lo, t_hi):
    return omega_gt_at(om, t_lo, t_hi)


def build_net(seg, seq):
    c = seq.calib
    return InteractingMapsThesis(
        H=seq.H, W=seq.W, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy,
        frame_duration=seg['dt'],
        dist_coeffs=None, poisson=config.POISSON_MODE,
        **config.THESIS_PARAMS,
    )


def gt_flow(net, R_gt):
    return np.einsum('hwij,j->hwi', net._C_mat, R_gt)


def score_frames(est_list, ref_list):
    est = np.asarray(est_list); ref = np.asarray(ref_list)
    errs, dirs, betas = [], [], []
    for e, r in zip(est, ref):
        err, d, b = compute_metrics(e, r)
        errs.append(err); dirs.append(d); betas.append(b)
    axis_r = []
    for a in range(3):
        ea, ra = est[:, a], ref[:, a]
        ea = ea - ea.mean(); ra = ra - ra.mean()
        den = np.sqrt((ea * ea).sum() * (ra * ra).sum())
        axis_r.append(float((ea * ra).sum() / den) if den > 0 else float('nan'))
    return {
        'mean_err': float(np.mean(errs)), 'median_err': float(np.median(errs)),
        'mean_dir': float(np.mean(dirs)), 'median_dir': float(np.median(dirs)),
        'mean_beta': float(np.mean(betas)),
        'axis_r': axis_r,
        'axis_bias': (est - ref).mean(axis=0).tolist(),
    }


def _selfcheck():
    for seg in SEGMENTS:
        loaded = load_segment(seg)
        if loaded is None:
            continue
        seq, paths, imu, om, frames, n = loaded
        net = build_net(seg, seq)
        assert net._C_mat.shape == (seq.H, seq.W, 2, 3)
        w0 = ref_omega(om, *frame_windows(seg, n)[0])
        print(f"[ok] {seg['name']}: {n} frames, |omega_gt[0]|="
              f"{np.linalg.norm(w0):.3f} rad/s")
```

- [ ] **Step 2: Rewrite the three rung functions to take `seg` and apply `--limit`**

Replace the bodies of `rung0_baseline`, `rung1a_kinematics`, and `rung1b_clamped_flow` so each takes `seg` (and reads a module-level `LIMIT` for a fast local smoke test). Add `LIMIT = None` near the top of the first cell.

```python
def _nlim(n):
    return n if LIMIT is None else min(n, LIMIT)


def rung0_baseline(seg):
    loaded = load_segment(seg)
    if loaded is None:
        return None, None, None
    seq, paths, imu, om, frames, n = loaded
    n = _nlim(n); frames = frames[:n]
    wins = frame_windows(seg, n)
    net = build_net(seg, seq)
    R_init = get_gyro_for_frame(imu, *wins[0]) * seg['dt']
    net.initialize_from_rotation(R_init)
    cond_M = float(np.linalg.cond(build_R_normal_equations(net._C_mat)))
    est, ref, curls = [], [], []
    for (V, _t), (t_lo, t_hi) in zip(frames, wins):
        net.step(V, n_iters=config.ITERS_PER_FRAME)
        est.append(net.R / seg['dt'])
        ref.append(ref_omega(om, t_lo, t_hi))
        curls.append(curl_share(net.G))
    s = score_frames(est, ref)
    s['cond_M'] = cond_M
    s['mean_curl_share'] = float(np.mean(curls))
    print(f"[rung0] {seg['name']:12s} err={s['mean_err']:6.2f} "
          f"dir={s['mean_dir']:6.2f} beta={s['mean_beta']:.3f} "
          f"axis_r={[round(v,2) for v in s['axis_r']]}")
    return s, np.asarray(est), np.asarray(ref)


def rung1a_kinematics(seg):
    loaded = load_segment(seg)
    if loaded is None:
        return None, False
    seq, paths, imu, om, frames, n = loaded
    n = _nlim(n)
    wins = frame_windows(seg, n)
    net = build_net(seg, seq)
    C = net._C_mat; M_inv = build_R_normal_equations(C); dt = seg['dt']
    est, ref = [], []
    for (t_lo, t_hi) in wins:
        w = ref_omega(om, t_lo, t_hi)
        R_hat = solve_R_lstsq(M_inv, C, gt_flow(net, w * dt))
        est.append(R_hat / dt); ref.append(w)
    s = score_frames(est, ref)
    s['cond_M'] = float(np.linalg.cond(M_inv))
    ok = s['mean_dir'] < 1.0 and abs(s['mean_beta'] - 1.0) < 0.05
    print(f"[rung1a] {seg['name']:12s} dir={s['mean_dir']:.4f} "
          f"beta={s['mean_beta']:.4f} [{'ok' if ok else 'FAIL'}]")
    return s, ok


def rung1b_clamped_flow(seg, anchored):
    loaded = load_segment(seg)
    if loaded is None:
        return None
    seq, paths, imu, om, frames, n = loaded
    n = _nlim(n); frames = frames[:n]
    wins = frame_windows(seg, n)
    net = build_net(seg, seq); dt = seg['dt']
    net.initialize_from_rotation(get_gyro_for_frame(imu, *wins[0]) * dt)
    net.q_F.update = lambda lr: None
    est, ref = [], []; max_drift = 0.0
    for (V, _t), (t_lo, t_hi) in zip(frames, wins):
        w = ref_omega(om, t_lo, t_hi)
        F_star = gt_flow(net, w * dt)
        net.q_F.value = F_star.copy()
        net.step(V, n_iters=config.ITERS_PER_FRAME,
                 omega_imu=(w if anchored else None))
        max_drift = max(max_drift, float(np.max(np.abs(net.q_F.value - F_star))))
        est.append(net.R / dt); ref.append(w)
    s = score_frames(est, ref)
    s['cond_M'] = float(np.linalg.cond(build_R_normal_equations(net._C_mat)))
    tag = 'anchored' if anchored else 'free'
    print(f"[rung1b/{tag}] {seg['name']:12s} err={s['mean_err']:6.2f} "
          f"dir={s['mean_dir']:6.2f} beta={s['mean_beta']:.3f} "
          f"(clamp drift {max_drift:.1e})")
    return s
```

- [ ] **Step 3: Point `__main__` setup at the new `_selfcheck()` and regenerate the notebook**

Leave the `if __name__ == '__main__':` block from the previous plan in place for now (Task 2 replaces it). Regenerate the notebook:

Run: `python "<scratchpad>/nb_sync.py"`
(where `<scratchpad>` is `C:/Users/gia80970/AppData/Local/Temp/claude/C--Users-gia80970-dev-EventVision-InteractingMaps/f31d0ce0-972c-469e-b947-fef6463d29b7/scratchpad`)

- [ ] **Step 4: Verify setup iterates segments (city ok, street skipped locally)**

Run: `python diag_omega_ladder.py --rung setup`
Expected: `[ok] city_short: 75 frames ...`, a `[warn]` capping `city_long` to ~75 frames locally then `[ok] city_long: ...`, and `[skip] street_short` / `[skip] street_long` (no local data). Exit 0.

- [ ] **Step 5: Commit**

```bash
git add diag_omega_ladder.py diag_omega_ladder.ipynb
git commit -m "Diagnostic: parametrize omega ladder by a SEGMENTS list (city/street, short/long)"
```

---

### Task 2: Aggregate run across segments + CSV outputs

**Files:**
- Modify: `diag_omega_ladder.py`

**Interfaces:**
- Consumes: `SEGMENTS`, `rung0_baseline`, `rung1a_kinematics`, `rung1b_clamped_flow`.
- Produces: `RESULTS_CSV='diag_ladder_results.csv'`, `FRAMES_CSV='diag_ladder_frames.csv'`, `run_all_segments(out_results, out_frames)`.

- [ ] **Step 1: Add the aggregate runner that writes both CSVs**

Add a new `# %%` cell before the `__main__` block:

```python
import csv

RESULTS_CSV = 'diag_ladder_results.csv'
FRAMES_CSV = 'diag_ladder_frames.csv'

RESULTS_HEADER = ['segment', 'ds', 'n_frames', 'rung', 'mean_err', 'median_err',
                  'mean_dir', 'median_dir', 'mean_beta',
                  'axis_r_x', 'axis_r_y', 'axis_r_z', 'cond_M', 'curl_share']


def _row(seg, n, rung, s):
    return [seg['name'], seg['ds'], n, rung,
            round(s['mean_err'], 4), round(s['median_err'], 4),
            round(s['mean_dir'], 4), round(s['median_dir'], 4),
            round(s['mean_beta'], 4),
            round(s['axis_r'][0], 4), round(s['axis_r'][1], 4),
            round(s['axis_r'][2], 4),
            round(s.get('cond_M', float('nan')), 3),
            round(s['mean_curl_share'], 4) if 'mean_curl_share' in s else '']


def run_all_segments(out_results=RESULTS_CSV, out_frames=FRAMES_CSV):
    rrows, frows = [], []
    for seg in SEGMENTS:
        s0, est, ref = rung0_baseline(seg)
        if s0 is None:
            continue
        n = len(est)
        rrows.append(_row(seg, n, '0', s0))
        for k in range(n):
            frows.append([seg['name'], k,
                          est[k, 0], est[k, 1], est[k, 2],
                          ref[k, 0], ref[k, 1], ref[k, 2]])
        s1a, _ok = rung1a_kinematics(seg)
        if s1a is not None:
            rrows.append(_row(seg, n, '1a', s1a))
        sfree = rung1b_clamped_flow(seg, anchored=False)
        if sfree is not None:
            rrows.append(_row(seg, n, '1b_free', sfree))
        sanch = rung1b_clamped_flow(seg, anchored=True)
        if sanch is not None:
            rrows.append(_row(seg, n, '1b_anchored', sanch))
    with open(out_results, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(RESULTS_HEADER); w.writerows(rrows)
    with open(out_frames, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['segment', 'frame', 'est_x', 'est_y', 'est_z',
                    'gt_x', 'gt_y', 'gt_z'])
        w.writerows(frows)
    print(f"\n[ok] wrote {out_results} ({len(rrows)} rows) and "
          f"{out_frames} ({len(frows)} rows)")
```

- [ ] **Step 2: Replace the `__main__` block to add `--run`, `--limit`, `--plots`**

Replace the entire `if __name__ == '__main__':` block with:

```python
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='run',
                    choices=['setup', 'run', 'plots'])
    ap.add_argument('--limit', type=int, default=None,
                    help='cap frames per segment (fast local smoke test)')
    args = ap.parse_args()
    LIMIT = args.limit
    if args.mode == 'setup':
        _selfcheck()
    elif args.mode == 'run':
        run_all_segments()
    elif args.mode == 'plots':
        make_all_plots()
```

> `LIMIT` is read inside `_nlim`; assigning it in `__main__` sets the module global. `make_all_plots` is defined in Task 3 — until then, run with `--mode run`/`setup` only.

- [ ] **Step 3: Regenerate notebook and verify a fast local run writes CSVs**

Run: `python "<scratchpad>/nb_sync.py"`
Run: `python diag_omega_ladder.py --mode run --limit 3`
Expected: city rows print `[rung0] city_short ...`, `[rung1a] ... [ok]`, `[rung1b/free] ...`, `[rung1b/anchored] ...` for city_short and city_long; street `[skip]` lines; final `[ok] wrote diag_ladder_results.csv (...) and diag_ladder_frames.csv (...)`. Confirm both CSVs exist and the header matches `RESULTS_HEADER`.

- [ ] **Step 4: Commit**

```bash
git add diag_omega_ladder.py diag_omega_ladder.ipynb
git commit -m "Diagnostic: aggregate multi-segment run writing results + frames CSVs"
```

---

### Task 3: The three dataviz plots (`--plots` mode)

**Files:**
- Modify: `diag_omega_ladder.py`

**Interfaces:**
- Consumes: `RESULTS_CSV`, `FRAMES_CSV`.
- Produces: `make_all_plots()` → saves `diag_P1_dir_beta.png`, `diag_P2_axis_heatmap.png`, `diag_P3_omega_timeseries.png`.

- [ ] **Step 1: Add the plotting cell (three figures, validated palette)**

Add a new `# %%` cell before the `__main__` block:

```python
RUNG_ORDER = ['0', '1a', '1b_free', '1b_anchored']
RUNG_COLORS = {'0': '#2a78d6', '1a': '#eb6834',
               '1b_free': '#1baf7a', '1b_anchored': '#eda100'}
RUNG_LABEL = {'0': 'rung 0 (vision-only)', '1a': 'rung 1a (kinematics)',
              '1b_free': 'rung 1b (flow, free)',
              '1b_anchored': 'rung 1b (flow, anchored)'}


def _read_results(path=RESULTS_CSV):
    import csv
    rows = list(csv.DictReader(open(path, newline='')))
    segs = []
    for r in rows:
        if r['segment'] not in segs:
            segs.append(r['segment'])
    return rows, segs


def plot_P1(path=RESULTS_CSV, out='diag_P1_dir_beta.png'):
    """Grouped bars: dir (deg) and beta by segment, grouped by rung.
    Shows the oracle rungs collapsing to ~0 across every scene."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, segs = _read_results(path)
    val = {(r['segment'], r['rung']): r for r in rows}
    x = np.arange(len(segs)); w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [(axes[0], 'mean_dir', 'Direction error (deg)'),
                              (axes[1], 'mean_beta', 'Scale  beta  (1.0 = exact)')]:
        for i, rung in enumerate(RUNG_ORDER):
            ys = [float(val[(s, rung)][metric]) if (s, rung) in val else 0.0
                  for s in segs]
            ax.bar(x + (i - 1.5) * w, ys, w, color=RUNG_COLORS[rung],
                   label=RUNG_LABEL[rung])
        ax.set_xticks(x); ax.set_xticklabels(segs, rotation=20, ha='right')
        ax.set_title(title); ax.spines[['top', 'right']].set_visible(False)
    axes[1].axhline(1.0, color='#52514e', lw=1, ls='--')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}")


def plot_P2(path=RESULTS_CSV, out='diag_P2_axis_heatmap.png'):
    """Heatmap segment x axis of rung-0 per-axis correlation axis_r.
    Diverging (−1..1, 0 neutral): reveals a recurring dead axis across scenes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rows, segs = _read_results(path)
    r0 = {r['segment']: r for r in rows if r['rung'] == '0'}
    segs = [s for s in segs if s in r0]
    M = np.array([[float(r0[s]['axis_r_x']), float(r0[s]['axis_r_y']),
                   float(r0[s]['axis_r_z'])] for s in segs])
    cmap = LinearSegmentedColormap.from_list(
        'div', ['#eb6834', '#f0efec', '#2a78d6'])
    fig, ax = plt.subplots(figsize=(5, 0.7 * len(segs) + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['ωx', 'ωy', 'ωz'])
    ax.set_yticks(range(len(segs))); ax.set_yticklabels(segs)
    for i in range(len(segs)):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center',
                    color='#0b0b0b', fontsize=9)
    ax.set_title('Rung-0 per-axis correlation of ω\u0302 with ω')
    fig.colorbar(im, ax=ax, label='correlation')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}")


def plot_P3(path=FRAMES_CSV, out='diag_P3_omega_timeseries.png'):
    """Small multiples: rung-0 estimated vs GT omega per axis over frames,
    one row per segment, one column per axis. Shows which axis diverges."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    rows = list(csv.DictReader(open(path, newline='')))
    segs = []
    for r in rows:
        if r['segment'] not in segs:
            segs.append(r['segment'])
    fig, axes = plt.subplots(len(segs), 3, figsize=(12, 2.6 * len(segs)),
                             squeeze=False)
    axkeys = [('est_x', 'gt_x', 'ωx'), ('est_y', 'gt_y', 'ωy'),
              ('est_z', 'gt_z', 'ωz')]
    for i, seg in enumerate(segs):
        sr = [r for r in rows if r['segment'] == seg]
        k = [int(r['frame']) for r in sr]
        for j, (ek, gk, title) in enumerate(axkeys):
            ax = axes[i][j]
            ax.plot(k, [float(r[gk]) for r in sr], color='#52514e', lw=1.5,
                    label='GT')
            ax.plot(k, [float(r[ek]) for r in sr], color='#2a78d6', lw=1.5,
                    label='est')
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(seg, fontsize=9)
    axes[0][2].legend(frameon=False, fontsize=8)
    fig.suptitle('Rung 0: estimated vs ground-truth ω (rad/s) over frames')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}")


def make_all_plots():
    plot_P1(); plot_P2(); plot_P3()
```

- [ ] **Step 2: Regenerate notebook and verify the plots render from the CSVs**

Run: `python "<scratchpad>/nb_sync.py"`
Run: `python diag_omega_ladder.py --mode plots`
Expected: `[ok] diag_P1_dir_beta.png`, `[ok] diag_P2_axis_heatmap.png`, `[ok] diag_P3_omega_timeseries.png`, exit 0. (Uses the CSVs from the Task 2 `--limit 3` run — city rows only locally; the full 4-segment figures come from the cluster run.)

- [ ] **Step 3: Open the three PNGs and eyeball them (dataviz check 7)**

Confirm no label collisions/overflow, legends present, β reference line at 1.0, heatmap annotations readable. Fix layout if needed and re-run.

- [ ] **Step 4: Commit**

```bash
git add diag_omega_ladder.py diag_omega_ladder.ipynb
git commit -m "Diagnostic: add P1/P2/P3 comparison plots and --plots mode"
```

---

## Self-Review

**1. Spec coverage** (against the narrowed brainstorm): multiple datasets + longer sequences → `SEGMENTS` (city/street × short/long) in Task 1. ✓ The 3 plots → Task 3 (P1 grouped bars, P2 diverging heatmap, P3 time-series small multiples). ✓ Keep single-file `.py`/`.ipynb` → all changes in `diag_omega_ladder.py`, notebook regenerated. ✓ Cluster-ready + local safety → `--limit`, `[skip]` on missing data, coverage cap. ✓ Rung 2/3 deferred → not in plan (filed separately). ✓

**2. Placeholder scan:** No TBD/TODO. `make_all_plots` is referenced in Task 2's `__main__` but defined in Task 3; Task 2 Step 2's note says to run only `run`/`setup` until Task 3 lands — an ordering note, not a placeholder. All code blocks are complete.

**3. Type consistency:** `rung0_baseline` returns `(s, est, ref)`; `run_all_segments` unpacks exactly that and indexes `est[k,0..2]`. `rung1a_kinematics` returns `(s, ok)`; unpacked as `s1a, _ok`. `rung1b_clamped_flow` returns `s`. `_row` reads `s['axis_r']` (list of 3), `s.get('cond_M')`, `s['mean_curl_share']` (rung 0 only) — all set by `score_frames`/the rung functions. CSV `RESULTS_HEADER` columns match `_row`'s order (14 fields). Plot readers key on `segment`/`rung`/`mean_dir`/`mean_beta`/`axis_r_{x,y,z}` and the frames columns `est_*`/`gt_*` — all present in the writers. `RUNG_ORDER` values match the `rung` labels written by `run_all_segments` (`'0'`,`'1a'`,`'1b_free'`,`'1b_anchored'`). ✓
