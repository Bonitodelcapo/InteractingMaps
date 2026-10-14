# G-flipping (OFCE step instability) Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the "indicative, not proven" one-pixel explanation of G flipping its sign every iteration into a measured result in the full network, and test whether that instability is what produces rung 0's β ≈ 3 scale error.

**Architecture:** Everything lives in the existing `--mode ofce` path of `diag_omega_ladder.py` (seven variants: `δ_VFG` sweep, no clip, normalised step, dt 10/5 ms), which already ran on the cluster. Task 1 reads those existing results. Only if they leave the mechanism open (gate after Task 1) do Tasks 2–5 add three measurements to the same runs: the per-pixel contraction factor of G along F against the linear prediction, flip rates binned by |F|/bound instead of px, and a row with F held fixed. These feed three figures, a notebook step and a findings write-up.

**Tech Stack:** Python 3 / NumPy / matplotlib; `interacting_maps.style` (`use_style`, `save`, `SCREEN_WIDTH`); existing diagnostic helpers (`stability_bound`, `flip_rate`, `_map_run`, `_ofce_g_only`, `run_ofce_variant`, `ofce_summary`).

## Background (what is already established)

From `docs/superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md`, "Findings":
- `Cost_OFCE` moves G by an explicit gradient step. Linearised per pixel along F (g = F̂·G), each iteration multiplies the error of g by **λ = 1 − 2·δ_VFG·|F|² − δ_IG**.
- Stable iff |λ| < 1, i.e. |F| < b = √((2−δ_IG)/(2δ_VFG)) ≈ 2.8 px/frame. Sign flips (λ < 0) start at |F| ≈ 1.9 px/frame. With x = |F|/b this is **λ(x) = 1 − δ_IG − (2 − δ_IG)·x²**, which is the same curve for every δ_VFG and dt. Flips start at x₀ = √((1−δ_IG)/(2−δ_IG)) ≈ 0.69.
- Measured: G flips on 0.80–0.99 of iterations where F is the true flow (rows 1b, 1c), and 0.23 above the bound in row 0. The true flow on `city_short` is ~4–10 px/frame at dt = 20 ms.

**Research questions**

- **RQ1 (mechanism):** In the full network, does G's per-pixel contraction factor along F follow λ(x)? Expected pattern: on the curve for x < 1, pinned near −1 above it (clip-bounded), below −1 without the clip.
- **RQ2 (scaling):** Do flip rates collapse onto one curve in x across δ_VFG and dt, with onset at x₀ ≈ 0.69?
- **RQ3 (consequence):** Does the instability cap |F| in the free network (row 0) near the bound, so that β ≈ max(1, mean|F\*|/b)? If so, the stable variants (normalised step, dt = 5 ms) should give β ≈ 1.

## Global Constraints

- **Do not change the model.** No edits to `interacting_maps/*.py` or `config.py`; `THESIS_PARAMS` drive every reported number. Only `diag_omega_ladder.py`, `diag_omega_ladder.ipynb` and docs change.
- **Compute runs on the cluster.** Locally only `--limit` smoke runs. Delete local smoke outputs in `diag_ofce/` after verifying; the directory is git-ignored (`.gitignore`: `/diag_ofce/`).
- **Do not edit `report/run_all.sh`** (it has uncommitted user edits). Its jobs `ofce_<variant>` (DIAG) and `ofce_summary` (FIGURES) already cover this work. Changing `diag_omega_ladder.py` changes `HASH_DIAG`, so the OFCE jobs rerun on the next `bash report/run_all.sh`.
- **Code-quality rules (mandatory):** reuse existing functions; new functions atomic; minimal diffs; import only from core modules.
- **Tests follow the repo idiom:** standalone `[ok]`/`[FAIL]` self-checks inside `_selfcheck()`, run with `python diag_omega_ladder.py --mode setup --limit 3`. Not pytest.
- **Notebook is authored, not generated:** when a function changes, update the cells that call it (memory: dual `.py` + `.ipynb`, walkthrough pattern).
- **Commits:** authored by the local git config, no attribution lines, never push.

---

## File Structure

- **Modify:** `diag_omega_ladder.py`
  - OFCE section: `_g_par`, `contraction`, `predicted_lambda`, `flip_onset`, `_xbin_stats`, `_xbin_occ`, `XBINS`, `HEAD_ITERS`, new `OFCE_HEADER` columns, `run_ofce_variant` (adds row `1b_free`, head recording, new columns), `plot_ofce_findings`, `VARIANT_COLORS`, `ofce_summary` (calls the plot).
  - `_ofce_g_only` gains an `on_iter` hook; new `_selfcheck_contraction`; `__main__` sets Agg for `ofce_summary`.
- **Modify:** `diag_omega_ladder.ipynb` — one new step (Step 11) before "Across segments".
- **Modify:** `docs/superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md` — the measured findings (Task 6).

---

### Task 1: Read the existing cluster results (no code)

**Files:** none (analysis only). Input: the cluster's `diag_ofce/summary.csv` and `diag_ofce/E*.png`.

**Interfaces:** Produces a decision: are Tasks 2–5 needed?

The existing summary has, per variant × row (`1c_R`, `0`): `bound_px`, `frac_Fgt_above`, `F_epe_rel`, `stripe_I`, `G_at_clip`, `finite`, `mean_err`, `mean_dir`, `mean_beta`, and flip rates in px bins `flip_0_1 … flip_6_inf`.

- [ ] **Step 1: Check every variant ran**

On the cluster:
```bash
python diag_omega_ladder.py --mode ofce_summary
```
Expected: one printed row per variant × row, 7 variants × 2 rows = 14. If `E4_dt10` / `E4_dt05` are missing, run them:
```bash
python diag_omega_ladder.py --mode ofce --variant E4_dt10
```
```bash
python diag_omega_ladder.py --mode ofce --variant E4_dt05
```
then re-run `--mode ofce_summary`. Copy `diag_ofce/summary.csv` and `diag_ofce/E*.png` to the local `diag_ofce/`.

- [ ] **Step 2: RQ3 first look — does rung-0 β follow the bound?**

```bash
python - <<'PY'
import csv, numpy as np
rows = [r for r in csv.DictReader(open('diag_ofce/summary.csv')) if r['rung'] == '0']
print(f"{'variant':14s}{'bound px':>9s}{'beta':>7s}{'stripe_I':>9s}{'F_epe_rel':>10s}")
for r in rows:
    print(f"{r['variant']:14s}{float(r['bound_px']):9.2f}{float(r['mean_beta']):7.2f}"
          f"{float(r['stripe_I']):9.3f}{float(r['F_epe_rel']):10.3f}")
PY
```
Read it against these predictions. If the bound sets β: E1_dVFG024 (b ≈ 2.0) has the largest β, E1_dVFG006 (b ≈ 4.0) a smaller β than E0_base (b ≈ 2.8), and E3_normalize and E4_dt05 have β ≈ 1.

- [ ] **Step 3: RQ2 first look — does the flip onset move with the bound?**

```bash
python - <<'PY'
import csv
rows = list(csv.DictReader(open('diag_ofce/summary.csv')))
fc = [k for k in rows[0] if k.startswith('flip_')]
print(f"{'variant':14s}{'row':6s}{'bound':>6s}  " + ' '.join(f"{k[5:]:>8s}" for k in fc))
for r in rows:
    print(f"{r['variant']:14s}{r['rung']:6s}{float(r['bound_px']):6.2f}  "
          + ' '.join(f"{r[k] or '-':>8s}" for k in fc))
PY
```
If the theory holds, flips start near 0.69·bound: about 1.4 px for E1_dVFG024, 1.9 px for E0_base, 2.7 px for E1_dVFG006.

- [ ] **Step 4: Decide (gate)**

- **RQ1–RQ3 all clearly answered** (β tracks the bound, onsets move as predicted, the stable variants give β ≈ 1): skip Tasks 2–5 and go to Task 6 with these numbers.
- **Otherwise** (px bins too coarse to see the onset, rows with F unfixed blur the contraction, or β does not follow): do Tasks 2–6.

Record the decision and the printed tables in the session before continuing.

---

### Task 2: Contraction factor — helpers and self-checks

**Files:**
- Modify: `diag_omega_ladder.py` (OFCE section: `flip_rate` and below; `_ofce_g_only`; `_selfcheck`)

**Interfaces:**
- Produces:
  - `_g_par(tail) -> ndarray (T, H, W)`: G projected on F̂ at each recorded iteration; `tail` is a list of `(G (H,W,2), F (H,W,2))`.
  - `contraction(tail, eps=1e-12) -> ndarray (H, W)`: per-pixel λ̂, nan where g has stopped moving.
  - `predicted_lambda(x, delta_IG) -> ndarray`: 1 − δ_IG − (2 − δ_IG)·x².
  - `flip_onset(delta_IG) -> float`: x₀ = √((1−δ_IG)/(2−δ_IG)).
  - `_ofce_g_only(F_mag, normalize, n_iters=200, V=1.0, on_iter=None)`: `on_iter(G, F)` after every iteration (copies).

- [ ] **Step 1: Write the self-check (fails: the functions do not exist yet)**

Add after `_selfcheck_ofce_step` in `diag_omega_ladder.py`:
```python
def _selfcheck_contraction():
    """contraction() recovers lam from a linear sequence and from the real
    one-pixel OFCE step (no spatial term there: lam = 1 - 2 delta_VFG |F|^2);
    predicted_lambda is 0 at flip_onset and -1 at the bound."""
    F = np.zeros((1, 1, 2)); F[..., 0] = 2.0
    for lam in (0.5, -0.8):
        tail = []
        for t in range(12):
            G = np.zeros((1, 1, 2)); G[..., 0] = 0.3 + 0.7 * lam ** t
            tail.append((G, F))
        got = float(contraction(tail)[0, 0])
        print(f"  [{'ok' if abs(got - lam) < 1e-9 else 'FAIL'}] contraction recovers "
              f"lam={lam} (got {got:.6f})")
    d = config.THESIS_PARAMS['delta_VFG']
    b = 1.0 / np.sqrt(d)
    for F_mag in (0.5 * b, 0.8 * b):
        tail = []
        _ofce_g_only(F_mag, False, n_iters=12, on_iter=lambda G, F: tail.append((G, F)))
        want = 1.0 - 2.0 * d * F_mag ** 2
        got = float(contraction(tail)[0, 0])
        print(f"  [{'ok' if abs(got - want) < 1e-6 else 'FAIL'}] one-pixel OFCE step "
              f"|F|={F_mag:.2f}: contraction {got:.4f} vs 1-2*delta_VFG*|F|^2 = {want:.4f}")
    dIG = config.THESIS_PARAMS['delta_IG']
    ok = (abs(predicted_lambda(1.0, dIG) + 1.0) < 1e-12
          and abs(predicted_lambda(flip_onset(dIG), dIG)) < 1e-12)
    print(f"  [{'ok' if ok else 'FAIL'}] predicted_lambda: -1 at the bound, "
          f"0 at flip_onset = {flip_onset(dIG):.3f}")
```
In `_selfcheck()`, after the existing `_selfcheck_ofce_step()` call, add:
```python
    _selfcheck_contraction()
```

- [ ] **Step 2: Run it and see it fail**

Run: `python diag_omega_ladder.py --mode setup --limit 3`
Expected: `NameError: name 'contraction' is not defined`.

- [ ] **Step 3: Implement the helpers**

Add `import warnings` to the top-level imports. Replace `flip_rate` with the version below and add the new functions directly after it:
```python
def _g_par(tail):
    """G along F at each recorded iteration, (T, H, W)."""
    return np.stack([np.sum(G * F, -1) / np.maximum(np.linalg.norm(F, axis=-1), 1e-9)
                     for G, F in tail])


def flip_rate(tail):
    """Per-pixel rate of sign changes of G along F over the recorded
    iterations, and |F| at the last one."""
    par = _g_par(tail)
    return (np.sign(par[1:]) != np.sign(par[:-1])).mean(0), np.linalg.norm(tail[-1][1], axis=-1)


def contraction(tail, eps=1e-12):
    """Per-pixel factor by which G along F changes per iteration: the median
    over t of (g[t+2]-g[t+1]) / (g[t+1]-g[t]). For a linear step g <- lam*g + c
    it is exactly lam; nan where g no longer moves."""
    d = np.diff(_g_par(tail), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = np.where(np.abs(d[:-1]) > eps, d[1:] / d[:-1], np.nan)
        return np.nanmedian(r, axis=0)


def predicted_lambda(x, delta_IG):
    """Linearised contraction of G along F at |F| = x * stability_bound:
    1 - delta_IG - (2 - delta_IG) x^2 (0 at the flip onset, -1 at the bound)."""
    return 1.0 - delta_IG - (2.0 - delta_IG) * np.asarray(x, dtype=np.float64) ** 2


def flip_onset(delta_IG):
    """x = |F| / bound above which the linear G step flips sign (lambda < 0)."""
    return float(np.sqrt((1.0 - delta_IG) / (2.0 - delta_IG)))
```
Give `_ofce_g_only` the hook (signature and loop only; body otherwise unchanged):
```python
def _ofce_g_only(F_mag, normalize, n_iters=200, V=1.0, on_iter=None):
    """One pixel, F fixed along x, only the OFCE step on G (q.update(1.0) as in
    the network). Returns |V + F.G| after n_iters: ~0 iff the step converged.
    on_iter(G, F), if given, gets copies after every iteration."""
    q = {k: Quantity(s, k) for k, s in (('V', (1, 1)), ('F', (1, 1, 2)), ('G', (1, 1, 2)))}
    q['V'].value[:] = V
    q['F'].value[..., 0] = F_mag
    cost = Cost_OFCE(q, config.THESIS_PARAMS['delta_VFG'], normalize=normalize)
    for _ in range(n_iters):
        q['G'].reset_gradient()
        cost.compute_and_send_gradients()
        q['G'].update(1.0)
        if on_iter is not None:
            on_iter(q['G'].value.copy(), q['F'].value.copy())
    return float(abs(V + np.sum(q['F'].value * q['G'].value)))
```

- [ ] **Step 4: Run it and see it pass**

Run: `python diag_omega_ladder.py --mode setup --limit 3`
Expected: the existing `[ok]` lines, plus:
```
  [ok] contraction recovers lam=0.5 (got 0.500000)
  [ok] contraction recovers lam=-0.8 (got -0.800000)
  [ok] one-pixel OFCE step |F|=1.44: contraction 0.5000 vs 1-2*delta_VFG*|F|^2 = 0.5000
  [ok] one-pixel OFCE step |F|=2.31: contraction -0.2800 vs 1-2*delta_VFG*|F|^2 = -0.2800
  [ok] predicted_lambda: -1 at the bound, 0 at flip_onset = 0.688
```
No `[FAIL]` anywhere (the three existing OFCE-step checks still pass).

- [ ] **Step 5: Commit**

```bash
git add diag_omega_ladder.py
git commit -m "Diagnostic: contraction factor of G along F, predicted lambda(|F|/bound), self-checks"
```

---

### Task 3: Measure it in the full network (`--mode ofce`)

**Files:**
- Modify: `diag_omega_ladder.py` (`OFCE_HEADER` block, `run_ofce_variant`, the OFCE comment and docstring)

**Interfaces:**
- Consumes: `contraction`, `flip_rate`, `stability_bound`, `_map_run` (supports `'1b_free'`, `'1c_R'`, `'0'`).
- Produces: per-variant CSV rows for `1b_free`, `1c_R`, `0` with new columns
  `Fgt_over_b`, `flipx_<lo>_<hi>`, `lamx_<lo>_<hi>`, `occx_<lo>_<hi>` over `XBINS`;
  `XBINS = (0, 0.5, 0.69, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, inf)`; `_XB` = their column suffixes;
  `_xbin_stats(v, x, stat, min_px=50) -> list`, `_xbin_occ(x) -> list`.

Why each addition:
- **Row 1b_free (F clamped to F\*):** F is exactly fixed, so the linear analysis applies cleanly.
- **Head recording:** the first `HEAD_ITERS` iterations of frame K, where G is still moving after the frame change. In the last 20 iterations stable pixels have converged and λ̂ is undefined.
- **x-bins:** let all variants be compared on one axis.

- [ ] **Step 1: Replace the header block**

Replace
```python
FLIP_BINS = (0, 1, 2, 3, 4, 6, np.inf)          # |F| px/frame
OFCE_HEADER = (['variant', 'rung', 'dt_ms', 'n_frames', 'delta_VFG', 'max_grad',
                'normalize', 'bound_px', 'frac_Fgt_above', 'F_epe', 'F_epe_rel',
                'stripe_I', 'G_at_clip', 'finite', 'mean_err', 'mean_dir', 'mean_beta']
               + [f'flip_{lo}_{hi}' for lo, hi in zip(FLIP_BINS[:-1], FLIP_BINS[1:])]
               + ['secs'])
```
with
```python
FLIP_BINS = (0, 1, 2, 3, 4, 6, np.inf)          # |F| px/frame
XBINS = (0, 0.5, 0.69, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, np.inf)   # x = |F| / bound
_XB = [f'{lo}_{hi}' for lo, hi in zip(XBINS[:-1], XBINS[1:])]
HEAD_ITERS = 12      # first iterations of frame K, for contraction()
OFCE_HEADER = (['variant', 'rung', 'dt_ms', 'n_frames', 'delta_VFG', 'max_grad',
                'normalize', 'bound_px', 'frac_Fgt_above', 'F_epe', 'F_epe_rel',
                'stripe_I', 'G_at_clip', 'finite', 'mean_err', 'mean_dir', 'mean_beta']
               + [f'flip_{lo}_{hi}' for lo, hi in zip(FLIP_BINS[:-1], FLIP_BINS[1:])]
               + ['Fgt_over_b'] + [f'flipx_{s}' for s in _XB]
               + [f'lamx_{s}' for s in _XB] + [f'occx_{s}' for s in _XB] + ['secs'])


def _xbin_stats(v, x, stat, min_px=50):
    """stat(v) over the finite pixels in each XBINS bin of x ('' if < min_px)."""
    out = []
    for lo, hi in zip(XBINS[:-1], XBINS[1:]):
        sel = (x >= lo) & (x < hi) & np.isfinite(v)
        out.append(round(float(stat(v[sel])), 3) if sel.sum() >= min_px else '')
    return out


def _xbin_occ(x):
    """Share of pixels in each XBINS bin of x."""
    return [round(float(((x >= lo) & (x < hi)).mean()), 3)
            for lo, hi in zip(XBINS[:-1], XBINS[1:])]
```

- [ ] **Step 2: Change the row loop in `run_ofce_variant`**

Replace
```python
    for rung in ('1c_R', '0'):
        tail, t0 = deque(maxlen=20), time.time()
        s = _map_run(ctx, rung,
                     on_frame=lambda k, nt: last.update({rung: _snap(nt, H, W)}),
                     on_iter=lambda nt: tail.append((nt.G.copy(), nt.F.copy())),
                     iter_frame=K)
        flips, mag = flip_rate(tail)
```
with
```python
    for rung in ('1b_free', '1c_R', '0'):
        tail, head, t0 = deque(maxlen=20), [], time.time()

        def on_iter(nt, tail=tail, head=head):
            tail.append((nt.G.copy(), nt.F.copy()))
            if len(head) < HEAD_ITERS:
                head.append(tail[-1])

        s = _map_run(ctx, rung,
                     on_frame=lambda k, nt: last.update({rung: _snap(nt, H, W)}),
                     on_iter=on_iter, iter_frame=K)
        flips, mag = flip_rate(tail)
        lam = contraction(head)
        x_tail = mag / bound
        x_head = np.linalg.norm(head[-1][1], axis=-1) / bound
```
and replace the row's last line
```python
                    + bins + [round(time.time() - t0)])
```
with
```python
                    + bins
                    + [round(float(np.mean([g.mean() for g in mag_gt])) / bound, 3)]
                    + _xbin_stats(flips, x_tail, np.mean)
                    + _xbin_stats(lam, x_head, np.median)
                    + _xbin_occ(x_tail)
                    + [round(time.time() - t0)])
```
Update the comment above `OFCE_DIR` to
`# (one report/run_all.sh job); rows 1b_free (F exact: the clean linear test), 1c_R (omega exact) and 0.`
and the `run_ofce_variant` docstring's first line to
`"""Run one OFCE_VARIANTS entry on rows 1b_free, 1c_R and 0 over the segment's span;`.
The last-frame maps figure keeps `rungs = ['1c_R', '0']` (unchanged).

- [ ] **Step 3: Run a local smoke test on an unstable and a stable variant**

```bash
python diag_omega_ladder.py --mode ofce --variant E0_base --limit 6
```
```bash
python diag_omega_ladder.py --mode ofce --variant E4_dt05 --limit 24
```
Expected for each: `[ok] diag_ofce/<variant>.csv + .png`, no traceback.

- [ ] **Step 4: Check the new columns make sense**

```bash
python - <<'PY'
import csv
import diag_omega_ladder as d
for v in ('E0_base', 'E4_dt05'):
    for r in csv.DictReader(open(f'diag_ofce/{v}.csv')):
        lam = [r[f'lamx_{s}'] or '-' for s in d._XB]
        occ = [r[f'occx_{s}'] for s in d._XB]
        print(v, r['rung'], 'Fgt/b', r['Fgt_over_b'], '| lam', lam, '| occ', occ)
print('predicted lam at bin centres:',
      [round(float(d.predicted_lambda(x, 0.1)), 2) for x in (0.25, 0.6, 0.8, 0.95, 1.05, 1.2)])
PY
```
Expected:
- Three rows per variant, in the order `1b_free`, `1c_R`, `0`.
- **E0_base row `1b_free`:** occupancy almost entirely in bins with x > 1.3 (|F\*| 4–10 px against b ≈ 2.81). λ̂ there clusters near −1, bounded by the clip.
- **E4_dt05 row `1b_free`:** occupancy in x < 1. λ̂ follows the printed predictions, e.g. ≈ 0.2 near x = 0.6 and ≈ −0.3 near x = 0.8, within about ±0.15.

A large mismatch below x = 1 is itself a finding: write it down; do not tune the code to hide it.

- [ ] **Step 5: Commit**

```bash
git add diag_omega_ladder.py
git commit -m "Diagnostic: OFCE runs add row 1b_free, contraction and flips in |F|/bound bins, F occupancy"
```

---

### Task 4: The three figures

**Files:**
- Modify: `diag_omega_ladder.py` (after `ofce_summary`; `ofce_summary` itself; `__main__` `ofce_summary` branch)

**Interfaces:**
- Consumes: `diag_ofce/summary.csv` with the Task 3 columns; `predicted_lambda`, `flip_onset`, `_XB`, `XBINS`, `use_style`, `save`, `SCREEN_WIDTH`.
- Produces: `plot_ofce_findings(path=os.path.join(OFCE_DIR, 'summary.csv'), out_dir=OFCE_DIR)`, which writes `ofce_lambda.png`, `ofce_flips.png`, `ofce_beta.png`; `VARIANT_COLORS`.

- [ ] **Step 1: Add the plotting function**

Add after `ofce_summary`:
```python
VARIANT_COLORS = dict(zip(OFCE_VARIANTS, ('#000000', '#56B4E9', '#0072B2', '#D55E00',
                                          '#009E73', '#E69F00', '#CC79A7')))  # Okabe-Ito


def _xcenters():
    """Plot position of each XBINS bin (the open last bin at 2.25)."""
    return np.array([(lo + hi) / 2 if np.isfinite(hi) else 2.25
                     for lo, hi in zip(XBINS[:-1], XBINS[1:])])


def _num(row, cols):
    """Floats of row[cols]; nan for empty cells."""
    return np.array([float(row[c]) if row.get(c) not in ('', None) else np.nan for c in cols])


def plot_ofce_findings(path=os.path.join(OFCE_DIR, 'summary.csv'), out_dir=OFCE_DIR):
    """From the OFCE summary: (1) measured vs predicted contraction of G along F
    (row 1b_free), (2) flip rate vs |F|/bound (rows 1b_free and 0), (3) rung-0
    beta vs mean|F*|/bound across variants."""
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(path, newline='')))
    if not rows or 'lamx_0_0.5' not in rows[0]:
        print(f"[skip] {path}: no |F|/bound columns (rerun --mode ofce)")
        return
    use_style('screen')
    xc = _xcenters()
    dIG = config.THESIS_PARAMS['delta_IG']
    x0 = flip_onset(dIG)

    def by(rung):
        return [r for r in rows if r['rung'] == rung]

    def marks(ax):
        ax.axvline(x0, color='black', lw=0.8, ls=':')
        ax.axvline(1.0, color='black', lw=0.8, ls='--')

    fig, ax = plt.subplots(figsize=(SCREEN_WIDTH * 0.6, 4.2))
    xs = np.linspace(0, 2.3, 200)
    ax.plot(xs, predicted_lambda(xs, dIG), color='#888888', lw=2.5, label='linear prediction')
    for r in by('1b_free'):
        ax.plot(xc, _num(r, [f'lamx_{s}' for s in _XB]), 'o-', lw=1.5,
                color=VARIANT_COLORS.get(r['variant'], '#333333'), label=r['variant'])
    ax.axhline(0, color='black', lw=0.8, ls=':')
    ax.axhline(-1, color='black', lw=0.8, ls='--')
    marks(ax)
    ax.set_ylim(-3, 1.1)
    ax.set_xlabel('|F| / stability bound')
    ax.set_ylabel('change of G along F per iteration')
    ax.set_title('Row 1b (F fixed): measured vs predicted G step')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, os.path.join(out_dir, 'ofce_lambda.png'))

    fig, axs = plt.subplots(1, 2, figsize=(SCREEN_WIDTH, 3.8), sharey=True)
    for ax, rung, title in ((axs[0], '1b_free', 'Row 1b (F fixed)'),
                            (axs[1], '0', 'Row 0 (free)')):
        for r in by(rung):
            ax.plot(xc, _num(r, [f'flipx_{s}' for s in _XB]), 'o-', lw=1.5,
                    color=VARIANT_COLORS.get(r['variant'], '#333333'), label=r['variant'])
        marks(ax)
        ax.set_xlabel('|F| / stability bound')
        ax.set_title(title)
    axs[0].set_ylabel('G sign flips per iteration')
    axs[0].legend(frameon=False, fontsize=8)
    save(fig, os.path.join(out_dir, 'ofce_flips.png'))

    fig, ax = plt.subplots(figsize=(SCREEN_WIDTH * 0.5, 4.2))
    pts = [(float(r['Fgt_over_b']), float(r['mean_beta']), r['variant'])
           for r in by('0') if r.get('Fgt_over_b') not in ('', None)]
    if pts:
        hi = 1.1 * max(max(p[0] for p in pts), max(p[1] for p in pts))
        xx = np.linspace(0, hi, 100)
        ax.plot(xx, np.maximum(1.0, xx), color='#888888', lw=2.5, label='F capped at the bound')
        for fx, beta, v in pts:
            ax.plot(fx, beta, 'o', color=VARIANT_COLORS.get(v, '#333333'))
            ax.annotate(v, (fx, beta), textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax.set_xlabel('mean |F*| / stability bound')
    ax.set_ylabel(r'rung-0 scale $\beta$')
    ax.set_title('Does the bound set the scale error?')
    ax.legend(frameon=False, fontsize=8)
    save(fig, os.path.join(out_dir, 'ofce_beta.png'))
    print(f"[ok] {out_dir}/ofce_lambda.png, ofce_flips.png, ofce_beta.png")
```
At the end of `ofce_summary` (after its `print(f"[ok] {out}")`), add:
```python
    plot_ofce_findings(out)
```
In `__main__`, change
```python
    elif args.mode == 'ofce_summary':
        ofce_summary()
```
to
```python
    elif args.mode == 'ofce_summary':
        import matplotlib
        matplotlib.use('Agg')
        ofce_summary()
```

- [ ] **Step 2: Render from the Task 3 smoke CSVs**

Run: `python diag_omega_ladder.py --mode ofce_summary`
Expected: the summary table, then `[ok] diag_ofce/ofce_lambda.png, ofce_flips.png, ofce_beta.png`.

- [ ] **Step 3: Look at the three PNGs**

Open each. Check: the grey prediction curve crosses 0 at x ≈ 0.69 and −1 at x = 1; points are coloured per variant with a legend (figs 1–2) or direct labels (fig 3); no clipped or overlapping labels; the axis text matches the content. Fix layout issues and re-run Step 2.

- [ ] **Step 4: Delete the local smoke outputs**

These come from `--limit` runs and must not be mistaken for results:
```bash
rm -rf diag_ofce
```

- [ ] **Step 5: Commit**

```bash
git add diag_omega_ladder.py
git commit -m "Diagnostic: plot_ofce_findings (contraction vs prediction, flips vs |F|/bound, beta vs bound)"
```

---

### Task 5: Notebook step

**Files:**
- Modify: `diag_omega_ladder.ipynb` (insert before the `## Across segments` cell; nothing else changes)

**Interfaces:**
- Consumes: `dol._ofce_g_only(..., on_iter=...)`, `dol.contraction`, `dol.stability_bound`, `dol.flip_onset`, `dol.plot_ofce_findings`, `dol.OFCE_DIR`; notebook names `config`, `np`, `plt`, `os` from the setup cell.

- [ ] **Step 1: Insert the cells**

```bash
python - <<'PY'
import json
NB = 'diag_omega_ladder.ipynb'
md1 = r'''## Step 11 — Why does G flip? The OFCE step, one pixel at a time

`Cost_OFCE` moves G by an explicit gradient step on `(V + F·G)²`. Along F this multiplies the remaining error of G by λ = 1 − 2·δ_VFG·|F|² − δ_IG every iteration. For 0 < λ < 1 the error shrinks steadily; for −1 < λ < 0 it still shrinks but **flips sign every iteration**; below −1 it grows, and only the `max_grad` clip holds it. In units of the stability bound, x = |F|/b, this is λ = 1 − δ_IG − (2 − δ_IG)·x², the same curve for every δ_VFG and dt.

**Look for:** one pixel with F fixed (OFCE term only): at x = 0.5 the residual decays smoothly, at x = 0.8 it zig-zags to zero, at x = 1.1 it keeps flipping.'''
code1 = r'''d_vfg, d_ig = config.THESIS_PARAMS['delta_VFG'], config.THESIS_PARAMS['delta_IG']
b_net = dol.stability_bound(d_vfg, d_ig)
print(f"delta_VFG = {d_vfg}, delta_IG = {d_ig}: bound b = {b_net:.2f} px/frame; "
      f"flips from x = {dol.flip_onset(d_ig):.2f} (|F| = {dol.flip_onset(d_ig) * b_net:.2f} px/frame)")

b1 = 1.0 / np.sqrt(d_vfg)            # one-pixel bound (OFCE term only, no delta_IG)
fig, ax = plt.subplots(figsize=(10, 3.4))
for xm, col in ((0.5, '#2a78d6'), (0.8, '#eb6834'), (1.1, '#d62728')):
    hist = []
    dol._ofce_g_only(xm * b1, False, n_iters=30, on_iter=lambda G, F: hist.append((G, F)))
    r = 1.0 + np.array([float(np.sum(F * G)) for G, F in hist])     # V + F.G with V = 1
    lam = float(dol.contraction(hist)[0, 0])
    ax.plot(np.arange(1, len(r) + 1), r, 'o-', ms=3, lw=1.2, color=col,
            label=f"x = {xm}: measured {lam:.2f}, predicted {1 - 2 * d_vfg * (xm * b1) ** 2:.2f}")
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('iteration'); ax.set_ylabel('residual V + F·G')
ax.set_title('One pixel, F fixed: the G step settles, zig-zags, or keeps flipping')
ax.legend(frameon=False, fontsize=8)
plt.show()'''
md2 = r'''**Across the variants (cluster).** `python diag_omega_ladder.py --mode ofce` measures the same thing in the full network for every variant (δ_VFG sweep, no clip, normalised step, dt 10/5 ms) and `--mode ofce_summary` draws three figures: measured vs predicted contraction (row 1b, F fixed), flip rate against x (if the theory holds, all variants fall on one curve), and rung-0 β against mean|F*|/b (on the grey line if the bound sets the scale error).'''
code2 = r'''summ = os.path.join(dol.OFCE_DIR, 'summary.csv')
if os.path.exists(summ):
    from IPython.display import Image, display
    dol.plot_ofce_findings(summ)
    for f in ('ofce_lambda.png', 'ofce_flips.png', 'ofce_beta.png'):
        display(Image(os.path.join(dol.OFCE_DIR, f)))
else:
    print("No diag_ofce/summary.csv here yet. On the cluster: python diag_omega_ladder.py --mode ofce")'''

def cell(kind, text):
    lines = text.split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    if kind == 'md':
        return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src}

nb = json.load(open(NB, encoding='utf-8'))
srcs = [''.join(c['source']) for c in nb['cells']]
assert not any(s.startswith('## Step 11') for s in srcs), 'Step 11 already present'
at = next(i for i, s in enumerate(srcs) if s.startswith('## Across segments'))
nb['cells'][at:at] = [cell('md', md1), cell('code', code1), cell('md', md2), cell('code', code2)]
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False); f.write('\n')
print('inserted 4 cells at', at)
PY
```
Expected: `inserted 4 cells at 28`.

- [ ] **Step 2: Run the new code cells**

Run the setup cell (to define `config`, `np`, `plt`, `os`, `dol`), then the two new code cells, either in Jupyter or with:
```bash
python - <<'PY'
import json, matplotlib
matplotlib.use('Agg')
nb = json.load(open('diag_omega_ladder.ipynb', encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
ns = {}
for src in [cells[0]] + [c for c in cells if c.startswith(("d_vfg, d_ig", "summ = os.path"))]:
    exec(src.replace('dol.LIMIT = 10', 'dol.LIMIT = 3'), ns)
print('[ok] Step 11 cells ran')
PY
```
Expected: the bound line printed, labels showing measured ≈ predicted for x = 0.5 and 0.8, and either the three figures or the "No diag_ofce/summary.csv here yet" line; ending with `[ok] Step 11 cells ran`.

- [ ] **Step 3: Commit**

```bash
git add diag_omega_ladder.ipynb
git commit -m "Diagnostic notebook: Step 11, why G flips (one-pixel demo + cluster OFCE figures)"
```

---

### Task 6: Cluster run and findings

**Files:**
- Modify: `docs/superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md` (new section after "Findings (2026-10-12)")

- [ ] **Step 1: Run on the cluster** (skip if Task 1's gate said Tasks 2–5 were not needed)

```bash
bash report/run_all.sh
```
The changed `diag_omega_ladder.py` invalidates the OFCE stamps, so all seven variants and `ofce_summary` rerun. To run only the diagnostic:
```bash
python diag_omega_ladder.py --mode ofce
```
Copy back `diag_ofce/summary.csv` and `diag_ofce/ofce_*.png`.

- [ ] **Step 2: Answer RQ1–RQ3**

Read the answers against these criteria, chosen before seeing the results:

| RQ | supported if | refuted if |
|---|---|---|
| 1 mechanism | row 1b λ̂ within ±0.15 of λ(x) for x < 1, in every variant except E3_normalize; near −1 above x = 1 with the clip, below −1 for E2_noclip | λ̂ off the curve below x = 1 |
| 2 scaling | flip rate < 0.1 below x ≈ 0.6 and > 0.5 above x ≈ 0.8, in all clipped variants (one curve) | onset at the same px across δ_VFG |
| 3 consequence | row-0 occupancy above x = 1 under 10 % while mean\|F\*\|/b > 1; β within ±25 % of max(1, mean\|F\*\|/b); E3_normalize and E4_dt05 β < 1.3 | β unchanged across variants |

- [ ] **Step 3: Write the findings**

Append a section `## Findings: G flipping measured (YYYY-MM-DD)` with the three answers, one table of the numbers behind each, and the figure file names. Add one paragraph on what follows:
- if RQ3 is supported, the β scale error is a numerical artefact of the explicit step, and the remedies (normalised step, shorter dt, smaller δ_VFG) become report-relevant;
- if not, record what was ruled out.
Update the memory note `ofce-step-instability` to match.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-10-11-ecrot-clamped-omega-maps-design.md
git commit -m "Spec: G-flipping findings (contraction, onset scaling, beta vs bound)"
```

---

## Self-Review

**1. Coverage.** RQ1 → Tasks 2–3 (`contraction` on row 1b with head recording) + figure 1. RQ2 → Task 3 x-bins + figure 2. RQ3 → Task 3 occupancy and `Fgt_over_b` + figure 3, and Task 1 Step 2 on existing data. Existing results first → Task 1 with an explicit gate. Notebook → Task 5. Write-up → Task 6. No model changes; `run_all.sh` untouched.

**2. Placeholders.** None. Every code step has the full code. Task 6's findings text depends on results by design; its criteria are fixed in advance.

**3. Consistency.**
- **Names:** `_g_par`, `contraction`, `predicted_lambda`, `flip_onset` (Task 2) are used unchanged in Tasks 3–5.
- **Columns:** `XBINS` / `_XB`, `_xbin_stats`, `_xbin_occ`, `HEAD_ITERS` (Task 3) are used by `plot_ofce_findings` (Task 4) through the column names `lamx_*`, `flipx_*`, `occx_*` and `Fgt_over_b`.
- **Hooks:** `_ofce_g_only(..., on_iter)` (Task 2) is used in `_selfcheck_contraction` and the notebook.
- **Rows:** `_map_run` already accepts `'1b_free'`.
- **Guard:** `plot_ofce_findings` skips summaries without the new columns, so old cluster CSVs don't crash it.
