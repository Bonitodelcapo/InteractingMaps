"""
analyze_grid.py — Post-hoc analysis of parameter_grid.csv

Usage:
    python analyze_grid.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ─── Load data ────────────────────────────────────────────────────────
df = pd.read_csv('results/parameter_grid.csv')

# ─── Add derived columns ──────────────────────────────────────────────
# Expected omega magnitudes per segment (from config.py)
omega_magnitudes = {
    ('boxes_rotation', 'seg_A'): 1.491,
    ('boxes_rotation', 'seg_B'): 0.869,
    ('boxes_rotation', 'seg_C'): 0.574,
    ('boxes_rotation', 'seg_D'): 0.358,
    ('boxes_rotation', 'seg_E'): 0.418,
    ('shapes_rotation', 'seg_A'): 0.757,
    ('shapes_rotation', 'seg_B'): 0.925,
    ('shapes_rotation', 'seg_C'): 0.646,
    ('shapes_rotation', 'seg_D'): 0.066,
    ('poster_rotation', 'seg_A'): 1.123,
    ('poster_rotation', 'seg_B'): 0.671,
    ('poster_rotation', 'seg_C'): 0.546,
    ('poster_rotation', 'seg_D'): 0.474,
    ('poster_rotation', 'seg_E'): 0.516,
}

df['omega_gt_mag'] = df.apply(
    lambda r: omega_magnitudes.get((r['dataset'], r['segment_id']), np.nan), axis=1
)
df['omega_gt_deg_s'] = df['omega_gt_mag'] * 180 / np.pi
df['relative_err_pct'] = df['mean_err_deg_s'] / df['omega_gt_deg_s'] * 100

os.makedirs('results/analysis', exist_ok=True)

# ─── 1. Best config per model × dataset × segment ────────────────────
print("\n" + "="*100)
print("BEST CONFIGURATION PER MODEL × DATASET × SEGMENT")
print("="*100)
print(f"{'Dataset':<18} {'Seg':<6} {'Model':<12} {'n_fr':>5} {'iter':>5} "
      f"{'δ_FR':>5} {'δ_IMU':>6} | {'err°/s':>7} {'rel%':>6} {'dir°':>6}")
print("-"*100)

best_rows = []
for (ds, seg, model), group in df.groupby(['dataset', 'segment_id', 'model']):
    best = group.loc[group['mean_err_deg_s'].idxmin()]
    best_rows.append(best)
    print(f"{ds:<18} {seg:<6} {model:<12} "
          f"{int(best['n_frames']):>5} {int(best['n_iters']):>5} "
          f"{best['delta_FR']:>5.2f} {best['delta_IMU']:>6.2f} | "
          f"{best['mean_err_deg_s']:>7.2f} {best['relative_err_pct']:>5.1f}% "
          f"{best['mean_dir_err_deg']:>6.1f}")

# ─── 2. Parameter sensitivity (marginal effect) ──────────────────────
print("\n\n" + "="*80)
print("PARAMETER SENSITIVITY (mean error across all runs with that value)")
print("="*80)

for model in ['cook', 'thesis', 'thesis_imu']:
    mdf = df[df['model'] == model]
    print(f"\n  [{model}]")
    
    for param in ['n_frames', 'n_iters', 'delta_FR', 'delta_IMU']:
        if param == 'delta_IMU' and model != 'thesis_imu':
            continue
        grouped = mdf.groupby(param)['mean_err_deg_s'].mean()
        print(f"    {param}:")
        for val, err in grouped.items():
            print(f"      {val:>6} → {err:.2f}°/s")

# ─── 3. Heatmap: delta_FR vs delta_IMU (thesis_imu only) ─────────────
imu_df = df[df['model'] == 'thesis_imu']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('thesis_imu: Error by δ_FR × δ_IMU (averaged over all segments)', fontsize=12)

for ax, n_fr in zip(axes, [25, 50, 150]):
    sub = imu_df[imu_df['n_frames'] == n_fr]
    pivot = sub.pivot_table(values='mean_err_deg_s', 
                            index='delta_FR', columns='delta_IMU', 
                            aggfunc='mean')
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=15)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index])
    ax.set_xlabel('δ_IMU')
    ax.set_ylabel('δ_FR')
    ax.set_title(f'n_frames={n_fr}')
    
    # Annotate
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.values[i,j]:.1f}', 
                   ha='center', va='center', fontsize=8)

plt.colorbar(im, ax=axes, label='Mean error (°/s)')
plt.tight_layout()
plt.savefig('results/analysis/heatmap_FR_vs_IMU.png', dpi=150)
plt.close()

# ─── 4. Error vs |ω| (non-IMU models) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for model, marker in [('cook', 'o'), ('thesis', 's')]:
    mdf = df[df['model'] == model]
    best_per_seg = mdf.groupby(['dataset', 'segment_id']).apply(
        lambda g: g.loc[g['mean_err_deg_s'].idxmin()])
    ax.scatter(best_per_seg['omega_gt_deg_s'], best_per_seg['mean_err_deg_s'],
              marker=marker, s=80, label=model, alpha=0.7)

ax.set_xlabel('|ω_GT| (°/s)')
ax.set_ylabel('Best achievable error (°/s)')
ax.set_title('Error scales with rotation speed (no IMU)')
ax.legend()
ax.grid(True, alpha=0.3)
# Add y=x reference
x_range = np.linspace(0, ax.get_xlim()[1], 50)
ax.plot(x_range, x_range, 'k--', alpha=0.3, label='err = |ω|')
plt.tight_layout()
plt.savefig('results/analysis/error_vs_omega.png', dpi=150)
plt.close()

# ─── 5. Relative error comparison ────────────────────────────────────
print("\n\n" + "="*80)
print("RELATIVE ERROR (% of |ω_GT|) — Best config per model × segment")
print("="*80)
print(f"{'Dataset':<18} {'Seg':<6} {'|ω|°/s':>7} | "
      f"{'Cook':>8} {'Thesis':>8} {'Th+IMU':>8}")
print("-"*80)

for (ds, seg), group in df.groupby(['dataset', 'segment_id']):
    omega_deg = group['omega_gt_deg_s'].iloc[0]
    errs = {}
    for model in ['cook', 'thesis', 'thesis_imu']:
        mg = group[group['model'] == model]
        if len(mg) > 0:
            errs[model] = mg['mean_err_deg_s'].min() / omega_deg * 100
        else:
            errs[model] = float('nan')
    print(f"{ds:<18} {seg:<6} {omega_deg:>7.1f} | "
          f"{errs.get('cook', float('nan')):>7.1f}% "
          f"{errs.get('thesis', float('nan')):>7.1f}% "
          f"{errs.get('thesis_imu', float('nan')):>7.1f}%")

# ─── 6. n_frames effect (divergence) ─────────────────────────────────
print("\n\n" + "="*80)
print("EFFECT OF SEQUENCE LENGTH (n_frames)")
print("="*80)
for model in ['cook', 'thesis', 'thesis_imu']:
    mdf = df[df['model'] == model]
    print(f"\n  [{model}] — mean error by n_frames (best δ params):")
    for n_fr in [25, 50, 150]:
        sub = mdf[mdf['n_frames'] == n_fr]
        # For each segment, take the best delta config
        best_per_seg = sub.groupby(['dataset', 'segment_id'])['mean_err_deg_s'].min()
        print(f"    n_frames={n_fr:>3}: mean_best={best_per_seg.mean():.2f}°/s "
              f"(range: {best_per_seg.min():.2f} – {best_per_seg.max():.2f})")

# ─── 7. Save enhanced CSV ────────────────────────────────────────────
df.to_csv('results/analysis/parameter_grid_enhanced.csv', index=False)
print(f"\n\nEnhanced CSV saved: results/analysis/parameter_grid_enhanced.csv")
print("Plots saved in results/analysis/")