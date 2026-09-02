"""
Aggregate the exp-12 runs into report-ready tables (D3).

Pure re-analysis: reads results/report/report_runs.csv (written by
`evaluation.py --exp 12`) and emits per-(dataset, model) and per-model summary
tables with min / max / mean / std / median ACROSS segments, plus the raw
per-segment long table.

Outputs (results/report/):
    table_main.{csv,md}         one row per (dataset, model)
    table_overall.{csv,md}      one row per model, pooled over datasets
    table_per_segment.{csv,md}  the raw long table
    table_main.tex              booktabs LaTeX for the thesis

Usage:
    python report_tables.py
    python report_tables.py --runs results/report/report_runs.csv --exclude shapes_rotation
"""

import argparse
import os

import numpy as np
import pandas as pd

REPORT_DIR = os.path.join('results', 'report')
DEFAULT_RUNS = os.path.join(REPORT_DIR, 'report_runs.csv')

# Order models consistently everywhere (pure vision -> anchored -> CMax).
MODEL_ORDER = ['cook', 'thesis', 'thesis_imu', 'thesis_cmax', 'thesis_cmax_v2']


def load_runs(path=DEFAULT_RUNS):
    """Load the exp-12 CSV, keeping only successful runs."""
    if not os.path.exists(path):
        raise SystemExit(f"No runs file at {path}. Run: python evaluation.py --exp 12 ...")
    df = pd.read_csv(path)
    n_all = len(df)
    if 'status' in df.columns:
        bad = df[df['status'] != 'ok']
        if len(bad):
            print(f"  {len(bad)}/{n_all} runs failed and are excluded:")
            for _, r in bad.iterrows():
                print(f"    {r['dataset']}/{r['segment_id']}/{r['model']}: "
                      f"{str(r['status'])[:80]}")
        df = df[df['status'] == 'ok'].copy()
    for c in df.columns:
        if c not in ('dataset', 'segment_id', 'model', 'status',
                     'distortion_mode', 'ref_source'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(['dataset', 'model', 'segment_id'])


def table_main(df):
    """Per (dataset, model): statistics of the per-segment mean error."""
    g = df.groupby(['dataset', 'model'], observed=True)
    out = g.agg(
        n_segments=('mean_err_deg_s', 'size'),
        err_mean=('mean_err_deg_s', 'mean'),
        err_std=('mean_err_deg_s', 'std'),
        err_median=('mean_err_deg_s', 'median'),
        err_min=('mean_err_deg_s', 'min'),
        err_max=('mean_err_deg_s', 'max'),
        dir_mean=('mean_dir_err_deg', 'mean'),
        beta_mean=('mean_beta', 'mean'),
        drift=('final_err_deg_s', 'mean'),
        t_frame=('mean_frame_time_s', 'mean'),
    ).reset_index()
    return out.round(3)


def table_overall(df):
    """Per model, pooled over every dataset and segment."""
    g = df.groupby('model', observed=True)
    out = g.agg(
        n_runs=('mean_err_deg_s', 'size'),
        err_mean=('mean_err_deg_s', 'mean'),
        err_std=('mean_err_deg_s', 'std'),
        err_median=('mean_err_deg_s', 'median'),
        err_min=('mean_err_deg_s', 'min'),
        err_max=('mean_err_deg_s', 'max'),
        dir_mean=('mean_dir_err_deg', 'mean'),
        beta_mean=('mean_beta', 'mean'),
        t_frame=('mean_frame_time_s', 'mean'),
    ).reset_index()
    return out.round(3)


def table_per_segment(df):
    cols = ['dataset', 'segment_id', 'model', 'n_frames', 'duration_s',
            'omega_gt_mag_mean', 'mean_err_deg_s', 'median_err_deg_s',
            'min_err_deg_s', 'max_err_deg_s', 'final_err_deg_s',
            'mean_dir_err_deg', 'mean_beta', 'mean_frame_time_s', 'ref_source']
    have = [c for c in cols if c in df.columns]
    return df[have].round(3)


def _to_markdown(df):
    """Markdown table without requiring the optional `tabulate` package."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = [str(c) for c in df.columns]
        lines = ['| ' + ' | '.join(cols) + ' |',
                 '|' + '|'.join('---' for _ in cols) + '|']
        for _, row in df.iterrows():
            lines.append('| ' + ' | '.join(
                '' if pd.isna(v) else str(v) for v in row.tolist()) + ' |')
        return '\n'.join(lines)


def _write(df, name, caption=None):
    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_p = os.path.join(REPORT_DIR, f'{name}.csv')
    md_p = os.path.join(REPORT_DIR, f'{name}.md')
    df.to_csv(csv_p, index=False)
    with open(md_p, 'w') as f:
        if caption:
            f.write(f'### {caption}\n\n')
        f.write(_to_markdown(df))
        f.write('\n')
    print(f"  wrote {csv_p} and {md_p}")


def main():
    ap = argparse.ArgumentParser(description='Build report tables from exp-12 runs')
    ap.add_argument('--runs', default=DEFAULT_RUNS)
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='Datasets to drop from the headline tables '
                         '(e.g. shapes_rotation, whose segments are lower quality)')
    args = ap.parse_args()

    df = load_runs(args.runs)
    print(f"  {len(df)} successful runs, "
          f"{df['dataset'].nunique()} datasets, {df['model'].nunique()} models")

    _write(table_per_segment(df), 'table_per_segment',
           'Per-segment results (raw)')

    head = df[~df['dataset'].isin(args.exclude)] if args.exclude else df
    if args.exclude:
        print(f"  headline tables exclude: {', '.join(args.exclude)}")

    main_t = table_main(head)
    _write(main_t, 'table_main',
           'Angular-velocity error by dataset and model '
           '(statistics across segments, deg/s)')
    _write(table_overall(head), 'table_overall',
           'Overall model comparison (pooled across datasets)')

    # LaTeX for the thesis
    tex_p = os.path.join(REPORT_DIR, 'table_main.tex')
    with open(tex_p, 'w') as f:
        f.write(main_t.to_latex(index=False, escape=False, float_format='%.2f'))
    print(f"  wrote {tex_p}")

    print('\n' + table_overall(head).to_string(index=False))


if __name__ == '__main__':
    main()
