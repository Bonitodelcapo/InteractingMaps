# Running the report on the cluster: `report/run_all.sh`

One script produces every number and figure in the report. It runs all jobs in
parallel, skips the ones whose results are still valid, then draws the figures.

Run everything from the repository root.

## 1. What it does

Two stages, each a list of independent jobs run up to `PARALLEL` at a time:

| stage | jobs | what they produce |
|---|---|---|
| `runs` | `main_*`, `dissoc_*`, `curl_*`, `sweep_*`, `grid_*`, `anchor_*`, `ablation_*`, `beta_*`, `baseline`, `window`, `ofce_*` | tracking runs and the CSVs behind every table |
| `figures` | `converge_*`, `recon`, `dissoc`, `regfig`, `make_figures`, `make_tables`, `ofce_summary` | `report/figures/*`, `report/experiments/table_main.tex` |

The `figures` stage starts only if every `runs` job succeeded.

## 2. Stamps: why a job is skipped

After a job succeeds, `run_all.sh` writes a fingerprint to `report/.stamps/<job>`.
The fingerprint is a hash of:

1. the job's command line,
2. the code the job depends on,
3. the data files (names, sizes, times).

On the next run, a job whose fingerprint is unchanged, and whose `--out` file
still exists with at least one data row, is skipped (`[skip]`). The file is
looked up only where the job writes it: `report/experiments/` or
`experiments/<name>/`. Copies moved to a subfolder such as `_old/` do not count. The runs are deterministic, so an unchanged
fingerprint means unchanged numbers.

"The code the job depends on" is coarse. It is one hash over whole file groups:

| group | files hashed | used by |
|---|---|---|
| `HASH_REPORT` | every `*.py` except `tests/`, `diag_*.py`, `demo.py`, `debug_*.py`, `report/make_figures.py`, `report/make_tables.py`, `report/analyze_grid.py` | all `runs` jobs except `ofce_*` |
| `HASH_DIAG` | `interacting_maps/*.py`, `config.py`, `diag_omega_ladder.py` | `ofce_*` |
| `HASH_FIGURES` | every `*.py` except `tests/`, plus the contents of all run stamps | all `figures` jobs |

Consequence: **any edit to any file in `HASH_REPORT`, even a comment or a
plotting-only function in `report/run_experiments.py`, changes the fingerprint
of every run job.** A plain `run_all.sh` then reruns all of them (about 2.5 h),
although their results would be identical.

`run_all.sh` itself is not hashed. Editing it does not trigger reruns.

## 3. The three modes

| command | effect | use when |
|---|---|---|
| `bash report/run_all.sh` | runs every job whose fingerprint changed | normal case |
| `FORCE=1 bash report/run_all.sh` | ignores stamps, reruns everything | you distrust the stamps |
| `ADOPT=1 bash report/run_all.sh` | runs nothing; writes the new fingerprint for every job that an earlier log shows as `[ok]` and whose output exists | you changed code that cannot change the results |

`ADOPT` means: "I declare the existing results current for the present code."
Use it only when you are sure the change does not affect the numbers. Examples:
a comment, a docstring, a plotting function, a new figure. Never use it after a
change to the model, `config.py`, the metrics or the data loading. Those changes
need a real rerun.

`ADOPT` also stamps the `figures` jobs. If the change was in a figure function,
remove those stamps afterwards so the figures are redrawn (recipe 4.3).

## 4. Recipes

Start every long run detached, so it survives closing the terminal:

```bash
nohup bash report/run_all.sh > run_all.out 2>&1 &
```

`[1] 12345` is the background job number and the process ID. It is not an error.

### 4.1 Full run after a model or config change

```bash
git pull
nohup bash report/run_all.sh > run_all.out 2>&1 &
```

### 4.2 Rerun one job

```bash
rm report/.stamps/<job>
nohup bash report/run_all.sh > run_all.out 2>&1 &
```

Every other job is skipped.

### 4.3 Code change that cannot affect the results (only figures need redrawing)

Run these in this order:

```bash
git pull
ADOPT=1 bash report/run_all.sh
rm -f report/.stamps/{converge_dynamic,converge_bicycle,recon,dissoc,regfig,make_figures,make_tables,ofce_summary}
nohup bash report/run_all.sh > run_all.out 2>&1 &
```

Step 2 stamps every job as current. Step 3 un-stamps the figure jobs. Step 4
then runs only the `figures` stage (about 5–10 min).

### 4.4 Rerun only the failed jobs

Nothing to do: failed jobs have no stamp. Fix the cause, then run
`run_all.sh` again. Successful jobs are skipped.

## 5. Choosing `PARALLEL` and `THREADS`

Keep `PARALLEL × THREADS` at or below the cores you actually get. Defaults are
80 × 4 = 320.

```bash
nproc
cat /sys/fs/cgroup/cpu.max
```

`cpu.max` printing `max 100000` means no CPU quota. A value such as
`3200000 100000` means a quota of 32 cores; set `PARALLEL` to about 32 / 4 = 8.

The wall time is bounded by the slowest single jobs (street runs, about
25–80 min each), not by the number of jobs.

## 6. Monitoring

```bash
tail -f run_all.out                      # live progress; Ctrl+C stops only tail
pgrep -af run_all.sh                     # is a run still alive?
L=$(ls -d report/logs/*/ | tail -1)
grep -c '^\[ok\]' $L/runs.txt $L/figures.txt
grep '^\[FAIL\]' $L/*.txt                # failed jobs, each with its log file
tail -20 $L/<job>.log                    # why a job failed
```

The run is complete when `run_all.out` ends with `figures done` and `total:`.

**Never start a second `run_all.sh` while one is running.** Both would run the
same jobs and write the same files at the same time. Check with `pgrep` first.

## 7. Where the outputs are

| path | content |
|---|---|
| `results/` | run folders (`run.json`, `tracking.csv`, `tracking_plot.png`, `maps_final.npz`) of main, dissoc, curl, beta, anchor, converge |
| `report/experiments/` | table CSVs, `baseline_dt20ms*.npz`, `table_main.tex`, `window_choice.csv` |
| `report/figures/` | every report figure, PDF and PNG |
| `experiments/sweep/`, `experiments/grid/`, `experiments/ablation/`, `experiments/recon_figure/` | sweep, grid and ablation CSVs and their runs |
| `diag_ofce/` | OFCE diagnostic CSVs, plots, `summary.csv` |
| `report/logs/<timestamp>/` | per-job logs, `runs.txt`, `figures.txt` |
| `report/.stamps/` | fingerprints |
| `run_all.out` | console output |

All jobs use `--no-frames`: no per-frame images are saved.

## 8. Download

```bash
zip -qr cluster_out_$(date +%Y%m%d).zip results report/experiments report/figures report/logs report/.stamps experiments diag_ofce run_all.out
ls -lh cluster_out_*.zip
```

In JupyterLab, right-click the zip in the file browser and choose **Download**.
Unzip it in the root of the local repository so the paths match. `zip` keeps
file times, so the age of each output stays visible.

## 9. What is still manual

`make_tables.py` writes only the main table (`table_main.tex`). The numbers in
`tab:baseline`, `tab:sweep`, `tab:params`, `tab:curl` and `tab:anchor` are typed
into `report/bare_jrnl.tex` by hand from the CSVs above.

## 10. Rules that keep the stamps honest

- A job must recompute its output every time it runs. Do not add "skip if the
  output file exists" inside a job: that check is keyed by a file name, not by
  the code, so it reuses stale results after a code change. The stamps already
  decide whether a job runs.
- A job that reads another job's runs must build the exact run folder through
  `RunConfig`/`_cfg`, not a glob. A glob also matches older runs with other
  parameters.
- A job line must not depend on quote removal: `xargs -d '\n'` passes each line
  through unchanged, then `eval` parses it. A `--vary` value with `;` must stay
  in single quotes.
