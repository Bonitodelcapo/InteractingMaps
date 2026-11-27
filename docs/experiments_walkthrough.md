# Experiments walkthrough — how to use it

One notebook, one section per experiment (A0–A19, index in [experiments.md](../experiments.md)).
Each section shows how its runs are made and unrolls how each report table, figure and text
number comes from them. It ends with a check against the report's own function.

## 1. Get the inputs

- **Redraw** (default) reads the report's saved runs; nothing is computed. Copy back from the cluster:
  `results/`, `experiments/` (sens, grid, ablation, best_image, maps_poster, recon_figure,
  readout_*, poisson_solvers, converge) and `report/experiments/*.csv`.
- **Live** makes short runs now; it needs the dataset in `data/<dataset>/`.
- A section with missing inputs prints `missing: <path>` and the `run_all.sh` job that makes
  them, then skips. It does not crash.

## 2. Open it

1. Python with `ipywidgets` (`pip install ipywidgets`); after installing, **restart the kernel**.
2. Rebuild if you changed a cell file in `walkthrough/cells/`: `python walkthrough/build_nb.py`.
3. Open `experiments_walkthrough.ipynb` from the repo root. Run the setup cells at the top first;
   they show the panel.
4. Then run the whole notebook or one section. Every section stands alone after the setup cells.

## 3. The panel (Live)

| Control | Does |
|---|---|
| *mode* | **Redraw**: the saved-runs lookup (below) · **Live**: this panel; runs go into `experiments/walkthrough/<ID>/` |
| *segment* (top, both modes) | the Table II segment the sections show and Live runs |
| *model* | which model Live runs; Redraw sections show every model |
| *dt*, *n_iters*, *poisson* | window (s), iterations per frame, how I is recovered from G |
| *n_frames* | length of a Live run (default 10; larger on the cluster) |
| *sweep* | sweep size in Live (A8–A13): *one point* · *one axis* · *full* |
| *delta_…* | the relaxation step sizes; the defaults are the report's |
| *reset to report defaults* | back to the report's settings |

The panel is read when a cell runs: change a value, then re-run that section.

**Redraw shows the lookup of saved runs** (every `run.json` under `results/` and `experiments/`):

1. pick a *model*: the values its saved runs have are printed as `{param: [values]}`;
2. the other dropdowns offer only those values; each choice narrows the rest (*any* undoes it);
3. the table below lists the matching runs, best mean ω error (deg/s) first, with direction
   error and β. Play one back: open its folder in the inspection viewer
   ([inspection_viewer.md](inspection_viewer.md)). *rescan runs* after copying new runs in.

The sections themselves still reproduce the report from its own runs.

**Live shows the panel above.** A configuration that is already saved is not run again:
it prints `already saved: <folder> -- use Redraw` and stops that cell (a run without a
summary, e.g. one that crashed, does not count as saved).

## 4. Read it

Each section has the same shape:

- **header**: the question, what makes the runs, which report functions read them
- **step 1**: find the runs (Redraw) or make them (Live)
- **steps 2…**: one cell per unrolled step, named after the report function (`file.py::function`)
- **outputs**: tables, figures (saved to `experiments/walkthrough/<ID>/`), text numbers
- **check**: same result as the report's function? `[ok]` · `[DIFF]` (often stale local runs:
  re-sync from the cluster) · `[skip]` (input missing)

The overview table at the top, "What you see, cell by cell", lists every section's cells.
Every ω, speed and error is in deg/s.

Never writes to `results/`, `report/experiments/` or `report/figures/`.

## 5. Test it

- Locally, one section as a smoke test: `PYTHONPATH=. python tests/test_experiments_walkthrough.py A5`
- On the cluster: the groups `live` (tracking runs) and `full` (the whole notebook).
