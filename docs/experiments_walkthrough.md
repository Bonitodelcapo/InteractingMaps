# Experiments walkthrough — how to use it

One notebook, one section per experiment (A0–A19). It starts with the **overview**, the
content of [experiments.md](../experiments.md) (copied in at build time); each experiment ID
there links to its section, and each section header links back (*↑ overview*).
Each section shows how its runs are made and unrolls how each report table, figure and text
number comes from them.

## 1. Get the inputs

- **Redraw** reads saved runs; nothing is computed. Copy back from the cluster:
  `results/`, `experiments/` (sens, grid, ablation, best_image, maps_poster, recon_figure,
  readout_*, poisson_solvers, converge) and `report/experiments/*.csv`.
- **Live** computes; it needs the dataset in `data/<dataset>/`.
- A section with missing inputs prints `missing: <path>` and the `run_all.sh` job that makes
  them, then skips. It does not crash.

## 2. Open it

1. Python with `ipywidgets` (`pip install ipywidgets`); after installing, **restart the kernel**.
2. Rebuild if you changed a cell file in `walkthrough/cells/` or `experiments.md`:
   `python walkthrough/build_nb.py`.
3. Open `experiments_walkthrough.ipynb` from the repo root; run the setup cells at the top.
4. Jump from the overview to an experiment and run its cells. Every section stands alone
   after the setup cells. One experiment at a time.

## 3. The experiment's form

Each converted experiment (so far **A2, A3**) starts with its own form: its knobs only,
pre-set to the report's values, its own **Redraw / Live** switch and *reset to report*.
Changing a form never changes another experiment.

- **Redraw = browse what is already computed for this experiment.** The dropdowns offer only
  values that saved runs have, and each choice narrows the others. The section then draws its
  tables, figures and numbers from the runs the form selects. No match →
  `no saved run matches the <ID> form`.
- **Live = compute something new.** Free values; *n_frames* starts at **10** (smoke size;
  raise it on the cluster), the report's value is shown under each knob. A configuration that
  is already saved is **not rerun**: the section shows
  **`SAVED RUN (not rerun): <folder>`** and uses that run. New runs go to
  `experiments/walkthrough/<dataset>/<model>/<run folder>`, the same format as `results/`, so
  they can be moved there later as they are.
- **Away from the report's configuration** the steps that only read the report's fixed runs
  print `[skip] report-only step`, and the comparison with the report's numbers prints
  `[skip] not the report configuration`.

The form is read when its section runs: change a value, then re-run that section.

**Sections not yet converted** (A1, A4–A19) still read the old global panel: *mode*,
*segment*, *model*, *dt*, *n_iters*, *poisson*, *n_frames*, *sweep*, the deltas. It goes away
once every section has a form. Under the overview, the **saved-runs lookup** browses every
saved `run.json` (model first, then narrowing dropdowns, then the matching runs, best ω error
first); open a run's folder in the inspection viewer ([inspection_viewer.md](inspection_viewer.md))
to play it back.

## 4. Read it

Each section has the same shape:

- **header**: the question, what makes the runs, which report functions read them
- **form** (converted sections) and **step 1**: find the runs (Redraw) or make them (Live)
- **steps 2…**: one cell per unrolled step, named after the report function (`file.py::function`)
- **outputs**: tables, figures (saved to `experiments/walkthrough/<ID>/`), text numbers
- **check**: same result as the report's function? `[ok]` · `[DIFF]` (often stale local runs:
  re-sync from the cluster) · `[skip]` (input missing, or not the report configuration)

Every ω, speed and error is in deg/s. Never writes to `results/`, `report/experiments/` or
`report/figures/`.

## 5. Test it

- Locally, one group as a smoke test: `PYTHONPATH=. python tests/test_experiments_walkthrough.py forms`
- On the cluster: the groups `live` (tracking runs) and `full` (the whole notebook).
