# experiments_walkthrough.ipynb — sources

The notebook at the repo root is **generated** from the files here. Edit a cell
file, then rebuild from the repo root:

```bash
python walkthrough/build_nb.py                        # all sections -> experiments_walkthrough.ipynb
python walkthrough/build_nb.py tmp.ipynb A2 A5        # core + A2 + A5 only (a test build)
```

One file per section, `cells/cells_<NN>_<ID>.py`; `NN` orders the sections and
the index of IDs is [experiments.md](../experiments.md). To add an experiment, add
its row there and a new cell file here; to remove one, delete both.

## Shared cells (read before writing a section)
- `cells_00_core.py` — setup, panel `W`, the "What you see, cell by cell" overview,
  helpers: `params()`, `walk_dir(ID)`, `need(paths, jobs)`, `check(what, got, want,
  rtol=1e-2)` (loose, never raises, prints [ok]/[DIFF]/[skip]), `nums(lines)`,
  `note_redraw(p, nf)`, `redirect(module, **attrs)`, `READY`, the `%%section ID` magic,
  `to_deg_s`.
- `cells_02_A2.py` — the template section and the `A2-defs` cell: `reference_and_gyro`,
  `gyro_floor`, `main_cfg`, `main_row`, `run_dir` (= `text_numbers.run`'s lookup),
  `tracking`, `run_tn` (= `text_numbers.run`), `find_runs`, `check_live_main`, `REAL`/`SYN`.
- `cells_08_A8.py` holds the `sweep-defs` cell (sweep jobs and helpers, A8–A13);
  `cells_14_A14.py` holds `image-defs` (figure-check helpers, A14–A18).

## What a section is
`CELLS = [(kind, tags, src), ...]`, kind `'md'` or `'code'`, tags include the ID.
1. md header `## <ID> · <title>`: **Question**, **Made by** (command, `run_all.sh`
   job → outputs), **Read by** (report functions → tex labels; say "not in the tex").
2. code: `p = params()` and step 1 — find (Redraw) or make (Live) the runs; sets
   `READY[ID]` (False before a Live run, True after it succeeds). Missing inputs →
   `need([...], [jobs])`. Live writes only under `walk_dir(ID)`, sized by `n_frames`
   (and `sweep` for sweeps).
3. `%%section <ID>` cells, one per unrolled step, `# step k -- what`, naming the
   report function unrolled (`file.py::function`).
4. outputs: tables, figures (saved to `walk_dir(ID)`), text lines.
5. last `%%section <ID>` cell: the same-code check against the report's own
   function, its output globals redirected (`make_tables.OUTDIR`, `make_figures.FIG`
   / `save`, `text_numbers.OUT`, `run_experiments.OUTDIR` / `_run`). Never write into
   `results/`, `report/experiments/`, `report/figures/`.
6. add the section's rows to the overview table in `cells_00_core.py`.

## Rules
- Mirror the report code (same formulas, same lookups); reuse the shared cells and
  the report modules' small helpers.
- Every shown angular velocity, speed or ω error is in deg/s; never print rad/s.
- Section globals carry the ID suffix (`rows_A5`) or stay inside functions.
- Missing inputs or datasets must not raise: print `missing:` / `[skip]`.
- Redraw never runs the network.

## Tests
`tests/test_experiments_walkthrough.py` (core, A2, A3, harness) and
`tests/walk_A*.py` (the other sections). Locally run one group at a time as a
smoke test, e.g. `PYTHONPATH=. python tests/test_experiments_walkthrough.py A5`;
`WALK_NB=<test build>` points the harness at a test build. The groups `live`
(tracking runs) and `full` (the whole notebook) run on the cluster.
