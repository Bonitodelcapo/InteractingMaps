# Experiments — index

Every active experiment, research diagnostic and investigation in the repo: what
question it answers, what produces it, where its output lands and who reads it.
One row per experiment; the walkthrough notebook has one section per ID, in the
same order ([experiments_walkthrough.ipynb](experiments_walkthrough.ipynb), generated
from [walkthrough/](walkthrough/README.md): A0–A19
built). Add or remove an experiment here first.

Scanned 2026-11-25. Orchestration: [report/run_all.sh](report/run_all.sh)
(guide [report/RUN_ALL.md](report/RUN_ALL.md)); `R` below =
`python report/run_experiments.py --no-frames`.

## A. Report experiments

Run by `run_all.sh`; tables by [make_tables.py](report/make_tables.py), figures by
[make_figures.py](report/make_figures.py), quoted numbers by
[text_numbers.py](report/text_numbers.py). All on the Table II segments
(`report_runs.SEGMENTS`), `C_full`, 20 ms, 75 iterations unless stated.

| ID | Experiment | Question | Produced by (`run_all.sh` job) | Output | Read by | In report |
|----|-----------|----------|------------------|--------|---------|-----------|
| A1 | Window choice | How constant is ω over a segment, how far does the scene move in one window? (GT only, no network) | `window`: `R --what window` | `report/experiments/window_choice.csv` | text_numbers (`segment_speeds`; the window lines of `intensity` and `track_length`) | text |
| A2 | Every model, 3 s | Every model on every segment, 150 frames | `main_<seq>_<model>`: `R --what main` | `results/`, `main_base_*.csv` | text_numbers (`poster_segments`, `inloop_identical`, `track_length`; also the 3 s β runs of A6) | text |
| A3 | Main table (Table III), 1.5 s | Every model on every segment, 75 frames; also the network column of the baseline table | `dissoc_<seq>_<model>`: `R --what main --duration 1.5` | `results/`, `dissoc_1p5s_*.csv` | table_main, table_baseline, fig_tracking, fig_dissociation, fig_maps/fig_appendix (with A13), fig_image_settings (with A12), text_numbers (`unanchored`, `reversal`, `reference`, first part of `intensity`) | tab:main, tab:baseline, fig:tracking, fig:dissociation |
| A4 | Curl regulariser | Same with `delta_curl=0.6` | `curl_<seq>_<model>`: `R --what main --duration 1.5 --vary delta_curl=0.6` | `main_curl_*.csv` | table_curl, table_image | tab:curl |
| A5 | Anchor alone (baseline) | What do CMax and the gyro score with no network? | `baseline`: `R --what baseline --duration 1.5` | `baseline_dt20ms*.csv/.npz` | table_baseline, fig_cmax_baseline, text_numbers | tab:baseline |
| A6 | β vs speed | Does β depend on rotation speed? 7 real segments × 3 anchor settings | `beta_<i>`: `R --what beta --index i` | `results/` | fig_beta_speed, text_numbers | fig:beta_speed |
| A7 | Anchor across poster segments | Every model on poster seg_A / seg_D (3 s) | `anchor_<seg>_<model>`: `evaluation.py --exp 2` | `results/poster_rotation/` | table_anchor | (table not in tex) |
| A8 | Sensitivity sweep | One parameter at a time around the adopted config (δ_IMU, δ_FR, n_iters, dt, relaxation rates) | `sens_{cmax,imu,rates}_<seq>`: `R --what sweep --name sens` | `experiments/sens/` | table_sweep, fig_sensitivity, text_numbers | tab:sweep |
| A9 | (δ_FR, anchor) grid | Full grid, CMax-anchor, 1.5 s | `grid_<seq>_FR<x>`: `R --what sweep --name grid --mode grid` | `experiments/grid/` | table_grid, table_params, fig_sensitivity, text_numbers | tab:params |
| A10 | Old coordinate sweep | One axis at a time around δ_FR 0.5 / anchor 0.3 | `sweep_<seq>_{cmax,imu}_<v>`: `R --what sweep --name sweep` | `experiments/sweep/` | text_numbers reads only `…_cmax_delta_FR=0.5.csv` | ⚠ mostly unread, see below |
| A11 | Regulariser ablation | What do shrink-I, MAP and curl each do? | `ablation_<seq>_<v>`: `R --what sweep --name ablation` | `experiments/ablation/` | table_regularisers, fig_regularisers, text_numbers | tab:regularisers, fig:regularisers |
| A12 | Image settings | Best image: dt 10 ms × curl on/off | `best_image_<seq>_<model>`: `R --what sweep --name best_image --mode grid` | `experiments/best_image/` | table_image, fig_image_settings, text_numbers | tab:image, fig:image_settings |
| A13 | Maps on poster | Every model stopped at frame 44 | `maps_poster_<model>`: `R --what sweep --name maps_poster` | `experiments/maps_poster/` | fig_maps, fig_appendix, text_numbers (`polarity`) | fig:maps, appendix |
| A14 | Convergence within a frame | How does I sharpen over the iterations? | `converge_{dynamic,bicycle}`: `R --what converge` | `converge_path()` npz | fig_converge | fig:converge_dynamic, appendix |
| A15 | Reconstruction | dt 20 vs 10 ms, iterative vs exact Poisson | `recon`: `R --what recon` | `experiments/recon_figure/` | fig_recon | fig:recon |
| A16 | Read-out fft vs dct | Which exact solver gives the better image? | `readout_{20ms,10ms}`: `R --what readout` | `experiments/readout_fft_vs_dct/`, `readout_dt10ms/` | table_recon, text_numbers (10-vs-20 ms lines of `intensity`) | tab:recon |
| A17 | In-loop Poisson | Where to apply the exact solve: in the loop or as a read-out? | `poisson`: `R --what poisson` | `experiments/poisson_solvers/` | text_numbers (`intensity_update`) | text |
| A18 | Lens distortion | Forward model vs OpenCV, warp check; the figure needs only raw events | `distortion_test`: `tests/test_distortion_cmax.py`; figure: fig_distortion | stdout; figure | fig_distortion | fig:distortion |
| A19 | Adopted parameters | The step sizes the report uses | `config.py` (no run) | n/a | table_params | tab:params |

Figures drawn but not included in `bare_jrnl.tex`: `fig_cmax_baseline`,
`fig_sensitivity`. Tables written but not in the tex: `table_anchor`, `table_grid`.

## B. Research diagnostics (not in the report)

| ID | Diagnostic | Question | Entry point | Output | Walkthrough |
|----|-----------|----------|-------------|--------|-------------|
| B1 | ω-recovery oracle ladder | Where does ω recovery fail? Rungs 0 / 1a / 1b / 1c inject GT flow or GT R | `diag_omega_ladder.py --mode run / maps / plots` | `diag_ladder_*.csv`, `diag_maps_<seg>.npz`, P1–P5 plots | [diag_omega_ladder.ipynb](diag_omega_ladder.ipynb) |
| B2 | OFCE step instability | Is the explicit G step unstable above \|F\| ≈ b? 7 variants | `run_all.sh` jobs `ofce_<v>`, `ofce_summary`: `diag_omega_ladder.py --mode ofce / ofce_summary` | OFCE dir `summary.csv` + plots | none yet |
| B3 | Distortion ablation | `undistort_events` vs `C_full` | `evaluation.py --exp 9` | `results/…` | none |
| B4 | Basin of attraction | From how far off does R converge? | `evaluation.py --exp 6` | `results/…` | none |

## C. Validation checks (tests that answer a question)

| ID | Check | Question | Script |
|----|-------|----------|--------|
| C1 | CMax front end | Does CMax ω match GT and IMU? | `tests/test_cmax_frontend.py` |
| C2 | CMax gradient | Analytic gradient vs finite differences, synthetic and real | `tests/test_cmax_gradient.py` |
| C3 | CMax V1 vs V2 | Do the two CMax integrations agree? | `tests/test_cmax_v1_vs_v2.py` |
| C4 | Poisson solvers | Do iterative / fft / dct recover I from G? | `tests/test_poisson_solvers.py` |
| C5 | Golden tracking | Do the tracking numbers still match the golden values? | `tests/test_golden_tracking.py` |

## D. Model walkthroughs (existing notebooks)

| Notebook | Covers | Status |
|----------|--------|--------|
| [cook_walkthrough.ipynb](cook_walkthrough.ipynb), [thesis_walkthrough.ipynb](thesis_walkthrough.ipynb), [thesis_imu_walkthrough.ipynb](thesis_imu_walkthrough.ipynb), [cmax_v1_walkthrough.ipynb](cmax_v1_walkthrough.ipynb), [cmax_v2_walkthrough.ipynb](cmax_v2_walkthrough.ipynb) | The inside of each model, one frame | current (first three untracked) |
| [demo.ipynb](demo.ipynb) | Live animation of the maps | current; a demo, not an experiment |

Plumbing, not experiments: [experiment_tracking.py](experiment_tracking.py) (run.json,
MLflow, `find_run`), `report/report_runs.py`, `evaluation.py --exp 5` (videos).

## E. Open investigations (no code yet)

Tracked in [docs/notes/todo.md](docs/notes/todo.md) (ω recovery: clip dependence P1,
overshoot, direction error, no-clip blow-up, zig-zag residuals) and
[docs/notes/cmax_todo.md](docs/notes/cmax_todo.md) (CMax bias, IWE borders, V1/V2 t_ref, …).

## Flags found while scanning

- **A10 orphan runs:** `run_all.sh` writes ~80 `sweep_*` jobs, but only the
  `delta_FR=0.5` base point is read (text_numbers). The sens sweep (A8) replaced
  the rest. Candidate for removal from `run_all.sh`.
- **Not yet in the walkthrough:** `text_numbers.little_effect_and_window` (δ_GI
  and the reconstruction; reads the A8 sens rows) has no section yet.
- **`fig_sensitivity` has no fallback:** it reads only `experiments/sens/`, while
  `table_sweep` and `text_numbers` fall back to `report/experiments/sens_*.csv`.
- **Old dataset name in local sens csvs:** the synthetic rows say
  `bycicle_sinthetic`, so their means are nan (report code and walkthrough alike)
  until the sens csvs are re-synced.
- **Report functions that raise on a missing dataset:** `fig_dissociation`,
  `fig_recon`, `compute_maps`, `fig_regularisers`, `text_numbers.reference` and
  `image_window` (and `report_runs.image_search` when a kind has no runs). The
  walkthrough checks them only where the data is present.
