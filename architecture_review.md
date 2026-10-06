# Architecture Review — InteractingMaps

> Review of the `main`-branch structure as of 2026-10-06, written to decide which
> refactors are worth doing before the thesis is finalized. Companion to
> [architecture.md](architecture.md) (what the system does) and
> [cmax/cmax.md](cmax/cmax.md) (the CMax branch). This document is about *how the
> code is organized* and where that organization creates risk — not about the
> model.

> **Implementation status (2026-10-07).**
> - **R1 Reproducibility** — ✅ done. One merged `run.json` (provenance + config +
>   summary) with git commit/dirty flag; local MLflow mirror (`provenance.py`).
> - **R2 Doc drift** — ✅ done. architecture.md reconciled to `C_full` default +
>   `POISSON_MODE`, 5 models, exp 9; CLAUDE.md updated.
> - **R3 Split `evaluation.py`** — ✅ done (1919→1659 lines). Pure move into
>   `eval_io.py` / `metrics.py` / `viz.py`, re-exported; golden test unchanged.
> - **R4 Network contract** — ✅ done. Shared clip constants + R-least-squares in
>   `camera.py` (`CLIP_*`, `build_R_normal_equations`, `solve_R_lstsq`); the one
>   intended Gauss-Seidel/Jacobi difference stays local.
> - **R6 Regression guard** — ✅ done (`test_golden_tracking.py`); all three
>   models match golden to ~1e-11 before and after R3/R4. Cook init seeded for
>   determinism; a Windows-console UTF-8 crash in the print path was fixed.
> - **R5 Dispatch & sprawl** — ◑ partial. Done: `demo.py` no longer imports
>   `archive_/` (uses `eval_io`), `archive_/README` marks it frozen, dead
>   commented block removed from `camera.py`. **Not done (deferred, see R5):** the
>   `--exp N` → name-based CLI change (breaking, heavily documented) and the
>   debug-viewer consolidation — left for explicit go-ahead.

## Executive summary

The foundations are sound for a research codebase: **low coupling (13/100), no
circular dependencies, a clean 7-package dependency set**, and the report tooling
already imports the pipeline (`import evaluation as E`) rather than duplicating it.
This is better than most thesis code.

The risks are not structural rot — they are **concern-mixing in one god-module,
reproducibility provenance gaps, documentation drift, and two parallel network
implementations held in sync only by convention**. For a thesis, where the
deliverable is *numbers you can defend* and *documentation that matches the code*,
those four are the ones that bite.

Priority order (rationale in the roadmap at the end):
**R2 (doc drift) → R1 (provenance) → R3 (split evaluation.py) → R6 (golden test)
→ R4 (network contract) → R5 (dispatch/sprawl).**

| # | Review track | Severity | Effort | Behavior-changing? |
|---|---|---|---|---|
| R1 | Reproducibility & provenance | High | Low–Med | No |
| R2 | Documentation ↔ code drift | High | Low | No |
| R3 | `evaluation.py` decomposition | Medium | Medium | No (pure move) |
| R4 | Two-network shared contract | Medium | Medium | No (if careful) |
| R5 | Experiment dispatch & script sprawl | Low–Med | Medium | No |
| R6 | Regression guard on headline numbers | Medium | Low–Med | Adds tests |

A note on the automated scan: `project_architect.py` reported `RunConfig` as
"~1850 lines" — that is a **false positive** (RunConfig is `evaluation.py:71–261`,
~190 lines; the tool mis-attributed the whole module to the first class). The real
finding it stumbled onto is correct: `evaluation.py` is 1919 lines and does too
much. Everything below is verified against the source, not the scan.

---

## R1 — Reproducibility & provenance *(High / Low–Med)*

**What is already good (do not rebuild this).** Every tracking run writes
`params.json` (`RunConfig.to_dict()`, `evaluation.py:216`) capturing
`distortion_mode`, `poisson`, the full `params` dict, `n_iters`, `initial_R`,
segment timing — and a separate `summary.json` with the metrics. So an individual
run *is* self-describing about its configuration.

**The gaps.**
1. **No code provenance.** Neither file records the git commit, a dirty-tree flag,
   or a timestamp. `run_experiments.py` recomputes every reported number precisely
   *because* "`results/` mixes runs from configurations that have since changed" —
   that comment is the tell: there is no way, looking at a run, to know *which code*
   produced it. A thesis reviewer asking "reproduce Table III" has no anchor.
2. **The global-default trap.** `DISTORTION_MODE`, `POISSON_MODE`, `THESIS_PARAMS`,
   `COOK_PARAMS`, `ITERS_PER_FRAME` are module-level mutable globals in `config.py`.
   `RunConfig` reads them as fallbacks (`evaluation.py:85,89,113,124…`). An
   interactive `python evaluation.py --exp 2` uses whatever the globals currently
   are; a report run pins `C_full` explicitly. The two can silently diverge, and
   only the explicit report path is safe.
3. **`summary.json` and `params.json` are separate files.** Any aggregation that
   reads one CSV of summaries loses the config link unless it re-opens `params.json`
   per row. (The report CSVs do carry some columns, but not the full config.)

**Recommendation.**
- Add a `provenance` block to `to_dict()`: `git rev-parse HEAD`, a dirty flag
  (`git status --porcelain` non-empty), ISO timestamp, and the Python/NumPy
  versions. One helper, written once, read everywhere.
- Fold the config into `summary.json` (or write a single `run.json`) so a result
  is readable standalone.
- Treat the `config.py` globals as **defaults for interactive use only** and have
  the report/experiment entry points assert the pinned values (or log a loud
  warning when a run uses a non-report default). This makes the "safe path" the
  default path.

**Why first-tier:** protects the integrity of the numbers in the thesis, and is
non-behavioral (nothing in the model changes).

---

## R2 — Documentation ↔ code drift *(High / Low)*

For a thesis, `architecture.md` is a deliverable, and it has drifted from the code:

- **Distortion path.** `architecture.md` §6.2/§6.3/§7 describe `undistort_events`
  (cv2, before binning) as *the* distortion handling. The code now defaults to
  `DISTORTION_MODE='C_full'` (`config.py:214`) — distortion carried inside the `C`
  matrix. The doc describes the non-default path as if it were current.
- **`POISSON_MODE`** (`iterative`/`fft`/`dct`, `config.py:224`) is not mentioned in
  architecture.md at all, yet it changes how `I` is recovered from `G`.
- **Experiment list.** architecture.md §6.6 lists experiments 1–8; the code has
  `--exp 9` (distortion ablation), and `--exp 5` is reassigned to the batch video
  builder (`evaluation.py:1898`).
- **Model list.** architecture.md §3 lists three variants; the harness now has five
  (`+thesis_cmax`, `+thesis_cmax_v2`). (cmax.md covers these, but the top-level doc
  doesn't cross-reference them.)

**Recommendation.** One editing pass on architecture.md: mark `C_full` as the
default and `undistort_events` as the ablation alternative; add a short
`POISSON_MODE` paragraph; sync the experiment and model tables; add a pointer to
cmax.md. (I already flagged the first two in the new `CLAUDE.md`, but the source
docs are what a reader cites.)

**Why first-tier:** cheapest high-value item; a wrong architecture doc is worse
than none when defending the work.

---

## R3 — `evaluation.py` decomposition *(Medium / Medium)*

`evaluation.py` is 1919 lines and mixes six distinct concerns in one file:

| Concern | Examples | Belongs in |
|---|---|---|
| Run configuration | `RunConfig`, `resolve_best_kwargs` | stays (`evaluation.py` or `run_config.py`) |
| Data I/O | `load_imu`, `load_groundtruth`, `load_omega_gt`, `get_gyro_for_frame`, `gt_omega_body`, `get_reference_omega` | `io.py` |
| Metrics | `compute_metrics`, `curl_share`, `_recon_scores`, `_corr_to_aps`, `_intensity_stats` | `metrics.py` |
| Visualization | `flow_to_rgb`, `grad_to_rgb`, `normalise`, `normalise_robust`, `_plot_tracking`, `_save_3col_frame` | `viz.py` |
| Network factory | `make_network` | `factory.py` or stays |
| Experiments | 12 × `experiment_*` | stays |

The seams are already real but implicit: `report/run_experiments.py` reaches into
`E.RunConfig`, `E.make_network`, `E.experiment_tracking`, `E.load_*`, etc. Extracting
those into named modules makes the report's true dependency surface explicit and
shrinks the file a maintainer has to hold in their head.

**Recommendation.** Pure *move* refactor (no logic change): pull `io.py`,
`metrics.py`, `viz.py` out, re-export from `evaluation` for backward compatibility
(`from .io import *`) so `report/` keeps working unchanged. Low risk given zero
circular dependencies. **Do R6 first** so the move is provably behavior-preserving.

---

## R4 — Two-network shared contract *(Medium / Medium)*

`network.py` (Cook) and `network_dissertation.py` (thesis) share real logic that is
duplicated, not shared:

- **Identical clip bounds, as magic numbers in two files.** Cook:
  `F∈[−10,10], G∈[−5,5], I∈[−10,10]` (`network.py:267–269`); thesis:
  `I∈[−10,10], G∈[−5,5], F∈[−10,10]` (`network_dissertation.py:638–640`). Same
  numbers, two sources of truth. Change one, forget the other → the two networks
  silently stop being comparable.
- **Same R least-squares.** Both solve `R = M⁻¹·ΣCᵀF` with precomputed `M⁻¹`
  (architecture.md §6.4/§6.5).
- **Same kinematic matrix** `build_kinematic_matrix` (shared already — good).

`config.py:203` literally asserts the design intent: *"matched to THESIS_PARAMS so
the two networks differ only in the update scheme, as the report requires."* That
invariant is currently enforced by hand, in comments.

**Recommendation.** Extract the shared pieces — clip bounds (as named constants),
the `M⁻¹` R-update, perhaps a tiny `InteractingMapsBase` — so "only the update
scheme differs" is structural. Keep Gauss-Seidel vs Jacobi as the one documented
difference. **Behavior-sensitive** — guard with R6 and diff the output numbers
before/after.

---

## R5 — Experiment dispatch & script sprawl *(Low–Med / Medium)*

- **Hand-numbered dispatch.** `experiments = {1:…, 2:…, …, 9:…}` of lambdas
  (`evaluation.py:1892`) with reassigned/aliased slots (4 → tracking-with-frames,
  5 → batch video). Numbers carry no meaning; adding one means editing a dict and
  the `--help` string. `report/run_experiments.py` already uses the better pattern
  (`--what main|grid|window|…` dispatched by name).
- **Debug/demo overlap.** `demo.py`, `debug_pipeline.py`, `debug_visual.py` are
  three partly-overlapping ad-hoc viewers at the root.
- **Stale `archive_/`.** `validation.py`, `validation_convergence.py` (1176 lines),
  `validation_convergence2.py` (746 lines) still `from config import …` live
  symbols. They are dead but *importable*, so they rot silently and mislead a
  `grep` (e.g. the `RESEED_R_FROM_GT` oracle architecture.md §6.6 warns about lives
  in this archived lineage, not on `main`).

**Recommendation.** Name-based subcommands for `evaluation.py` (mirror
`run_experiments.py`); consolidate the three debug viewers into one entry point;
either delete `archive_/` or move it outside the import path and drop a one-line
`archive_/README` saying it is frozen and not maintained.

---

## R6 — Regression guard on headline numbers *(Medium / Low–Med)*

Tests are standalone `print([ok]/[FAIL])` scripts (`test_poisson_solvers.py`,
`test_cmax_frontend.py`, …) — no runner, no gate, and **nothing asserts the
reported ω-error stays put across a refactor.** R3 and R4 both move code that feeds
the thesis numbers; today there is no automated way to prove they didn't shift a
result.

**Recommendation.** One golden-value test: run `experiment_tracking` on a single
small segment (few frames, fixed seed/params) and assert `mean_err_deg_s` within a
tolerance of a committed golden value. Cheap, and it is the safety net that makes
R3/R4 safe to attempt. Optionally wire the existing `test_*.py` into a trivial
runner so "run the tests" is one command.

---

## Roadmap

```
Phase 0 (cheap, non-behavioral, do now)
  R2  fix architecture.md drift          ~1 editing pass
  R1  add provenance + fold config        one helper, touch to_dict/summary

Phase 1 (safety net, then the big move)
  R6  golden-value regression test        establishes the invariant
  R3  split evaluation.py (io/metrics/viz) pure move, guarded by R6

Phase 2 (behavior-sensitive, guarded)
  R4  shared network base/constants        diff numbers before/after via R6
  R5  name-based dispatch + archive_ purge  maintainability cleanup
```

**Not recommended.** No database, microservices, or service-boundary work — this is
a single-process numerical pipeline and those patterns do not apply. No rewrite of
the model or the CMax design; those are correct per their own validated references.
The entire value here is in *provenance, documentation fidelity, and keeping the two
networks honestly comparable* — everything a thesis is judged on.
