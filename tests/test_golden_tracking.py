"""
golden-value regression guard (R6 of architecture_review.md).

Runs experiment_tracking on one small fixed configuration for each of the three
core models and checks that the headline metric (mean angular-velocity error,
deg/s) still matches a committed golden value. This is the safety net for the
R3/R4 refactors: a move that silently shifts a reported number trips here.

The pipeline is deterministic for these models (the thesis init seeds
default_rng(42); the Cook init was seeded to match), so the tolerance is tight --
a real regression moves the number by far more than ATOL.

Standalone script, like the other test_*.py:  python test_golden_tracking.py
Exits non-zero on any mismatch. Skips (exit 0) if the dataset is not present
locally -- data/ is git-ignored, so this guard only runs where the data lives.
"""

import os
import sys
import shutil

# Keep the probe out of the MLflow store and the real results/ tree.
os.environ.setdefault('IM_MLFLOW', '0')

import evaluation as E

# Fixed small config. poster_rotation seg_C exists in the committed config.
DATASET = 'poster_rotation'
SEGMENT = 'seg_C'
N_FRAMES = 3
N_ITERS = 10
OUT_ROOT = 'experiments/golden_test'
ATOL = 0.02  # deg/s; deterministic runs, so this only absorbs BLAS/FP noise.

# Golden mean_err_deg_s, measured 2026-10-06 (commit before the R3/R4 refactors).
# Regenerate deliberately (and note why in the commit) if the model is changed:
#   for m: RunConfig(model=m, dataset=.., segment=.., n_frames=3, n_iters=10)
GOLDEN = {
    'thesis':     8.8073779271,
    'thesis_imu': 4.6292360480,
    'cook':       5.6864297554,
}


def main():
    paths = E.get_dataset_paths(DATASET)
    if not os.path.exists(paths['events']):
        print(f"[skip] {DATASET} not present locally ({paths['events']}); "
              f"golden guard needs the dataset. Skipping.")
        return 0

    print(f"golden tracking guard: {DATASET}/{SEGMENT}, "
          f"{N_FRAMES} frames x {N_ITERS} iters, atol={ATOL} deg/s\n")

    failures = 0
    try:
        for model, golden in GOLDEN.items():
            rc = E.RunConfig(dataset=DATASET, segment=SEGMENT, model=model,
                             n_frames=N_FRAMES, n_iters=N_ITERS, out_root=OUT_ROOT)
            summary = E.experiment_tracking(rc, save_frames=False)
            got = summary['mean_err_deg_s']
            delta = abs(got - golden)
            ok = delta <= ATOL
            failures += not ok
            print(f"  [{'ok ' if ok else 'FAIL'}] {model:12s} "
                  f"mean_err={got:.6f} deg/s  (golden {golden:.6f}, "
                  f"|d|={delta:.2e})")
    finally:
        shutil.rmtree(OUT_ROOT, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED: {failures}/{len(GOLDEN)} models drifted beyond {ATOL} deg/s.")
        return 1
    print(f"OK: all {len(GOLDEN)} models match their golden values.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
