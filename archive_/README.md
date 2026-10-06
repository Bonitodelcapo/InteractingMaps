# archive_ — frozen, not maintained

Superseded validation scripts kept only for reference. **Nothing in the live
pipeline imports this directory** (as of the R5 cleanup in
[architecture_review.md](../architecture_review.md); `demo.py` used to pull
`load_imu`/`get_gyro_for_frame` from here and now uses `eval_io.py`).

Do not build on these files:

- They predate the current model defaults (`DISTORTION_MODE='C_full'`,
  `POISSON_MODE`, the tuned `delta_*`), so any numbers they produce are stale.
- They still `from config import ...` live symbols, so they *look* runnable but
  are not kept in sync with the pipeline and will drift.
- `validation.py` here is the lineage of the `RESEED_R_FROM_GT` "perfect oracle"
  that does **not** exist on `main` (see architecture.md §6.6) — a grep hit here
  is not evidence of a feature in the live code.

Canonical equivalents live in `evaluation.py`, `eval_io.py`, `metrics.py`,
`viz.py` and `provenance.py`.
