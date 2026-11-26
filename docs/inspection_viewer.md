# Inspection viewer — how to use it

Click a spike in a curve → see the model's maps, the events and the APS image of that frame.
1 model, or 2–5 models on the same segment.

## 1. Make runs (cluster)

```bash
python report/run_experiments.py --what inspect --sequences poster --models cook,thesis_imu --replay-frames 6,14
```

- Saves the model state at every frame into `experiments/inspect/<dataset>/<model>/<run>/states/`.
- `--replay-frames` also stores every iteration of those frames (`replay/`). More later:
  `python replay_frame.py --run <run folder> --frame 20`
- Never writes to `results/`. Size ≈ 1–2 MB per frame.

Copy the run folders back to the laptop. Keep the dataset in `data/<dataset>/` (or pass `data_root=`)
for the events/APS panels; without it, maps and curves still show.

## 2. Open the viewer (laptop)

1. Python: the one with `ipywidgets` + `ipympl` (`pip install ipywidgets ipympl`).
   After installing, **restart the kernel**.
2. Open `inspect_viewer.ipynb` from the repo folder that contains `inspect_viewer.py`.
3. Run the cells top to bottom. Cell 2 lists the runs found; in cell 3 pick them by index,
   e.g. `RUNS = [runs_found[i] for i in (0, 3)]`. No runs found → a demo on fake data opens.

Or in any notebook:

```python
%matplotlib widget
from inspect_viewer import Inspector
Inspector(['experiments/inspect/poster_rotation/cook/<run>',
           'experiments/inspect/poster_rotation/thesis_imu/<run>']).show()
```

## 3. Use it

| Control | Does |
|---|---|
| **frame** slider / **click on a curve** | jump to that frame (all panels + the dashed marker) |
| **iteration** slider | steps through a replayed frame (0 = saved end-of-frame state); greyed out when no run has a replay of this frame |
| **curves** | any `tracking.csv` column, `n_events` (events per frame), `omega_fullrate` (gyro and reference \|ω\|) |
| **maps** | I, G, F |
| **share scale** | one colour scale across models (default: fixed per model over the whole sequence) |
| **export frames / iterations** | GIF (or MP4 by extension) of exactly what is shown |

Rows: one per model (maps), then events V + nearest APS, then the curves.
Runs of different segments, dt or length are refused.
