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

Copy the run folders back to the laptop. Keep the dataset in `data/<dataset>/` (or set the
*data in* box) for the events/APS panels; without it, maps and curves still show.

## 2. Open it (laptop)

1. Python with `ipywidgets` + `ipympl` (`pip install ipywidgets ipympl`); after installing,
   **restart the kernel**.
2. Open `inspect_viewer.ipynb` from the repo folder that contains `inspect_viewer.py`; run both cells.
   No runs found → a demo on random maps opens.

Or anywhere: `%matplotlib widget` then `Inspector(root='experiments/inspect').show()`.

## 3. Use it

**Runs tab** — *runs in*: folder to scan (*scan*) · *data in*: dataset folder ·
*dataset* → *segment* → *models* (only runs of one segment can be picked; up to 5) → *load*.
The table shows per run: model, iterations, deltas, git commit, mean error, β, replayed frames.

**View tab**

| Control | Does |
|---|---|
| ▷ play, *ms/frame*, ◀ ▶, *frame* | move through the frames |
| click on a curve | jump to that frame |
| *spikes in* + *above* · *◀ spike* / *worst frame* / *spike ▶* | jump to the previous/next frame above the threshold, or to the largest value (max over the loaded runs) |
| *iteration* | steps through a replayed frame (0 = end-of-frame state); greyed out without a replay |
| *scale* | *fixed* per model over the sequence (default) · *shared* across models · *per frame* (re-stretches: hides breakdowns) |
| *curves*, *maps* | any `tracking.csv` column, `n_events`, `omega_fullrate`; maps I, G, F |
| click on a map pixel | **pixel probe**: I, \|G\|, \|F\| of that pixel over the frames (+ over the iterations of a replayed frame), in the figure below; *clear pixel probe* removes it |

On a replayed frame the figure below also shows **mean \|V+F·G\|, mean \|G−∇I\| and \|ω\| per iteration**.

**Export tab** — *frames* range, *fps*, *format* (gif / mp4 / png), *to folder* ·
*export frames* (the range) · *export iterations* (of the current frame) · *save current view* (PNG).
File names carry dataset, segment, models and frames.
