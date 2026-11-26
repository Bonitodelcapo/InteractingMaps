CELLS = [
('md', ('A0',), r'''
## A0 · Inside one tracking run (linked, not unrolled)

Every section below calls the tracking run `evaluation.py::experiment_tracking`
as one step: events → frames → network → per-frame ω → `tracking.csv` +
`run.json`. What happens inside it, model by model, is unrolled in:

- [cook_walkthrough.ipynb](cook_walkthrough.ipynb)
- [thesis_walkthrough.ipynb](thesis_walkthrough.ipynb)
- [thesis_imu_walkthrough.ipynb](thesis_imu_walkthrough.ipynb)
- [cmax_v1_walkthrough.ipynb](cmax_v1_walkthrough.ipynb) (`thesis_cmax`)
- [cmax_v2_walkthrough.ipynb](cmax_v2_walkthrough.ipynb) (`thesis_cmax_v2`)
'''),
]
