# InteractingMaps

Code developed during the project **Event-based Robot Vision** at TU Berlin,
under the supervision of Prof. Guillermo Gallego.

Joint estimation of **optical flow, image intensity and angular velocity** from a
rotating event camera, following:

1. Cook et al., *Interacting maps for fast visual interpretation*, IJCNN 2011.
2. Martel, PhD thesis (2019), Chapter 6 — the message-passing formulation and the
   IMU sensor-fusion variant.

Plus a **Contrast Maximization** extension (Prof. Gallego's suggestion) that
replaces the IMU with a purely visual, sensor-free angular-velocity anchor.

---

## The idea in one paragraph

An event camera reports only the *sign of the temporal change* of log-intensity.
Binned over a short window this gives `V ≈ ∂I/∂t` — the **sole input**. From it
the network jointly infers the intensity `I`, its gradient `G`, the optical flow
`F` and the camera's angular velocity `R` (= ω). There is no feed-forward path:
the four maps are tied by three mutual constraints and refined by iterative
relaxation until mutually consistent.

| Constraint | Equation |
|---|---|
| OFCE (brightness conservation) | `V + F·G = 0` |
| Spatial | `G = ∇I` |
| Kinematics (pure rotation) | `F = C·R` |

**The central difficulty — the β-scale ambiguity.** For any `β ≠ 0`, the
substitution `(G→βG, I→βI, F→F/β, R→R/β)` satisfies all three constraints
identically. `V` alone cannot fix the scale or the sign. This is not a bug — both
papers state it — and it is the root cause of the pure-vision drift. Every model
variant beyond the first two is an answer to it.

---

## Model variants

| `--model` | Anchor for ω | Notes |
|---|---|---|
| `cook` | none | Gauss-Seidel (sequential) updates |
| `thesis` | none | Jacobi two-phase updates |
| `thesis_imu` | IMU gyro | the thesis's own fix for the β-ambiguity |
| `thesis_cmax` | full CMax solve per frame | IMU-free (V1) |
| `thesis_cmax_v2` | 1 CMax gradient step per iteration | IMU-free (V2) |

---

## Quick start

```bash
pip install -r requirements.txt
```

Put a dataset in `data/<name>/` (`events.txt`, `calib.txt`, `imu.txt`, and
`groundtruth.txt` or `omega_gt.txt`). RPG datasets work as distributed; ECRot ROS
bags are converted with `convert_ecrot.py`.

**Track angular velocity over a segment:**
```bash
python evaluation.py --exp 2 --dataset poster_rotation --segment seg_A --model thesis_imu
```

**Find constant-ω segments in a new dataset:**
```bash
python find_segments.py data/<name>/imu.txt --n-segments 5 --export
```

**Full report pipeline:**
```bash
python evaluation.py --exp 12 --model all --n_frames 25 --no-frames  # results table
python report_tables.py --exclude shapes_rotation                    # aggregate
python evaluation.py --exp 10 --dataset poster_rotation --segment seg_A --model all
python report_analysis.py --all                                      # extra figures
```

---

## Documentation

| Document | Contents |
|---|---|
| [`architecture.md`](architecture.md) | **Start here.** End-to-end reference: the model, the pipeline from `events.txt` to a metric, every module, and all 13 experiments. |
| [`cmax/cmax.md`](cmax/cmax.md) | Contrast Maximization: design decisions, wiring of V1/V2, and measured findings. |
| [`cmax/distortion_validation.md`](cmax/distortion_validation.md) | Network-free check of the two distortion-handling strategies. |

---

## A note on segments

A *segment* is an interval over which ω is approximately constant — a property of
the **data**, recorded as `duration`. The number of frames and the frame duration
are **hyperparameters** and are clamped so a run never extends past the validated
interval. Measured constant-ω windows are **0.35–1.3 s** on the RPG datasets and
**2.4 s** on ECRot; earlier configurations ran every segment for 3.0 s regardless,
which broke the constant-ω premise.
