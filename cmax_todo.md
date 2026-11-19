# CMax — TODO

Working list for the CMax angular-velocity work (V1 `thesis_cmax`, V2 `thesis_cmax_v2`).
Design and background: [cmax.md](cmax.md). Step-by-step inspection:
[cmax_v1_walkthrough.ipynb](cmax_v1_walkthrough.ipynb), [cmax_v2_walkthrough.ipynb](cmax_v2_walkthrough.ipynb).

**Priority:** **P1** = likely affects the reported ω accuracy; do first. **P2** = correctness, consistency or
a design question; cheap or second-order. **P3** = nice to have.

**Categories:**
- **Validation:** is the yardstick right?
- **Accuracy:** the score or optimizer shifts ω.
- **Parity:** differs from Gallego's `cmax_slam`.
- **Consistency:** V1 and V2 disagree.
- **Design:** a question about how CMax is coupled to the network.
- **Performance**

Numbers come from the walkthrough smoke run (poster seg_C, 3 frames, °/s) unless stated otherwise.

## To fix

- [ ] **P2 · Consistency — Make V1 and V2 use the same CMax reference time `t_ref`.**
  V1 (`evaluation.py::experiment_tracking`) passes the frame centre `0.5·(t_lo+t_hi)` to
  `CMaxAngularVelocity.estimate`. V2 (`network_dissertation.py::InteractingMapsThesis.step`) uses the
  midpoint of the event timestamps `0.5·(t.min()+t.max())`. `estimate()` also defaults to the event
  midpoint. Pick one convention and use it in all three places.
  The gap was 0.001 ms on seg_C frame 0, but it can be larger on sparse windows.

## To investigate

- [ ] **P1 · Validation — Check the reference before calling CMax biased.** On seg_C frames 0→2:
  - reference: 46.5 → 52 → 57.7
  - gyro: 48.2 → 60 → 67.4
  - CMax: 54.7 → … → 75

  The gyro error also grows (2.2 → 8.1 → 10.1). Two independent sensors are drifting away from the
  smoothed Vicon in the same direction, so part of the "overestimate" may be the reference. Check:
  - the time offset between Vicon and the gyro
  - the Vicon (±50 ms) smoothing under acceleration (seg_C speeds up)
  - the gyro-to-camera frame

  Three frames on one segment is too little to say anything firm.
- [ ] **P1 · Accuracy — CMax overestimates |ω_x|** (pending the validation item above). On frame 0, CMax
  gives ω_x = 54.7, the reference is 46.5 and the gyro 48.2. The NM optimum on the blurred score and the CG
  optimum on the unblurred score agree, so the score's peak itself is shifted. Over 3 frames: CMax error
  13.8, gyro error 6.8.
- [ ] **P1 · Accuracy — The contrast peak is flat because the window is short.** In 20 ms, a 10 °/s error moves an
  edge event about 0.35 px, which is under one pixel and under the blur. The score at the reference is
  only about 1% below the peak. With a peak that shallow, any small bias in the score moves the optimum
  a long way.
  Fix: a longer window centred on `t_mid` (`± {10, 20, 40} ms`), which sharpens the peak roughly in
  proportion to its length; ω must stay roughly constant across it. Gallego uses event-count packets,
  not a fixed time; see the parity item on settings.
- [ ] **P1 · Accuracy / Parity — Border handling of the IWE.**
  Pipeline: events → undistort (20.5% land outside 240×180) → warp by ω → bilinear voting → out-of-frame
  votes dropped.
  - Events already outside after undistortion are wasted computation. Dropping them at the source would
    be cheaper but would not remove the bias.
  - The bias comes from events near the edge: whether they count depends on ω, so the score changes for a
    reason that has nothing to do with sharpness. With signed polarity, losing events can push the score
    up or down.
  - Ours keeps each of the 4 corners separately if it is inside. `cmax_slam` keeps a whole event only if
    `1 ≤ x < W−2` and `1 ≤ y < H−2`. That is a cleaner cut, but its event set still depends on ω.
  - Fix options: (a) vote into a padded canvas, larger than the sensor by the undistortion extent plus the
    largest warp, so no vote is lost; or (b) use a fixed mask of events that stay inside for every
    candidate ω. Option (a) is the cleaner one.
- [ ] **P2 · Parity / Accuracy — Polarity weighting.** `cmax_slam` votes unsigned (b_k = 1; Gallego's
  default, CVPR 2018 p. 3), despite the "(polarity)" comment in the C++. Ours uses `use_polarity=True`
  (b_k = ±1).
  `cmax.md` reported this as negligible (24.2 vs 24.1), but that was measured against the noisy 20 ms
  ground truth. Re-test against the smoothed ground truth. The flag already exists, so this costs one run.
  Unsigned weights also change the border item: with b_k = 1, losing events always lowers the IWE mass.
- [ ] **P2 · Parity — Bearing vectors.** `cmax_slam` looks up a precomputed bearing per integer raw pixel
  (`precomputed_bearing_vectors_[y*W+x]`). Ours back-projects the undistorted float coordinates.
  Raw event coordinates are integers, so the two are equivalent if `cmax_slam` undistorts in that
  precomputation. Check where `precomputed_bearing_vectors_` is filled.
- [ ] **P2 · Parity — Settings defined elsewhere in `cmax_slam`.** Check these against our values:
  - `time_packet_` (reference time)
  - blur σ
  - batch / packet size (event count vs our fixed 20 ms)
  - optimizer settings

  The packet size feeds directly into the flat-peak item.
- [ ] **P2 · Accuracy — V2 drifts upward.** ω_x goes 55.7 → 68.0 → 75.2 while the reference goes
  46.5 → 52 → 57.7. V1's CMax anchor shows the same growing bias, so this is probably the P1 items
  accumulating through the warm start, or the reference itself. Check this again after the P1 items.
- [ ] **P2 · Design — V1 result is a 5:1 blend.** R ≈ (δ_FR·R* + δ_IMU·ω_cmax·dt)/(δ_FR+δ_IMU). V1 currently
  scores better than CMax alone (10.3 vs 13.8) only because the flow-implied R* (42.8) is biased the other way.
- [ ] **P2 · Design — V2 learning rate.** With `lr = 1e-4`, the first iteration moves ω_x by 6.2 and
  contrast stops rising after about 3 iterations (ω_z needs about 40). So V2 is effectively a full
  unblurred CMax solve per frame, not a gradual coupling.
- [ ] **P2 · Design — In V2 the maps don't affect ω.** Only `Cost_CMax` changes R. Decide whether the
  kinematics cost should feed back into R.
- [ ] **P3 · Parity / Performance — One time offset per batch.** `cmax_slam` gives every event in a batch
  the batch-midpoint time (`time_first + 0.5·Δt`) for speed. Ours uses each event's own timestamp, which
  is more exact. No action, unless speed becomes a concern.
- [ ] **P3 · Accuracy — Blur `σ`.** `blur_sigma = 1 px` (V1) smooths away sub-pixel differences, but the
  unblurred optimum (55.7) is shifted too, so this is not the main cause.
- [ ] **P3 · Consistency — V1 frame-0 CMax search starts from zero**, not from the gyro-initialised R. It
  made no difference on seg_C (109 score evaluations vs 105–108 on later frames).

## Open (carried over from cmax.md §5)

- [ ] **P1 · Validation** — Run V1 and V2 at scale (all poster segments, many frames, against the smoothed
  ground truth and the gyro). This is the baseline every P1 item is measured against, so run it first
  and rerun it after each fix. Use the cluster.
- [ ] **P2 · Design** — Make the V2 `lr` robust: normalize the step (`λ·grad/‖grad‖`) or scale it as `lr ∝ 1/N²`.
- [ ] **P3 · Design** — Initialise frame 0 from CMax instead of the gyro, for a fully IMU-free pipeline.
- [ ] **P3 · Performance** — Speed up the analytic gradient with a `np.bincount` scatter.

## Checked and fine

- Polarity is 0/1 in every local dataset, so `w = 2·pol − 1 ∈ {−1, +1}` is correct.
- Analytic gradient matches finite differences (relative difference 0.2%). The mean term the code drops
  is about 2e-4 of the gradient.
- All walkthrough checks (unrolled steps vs `estimate`, `_contrast_and_grad`, `step`,
  `experiment_tracking`) are exactly equal for V1 and V2.
