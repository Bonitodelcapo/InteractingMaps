A18 = ('A18',)

CELLS = [
('md', A18, r'''
## A18 · Lens distortion

**Question.** How far does the lens move the events, and does it matter how
the distortion is handled? The report carries it inside the kinematic matrix C
(`DISTORTION_MODE='C_full'`) instead of undistorting the events before binning;
undistortion would leave pixels of the frame empty and spread neighbouring
events apart.

**Made by** `report/make_figures.py::fig_distortion` (`python report/make_figures.py`;
needs only the raw events of `poster_rotation`) → `report/figures/fig_distortion.png/.pdf`;
it calls `make_figures.py::distortion_numbers` → `report/experiments/distortion_numbers.csv`.
Also `run_all.sh` job `distortion_test` = `python tests/test_distortion_cmax.py` (stdout).

**Read by** fig:distortion and the numbers of the Lens Distortion paragraph.

**Steps**: where undistortion sends the sensor pixels (`distortion_numbers`) →
the events of 0.2 s, binned raw and undistorted → the figure → same-code check.
'''),

('code', A18, r'''
# step 1 -- the raw events and the calibration: fig_distortion's fixed window (Redraw),
# or the panel's segment for n_frames * dt (Live). No run is made.
from interacting_maps.data_loader import (CameraCalibration, load_events_fast,
                                          undistort_events, events_to_vframe)
p = params()
if p['mode'] == 'Redraw':
    note_redraw(p, 10)                          # 0.2 s = 10 frames of 20 ms
    ds_A18, t0_A18, dur_A18 = 'poster_rotation', 8.816, 0.20   # fixed in fig_distortion
else:
    ds_A18, _sid, t0_A18, _lab = p['seg']
    dur_A18 = p['n_frames'] * p['dt']
H_A18, W_A18 = 180, 240
paths_A18 = E.get_dataset_paths(ds_A18)
READY['A18'] = need([paths_A18['events'], paths_A18['calib']])
if READY['A18']:
    calib_A18 = CameraCalibration(paths_A18['calib'])
    print(f'{ds_A18}: t {t0_A18} s + {dur_A18:.2f} s; fx {calib_A18.fx:.1f} fy {calib_A18.fy:.1f} '
          f'cx {calib_A18.cx:.1f} cy {calib_A18.cy:.1f}; dist {np.round(calib_A18.dist, 4)}')
'''),

('code', A18, r'''
%%section A18
# step 2 -- make_figures.py::distortion_numbers, unrolled: undistort every sensor pixel
import cv2
c_A18 = calib_A18
K_A18 = np.array([[c_A18.fx, 0, c_A18.cx], [0, c_A18.fy, c_A18.cy], [0, 0, 1]])
ys_A18, xs_A18 = np.mgrid[0:H_A18, 0:W_A18]
pts_A18 = np.stack([xs_A18.ravel(), ys_A18.ravel()], -1).astype(np.float64).reshape(-1, 1, 2)
u_A18 = cv2.undistortPoints(pts_A18, K_A18, c_A18.dist.astype(np.float64), P=K_A18)[:, 0, :].reshape(H_A18, W_A18, 2)
cells_A18 = u_A18.astype(np.int32)                  # binned as events_to_vframe bins
hit_A18 = ((cells_A18[..., 0] >= 0) & (cells_A18[..., 0] < W_A18) & (cells_A18[..., 1] >= 0) & (cells_A18[..., 1] < H_A18))
inside_A18 = ((u_A18[..., 0] >= 0) & (u_A18[..., 0] < W_A18) & (u_A18[..., 1] >= 0) & (u_A18[..., 1] < H_A18))
dx_A18 = np.linalg.norm(u_A18[:, 1:] - u_A18[:, :-1], axis=-1)[inside_A18[:, 1:] & inside_A18[:, :-1]]
dy_A18 = np.linalg.norm(u_A18[1:] - u_A18[:-1], axis=-1)[inside_A18[1:] & inside_A18[:-1]]
num_A18 = {'sensor_pixels': H_A18 * W_A18,
           'occupied_undistorted': int(np.unique(cells_A18[hit_A18][:, 1] * W_A18 + cells_A18[hit_A18][:, 0]).size),
           'max_spacing_px': round(float(max(dx_A18.max(), dy_A18.max())), 3)}
shift_A18 = np.linalg.norm(u_A18 - np.stack([xs_A18, ys_A18], -1), axis=-1)
print(', '.join(f'{k} {v}' for k, v in num_A18.items()))
print(f'{num_A18["sensor_pixels"] - num_A18["occupied_undistorted"]} frame pixels get no sensor pixel; '
      f'shift of a pixel by undistortion: median {np.median(shift_A18):.2f} px, max {shift_A18.max():.2f} px (corners)')
'''),

('code', A18, r'''
%%section A18
# step 3 -- fig_distortion's inputs: the events binned raw and after undistort_events
ev_A18 = load_events_fast(paths_A18['events'], t_start=t0_A18, duration=dur_A18 + 0.02)
ev_A18 = ev_A18[ev_A18[:, 0] < t0_A18 + dur_A18]
V_raw_A18 = events_to_vframe(ev_A18, H_A18, W_A18, clip_value=40.0, normalise=True)
V_und_A18 = events_to_vframe(undistort_events(ev_A18, calib_A18), H_A18, W_A18, clip_value=40.0, normalise=True)
print(f'{len(ev_A18)} events; pixels with events: raw {(V_raw_A18 != 0).sum()}, '
      f'undistorted {(V_und_A18 != 0).sum()} of {H_A18 * W_A18}')
'''),

('code', A18, r'''
%%section A18
# step 4 -- make_figures.py::fig_distortion, unrolled
from interacting_maps.style import use_style, save, PAPER_WIDTH
from interacting_maps.viz import draw_map, map_scale
panels_A18 = [(V_raw_A18, 'raw events\n' r'(lens in $\mathcal{C}$)'),
              (V_und_A18, 'undistorted\nevents'),
              (V_und_A18[10:80, 140:230], 'detail\n(undistorted)')]
scale_A18 = map_scale('V', V_und_A18)
with use_style('paper'):
    fig_A18, ax_A18 = plt.subplots(1, 3, figsize=(PAPER_WIDTH, 1.25))
    for a, (img, ttl) in zip(ax_A18, panels_A18):
        draw_map(a, 'V', img, title=ttl, scale=scale_A18)
    pix_A18 = fig_pixels(fig_A18)
    save(fig_A18, os.path.join(walk_dir('A18'), 'fig_distortion.png'), show=True)
'''),

('md', A18, r'''
### What `tests/test_distortion_cmax.py` asks (not unrolled)

The `distortion_test` job checks, on poster seg_C without the network, that the
distortion-aware warp CMax uses -- the same lens geometry as the
distortion-aware C matrix -- agrees with the trusted undistort-first path:

- **A, forward model.** Is the forward Brown–Conrady projection the same as
  `cv2.projectPoints`? (max error < 1e-6 px, pass/fail)
- **B, geometry.** For a fixed rotation (the gyro's), does warping the raw
  events the distortion-aware way and then undistorting land where the plain
  pinhole warp of the undistorted events lands? (max < 0.5 px, pass/fail: the
  real confirmation of the C-matrix re-distortion)
- **C, estimator (context).** Do the two CMax estimates of ω agree, and how far
  is each from the gyro? They need not match exactly: the CMax objective is
  computed on different pixel grids. Reported in deg/s, not a gate.

It prints `RESULT: PASS` when A and B pass. Run it with
`PYTHONPATH=. python tests/test_distortion_cmax.py`.
'''),

('code', A18, r'''
%%section A18
# same-code check -- distortion_numbers and fig_distortion themselves (their ROOT and
# FIG redirected), and the csv the report job wrote
chk = os.path.join(walk_dir('A18'), 'check')
csv_rel = os.path.join('report', 'experiments', 'distortion_numbers.csv')
read_num = lambda path: {k: float(v) for k, v in next(csv.DictReader(open(path))).items()}
if p['mode'] == 'Redraw' and need([csv_rel], ['make_figures (fig_distortion)']):
    check('distortion_numbers vs csv', num_A18, read_num(csv_rel))
with redirect(make_figures, ROOT=chk), contextlib.redirect_stdout(io.StringIO()):
    make_figures.distortion_numbers(calib_A18, H_A18, W_A18)
check('distortion_numbers', num_A18, read_num(os.path.join(chk, csv_rel)))
if p['mode'] == 'Redraw':
    with caught_figures(make_figures, FIG=chk, ROOT=chk) as got:
        make_figures.fig_distortion(E)
    check('fig_distortion pixels', pix_A18, got['pixels'].get('fig_distortion.png', []))
else:
    print("Live: fig_distortion's window is fixed (poster, 0.2 s); its pixels are checked in Redraw")
'''),
]
