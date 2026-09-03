"""
convert_ecrot.py — ECRot ROS bag  ->  RPG text format used by this pipeline.

ECRot ships sequences as ROS1 bags:
  events -> dvs_msgs/EventArray, IMU -> sensor_msgs/Imu, GT -> a pose topic
  and/or a twist topic, calibration -> a separate YAML or an in-bag CameraInfo.
This converts one bag into the layout the loader/config expect:
  data/<name>/events.txt       t x y pol            (pol in {0,1})
  data/<name>/imu.txt          t ax ay az gx gy gz  (gyro at cols 4:7)
  data/<name>/groundtruth.txt  t px py pz qx qy qz qw
  data/<name>/omega_gt.txt     t wx wy wz           (see below)
  data/<name>/calib.txt        fx fy cx cy k1 k2 p1 p2 k3
  data/<name>/images/          + images.txt         (unless --no-images)

omega_gt.txt — the CLEAN angular-velocity reference:
  ECRot's synthetic bags carry a twist topic holding the simulator's exact
  angular velocity. When present it is written straight out; otherwise omega is
  derived from the poses by CENTRAL differencing. This matters: differencing
  poses over a frame window is noise-dominated, and on the synthetic sequences
  the difference between a differenced reference and the exact twist is
  ~0.8 deg/s — comparable to the errors being measured. evaluation.py prefers
  omega_gt.txt when it exists.

Timestamps: shifted so the stream starts at t=0, UNLESS a groundtruth.txt is
already present (e.g. the one ECRot ships), in which case absolute bag time is
kept so every stream stays aligned with it and the provided file is not
overwritten.

Requires:  pip install rosbags pyyaml numpy opencv-python

Usage:
  python convert_ecrot.py --bag street.bag --list          # inspect topics only
  python convert_ecrot.py --bag street.bag --name street_sinthetic
  python convert_ecrot.py --bag street.bag --name street --calib cam.yaml
  python convert_ecrot.py --bag big.bag --name big --t-end 5.0   # partial
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_TWIST_TYPES = ('TwistStamped', 'Twist')
_POSE_TYPES = ('PoseStamped', 'TransformStamped', 'Odometry')


def _stamp(t):
    """rosbags Time / header stamp -> float seconds."""
    return t.sec + t.nanosec * 1e-9


def _quat_to_rotmat(qx, qy, qz, qw):
    n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def _omega_body(R1, R2, dt):
    """Body-frame angular velocity (rad/s) from two rotations: dR = R1^T R2."""
    if abs(dt) < 1e-12:
        return np.zeros(3)
    dR = R1.T @ R2
    cos_a = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-12:
        return np.zeros(3)
    skew = (dR - dR.T) / (2.0 * np.sin(angle) + 1e-15)
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return axis * angle / dt


def parse_calib_yaml(path):
    """Return (fx, fy, cx, cy, [k1,k2,p1,p2,k3]). Handles kalibr-style YAML,
    searching nested cam blocks if the intrinsics are not at the top level."""
    import yaml
    with open(path) as f:
        y = yaml.safe_load(f)

    def find_block(d):
        if isinstance(d, dict):
            if 'intrinsics' in d or 'camera_matrix' in d:
                return d
            for v in d.values():
                r = find_block(v)
                if r:
                    return r
        return None

    cam = find_block(y)
    if cam is None:
        raise ValueError(f"could not find intrinsics in {path}")
    intr = cam.get('intrinsics') or cam.get('camera_matrix')
    fx, fy, cx, cy = intr[:4]
    dist = list(cam.get('distortion_coeffs', cam.get('distortion', [])))
    dist = (dist + [0, 0, 0, 0, 0])[:5]          # pad to k1 k2 p1 p2 k3
    return float(fx), float(fy), float(cx), float(cy), [float(d) for d in dist]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True, help='path to the ECRot .bag')
    ap.add_argument('--name', help='dataset name -> data/<name>/ (default: bag stem)')
    ap.add_argument('--calib', help='calibration YAML -> calib.txt (else read from CameraInfo)')
    ap.add_argument('--no-images', action='store_true', help='skip extracting image_raw frames')
    ap.add_argument('--list', action='store_true', help='only list bag topics/types, then exit')
    ap.add_argument('--t-end', type=float, default=None,
                    help='only convert bag time up to T_END seconds (large bags)')
    ap.add_argument('--events-topic', default=None, help='override event topic')
    ap.add_argument('--imu-topic', default=None, help='override IMU topic')
    ap.add_argument('--gt-topic', default=None, help='override pose/twist topic')
    args = ap.parse_args()

    from rosbags.highlevel import AnyReader

    bag = Path(args.bag)
    name = args.name or bag.stem
    out = os.path.join(os.path.dirname(__file__), 'data', name)
    img_dir = os.path.join(out, 'images')

    with AnyReader([bag]) as reader:
        # ---- topic inventory ----
        print(f"\nBag: {bag}")
        print(f"{'topic':<32} {'msgtype':<34} {'count':>8}")
        print('-' * 78)
        for c in reader.connections:
            print(f"{c.topic:<32} {c.msgtype:<34} {c.msgcount:>8}")
        if args.list:
            return

        # Events are STREAMED to disk: a dense bag holds tens of millions of
        # events, which will not fit in a Python list.  The bag is read in time
        # order and each packet is internally ordered, so events.txt comes out
        # monotonic; the loader masks per window regardless.
        os.makedirs(out, exist_ok=True)
        ev_path_tmp = os.path.join(out, 'events.raw.tmp')
        imu_rows, gt_rows, tw_rows, img_rows = [], [], [], []
        cam_info, res = None, None
        n_ev, ev_t0, ev_t1 = 0, None, None

        with open(ev_path_tmp, 'w') as fev:
            for conn, _ts, raw in reader.messages():
                mt, topic = conn.msgtype, conn.topic
                if args.t_end is not None and _ts * 1e-9 > args.t_end:
                    break
                if args.events_topic and mt.endswith('EventArray') \
                        and topic != args.events_topic:
                    continue
                if args.imu_topic and mt.endswith('Imu') and topic != args.imu_topic:
                    continue
                try:
                    msg = reader.deserialize(raw, mt)
                except Exception as e:
                    print(f"  WARN could not deserialize {mt} on {topic}: {e}")
                    continue

                if mt.endswith('EventArray'):
                    res = res or (getattr(msg, 'height', None), getattr(msg, 'width', None))
                    ev = msg.events
                    if ev:
                        fev.write(''.join(
                            f"{e.ts.sec + e.ts.nanosec * 1e-9:.9f} {e.x} {e.y} "
                            f"{1 if e.polarity else 0}\n" for e in ev))
                        n_ev += len(ev)
                        if ev_t0 is None:
                            ev_t0 = _stamp(ev[0].ts)
                        ev_t1 = _stamp(ev[-1].ts)

                elif mt.endswith('Imu'):
                    t = _stamp(msg.header.stamp)
                    a, g = msg.linear_acceleration, msg.angular_velocity
                    imu_rows.append((t, a.x, a.y, a.z, g.x, g.y, g.z))

                elif mt.endswith('CameraInfo') and cam_info is None:
                    K = getattr(msg, 'k', None); K = K if K is not None else getattr(msg, 'K')
                    D = getattr(msg, 'd', None); D = D if D is not None else getattr(msg, 'D', [])
                    cam_info = (float(K[0]), float(K[4]), float(K[2]), float(K[5]), list(D))

                elif mt.endswith('Image') and topic.endswith('image_raw') and not args.no_images:
                    t = _stamp(msg.header.stamp)
                    H_, W_, step = msg.height, msg.width, msg.step
                    buf = np.frombuffer(msg.data, dtype=np.uint8)
                    enc = (msg.encoding or 'mono8').lower()
                    if enc in ('rgb8', 'bgr8'):
                        im = buf.reshape(H_, step)[:, :W_ * 3].reshape(H_, W_, 3)
                        if enc == 'rgb8':
                            im = im[:, :, ::-1]           # -> BGR for cv2.imwrite
                    else:                                  # mono8 / other single-channel
                        im = buf.reshape(H_, step)[:, :W_]
                    os.makedirs(img_dir, exist_ok=True)
                    fname = f"images/frame_{len(img_rows):08d}.png"
                    import cv2
                    cv2.imwrite(os.path.join(out, fname), im)
                    img_rows.append((t, fname))

                elif mt.endswith(_TWIST_TYPES):
                    if args.gt_topic and topic != args.gt_topic:
                        continue
                    tw = getattr(msg, 'twist', msg)
                    tw = getattr(tw, 'twist', tw)          # TwistStamped -> twist
                    t = _stamp(msg.header.stamp) if hasattr(msg, 'header') else _ts * 1e-9
                    tw_rows.append((t, tw.angular.x, tw.angular.y, tw.angular.z))

                elif mt.endswith(_POSE_TYPES):
                    if args.gt_topic and topic != args.gt_topic:
                        continue
                    t = _stamp(msg.header.stamp)
                    if mt.endswith('PoseStamped'):
                        p, q = msg.pose.position, msg.pose.orientation
                    elif mt.endswith('Odometry'):
                        p, q = msg.pose.pose.position, msg.pose.pose.orientation
                    else:  # TransformStamped
                        p, q = msg.transform.translation, msg.transform.rotation
                    gt_rows.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    if n_ev == 0:
        os.remove(ev_path_tmp)
        print("\nNo dvs_msgs/EventArray messages found — check the topic list above.")
        return

    imu = np.array(imu_rows, dtype=np.float64) if imu_rows else None
    gt = np.array(gt_rows, dtype=np.float64) if gt_rows else None
    tw = np.array(sorted(tw_rows), dtype=np.float64) if tw_rows else None
    img_rows = sorted(img_rows)
    if imu is not None:
        imu = imu[np.argsort(imu[:, 0], kind='stable')]
    if gt is not None:
        gt = gt[np.argsort(gt[:, 0], kind='stable')]

    gt_path = os.path.join(out, 'groundtruth.txt')
    preserve_gt = os.path.exists(gt_path)

    # If a groundtruth.txt was provided, keep ABSOLUTE bag time so all streams stay
    # aligned with it; otherwise shift everything so t starts at 0.
    if preserve_gt:
        t0 = 0.0
    else:
        t0 = min([ev_t0] + ([imu[0, 0]] if imu is not None else [])
                 + ([gt[0, 0]] if gt is not None else []))

    # rewrite the streamed events with the time offset applied
    ev_path = os.path.join(out, 'events.txt')
    with open(ev_path_tmp) as fin, open(ev_path, 'w') as fout:
        for line in fin:
            t, x, y, p = line.split()
            fout.write(f"{float(t) - t0:.9f} {x} {y} {p}\n")
    os.remove(ev_path_tmp)

    if imu is not None:
        imu[:, 0] -= t0
        np.savetxt(os.path.join(out, 'imu.txt'), imu, fmt='%.9f')
    if gt is not None:
        gt[:, 0] -= t0
        if not preserve_gt:
            np.savetxt(gt_path, gt, fmt='%.9f')
    if img_rows:
        with open(os.path.join(out, 'images.txt'), 'w') as f:
            for t, fname in img_rows:
                f.write(f"{t - t0:.9f} {fname}\n")

    # ---- omega_gt.txt: exact twist if the bag has one, else central difference
    omega_src = None
    if tw is not None and len(tw):
        tw[:, 0] -= t0
        np.savetxt(os.path.join(out, 'omega_gt.txt'), tw, fmt='%.9f')
        omega_src = f"twist topic, {len(tw)} samples (exact)"
    elif gt is not None and len(gt) > 2:
        R = [_quat_to_rotmat(*row[4:8]) for row in gt]
        om = [(gt[i, 0], *_omega_body(R[i - 1], R[i + 1], gt[i + 1, 0] - gt[i - 1, 0]))
              for i in range(1, len(gt) - 1)]
        np.savetxt(os.path.join(out, 'omega_gt.txt'), np.array(om), fmt='%.9f')
        omega_src = f"central difference of poses, {len(om)} samples"

    # calib: prefer --calib YAML, else the bag's CameraInfo
    if args.calib:
        fx, fy, cx, cy, dist = parse_calib_yaml(args.calib)
    elif cam_info is not None:
        fx, fy, cx, cy, d = cam_info
        dist = (list(d) + [0, 0, 0, 0, 0])[:5]
    else:
        fx = fy = cx = cy = None
    if fx is not None:
        np.savetxt(os.path.join(out, 'calib.txt'),
                   np.array([[fx, fy, cx, cy, *dist]]), fmt='%.9f')

    t_start, t_last = ev_t0 - t0, ev_t1 - t0
    dur = t_last - t_start
    H, W = (res if res and res[0] else (180, 240))
    print(f"\nWrote -> {out}/")
    print(f"  events {n_ev:>10}   imu {0 if imu is None else len(imu):>6}   "
          f"gt {'(kept provided)' if preserve_gt else (0 if gt is None else len(gt))}   "
          f"images {len(img_rows)}")
    print(f"  omega_gt.txt: {omega_src or 'NOT WRITTEN (no twist and no poses)'}")
    print(f"  events t range [{t_start:.3f}, {t_last:.3f}]s   duration {dur:.2f}s   "
          f"sensor {H}x{W}")
    if fx is not None:
        print(f"  calib fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} dist={dist}")
    if gt is None and tw is None:
        print("  WARNING: no GT topic found — scoring vs GT will not work.")
    if fx is None:
        print("  WARNING: no --calib given — create data/%s/calib.txt "
              "(fx fy cx cy k1 k2 p1 p2 k3) before running." % name)

    # ready-to-paste config snippet
    print("\nAdd to config.py DATASET_SEGMENTS:")
    print(f"    '{name}': [")
    print(f"        {{'id': 'seg_A', 't_start': {t_start:.3f}, 'frame_duration': 0.02, "
          f"'n_frames': {int(dur/0.02)}, 'initial_R': None, "
          f"'sensor_size': ({H}, {W})}},")
    print(f"    ],")


if __name__ == '__main__':
    main()
