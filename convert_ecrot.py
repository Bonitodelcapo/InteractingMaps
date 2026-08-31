"""
convert_ecrot.py — ECRot ROSbag -> InteractingMaps pipeline format.

ECRot (tub-rip/ECRot) ships its synthetic sequences (City, Street, …) as ROS1
bags. This script converts a bag into the flat text layout the pipeline reads
(data/<dataset>/), using the pure-Python `rosbags` library — no ROS install.

Outputs (into --out dir):
  events.txt      t x y polarity            (from dvs_msgs/EventArray)
  imu.txt         t ax ay az gx gy gz       (from sensor_msgs/Imu, if present)
  omega_gt.txt    t wx wy wz                (the CLEAN reference angular velocity)
  groundtruth.txt t tx ty tz qx qy qz qw    (only when GT is a pose topic)
  calib.txt       fx fy cx cy k1 k2 p1 p2 k3 (from the ECRot YAML, if --calib given)

Ground-truth ω (decided from the bag):
  - a direct twist / angular-velocity topic -> written straight to omega_gt.txt;
  - otherwise poses are exported to groundtruth.txt AND omega_gt.txt is derived
    from them by central differencing (body frame, R1ᵀR2 / dt). Because the
    synthetic poses are exact and dense, this ω is clean — no mocap noise.

Usage
  python convert_ecrot.py --topics city.bag                 # inspect topics first
  python convert_ecrot.py city.bag --out ecrot_city --calib DAVIS240C-synthetic.yaml
  # optional overrides: --events-topic /dvs/events --gt-topic /cam0/pose
"""

import argparse
import os
import numpy as np

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# ---------------------------------------------------------------------------
# Custom dvs_msgs types (rpg_dvs_ros) — not in the standard ROS typestore
# ---------------------------------------------------------------------------
_EVENT_MSG = """
uint16 x
uint16 y
time ts
bool polarity
"""
_EVENTARRAY_MSG = """
std_msgs/Header header
uint32 height
uint32 width
Event[] events
"""


def make_typestore():
    """ROS1 typestore with dvs_msgs/Event and dvs_msgs/EventArray registered."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    types = {}
    types.update(get_types_from_msg(_EVENT_MSG, 'dvs_msgs/msg/Event'))
    types.update(get_types_from_msg(_EVENTARRAY_MSG, 'dvs_msgs/msg/EventArray'))
    ts.register(types)
    return ts


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _stamp_to_sec(stamp):
    """ROS Time (sec/nanosec) -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _quat_to_rotmat(qx, qy, qz, qw):
    n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def _omega_body(R1, R2, dt):
    """Body-frame angular velocity (rad/s) from two rotations: dR = R1ᵀR2."""
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


# Message-type classification (suffix match, package-agnostic)
_EVENT_TYPES = ('dvs_msgs/msg/EventArray',)
_IMU_TYPES = ('sensor_msgs/msg/Imu',)
_TWIST_TYPES = ('geometry_msgs/msg/TwistStamped', 'geometry_msgs/msg/Twist')
_POSE_TYPES = ('geometry_msgs/msg/PoseStamped', 'geometry_msgs/msg/TransformStamped',
               'nav_msgs/msg/Odometry')


def _classify(topics):
    """Pick the first topic of each role. topics: {name: msgtype}."""
    def find(cands):
        return [t for t, mt in topics.items() if mt in cands]
    ev = find(_EVENT_TYPES)
    imu = find(_IMU_TYPES)
    twist = find(_TWIST_TYPES)
    pose = find(_POSE_TYPES)
    return (ev[0] if ev else None,
            imu[0] if imu else None,
            twist[0] if twist else None,
            pose[0] if pose else None)


# ---------------------------------------------------------------------------
# Topic inspection
# ---------------------------------------------------------------------------
def cmd_topics(bagpath):
    ts = make_typestore()
    with Reader(bagpath) as reader:
        print(f"Bag: {bagpath}")
        print(f"  duration: {reader.duration * 1e-9:.3f} s   messages: {reader.message_count}")
        print(f"  {'topic':<32} {'msgtype':<34} {'count'}")
        print("  " + "-" * 78)
        topics = {}
        for conn in reader.connections:
            topics[conn.topic] = conn.msgtype
            print(f"  {conn.topic:<32} {conn.msgtype:<34} {conn.msgcount}")
    ev, imu, twist, pose = _classify(topics)
    print("\n  Auto-detected roles:")
    print(f"    events : {ev}")
    print(f"    imu    : {imu}")
    print(f"    twist  : {twist}   (direct omega if present)")
    print(f"    pose   : {pose}   (omega derived by differencing if no twist)")


# ---------------------------------------------------------------------------
# Calibration YAML  ->  calib.txt
# ---------------------------------------------------------------------------
def convert_calib(yaml_path, out_dir):
    """
    Parse an ECRot/Kalibr-style intrinsics YAML -> calib.txt + return sensor_size.
    Flexible: searches for `intrinsics` [fx,fy,cx,cy], `distortion_coeffs`,
    `resolution` [W,H], scanning nested cam blocks (cam0, …) if needed.
    """
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    def find_block(d):
        if isinstance(d, dict):
            if 'intrinsics' in d and 'resolution' in d:
                return d
            for v in d.values():
                r = find_block(v)
                if r:
                    return r
        return None

    block = find_block(data)
    if block is None:
        raise ValueError(f"No intrinsics/resolution found in {yaml_path}")

    fx, fy, cx, cy = [float(v) for v in block['intrinsics'][:4]]
    W, H = [int(v) for v in block['resolution'][:2]]
    dist = [float(v) for v in block.get('distortion_coeffs', [])]
    dist = (dist + [0.0] * 5)[:5]   # pad/truncate to [k1,k2,p1,p2,k3]

    return _write_calib(out_dir, fx, fy, cx, cy, dist, W, H)


def _write_calib(out_dir, fx, fy, cx, cy, dist, W, H):
    dist = (list(dist) + [0.0] * 5)[:5]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'calib.txt'), 'w') as f:
        f.write(f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} "
                f"{dist[0]:.8f} {dist[1]:.8f} {dist[2]:.8f} {dist[3]:.8f} {dist[4]:.8f}\n")
    print(f"  calib.txt written (fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}, "
          f"{W}x{H}, dist={dist})")
    return (H, W)


def calib_from_caminfo(msg, out_dir):
    """
    sensor_msgs/CameraInfo → calib.txt + sensor_size. K = [fx 0 cx; 0 fy cy; 0 0 1];
    D = plumb_bob [k1,k2,p1,p2,k3]. Lets the bag be self-contained (no YAML).
    """
    K = list(msg.K)
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    return _write_calib(out_dir, fx, fy, cx, cy, list(msg.D), int(msg.width), int(msg.height))


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(bagpath, out_dir, calib_yaml=None,
            events_topic=None, imu_topic=None, gt_topic=None, t_end=None):
    ts = make_typestore()
    os.makedirs(out_dir, exist_ok=True)

    # --- classify topics ---
    with Reader(bagpath) as reader:
        topics = {c.topic: c.msgtype for c in reader.connections}
    ev_t, imu_t, twist_t, pose_t = _classify(topics)
    events_topic = events_topic or ev_t
    imu_topic = imu_topic or imu_t
    if gt_topic is None:
        gt_topic = twist_t or pose_t
    gt_is_twist = gt_topic is not None and topics.get(gt_topic) in _TWIST_TYPES

    if events_topic is None:
        raise ValueError("No dvs_msgs/EventArray topic found; pass --events-topic.")
    print(f"  events: {events_topic} | imu: {imu_topic} | "
          f"gt: {gt_topic} ({'twist' if gt_is_twist else 'pose'})")

    imu_rows = []        # (t, ax, ay, az, gx, gy, gz)
    twist_rows = []      # (t, wx, wy, wz)
    poses = []           # (t, tx,ty,tz, R)
    ev_count = 0
    ev_t0 = ev_t1 = None
    events_path = os.path.join(out_dir, 'events.txt')

    want = {events_topic, imu_topic, gt_topic} - {None}
    # Events are STREAMED to disk per packet (79M+ events won't fit a list). ROS
    # returns messages in time order and each packet's events are ordered, so the
    # file is time-monotonic; EventFrameSequence masks per window regardless.
    with Reader(bagpath) as reader, open(events_path, 'w') as fev:
        conns = [c for c in reader.connections if c.topic in want]
        for conn, tstamp, raw in reader.messages(connections=conns):
            if t_end is not None and tstamp * 1e-9 > t_end:
                break                          # windowed conversion (large bags)
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            if conn.topic == events_topic:
                evs = msg.events
                if evs:
                    fev.write(''.join(
                        f"{e.ts.sec + e.ts.nanosec * 1e-9:.6f} {e.x} {e.y} "
                        f"{1 if e.polarity else 0}\n" for e in evs))
                    ev_count += len(evs)
                    if ev_t0 is None:
                        ev_t0 = evs[0].ts.sec + evs[0].ts.nanosec * 1e-9
                    ev_t1 = evs[-1].ts.sec + evs[-1].ts.nanosec * 1e-9
            elif conn.topic == imu_topic:
                t = _stamp_to_sec(msg.header.stamp)
                a, w = msg.linear_acceleration, msg.angular_velocity
                imu_rows.append((t, a.x, a.y, a.z, w.x, w.y, w.z))
            elif conn.topic == gt_topic:
                if gt_is_twist:
                    tw = getattr(msg, 'twist', msg)
                    tw = getattr(tw, 'twist', tw)          # TwistStamped->twist
                    t = _stamp_to_sec(msg.header.stamp) if hasattr(msg, 'header') \
                        else tstamp * 1e-9
                    twist_rows.append((t, tw.angular.x, tw.angular.y, tw.angular.z))
                else:
                    t, tx, ty, tz, R = _extract_pose(msg, conn.msgtype, tstamp)
                    poses.append((t, tx, ty, tz, R))

    # --- events.txt already streamed above ---
    print(f"  events.txt: {ev_count} events"
          + (f" [{ev_t0:.3f}, {ev_t1:.3f}] s" if ev_count else " (EMPTY)"))

    # --- imu.txt ---
    if imu_rows:
        imu_rows.sort(key=lambda r: r[0])
        np.savetxt(os.path.join(out_dir, 'imu.txt'), np.array(imu_rows), fmt='%.6f')
        print(f"  imu.txt: {len(imu_rows)} samples")

    # --- GT -> omega_gt.txt (+ groundtruth.txt if poses) ---
    if gt_is_twist and twist_rows:
        twist_rows.sort(key=lambda r: r[0])
        np.savetxt(os.path.join(out_dir, 'omega_gt.txt'), np.array(twist_rows), fmt='%.6f')
        print(f"  omega_gt.txt: {len(twist_rows)} samples (direct twist)")
    elif poses:
        poses.sort(key=lambda r: r[0])
        # groundtruth.txt (poses)
        with open(os.path.join(out_dir, 'groundtruth.txt'), 'w') as f:
            for t, tx, ty, tz, R in poses:
                q = _rotmat_to_quat(R)
                f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                        f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
        # omega_gt.txt (central difference on exact poses -> clean ω)
        omega = []
        for i in range(1, len(poses) - 1):
            t0, _, _, _, R0 = poses[i - 1]
            t2, _, _, _, R2 = poses[i + 1]
            tm = poses[i][0]
            omega.append((tm, *_omega_body(R0, R2, t2 - t0)))
        if omega:
            np.savetxt(os.path.join(out_dir, 'omega_gt.txt'), np.array(omega), fmt='%.6f')
        print(f"  groundtruth.txt: {len(poses)} poses ; "
              f"omega_gt.txt: {len(omega)} samples (central diff)")
    else:
        print("  WARNING: no GT topic found -> no omega_gt.txt / groundtruth.txt")

    # --- calib.txt: YAML if given, else in-bag CameraInfo ---
    sensor_size = None
    if calib_yaml:
        sensor_size = convert_calib(calib_yaml, out_dir)
    else:
        cam_topics = [t for t, mt in topics.items() if mt == 'sensor_msgs/msg/CameraInfo']
        if cam_topics:
            with Reader(bagpath) as reader:
                conns = [c for c in reader.connections if c.topic == cam_topics[0]]
                for conn, _, raw in reader.messages(connections=conns):
                    sensor_size = calib_from_caminfo(ts.deserialize_ros1(raw, conn.msgtype), out_dir)
                    break
        else:
            print("  (no --calib YAML and no CameraInfo topic -> calib.txt not written)")

    if sensor_size:
        print(f"\n  -> add to config.py DATASET_SEGMENTS with sensor_size={sensor_size}")
    return out_dir


def _extract_pose(msg, msgtype, tstamp):
    """Return (t, tx, ty, tz, R) from a pose-like message."""
    if msgtype == 'geometry_msgs/msg/TransformStamped':
        t = _stamp_to_sec(msg.header.stamp)
        tr, r = msg.transform.translation, msg.transform.rotation
        return t, tr.x, tr.y, tr.z, _quat_to_rotmat(r.x, r.y, r.z, r.w)
    if msgtype == 'nav_msgs/msg/Odometry':
        t = _stamp_to_sec(msg.header.stamp)
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        return t, p.x, p.y, p.z, _quat_to_rotmat(o.x, o.y, o.z, o.w)
    # PoseStamped (default)
    t = _stamp_to_sec(msg.header.stamp)
    p, o = msg.pose.position, msg.pose.orientation
    return t, p.x, p.y, p.z, _quat_to_rotmat(o.x, o.y, o.z, o.w)


def _rotmat_to_quat(R):
    """Rotation matrix -> (qx, qy, qz, qw)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return np.array([qx, qy, qz, qw])


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='ECRot ROSbag -> pipeline format')
    ap.add_argument('bag', nargs='?', help='path to the ROS1 .bag')
    ap.add_argument('--topics', metavar='BAG',
                    help='just list topics/types of BAG and exit')
    ap.add_argument('--out', help='dataset name -> data/<out>/')
    ap.add_argument('--calib', help='ECRot intrinsics YAML')
    ap.add_argument('--events-topic', default=None)
    ap.add_argument('--imu-topic', default=None)
    ap.add_argument('--gt-topic', default=None)
    ap.add_argument('--t-end', type=float, default=None,
                    help='only convert bag time up to T_END seconds (large bags / quick example)')
    ap.add_argument('--base-dir', default=os.path.join(os.path.dirname(__file__), 'data'))
    args = ap.parse_args()

    if args.topics:
        cmd_topics(args.topics)
    elif args.bag and args.out:
        out_dir = os.path.join(args.base_dir, args.out)
        convert(args.bag, out_dir, calib_yaml=args.calib,
                events_topic=args.events_topic, imu_topic=args.imu_topic,
                gt_topic=args.gt_topic, t_end=args.t_end)
        print(f"\nDone -> {out_dir}")
    else:
        ap.error("give either --topics BAG, or BAG --out NAME")
