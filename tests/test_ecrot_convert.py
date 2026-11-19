"""
Self-test for convert_ecrot.py — no ECRot download required.

Writes a tiny ROS1 bag in-memory (a few dvs_msgs/EventArray + sensor_msgs/Imu +
geometry_msgs/PoseStamped messages describing a known constant rotation about z
at ω = 0.5 rad/s), runs the converter, and asserts the produced text files are
correct — including that omega_gt.txt recovers ≈ (0, 0, 0.5).

Run:  python test_ecrot_convert.py
"""

import os
import tempfile
import numpy as np

from rosbags.rosbag1 import Writer
from convert_ecrot import make_typestore, convert

TS = make_typestore()
OMEGA_Z = 0.5     # rad/s, constant rotation about z


def _T(cls, **kw):
    return TS.types[cls](**kw)


def time_msg(t):
    return _T('builtin_interfaces/msg/Time', sec=int(t), nanosec=int((t % 1) * 1e9))


def header(t, seq=0):
    return _T('std_msgs/msg/Header', seq=seq, stamp=time_msg(t), frame_id='cam')


def write_bag(path):
    zero9 = np.zeros(9, dtype=np.float64)
    with Writer(path) as w:
        c_ev = w.add_connection('/dvs/events', 'dvs_msgs/msg/EventArray', typestore=TS)
        c_imu = w.add_connection('/imu', 'sensor_msgs/msg/Imu', typestore=TS)
        c_gt = w.add_connection('/cam0/pose', 'geometry_msgs/msg/PoseStamped', typestore=TS)

        # 0..0.5 s at 100 Hz
        for i in range(51):
            t = i * 0.01
            tns = int(t * 1e9)

            # pose: rotation about z by OMEGA_Z * t
            ang = OMEGA_Z * t
            q = _T('geometry_msgs/msg/Quaternion',
                   x=0.0, y=0.0, z=np.sin(ang / 2), w=np.cos(ang / 2))
            pose = _T('geometry_msgs/msg/Pose',
                      position=_T('geometry_msgs/msg/Point', x=0.0, y=0.0, z=0.0),
                      orientation=q)
            ps = _T('geometry_msgs/msg/PoseStamped', header=header(t, i), pose=pose)
            w.write(c_gt, tns, TS.serialize_ros1(ps, 'geometry_msgs/msg/PoseStamped'))

            # imu: gyro = (0,0,OMEGA_Z)
            imu = _T('sensor_msgs/msg/Imu', header=header(t, i),
                     orientation=q, orientation_covariance=zero9.copy(),
                     angular_velocity=_T('geometry_msgs/msg/Vector3', x=0.0, y=0.0, z=OMEGA_Z),
                     angular_velocity_covariance=zero9.copy(),
                     linear_acceleration=_T('geometry_msgs/msg/Vector3', x=0.0, y=0.0, z=9.81),
                     linear_acceleration_covariance=zero9.copy())
            w.write(c_imu, tns, TS.serialize_ros1(imu, 'sensor_msgs/msg/Imu'))

            # a couple of events per packet
            evs = [_T('dvs_msgs/msg/Event', x=10 + i % 5, y=20, ts=time_msg(t), polarity=True),
                   _T('dvs_msgs/msg/Event', x=30, y=40, ts=time_msg(t + 0.002), polarity=False)]
            ea = _T('dvs_msgs/msg/EventArray', header=header(t, i),
                    height=180, width=240, events=evs)
            w.write(c_ev, tns, TS.serialize_ros1(ea, 'dvs_msgs/msg/EventArray'))


def write_calib(path):
    with open(path, 'w') as f:
        f.write("cam0:\n"
                "  intrinsics: [199.0, 198.8, 120.0, 90.0]\n"
                "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
                "  resolution: [240, 180]\n")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        bag = os.path.join(tmp, 'mini.bag')
        calib = os.path.join(tmp, 'calib.yaml')
        out = os.path.join(tmp, 'data', 'ecrot_test')
        write_bag(bag)
        write_calib(calib)

        convert(bag, out, calib_yaml=calib)

        # --- assertions ---
        ev = np.loadtxt(os.path.join(out, 'events.txt'))
        assert ev.shape == (102, 4), f"events shape {ev.shape}"
        assert set(np.unique(ev[:, 3])) <= {0.0, 1.0}, "polarity not 0/1"
        assert np.all(np.diff(ev[:, 0]) >= -1e-9), "events not time-sorted"

        imu = np.loadtxt(os.path.join(out, 'imu.txt'))
        assert imu.shape == (51, 7)
        assert abs(imu[:, 6].mean() - OMEGA_Z) < 1e-6, "imu gyro_z wrong"

        om = np.loadtxt(os.path.join(out, 'omega_gt.txt'))
        assert om.shape[1] == 4
        mean_w = om[:, 1:].mean(axis=0)
        assert np.allclose(mean_w, [0, 0, OMEGA_Z], atol=1e-3), f"omega_gt {mean_w}"

        cal = np.loadtxt(os.path.join(out, 'calib.txt'))
        assert cal.shape == (9,) and abs(cal[0] - 199.0) < 1e-6

        print("\n  events.txt   ✓ (102 rows, 0/1 polarity, sorted)")
        print("  imu.txt      ✓ (51 rows, gyro_z = 0.5)")
        print(f"  omega_gt.txt ✓ (central-diff ω = {mean_w.round(4)} ≈ [0,0,0.5])")
        print("  calib.txt    ✓ (fx=199, 240×180)")
        print("\nALL ASSERTIONS PASSED — converter logic verified.")


if __name__ == '__main__':
    main()
