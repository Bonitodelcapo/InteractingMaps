"""Replay frames of a saved run iteration by iteration (inspection viewer).

    python replay_frame.py --run experiments/inspect/<dataset>/<model>/<run> --frame 6 --frame 14

The run must have been made with save_states (run_experiments --what inspect).
Writes <run>/replay/frame_XXXX.npz; refuses anything under results/.
"""
import argparse

from interacting_maps.replay import replay_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='run folder with states/')
    ap.add_argument('--frame', type=int, action='append', required=True,
                    help='frame to replay (repeatable)')
    args = ap.parse_args()
    for k in args.frame:
        print(f'-> {replay_frame(args.run, k)}', flush=True)


if __name__ == '__main__':
    main()
