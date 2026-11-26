"""Where a run's per-frame states and iteration replays live, and how they are read.

    <run_dir>/states/init.npz          state before frame 0 (k = -1)
    <run_dir>/states/frame_0006.npz    state after frame 6's last iteration
    <run_dir>/replay/frame_0006.npz    every iteration of frame 6 (replay.py)

States are float64 so that replaying frame k from state k-1 is bit-exact.
Written only when RunConfig.save_states is on, and never under results/.
"""
import os

import numpy as np

_LOOP = 'loop_'
_HINT = ("no saved states here; rerun with RunConfig.save_states = True "
         "(python report/run_experiments.py --what inspect)")


def states_dir(run_dir) -> str:
    return os.path.join(run_dir, 'states')


def state_path(run_dir, k) -> str:
    name = 'init.npz' if k == -1 else f'frame_{k:04d}.npz'
    return os.path.join(states_dir(run_dir), name)


def replay_path(run_dir, k) -> str:
    return os.path.join(run_dir, 'replay', f'frame_{k:04d}.npz')


def has_states(run_dir) -> bool:
    return os.path.exists(state_path(run_dir, -1))


def assert_not_results(path) -> None:
    """results/ holds the reported runs; nothing of this kind is written there."""
    parts = os.path.normpath(os.path.relpath(os.path.abspath(path))).split(os.sep)
    if parts and parts[0] == 'results':
        raise ValueError(f'refusing to write states/replays under results/: {path}')


def save_state(run_dir, k, state: dict, loop: dict) -> None:
    os.makedirs(states_dir(run_dir), exist_ok=True)
    arrays = {key: np.asarray(v, dtype=np.float64) for key, v in state.items()}
    arrays.update({_LOOP + key: np.asarray(v, dtype=np.float64) for key, v in loop.items()})
    np.savez_compressed(state_path(run_dir, k), **arrays)


def load_state(run_dir, k):
    """(state, loop) saved for frame k (k = -1: the initial state)."""
    path = state_path(run_dir, k)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path}: {_HINT}')
    with np.load(path) as z:
        state = {key: z[key] for key in z.files if not key.startswith(_LOOP)}
        loop = {key[len(_LOOP):]: z[key] for key in z.files if key.startswith(_LOOP)}
    return state, loop
