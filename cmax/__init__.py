"""
Contrast-Maximization (CMax) angular-velocity estimation.

Python port of the front-end of Gallego et al.'s `cmax_slam`
(https://github.com/tub-rip/cmax_slam, src/frontend/), adapted to the
fixed-time-window frames used by the InteractingMaps pipeline.

`CMaxAngularVelocity` estimates ω (rad/s, body frame) from a window of events by
maximizing the contrast (variance) of the Image of Warped Events. It supports
both distortion modes (Way 1: pre-undistorted events, pinhole warp; Way 2:
raw events with the lens carried inside the warp).

Used by two integrations (see evaluation.py / network_dissertation.py):
  - V1 (thesis_cmax)    : a full CMax solve per frame feeds the R anchor.
  - V2 (thesis_cmax_v2) : one CMax gradient step per iteration drives R directly.
"""

from .angular_velocity import CMaxAngularVelocity

__all__ = ["CMaxAngularVelocity"]
