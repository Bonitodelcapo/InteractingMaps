"""
Contrast-Maximization (CMax) angular-velocity estimation.

Python port of the front-end of Gallego et al.'s `cmax_slam`
(https://github.com/tub-rip/cmax_slam, src/frontend/), adapted to the
fixed-time-window frames used by the InteractingMaps pipeline.

Used by the V1 integration: a "full CMax" angular-velocity estimate is computed
per frame and fed to the message-passing network as the R anchor (replacing the
IMU gyro).
"""

from .angular_velocity import CMaxAngularVelocity

__all__ = ["CMaxAngularVelocity"]
