"""
interacting_maps — the two Interacting-Maps networks and camera geometry.

  - network.py              : InteractingMaps (Cook 2011, Gauss-Seidel)
  - network_dissertation.py : InteractingMapsThesis (Martel 2019, Jacobi two-phase;
                              base for the thesis / thesis_imu / thesis_cmax[_v2] variants)
  - camera.py               : compute_calibration + build_kinematic_matrix (C matrix)

`InteractingMapsThesis` is intentionally not re-exported here to keep this import
lightweight; import it directly from interacting_maps.network_dissertation.
"""

from .network import InteractingMaps
from .camera import compute_calibration, build_kinematic_matrix
