"""Aerodynamic drag and lift forces."""

from __future__ import annotations
import numpy as np


def drag_force(
    vx: float,
    vy: float,
    air_density: float,
    cd: float,
    ref_area: float,
) -> tuple[float, float]:
    """Return (drag_x, drag_y) opposing the velocity vector."""
    speed = np.hypot(vx, vy)
    if speed < 1e-6:
        return 0.0, 0.0
    magnitude = 0.5 * air_density * speed**2 * cd * ref_area
    return -magnitude * vx / speed, -magnitude * vy / speed
