"""Episode termination conditions."""

from __future__ import annotations
import numpy as np
from ..physics.rocket_model import RocketState
from ..config import PhysicsConfig


X_BOUND = 1000.0   # m from centre


def check_termination(
    state: RocketState,
    pad_x: float,
    cfg: PhysicsConfig,
    step: int,
    max_steps: int,
) -> tuple[bool, bool, bool, bool]:
    """
    Returns (done, landed, crashed, out_of_bounds).
    """
    on_ground = state.is_on_ground(cfg.ground_y_m)
    speed = np.hypot(state.vx, state.vy)
    tilt_deg = abs(np.degrees(state.angle))
    on_pad = abs(state.x - pad_x) <= 5.0

    landed = on_ground and on_pad and speed <= cfg.max_landing_velocity_mps and tilt_deg <= cfg.max_landing_angle_deg
    crashed = on_ground and not landed
    out_of_bounds = abs(state.x) > X_BOUND or state.y > 3000.0
    timeout = step >= max_steps

    done = landed or crashed or out_of_bounds or timeout
    return done, landed, crashed, out_of_bounds
