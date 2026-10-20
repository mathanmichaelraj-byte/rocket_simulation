"""Builds the observation vector from rocket state."""

from __future__ import annotations
import numpy as np
from ..physics.rocket_model import RocketState


OBS_DIM = 9


def build_observation(state: RocketState, pad_x: float, initial_fuel: float) -> np.ndarray:
    """
    Returns a normalised observation vector:
      [x_rel, y, vx, vy, angle, omega, fuel_frac, legs_deployed, speed]
    """
    return np.array([
        (state.x - pad_x) / 200.0,          # horizontal offset to pad
        state.y / 1500.0,                    # normalised altitude
        state.vx / 100.0,                    # horizontal speed
        state.vy / 100.0,                    # vertical speed
        state.angle / np.pi,                 # tilt (-1 to 1)
        state.omega / 5.0,                   # angular rate
        state.fuel_kg / initial_fuel,        # fuel fraction
        float(state.legs_deployed),
        np.hypot(state.vx, state.vy) / 100.0,  # total speed
    ], dtype=np.float32)
