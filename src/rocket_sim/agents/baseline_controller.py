"""Simple PD baseline controller — used as a sanity-check before RL training."""

from __future__ import annotations
import numpy as np
from ..physics.rocket_model import RocketState


class PDController:
    """Proportional-derivative controller for vertical landing."""

    def __init__(self, kp_vy: float = 0.02, kd_vy: float = 0.01,
                 kp_angle: float = 0.5, kd_angle: float = 0.2):
        self.kp_vy = kp_vy
        self.kd_vy = kd_vy
        self.kp_angle = kp_angle
        self.kd_angle = kd_angle

    def act(self, state: RocketState) -> np.ndarray:
        """Return a continuous action [throttle_norm, gimbal_norm]."""
        # Throttle: fight gravity + damp vertical speed
        target_vy = -2.0  # desired descent rate (m/s)
        throttle = 0.5 + self.kp_vy * (state.vy - target_vy) - self.kd_vy * state.vy
        throttle = float(np.clip(throttle * 2.0 - 1.0, -1.0, 1.0))  # → [-1,1]

        # Gimbal: correct tilt
        gimbal = -(self.kp_angle * state.angle + self.kd_angle * state.omega)
        gimbal = float(np.clip(gimbal, -1.0, 1.0))

        return np.array([throttle, gimbal], dtype=np.float32)
