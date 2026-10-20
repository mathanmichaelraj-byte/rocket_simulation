"""Reward shaping for the landing task."""

from __future__ import annotations
import numpy as np
from ..physics.rocket_model import RocketState
from ..config import PhysicsConfig


def compute_reward(
    state: RocketState,
    prev_state: RocketState,
    pad_x: float,
    cfg: PhysicsConfig,
    throttle: float,
    landed: bool,
    crashed: bool,
    out_of_bounds: bool,
) -> float:
    reward = 0.0

    # Shaping: reward closing distance to pad
    prev_dist = np.hypot(prev_state.x - pad_x, prev_state.y)
    curr_dist = np.hypot(state.x - pad_x, state.y)
    reward += (prev_dist - curr_dist) * 0.1

    # Penalise tilt
    reward -= abs(state.angle) * 0.5

    # Penalise fuel use
    reward -= throttle * 0.02

    if landed:
        speed = np.hypot(state.vx, state.vy)
        tilt_deg = abs(np.degrees(state.angle))
        reward += 200.0
        reward += max(0.0, 100.0 - speed * 20.0)   # softer = more reward
        reward += max(0.0, 100.0 - tilt_deg * 10.0) # more upright = more reward

    if crashed:
        reward -= 100.0

    if out_of_bounds:
        reward -= 50.0

    return float(reward)
