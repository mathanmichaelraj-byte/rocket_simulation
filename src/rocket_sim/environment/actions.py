"""Defines and decodes the agent's action space."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class EngineCommand:
    throttle: float   # [0, 1]
    gimbal: float     # [-1, 1] normalised, maps to ±max_gimbal_angle_deg


ACTION_DIM = 2  # continuous: [throttle, gimbal]


def decode_action(action: np.ndarray, max_gimbal_rad: float) -> EngineCommand:
    """Map a raw policy output (tanh-squashed) to an EngineCommand."""
    throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))  # [-1,1] → [0,1]
    gimbal = float(np.clip(action[1], -1.0, 1.0)) * max_gimbal_rad
    return EngineCommand(throttle=throttle, gimbal=gimbal)
