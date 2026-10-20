"""Computes summary metrics from a recorded episode."""

from __future__ import annotations
import numpy as np
from .recorder import TelemetryFrame


def episode_summary(frames: list[TelemetryFrame]) -> dict:
    if not frames:
        return {}
    rewards = [f.reward for f in frames]
    speeds = [np.hypot(f.vx, f.vy) for f in frames]
    last = frames[-1]
    return {
        "total_reward": round(sum(rewards), 2),
        "steps": len(frames),
        "final_x": round(last.x, 2),
        "final_y": round(last.y, 2),
        "final_speed_mps": round(np.hypot(last.vx, last.vy), 2),
        "final_angle_deg": round(np.degrees(last.angle), 2),
        "fuel_remaining_kg": round(last.fuel_kg, 1),
        "max_speed_mps": round(max(speeds), 2),
    }
