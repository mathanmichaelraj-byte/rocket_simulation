"""Records per-step telemetry to an in-memory buffer and optionally to disk."""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from ..physics.rocket_model import RocketState


@dataclass
class TelemetryFrame:
    step: int
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    omega: float
    fuel_kg: float
    reward: float
    action_throttle: float
    action_gimbal: float


class TelemetryRecorder:
    def __init__(self):
        self._frames: list[TelemetryFrame] = []

    def record(self, step: int, state: RocketState, reward: float,
               throttle: float, gimbal: float) -> None:
        self._frames.append(TelemetryFrame(
            step=step, x=state.x, y=state.y, vx=state.vx, vy=state.vy,
            angle=state.angle, omega=state.omega, fuel_kg=state.fuel_kg,
            reward=reward, action_throttle=throttle, action_gimbal=gimbal,
        ))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([asdict(f) for f in self._frames], indent=2))

    def clear(self) -> None:
        self._frames.clear()

    @property
    def frames(self) -> list[TelemetryFrame]:
        return list(self._frames)
