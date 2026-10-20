"""Serialisable snapshot of rocket state for the viewer."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from ..physics.rocket_model import RocketState


@dataclass
class ViewerState:
    x: float
    y: float
    angle: float
    fuel_frac: float
    legs_deployed: bool
    vx: float
    vy: float
    step: int
    reward: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_rocket(state: RocketState, initial_fuel: float,
                    step: int, reward: float) -> "ViewerState":
        return ViewerState(
            x=state.x, y=state.y, angle=state.angle,
            fuel_frac=state.fuel_kg / initial_fuel,
            legs_deployed=state.legs_deployed,
            vx=state.vx, vy=state.vy,
            step=step, reward=reward,
        )
