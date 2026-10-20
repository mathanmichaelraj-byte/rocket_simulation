"""Rocket state container — position, velocity, orientation, fuel."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RocketState:
    x: float = 0.0          # horizontal position (m)
    y: float = 1500.0        # vertical position (m)
    vx: float = 0.0          # horizontal velocity (m/s)
    vy: float = -60.0        # vertical velocity (m/s)
    angle: float = 0.0       # tilt from vertical (rad)
    omega: float = 0.0       # angular velocity (rad/s)
    fuel_kg: float = 10000.0 # remaining fuel mass
    legs_deployed: bool = False

    def total_mass(self, dry_mass: float) -> float:
        return dry_mass + max(self.fuel_kg, 0.0)

    def is_on_ground(self, ground_y: float = 0.0) -> bool:
        return self.y <= ground_y
