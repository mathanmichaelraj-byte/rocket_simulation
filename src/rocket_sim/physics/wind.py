"""Stochastic wind model with gusts."""

from __future__ import annotations
import numpy as np


class WindModel:
    def __init__(self, max_speed_mps: float = 10.0, seed: int | None = None):
        self.max_speed = max_speed_mps
        self._rng = np.random.default_rng(seed)
        self._base_vx = self._rng.uniform(-max_speed_mps * 0.5, max_speed_mps * 0.5)

    def get_wind(self, altitude_m: float, dt: float) -> tuple[float, float]:
        """Return (wind_vx, wind_vy) at the given altitude."""
        # Slow random walk on base wind
        self._base_vx += self._rng.normal(0, 0.1) * dt
        self._base_vx = np.clip(self._base_vx, -self.max_speed, self.max_speed)

        # Gusts decrease with altitude (stronger near ground)
        gust_scale = max(0.0, 1.0 - altitude_m / 1000.0)
        gust = self._rng.normal(0, gust_scale * 2.0)
        return self._base_vx + gust, 0.0

    def reset(self) -> None:
        self._base_vx = self._rng.uniform(-self.max_speed * 0.5, self.max_speed * 0.5)
